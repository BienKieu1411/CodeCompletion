"""Dense retriever with full fine-tuning and DPO-style preference learning.

Architecture
------------
* **Encoder**: ``jinaai/jina-code-embeddings-1.5b`` (or any HF encoder) —
  ALL parameters trainable (full fine-tune).
* **Scoring**: cosine similarity between query and chunk embeddings.
* **DPO score**:
      S(q, C_retrieve) = mean(sim(q, snippet_i))
      S(q, C_stop)      = 0
* **DPO loss**: context preference ranking loss with frozen reference snapshot.
  The retrieval gate is trained separately from utility labels in the main
  pipeline so retrieve-vs-retrieve pairs do not accidentally supervise the
  skip/retrieve decision.
* **FAISS index**: rebuilt periodically (after retriever weights change).

Because the encoder is fully fine-tuned, chunk embeddings change every
time the weights update → the FAISS index must be rebuilt per epoch /
per DPO round.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

from co_retrieval.chunking import CodeChunk

logger = logging.getLogger(__name__)


# ── Pooling ───────────────────────────────────────────────────────────────────


def _mean_pool(
    last_hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pooling over non-padding tokens."""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
    return sum_hidden / sum_mask


# ── DenseRetriever ────────────────────────────────────────────────────────────


class DenseRetriever(nn.Module):
    """Full fine-tune bi-encoder retriever with DPO training.

    Parameters
    ----------
    model_name : str
        HuggingFace encoder model (e.g. ``jinaai/jina-code-embeddings-1.5b``).
    max_length : int
        Maximum token length for the encoder.
    device : str
        Target device.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-code-embeddings-1.5b",
        max_length: int = 512,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self._device = device

        # Full fine-tune encoder
        self.encoder = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.hidden_size: int = self.encoder.config.hidden_size

        # Reference copy for DPO (frozen snapshot)
        self._reference_encoder: Optional[nn.Module] = None

        # Frozen initial copy for C_jina baseline strategy
        self._initial_encoder: Optional[nn.Module] = None

        self.to(device)

    # ── Encoding ──────────────────────────────────────────────────────────

    def _encode_batch(
        self,
        texts: List[str],
        encoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Encode a batch of texts → (batch, hidden_size) L2-normalised.

        Parameters
        ----------
        encoder : optional
            If given, uses this encoder (e.g. reference or initial copy)
            instead of the trainable one.
        """
        enc = encoder if encoder is not None else self.encoder
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        if encoder is not None:
            with torch.no_grad():
                outputs = enc(**tokens)
        else:
            outputs = enc(**tokens)

        pooled = _mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
        return F.normalize(pooled, p=2, dim=-1)

    def encode_query(self, left_context: str) -> torch.Tensor:
        """Encode left_context → (hidden_size,) normalised vector."""
        return self._encode_batch([left_context])[0]

    def encode_chunks(
        self,
        chunks: Sequence[CodeChunk],
        batch_size: int = 32,
        encoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Encode chunks → (N, hidden_size) matrix."""
        texts = [c.retrieval_text() for c in chunks]
        return self.encode_texts(texts, batch_size=batch_size, encoder=encoder)

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        encoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Encode text list → (N, hidden_size) matrix."""
        all_vecs: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vecs = self._encode_batch(batch, encoder=encoder)
            all_vecs.append(vecs)
        if not all_vecs:
            return torch.zeros(0, self.hidden_size, device=self._device)
        return torch.cat(all_vecs, dim=0)

    def encode_texts_numpy(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts → CPU NumPy array.  For ``EmbeddingCache.build_from_chunks``."""
        with torch.no_grad():
            return self.encode_texts(texts, batch_size).cpu().numpy()

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
        top_k: int = 3,
        batch_size: int = 32,
    ) -> List[Tuple[float, CodeChunk]]:
        """Score, rank, and return top-k chunks."""
        if not chunks:
            return []
        with torch.no_grad():
            q_vec = self.encode_query(query)
            c_vecs = self.encode_chunks(chunks, batch_size=batch_size)
            scores = (q_vec.unsqueeze(0) @ c_vecs.T).squeeze(0)  # (N,)

        ranked = sorted(
            zip(scores.cpu().tolist(), list(chunks)),
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[: max(0, top_k)]

    def retrieve_chunks(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
        top_k: int = 3,
        batch_size: int = 32,
    ) -> List[CodeChunk]:
        """Return just the top-k CodeChunk objects."""
        return [c for _, c in self.retrieve(query, chunks, top_k, batch_size)]

    def retrieve_with_encoder(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
        top_k: int = 3,
        encoder: Optional[nn.Module] = None,
        batch_size: int = 32,
    ) -> List[CodeChunk]:
        """Retrieve using a specific encoder (e.g. initial frozen copy)."""
        if not chunks:
            return []
        with torch.no_grad():
            q_vec = self._encode_batch([query], encoder=encoder)[0]
            c_vecs = self.encode_chunks(chunks, batch_size=batch_size, encoder=encoder)
            scores = (q_vec.unsqueeze(0) @ c_vecs.T).squeeze(0)
            _, indices = scores.topk(min(top_k, len(chunks)))
        chunk_list = list(chunks)
        return [chunk_list[i] for i in indices.cpu().tolist()]

    # ── Temperature-based strategy sampling ───────────────────────────────

    def sample_strategies(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
        top_k: int = 3,
        temperatures: Sequence[float] = (0.0, 0.7, 2.0),
        batch_size: int = 32,
    ) -> List[List[CodeChunk]]:
        """Sample diverse retrieval strategies for DPO pair creation.

        Returns one list of chunks per temperature, plus empty (skip).
        """
        if not chunks:
            return [[]]

        with torch.no_grad():
            q_vec = self.encode_query(query)
            c_vecs = self.encode_chunks(chunks, batch_size=batch_size)
            scores = (q_vec.unsqueeze(0) @ c_vecs.T).squeeze(0)

        chunk_list = list(chunks)
        strategies: List[List[CodeChunk]] = []

        for temp in temperatures:
            k = min(top_k, len(chunk_list))
            if temp < 1e-6:
                _, indices = scores.topk(k)
                selected = [chunk_list[i] for i in indices.cpu().tolist()]
            else:
                probs = F.softmax(scores / temp, dim=-1)
                indices = torch.multinomial(
                    probs, k, replacement=False
                )
                selected = [chunk_list[i] for i in indices.cpu().tolist()]
            strategies.append(selected)

        strategies.append([])  # skip strategy
        return strategies

    # ── Combined scoring for DPO ──────────────────────────────────────────

    def retrieval_score(
        self,
        query_vec: torch.Tensor,
        chunk_vecs: torch.Tensor,
    ) -> torch.Tensor:
        """mean(sim(q, snippet_i)) — the retrieval component of S(q, C)."""
        if chunk_vecs.shape[0] == 0:
            return torch.tensor(0.0, device=self._device)
        sims = (query_vec.unsqueeze(0) @ chunk_vecs.T).squeeze(0)
        return sims.mean()

    def combined_score(
        self,
        query_text: str,
        context_chunks: List[CodeChunk],
        gate_log_continue: torch.Tensor,
        gate_log_stop: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined score S(q, C).

        This helper is kept for diagnostics or ablations that explicitly want
        a joint retriever/gate score.  The main novelty-aligned training path
        keeps gate supervision separate from retriever DPO.

        If context is non-empty (retrieve):
            S = log P_gate(continue|q) + mean(sim(q, snippet_i))
        If context is empty (stop):
            S = log P_gate(stop|q)
        """
        if not context_chunks:
            return gate_log_stop

        q_vec = self.encode_query(query_text)
        c_vecs = self.encode_chunks(context_chunks)
        r_score = self.retrieval_score(q_vec, c_vecs)
        return gate_log_continue + r_score

    def combined_score_from_vecs(
        self,
        query_vec: torch.Tensor,
        chunk_vecs: torch.Tensor,
        gate_log_continue: torch.Tensor,
        gate_log_stop: torch.Tensor,
        is_stop: bool = False,
    ) -> torch.Tensor:
        """Like ``combined_score`` but from pre-computed vectors."""
        if is_stop:
            return gate_log_stop
        r_score = self.retrieval_score(query_vec, chunk_vecs)
        return gate_log_continue + r_score

    # ── DPO loss ──────────────────────────────────────────────────────────

    def dpo_loss(
        self,
        query_text: str,
        chosen_chunks: List[CodeChunk],
        rejected_chunks: List[CodeChunk],
        gate: Optional[nn.Module] = None,
        beta: float = 0.1,
        chosen_is_stop: bool = False,
        rejected_is_stop: bool = False,
        include_gate_score: bool = False,
    ) -> torch.Tensor:
        """Compute DPO-style ranking loss.

        L = -log σ(β × [(S_θ(q,C⁺) - S_θ(q,C⁻)) - (S_ref(q,C⁺) - S_ref(q,C⁻))])

        By default, gradients flow only through the retriever encoder.  The
        main pipeline trains the gate separately from utility-derived labels.
        ``include_gate_score=True`` is reserved for ablations.
        """
        # Current policy scores
        q_vec = self.encode_query(query_text)
        if include_gate_score:
            if gate is None:
                raise ValueError("gate is required when include_gate_score=True")
            gate_log_continue, gate_log_stop = gate.log_probs(q_vec)
        else:
            gate_log_continue = q_vec.new_zeros(())
            gate_log_stop = q_vec.new_zeros(())

        if chosen_is_stop:
            s_chosen = gate_log_stop
        else:
            c_chosen_vecs = self.encode_chunks(chosen_chunks)
            s_chosen = gate_log_continue + self.retrieval_score(q_vec, c_chosen_vecs)

        if rejected_is_stop:
            s_rejected = gate_log_stop
        else:
            c_rejected_vecs = self.encode_chunks(rejected_chunks)
            s_rejected = gate_log_continue + self.retrieval_score(q_vec, c_rejected_vecs)

        # Reference policy scores (frozen)
        with torch.no_grad():
            if self._reference_encoder is not None:
                ref_q_vec = self._encode_batch([query_text], encoder=self._reference_encoder)[0]
            else:
                ref_q_vec = q_vec.detach()

            ref_gate_log_continue = gate_log_continue.detach()
            ref_gate_log_stop = gate_log_stop.detach()

            if chosen_is_stop:
                s_ref_chosen = ref_gate_log_stop
            else:
                ref_c_chosen = self.encode_chunks(
                    chosen_chunks, encoder=self._reference_encoder
                ) if self._reference_encoder else c_chosen_vecs.detach()
                s_ref_chosen = ref_gate_log_continue + self.retrieval_score(ref_q_vec, ref_c_chosen)

            if rejected_is_stop:
                s_ref_rejected = ref_gate_log_stop
            else:
                ref_c_rejected = self.encode_chunks(
                    rejected_chunks, encoder=self._reference_encoder
                ) if self._reference_encoder else c_rejected_vecs.detach()
                s_ref_rejected = ref_gate_log_continue + self.retrieval_score(ref_q_vec, ref_c_rejected)

        logit = beta * (
            (s_chosen - s_rejected) - (s_ref_chosen - s_ref_rejected)
        )
        return -F.logsigmoid(logit)

    # ── Reference management ──────────────────────────────────────────────

    def refresh_reference(self) -> None:
        """Snapshot current encoder as the frozen reference for DPO."""
        self._reference_encoder = copy.deepcopy(self.encoder)
        self._reference_encoder.eval()
        for p in self._reference_encoder.parameters():
            p.requires_grad = False
        logger.info("DenseRetriever: reference snapshot updated")

    def save_initial_copy(self) -> None:
        """Save the pretrained encoder as frozen baseline (for C_jina strategy)."""
        self._initial_encoder = copy.deepcopy(self.encoder)
        self._initial_encoder.eval()
        for p in self._initial_encoder.parameters():
            p.requires_grad = False
        logger.info("DenseRetriever: initial (pretrained) copy saved")

    @property
    def initial_encoder(self) -> Optional[nn.Module]:
        return self._initial_encoder

    # ── Serialisation ─────────────────────────────────────────────────────

    def save_pretrained(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(os.path.join(path, "encoder"))
        self.tokenizer.save_pretrained(os.path.join(path, "encoder"))
        logger.info("DenseRetriever: saved to %s", path)

    def load_pretrained(self, path: str) -> None:
        import os
        encoder_path = os.path.join(path, "encoder")
        self.encoder = AutoModel.from_pretrained(
            encoder_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_path, trust_remote_code=True
        )
        self.to(self._device)
        logger.info("DenseRetriever: loaded from %s", path)
