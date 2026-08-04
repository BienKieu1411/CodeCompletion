"""Dense retriever with full fine-tuning and DPO-style preference learning.

Architecture
------------
* **Encoder**: ``microsoft/unixcoder-base`` (or any HF encoder) —
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
        HuggingFace encoder model (e.g. ``microsoft/unixcoder-base``).
    max_length : int
        Maximum token length for the encoder.
    device : str
        Target device.
    """

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        max_length: int = 512,
        device: str = "cuda",
        encode_batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self._device = device
        self.encode_batch_size = max(1, int(encode_batch_size))
        self._load_dtype = (
            torch.bfloat16
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else None
        )

        # Full fine-tune encoder
        load_kwargs = {"trust_remote_code": True}
        if self._load_dtype is not None:
            load_kwargs["torch_dtype"] = self._load_dtype
        self.encoder = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.hidden_size: int = self.encoder.config.hidden_size

        # Reference copy for DPO (frozen snapshot)
        self._reference_encoder: Optional[nn.Module] = None

        # Frozen initial copy for the dense_frozen baseline strategy
        self._initial_encoder: Optional[nn.Module] = None

        self.to(device)

    def _configured_encode_batch_size(self) -> int:
        """Keep tiny test doubles/backward-compatible subclasses working."""
        return max(1, int(getattr(self, "encode_batch_size", 32)))

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
        batch_size: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Encode chunks → (N, hidden_size) matrix."""
        texts = [c.retrieval_text() for c in chunks]
        return self.encode_texts(texts, batch_size=batch_size, encoder=encoder)

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Encode text list → (N, hidden_size) matrix."""
        batch_size = (
            self._configured_encode_batch_size()
            if batch_size is None
            else int(batch_size)
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        all_vecs: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vecs = self._encode_batch(batch, encoder=encoder)
            all_vecs.append(vecs)
        if not all_vecs:
            return torch.zeros(0, self.hidden_size, device=self._device)
        return torch.cat(all_vecs, dim=0)

    def encode_texts_numpy(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode texts → CPU NumPy array.  For ``EmbeddingCache.build_from_chunks``."""
        with torch.no_grad():
            return self.encode_texts(texts, batch_size).cpu().numpy()

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
        top_k: int = 3,
        batch_size: Optional[int] = None,
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
        batch_size: Optional[int] = None,
    ) -> List[CodeChunk]:
        """Return just the top-k CodeChunk objects."""
        return [c for _, c in self.retrieve(query, chunks, top_k, batch_size)]

    def retrieve_with_encoder(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
        top_k: int = 3,
        encoder: Optional[nn.Module] = None,
        batch_size: Optional[int] = None,
    ) -> List[CodeChunk]:
        """Retrieve using a specific encoder (e.g. initial frozen copy)."""
        if not chunks:
            return []
        k = min(max(0, int(top_k)), len(chunks))
        if k == 0:
            return []
        with torch.no_grad():
            q_vec = self._encode_batch([query], encoder=encoder)[0]
            c_vecs = self.encode_chunks(chunks, batch_size=batch_size, encoder=encoder)
            scores = (q_vec.unsqueeze(0) @ c_vecs.T).squeeze(0)
            _, indices = scores.topk(k)
        chunk_list = list(chunks)
        return [chunk_list[i] for i in indices.cpu().tolist()]

    # ── Temperature-based strategy sampling ───────────────────────────────

    def sample_strategies(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
        top_k: int = 3,
        temperatures: Sequence[float] = (0.0, 0.7, 2.0),
        batch_size: Optional[int] = None,
    ) -> List[List[CodeChunk]]:
        """Sample diverse retrieval strategies for DPO pair creation.

        Returns one list of chunks per temperature, plus empty (skip).
        """
        if not chunks:
            return [[]]

        k = min(max(0, int(top_k)), len(chunks))
        if k == 0:
            return [[] for _ in temperatures] + [[]]

        with torch.no_grad():
            q_vec = self.encode_query(query)
            c_vecs = self.encode_chunks(chunks, batch_size=batch_size)
            scores = (q_vec.unsqueeze(0) @ c_vecs.T).squeeze(0)

        chunk_list = list(chunks)
        strategies: List[List[CodeChunk]] = []

        for temp in temperatures:
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
        c_vecs = self.encode_chunks(
            context_chunks, batch_size=self._configured_encode_batch_size()
        )
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
            c_chosen_vecs = self.encode_chunks(
                chosen_chunks, batch_size=self._configured_encode_batch_size()
            )
            s_chosen = gate_log_continue + self.retrieval_score(q_vec, c_chosen_vecs)

        if rejected_is_stop:
            s_rejected = gate_log_stop
        else:
            c_rejected_vecs = self.encode_chunks(
                rejected_chunks, batch_size=self._configured_encode_batch_size()
            )
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
                    chosen_chunks,
                    batch_size=self._configured_encode_batch_size(),
                    encoder=self._reference_encoder,
                ) if self._reference_encoder else c_chosen_vecs.detach()
                s_ref_chosen = ref_gate_log_continue + self.retrieval_score(ref_q_vec, ref_c_chosen)

            if rejected_is_stop:
                s_ref_rejected = ref_gate_log_stop
            else:
                ref_c_rejected = self.encode_chunks(
                    rejected_chunks,
                    batch_size=self._configured_encode_batch_size(),
                    encoder=self._reference_encoder,
                ) if self._reference_encoder else c_rejected_vecs.detach()
                s_ref_rejected = ref_gate_log_continue + self.retrieval_score(ref_q_vec, ref_c_rejected)

        logit = beta * (
            (s_chosen - s_rejected) - (s_ref_chosen - s_ref_rejected)
        )
        return -F.logsigmoid(logit)

    # ── LiPO loss ─────────────────────────────────────────────────────────

    def lipo_loss(
        self,
        query_text: str,
        candidate_chunks_list: List[List[CodeChunk]],
        utilities: List[float],
        tau: float = 1.0,
    ) -> torch.Tensor:
        """Listwise Preference Optimization with utility-derived soft labels.

        Instead of pairwise DPO, LiPO aligns the retriever's score distribution
        with a target distribution derived from context utility values.  This
        uses the full utility spectrum (soft labels) rather than discretising
        into chosen/rejected pairs.

        Parameters
        ----------
        query_text : str
            The retrieval query.
        candidate_chunks_list : list of list of CodeChunk
            One chunk list per candidate strategy.  Empty list = stop strategy.
        utilities : list of float
            Utility U(C) = NLL(stop) - NLL(C) for each candidate.
        tau : float
            Temperature for both the target utility distribution and the
            retriever score distribution.

        Returns
        -------
        loss : scalar tensor
            KL(target_dist || pred_dist).
        """
        if len(candidate_chunks_list) != len(utilities):
            raise ValueError(
                f"Mismatch: {len(candidate_chunks_list)} candidates vs "
                f"{len(utilities)} utilities"
            )
        if tau <= 0:
            raise ValueError("tau must be greater than zero")
        if not all(np.isfinite(value) for value in utilities):
            raise ValueError("utilities must contain only finite values")
        if len(candidate_chunks_list) < 2:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        q_vec = self.encode_query(query_text)

        # Target distribution from utility (soft labels)
        utility_tensor = torch.tensor(
            utilities, device=self._device, dtype=torch.float32
        )
        target_dist = F.softmax(utility_tensor / tau, dim=-1)

        # Retriever score distribution. Candidate strategies frequently share
        # the same top-k chunks (current/oracle/hard-neg). Encode each unique
        # chunk once per LiPO step instead of running the encoder once per
        # strategy; this is a large speedup without changing the objective.
        unique_chunks: List[CodeChunk] = []
        seen_chunk_ids: set[str] = set()
        for chunks in candidate_chunks_list:
            for chunk in chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk.chunk_id)
                    unique_chunks.append(chunk)

        encoded_by_id: Dict[str, torch.Tensor] = {}
        if unique_chunks:
            unique_vecs = self.encode_chunks(
                unique_chunks, batch_size=self._configured_encode_batch_size()
            )
            encoded_by_id = {
                chunk.chunk_id: unique_vecs[index]
                for index, chunk in enumerate(unique_chunks)
            }

        scores: List[torch.Tensor] = []
        for chunks in candidate_chunks_list:
            if not chunks:
                # Stop strategy: score = 0
                scores.append(torch.tensor(0.0, device=self._device))
            else:
                c_vecs = torch.stack(
                    [encoded_by_id[chunk.chunk_id] for chunk in chunks]
                )
                scores.append(self.retrieval_score(q_vec, c_vecs))

        score_tensor = torch.stack(scores)
        log_pred_dist = F.log_softmax(score_tensor / tau, dim=-1)

        # KL divergence: align retriever distribution with utility distribution
        return F.kl_div(
            log_pred_dist.unsqueeze(0),
            target_dist.unsqueeze(0),
            reduction="batchmean",
        )

    # ── Reference management ──────────────────────────────────────────────

    def refresh_reference(self) -> None:
        """Snapshot current encoder as the frozen reference for DPO."""
        self._reference_encoder = copy.deepcopy(self.encoder)
        self._reference_encoder.eval()
        for p in self._reference_encoder.parameters():
            p.requires_grad = False
        logger.info("DenseRetriever: reference snapshot updated")

    def save_initial_copy(self) -> None:
        """Save the pretrained encoder as the frozen baseline strategy."""
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
        load_kwargs = {"trust_remote_code": True}
        if self._load_dtype is not None:
            load_kwargs["torch_dtype"] = self._load_dtype
        self.encoder = AutoModel.from_pretrained(encoder_path, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_path, trust_remote_code=True
        )
        self.to(self._device)
        logger.info("DenseRetriever: loaded from %s", path)
