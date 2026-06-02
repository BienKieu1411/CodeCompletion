"""
UniXCoder Retriever (Production Implementation)
Pre-indexed chunking + BM25 Pre-filter + Batch Encoding.

Pipeline:
  1. build_index():  AST-based chunking → pre-compute BM25 index (1 lần duy nhất)
  2. retrieve_top_k(): BM25 filter → Dense encoding → cosine similarity → top-k
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


# ── BM25 Pre-filter ──────────────────────────────────────────────────────────

class _BM25Index:
    """Lightweight BM25 index using rank_bm25."""

    def __init__(self, corpus_tokens: List[List[str]]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")
        self._bm25 = BM25Okapi(corpus_tokens)

    def query(self, query_tokens: List[str], top_n: int) -> List[int]:
        scores = self._bm25.get_scores(query_tokens)
        top_n = min(top_n, len(scores))
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]


def _tokenize_for_bm25(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    import re
    return re.findall(r'\w+', text.lower())


# ── AST Chunking ─────────────────────────────────────────────────────────────

def _chunk_content_ast(filename: str, content: str, language: str = "python") -> List[Tuple[str, str]]:
    """
    Chunk file content by AST nodes tại cấp method/function.

    Chiến lược:
    - Top-level function → 1 chunk
    - Class → KHÔNG lấy cả class làm 1 chunk
      + Class header (tên + docstring) → 1 chunk riêng
      + Mỗi method bên trong → 1 chunk riêng, kèm class header prefix
    - Global code (không thuộc function/class) → gom thành 1 chunk

    Ví dụ:
        class Foo(Bar):
            '''Docstring'''
            def method_a(self):     → Chunk: "class Foo(Bar):\\n  def method_a(self):..."
                ...
            def method_b(self):     → Chunk: "class Foo(Bar):\\n  def method_b(self):..."
                ...
        def standalone():           → Chunk: "def standalone():..."
    """
    chunks: List[Tuple[str, str]] = []
    try:
        import tree_sitter_languages
        parser = tree_sitter_languages.get_parser(language)
        tree = parser.parse(bytes(content, "utf8"))
        lines = content.split('\n')

        def _get_text(node) -> str:
            start = node.start_point[0]
            end = node.end_point[0]
            return '\n'.join(lines[start:end + 1])

        def _get_class_header(class_node) -> str:
            """Lấy class header: dòng 'class Foo(Bar):' + docstring nếu có."""
            header_line = lines[class_node.start_point[0]].rstrip()
            # Tìm docstring (node con đầu tiên kiểu expression_statement chứa string)
            docstring = ""
            body = None
            for child in class_node.children:
                if child.type == "block":
                    body = child
                    break
            if body and body.children:
                first_stmt = body.children[0]
                if first_stmt.type == "expression_statement":
                    for sub in first_stmt.children:
                        if sub.type == "string":
                            docstring = sub.text.decode("utf8")
                            break
            if docstring:
                return f"{header_line}\n    {docstring}"
            return header_line

        def _extract_methods_from_class(class_node):
            """Tách từng method trong class thành chunk riêng."""
            class_header = _get_class_header(class_node)
            found_methods = False

            for child in class_node.children:
                if child.type == "block":
                    for block_child in child.children:
                        if block_child.type == "function_definition":
                            found_methods = True
                            method_text = _get_text(block_child)
                            n_lines = block_child.end_point[0] - block_child.start_point[0] + 1
                            if n_lines < 2:
                                continue  # Bỏ method quá ngắn (1 dòng)

                            # Prefix class header để retriever hiểu context
                            chunk_text = f"{class_header}\n\n{method_text}"
                            start = block_child.start_point[0]
                            end = block_child.end_point[0]
                            chunk_id = f"{filename}::L{start}-{end}"
                            chunks.append((chunk_id, chunk_text))

            # Nếu class không có method nào (chỉ có attributes)
            # → lấy cả class làm 1 chunk
            if not found_methods:
                n_lines = class_node.end_point[0] - class_node.start_point[0] + 1
                if 2 <= n_lines <= 100:
                    chunk_text = _get_text(class_node)
                    chunk_id = f"{filename}::L{class_node.start_point[0]}-{class_node.end_point[0]}"
                    chunks.append((chunk_id, chunk_text))

        # ── Traverse top-level nodes ──
        global_lines_used = set()

        for node in tree.root_node.children:
            if node.type == "class_definition":
                _extract_methods_from_class(node)
                # Mark these lines as used
                for ln in range(node.start_point[0], node.end_point[0] + 1):
                    global_lines_used.add(ln)

            elif node.type == "function_definition":
                n_lines = node.end_point[0] - node.start_point[0] + 1
                if n_lines >= 2:
                    chunk_text = _get_text(node)
                    chunk_id = f"{filename}::L{node.start_point[0]}-{node.end_point[0]}"
                    chunks.append((chunk_id, chunk_text))
                for ln in range(node.start_point[0], node.end_point[0] + 1):
                    global_lines_used.add(ln)

            elif node.type == "decorated_definition":
                # Python: @decorator\ndef func / class MyClass
                for child in node.children:
                    if child.type == "function_definition":
                        n_lines = node.end_point[0] - node.start_point[0] + 1
                        if n_lines >= 2:
                            chunk_text = _get_text(node)  # Lấy cả decorator
                            chunk_id = f"{filename}::L{node.start_point[0]}-{node.end_point[0]}"
                            chunks.append((chunk_id, chunk_text))
                    elif child.type == "class_definition":
                        _extract_methods_from_class(child)
                for ln in range(node.start_point[0], node.end_point[0] + 1):
                    global_lines_used.add(ln)

        # ── Gom global code còn lại (constants, module-level logic) ──
        global_code_lines = []
        for i, line in enumerate(lines):
            if i not in global_lines_used and line.strip():
                global_code_lines.append(line)

        # Tách global code thành chunks nhỏ (15 dòng/chunk)
        if global_code_lines:
            for i in range(0, len(global_code_lines), 15):
                block = global_code_lines[i:i + 15]
                chunk_text = '\n'.join(block)
                chunk_id = f"{filename}::global_{i}"
                chunks.append((chunk_id, chunk_text))

        # Final fallback: nếu file không parse được gì
        if not chunks:
            chunks.append((f"{filename}::all", content[:3000]))

    except Exception:
        # Fallback: fixed-size blocks (15 lines each)
        lines = content.split('\n')
        for i in range(0, len(lines), 15):
            chunk_text = '\n'.join(lines[i:i + 15])
            chunk_id = f"{filename}::L{i}-{min(i + 14, len(lines) - 1)}"
            chunks.append((chunk_id, chunk_text))

    return chunks


# ── Pre-built Chunk Index ─────────────────────────────────────────────────────

class ChunkIndex:
    """
    Pre-computed chunk index cho 1 repo.
    Build 1 lần, dùng cho tất cả queries trong repo đó.
    """

    def __init__(self, crossfile_dict: Dict[str, str]):
        """
        Args:
            crossfile_dict: Dict[filename, content] — tất cả file trong repo (trừ file hiện tại)
        """
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []
        self._bm25: Optional[_BM25Index] = None
        self._bm25_tokens: List[List[str]] = []

        # AST-chunk tất cả files 1 lần
        for filename, content in sorted(crossfile_dict.items()):
            lang = "python"
            ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            if ext == "java":
                lang = "java"
            elif ext == "js":
                lang = "javascript"
            elif ext == "ts":
                lang = "typescript"
            file_chunks = _chunk_content_ast(filename, content, lang)
            for cid, ctext in file_chunks:
                self.chunk_ids.append(cid)
                self.chunk_texts.append(ctext)

        # Pre-build BM25 index
        if self.chunk_texts:
            self._bm25_tokens = [_tokenize_for_bm25(t) for t in self.chunk_texts]
            try:
                self._bm25 = _BM25Index(self._bm25_tokens)
            except Exception:
                self._bm25 = None

        logger.debug(f"ChunkIndex built: {len(self.chunk_texts)} chunks from {len(crossfile_dict)} files")

    def __len__(self) -> int:
        return len(self.chunk_texts)

    def bm25_filter(self, query: str, top_n: int = 50) -> Tuple[List[str], List[str]]:
        """BM25 pre-filter → return (filtered_ids, filtered_texts)."""
        if self._bm25 is None or len(self) <= top_n:
            return self.chunk_ids, self.chunk_texts

        query_tokens = _tokenize_for_bm25(query)
        indices = self._bm25.query(query_tokens, top_n)
        return (
            [self.chunk_ids[i] for i in indices],
            [self.chunk_texts[i] for i in indices],
        )


# ── Quantum-inspired Scoring ──────────────────────────────────────────────────

class ZYZ(nn.Module):
    """
    Learnable ZYZ unitary rotation gate from the QIEPSM-style retriever.

    This is a quantum-inspired complex tensor transform implemented in PyTorch;
    it does not execute a real quantum circuit.
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.alpha_param = nn.Parameter(torch.zeros(n_qubits))
        self.beta_param = nn.Parameter(torch.zeros(n_qubits))
        self.gamma_param = nn.Parameter(torch.zeros(n_qubits))
        self.delta_param = nn.Parameter(torch.zeros(n_qubits))
        self.register_buffer(
            "identity",
            self._u_composition(
                torch.zeros(n_qubits),
                torch.zeros(n_qubits),
                torch.zeros(n_qubits),
                torch.zeros(n_qubits),
            ),
        )

    @staticmethod
    def _u_composition(alpha, beta, gamma, delta):
        cos_g = torch.cos(gamma / 2)
        sin_g = torch.sin(gamma / 2)
        e_00 = torch.exp(1j * (alpha - beta / 2 - delta / 2)) * cos_g
        e_01 = -torch.exp(1j * (alpha - beta / 2 + delta / 2)) * sin_g
        e_10 = torch.exp(1j * (alpha + beta / 2 - delta / 2)) * sin_g
        e_11 = torch.exp(1j * (alpha + beta / 2 + delta / 2)) * cos_g
        row0 = torch.cat((e_00.unsqueeze(-1), e_01.unsqueeze(-1)), dim=-1)
        row1 = torch.cat((e_10.unsqueeze(-1), e_11.unsqueeze(-1)), dim=-1)
        return torch.stack((row0, row1), dim=-2)

    @staticmethod
    def _clamp_angle(t, max_val):
        return torch.tanh(t) * max_val / 2 + max_val / 2

    def forward(self, qubits: torch.Tensor, controlled: bool = False) -> torch.Tensor:
        alpha = self._clamp_angle(self.alpha_param, 4 * torch.pi)
        beta = self._clamp_angle(self.beta_param, 4 * torch.pi)
        gamma = self._clamp_angle(self.gamma_param, 4 * torch.pi)
        delta = self._clamp_angle(self.delta_param, 4 * torch.pi)
        unitary = self._u_composition(alpha, beta, gamma, delta)

        if controlled:
            controlled_unitary = (
                F.pad(unitary, (2, 0, 2, 0), "constant", 0.0)
                + F.pad(self.identity, (0, 2, 0, 2), "constant", 0.0)
            )
            return controlled_unitary.matmul(qubits.unsqueeze(-1)).squeeze(-1)

        return unitary.matmul(qubits.unsqueeze(-1)).squeeze(-1)


# ── UniXCoder Retriever ───────────────────────────────────────────────────────

class CoarseDenseRetriever(torch.nn.Module):
    """
    Production UniXCoder retriever with optional quantum-inspired reranking:
    - Pre-indexed chunking (build once, query many)
    - BM25 pre-filter (reduces candidate set)
    - Batch encoding (single forward pass for all chunks)
    - Dense cosine, quantum log-fidelity, or z-normalized hybrid scoring
    """

    N_QUBITS = 256

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        device: str = "cuda",
        max_seq_len: int = 512,
        bm25_top_n: int = 50,
        scoring_mode: str = "dense",
        quantum_alpha: float = 0.5,
    ):
        super().__init__()
        if scoring_mode == "classical":
            scoring_mode = "dense"
        if scoring_mode not in ("dense", "quantum", "hybrid"):
            raise ValueError("scoring_mode must be one of: dense, quantum, hybrid")

        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.max_seq_len = max_seq_len
        self.bm25_top_n = bm25_top_n
        self.scoring_mode = scoring_mode

        init_alpha = max(0.01, min(0.99, float(quantum_alpha)))
        raw_alpha_init = math.log(init_alpha / (1.0 - init_alpha))
        self.raw_quantum_alpha = nn.Parameter(torch.tensor(float(raw_alpha_init)))

        logger.info(f"Loading {model_name} on {self.device} with coarse scoring={self.scoring_mode}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if self.scoring_mode in ("quantum", "hybrid"):
            self._init_quantum_layers()
        self.to(self.device)

        # Cache: repo_key → ChunkIndex
        self._index_cache: Dict[str, ChunkIndex] = {}

    @property
    def quantum_alpha_value(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_quantum_alpha)

    def _init_quantum_layers(self) -> None:
        n_qubits = self.N_QUBITS
        self.c_zyz = ZYZ(n_qubits)
        self.b_zyz = ZYZ(n_qubits)
        self.cb_controlled_zyz = ZYZ(n_qubits)
        self.b_cb_zyz = ZYZ(n_qubits)
        self.a_zyz = ZYZ(n_qubits)
        self.ba_controlled_zyz = ZYZ(n_qubits)

    # ── Index Management ──────────────────────────────────────────────────────

    def build_index(self, crossfile_dict: Dict[str, str], repo_key: str = "default") -> ChunkIndex:
        """
        Pre-chunk + pre-build BM25 index for a repo.
        Call this ONCE per repo, before running queries.
        """
        index = ChunkIndex(crossfile_dict)
        self._index_cache[repo_key] = index
        logger.info(f"Built index for '{repo_key}': {len(index)} chunks")
        return index

    def get_or_build_index(self, crossfile_dict: Dict[str, str], repo_key: str = "default") -> ChunkIndex:
        """Get cached index or build new one."""
        if repo_key not in self._index_cache:
            return self.build_index(crossfile_dict, repo_key)
        return self._index_cache[repo_key]

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _mean_pool_raw(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over non-padding tokens without normalization."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_batch_raw(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode a list of texts into raw pooled embeddings.
        Uses batched forward passes for efficiency.
        Returns: Tensor of shape (len(texts), hidden_dim)
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            outputs = self.model(**tokens)
            embeddings = self._mean_pool_raw(outputs.last_hidden_state, tokens["attention_mask"])
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode a list of texts into L2-normalized embeddings.
        Returns: Tensor of shape (len(texts), hidden_dim)
        """
        return F.normalize(self.encode_batch_raw(texts, batch_size=batch_size), p=2, dim=-1)

    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text. Kept for backward compatibility."""
        return self.encode_batch([text], batch_size=1)

    def _amplitude_encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        ae = torch.tanh(embeddings)
        magnitudes = ae * (torch.pi / 2) + (torch.pi / 2)
        phase_val = (2.0 * torch.pi - 1e-9) / 2.0
        phases = torch.full_like(ae, phase_val)
        amplitude_0 = torch.cos(magnitudes / 2).unsqueeze(-1)
        amplitude_1 = (torch.sin(magnitudes / 2) * torch.exp(1j * phases)).unsqueeze(-1)
        return torch.cat((amplitude_0, amplitude_1), dim=-1)

    def _quantum_transform(self, qubits: torch.Tensor) -> torch.Tensor:
        n_qubits = self.N_QUBITS
        if qubits.shape[1] < 3 * n_qubits:
            return qubits[:, :n_qubits, :]

        a = qubits[:, :n_qubits, :]
        b = qubits[:, n_qubits:2 * n_qubits, :]
        c = qubits[:, 2 * n_qubits:3 * n_qubits, :]

        c_rotated = self.c_zyz(c)
        b_rotated = self.b_zyz(b)
        cb_product = torch.einsum("bni,bnj->bnij", c_rotated, b_rotated)
        cb_state = cb_product.reshape(cb_product.shape[0], n_qubits, 4)
        cb_state = self.cb_controlled_zyz(cb_state, controlled=True)
        b_cb = (
            cb_state[:, :, 0:2].conj() * cb_state[:, :, 0:2]
            + cb_state[:, :, 2:4].conj() * cb_state[:, :, 2:4]
        ).real.sqrt().to(cb_state.dtype)

        b_cb_rotated = self.b_cb_zyz(b_cb)
        a_rotated = self.a_zyz(a)
        ba_product = torch.einsum("bni,bnj->bnij", b_cb_rotated, a_rotated)
        ba_state = ba_product.reshape(ba_product.shape[0], n_qubits, 4)
        ba_state = self.ba_controlled_zyz(ba_state, controlled=True)
        return (
            ba_state[:, :, 0:2].conj() * ba_state[:, :, 0:2]
            + ba_state[:, :, 2:4].conj() * ba_state[:, :, 2:4]
        ).real.sqrt().to(ba_state.dtype)

    @staticmethod
    def _quantum_fidelity(query_q: torch.Tensor, doc_q: torch.Tensor) -> torch.Tensor:
        q_conj = query_q.conj().unsqueeze(0)
        inner = (q_conj[..., 0] * doc_q[..., 0]) + (q_conj[..., 1] * doc_q[..., 1])
        return 2.0 * torch.log(inner.abs() + 1e-12).sum(dim=1)

    def _compute_quantum_scores(self, query_raw: torch.Tensor, doc_raw: torch.Tensor) -> torch.Tensor:
        query_qubits = self._amplitude_encode(query_raw)
        doc_qubits = self._amplitude_encode(doc_raw)
        if hasattr(self, "c_zyz"):
            query_qubits = self._quantum_transform(query_qubits)
            doc_qubits = self._quantum_transform(doc_qubits)
        return self._quantum_fidelity(query_qubits.squeeze(0), doc_qubits)

    @staticmethod
    def _zscore(scores: torch.Tensor) -> torch.Tensor:
        std = scores.std(unbiased=False)
        if bool(torch.isnan(std).item()) or float(std.item()) < 1e-8:
            return torch.zeros_like(scores)
        return (scores - scores.mean()) / std

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_top_k(
        self,
        query: str,
        crossfile_dict: Dict[str, str],
        top_k: int = 3,
        return_aux: bool = False,
        force_indices: Optional[torch.Tensor] = None,
        repo_key: str = "default",
    ):
        """
        Full retrieval pipeline:
          1. Get or build chunk index (pre-computed)
          2. BM25 pre-filter → top-N candidates
          3. Dense/quantum encode (batched) → similarity scoring
          4. Return top-k snippets + logprobs for PPO

        Returns:
            top_snippets: List[str]
            logprobs_tensor: Tensor (shape: top_k) — for PPO training
            aux (optional): Dict with all_logprobs, topk_indices, filenames
        """
        if not crossfile_dict:
            empty_lp = torch.zeros(1, requires_grad=True, device=self.device)
            if return_aux:
                return [], empty_lp, {"all_logprobs": empty_lp, "topk_indices": torch.zeros(1, dtype=torch.long, device=self.device), "filenames": []}
            return [], empty_lp

        # Step 1: Get pre-built index (chunk 1 lần, dùng cho mọi query)
        index = self.get_or_build_index(crossfile_dict, repo_key)

        if len(index) == 0:
            empty_lp = torch.zeros(1, requires_grad=True, device=self.device)
            if return_aux:
                return [], empty_lp, {"all_logprobs": empty_lp, "topk_indices": torch.zeros(1, dtype=torch.long, device=self.device), "filenames": []}
            return [], empty_lp

        # Step 2: BM25 Pre-filter
        filenames, contents = index.bm25_filter(query, top_n=self.bm25_top_n)

        # Step 3: Dense encoding (batched)
        safe_contents = [c[:1500] for c in contents]
        query_raw = self.encode_batch_raw([query[-1500:]], batch_size=1)
        doc_raw = self.encode_batch_raw(safe_contents, batch_size=32)

        # Step 4: Similarity scoring
        query_dense = F.normalize(query_raw, p=2, dim=-1)
        doc_dense = F.normalize(doc_raw, p=2, dim=-1)
        dense_scores = torch.matmul(doc_dense, query_dense.squeeze(0))

        quantum_scores = None
        if self.scoring_mode in ("quantum", "hybrid"):
            quantum_scores = self._compute_quantum_scores(query_raw, doc_raw)

        if self.scoring_mode == "dense":
            scores_tensor = dense_scores
        elif self.scoring_mode == "quantum":
            scores_tensor = quantum_scores
        else:
            alpha = self.quantum_alpha_value
            scores_tensor = alpha * self._zscore(dense_scores) + (1.0 - alpha) * self._zscore(quantum_scores)

        # Softmax → log-probabilities for PPO
        logprobs_all = F.log_softmax(scores_tensor, dim=-1)

        # Step 5: Top-k selection
        if force_indices is not None:
            topk_indices = force_indices
        else:
            k = min(top_k, len(filenames))
            _, topk_indices = torch.topk(scores_tensor, k)

        top_snippets = []
        top_logprobs = []
        for idx in topk_indices:
            idx_val = idx.item()
            sim_val = scores_tensor[idx_val].item()
            top_snippets.append(
                f"### File: {filenames[idx_val]} (score: {sim_val:.3f}) ###\n{contents[idx_val]}"
            )
            top_logprobs.append(logprobs_all[idx_val:idx_val + 1])

        logprobs_tensor = torch.cat(top_logprobs)

        if return_aux:
            aux: Dict[str, Any] = {
                "all_logprobs": logprobs_all,
                "topk_indices": topk_indices,
                "filenames": filenames,
                "scoring_mode": self.scoring_mode,
            }
            if quantum_scores is not None:
                aux["quantum_alpha"] = float(self.quantum_alpha_value.detach().cpu().item())
            return top_snippets, logprobs_tensor, aux
        return top_snippets, logprobs_tensor
