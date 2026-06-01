"""
Hybrid Quantum-Classical Retriever (QIEPSM)
Pipeline:
  1. build_index():  AST-based chunking → pre-compute BM25 index (1 lần duy nhất)
  2. retrieve_top_k(): BM25 filter → Dense/Quantum encoding → similarity → top-k
     - Classical mode:  L2 norm + Cosine Similarity (giống UniXCoder gốc)
     - Quantum mode:    Amplitude Encoding + ZYZ Gates + Quantum Fidelity
     - Hybrid mode:     Kết hợp cả hai với learnable alpha (sigmoid-clamped)
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


# ── ZYZ Unitary Rotation Gate (from QIEPSM) ──────────────────────────────────

class ZYZ(nn.Module):
    """
    Learnable ZYZ unitary rotation gate (quantum-inspired, NOT actual quantum).
    
    Mỗi qubit được xoay bởi: U = e^(iα) · Rz(β) · Ry(γ) · Rz(δ)
    Đây là ma trận unitary 2×2 với 4 tham số learnable per qubit.
    
    Tham khảo: QIEPSM paper, Section III.
    """
    
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        # 4 learnable rotation angles per qubit
        self.alpha_param = nn.Parameter(torch.randn(n_qubits))
        self.beta_param = nn.Parameter(torch.randn(n_qubits))
        self.gamma_param = nn.Parameter(torch.randn(n_qubits))
        self.delta_param = nn.Parameter(torch.randn(n_qubits))
        # Pre-compute identity for controlled variant
        self.register_buffer(
            "identity",
            self._u_composition(
                torch.zeros(n_qubits), torch.zeros(n_qubits),
                torch.zeros(n_qubits), torch.zeros(n_qubits),
            ),
        )

    @staticmethod
    def _u_composition(alpha, beta, gamma, delta):
        """Build 2×2 unitary matrix U per qubit from 4 rotation angles."""
        cos_g = torch.cos(gamma / 2)
        sin_g = torch.sin(gamma / 2)
        e_00 = torch.exp(1j * (alpha - beta / 2 - delta / 2)) * cos_g
        e_01 = -torch.exp(1j * (alpha - beta / 2 + delta / 2)) * sin_g
        e_10 = torch.exp(1j * (alpha + beta / 2 - delta / 2)) * sin_g
        e_11 = torch.exp(1j * (alpha + beta / 2 + delta / 2)) * cos_g
        row0 = torch.cat((e_00.unsqueeze(-1), e_01.unsqueeze(-1)), dim=-1)
        row1 = torch.cat((e_10.unsqueeze(-1), e_11.unsqueeze(-1)), dim=-1)
        return torch.stack((row0, row1), dim=-2)  # (..., n_qubits, 2, 2)

    @staticmethod
    def _clamp_angle(t, max_val):
        """Smooth clamping via tanh → angle ∈ [0, max_val]."""
        return torch.tanh(t) * max_val / 2 + max_val / 2

    def forward(self, qubits: torch.Tensor, controlled: bool = False) -> torch.Tensor:
        """
        Apply unitary rotation to qubit states.
        
        Args:
            qubits: (..., n_qubits, 2) or (..., n_qubits, 4) for controlled
            controlled: If True, apply Controlled-U (4×4 block diagonal)
        Returns:
            Rotated qubit states, same shape as input
        """
        a = self._clamp_angle(self.alpha_param, 4 * torch.pi)
        b = self._clamp_angle(self.beta_param, 4 * torch.pi)
        g = self._clamp_angle(self.gamma_param, 4 * torch.pi)
        d = self._clamp_angle(self.delta_param, 4 * torch.pi)
        U = self._u_composition(a, b, g, d)  # (n_qubits, 2, 2)

        if controlled:
            # Controlled-U: [[I, 0], [0, U]] as 4×4 block
            CU = (
                F.pad(U, (2, 0, 2, 0), "constant", 0.0)
                + F.pad(self.identity, (0, 2, 0, 2), "constant", 0.0)
            )
            return CU.matmul(qubits.unsqueeze(-1)).squeeze(-1)

        return U.matmul(qubits.unsqueeze(-1)).squeeze(-1)


# ── Hybrid Quantum-Classical Retriever ────────────────────────────────────────

class QuantumUniXCoderRetriever(nn.Module):
    """
    Production Quantum-Inspired Hybrid Retriever.

    Modes:
      - "classical": L2 norm + cosine similarity (tương đương UniXCoder gốc)
      - "quantum":   Amplitude encoding + ZYZ gates + quantum fidelity
      - "hybrid":    sigmoid(α) * classical + (1 - sigmoid(α)) * quantum

    Quantum-inspired approach: Mượn công thức toán cơ học lượng tử
    (Bloch sphere, unitary gates, fidelity) nhưng chạy hoàn toàn
    trên GPU bằng PyTorch complex tensors. KHÔNG dùng mạch lượng tử thật.
    """

    # Số qubits cho mỗi nhóm trong quantum circuit mô phỏng
    # UniXCoder hidden_dim = 768 → 3 nhóm × 256 qubits
    N_QUBITS = 256

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        device: str = "cuda",
        max_seq_len: int = 512,
        bm25_top_n: int = 50,
        mode: str = "hybrid",
        alpha: float = 0.5,
    ):
        """
        Args:
            model_name: HuggingFace model name for the encoder backbone.
            device: "cuda" or "cpu".
            max_seq_len: Maximum sequence length for tokenization.
            bm25_top_n: Number of BM25 pre-filter candidates.
            mode: "classical", "quantum", or "hybrid".
            alpha: Initial mixing weight for hybrid mode (will be sigmoid-clamped).
                   alpha → sigmoid(raw_alpha) ∈ (0, 1)
                   Higher alpha = more classical, lower = more quantum.
        """
        super().__init__()
        assert mode in ("classical", "quantum", "hybrid"), \
            f"mode must be 'classical', 'quantum', or 'hybrid', got '{mode}'"

        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.max_seq_len = max_seq_len
        self.bm25_top_n = bm25_top_n
        self.mode = mode

        # ── Learnable alpha (sigmoid-clamped to stay in [0, 1]) ──
        # Inverse sigmoid of desired initial alpha → raw parameter
        _init_alpha = max(0.01, min(0.99, alpha))  # clamp to avoid inf
        raw_alpha_init = math.log(_init_alpha / (1.0 - _init_alpha))
        self.raw_alpha = nn.Parameter(torch.tensor(float(raw_alpha_init)))

        # ── Encoder backbone ──
        logger.info(f"Loading {model_name} on {self.device} | Mode: {self.mode}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # ── Quantum-inspired layers (ZYZ gates) ──
        # Chỉ khởi tạo khi cần (quantum hoặc hybrid mode)
        if self.mode in ("quantum", "hybrid"):
            self._init_quantum_layers()

        # ── Move ALL parameters + buffers (encoder + ZYZ gates) to device ──
        # Phải gọi self.to() thay vì chỉ self.model.to() để đảm bảo
        # ZYZ params, identity buffer, raw_alpha đều trên đúng device.
        self.to(self.device)

        # ── Index cache (không phải nn parameter, không cần move) ──
        self._index_cache: Dict[str, ChunkIndex] = {}

    def _init_quantum_layers(self):
        """
        Khởi tạo ZYZ gates theo kiến trúc QIEPSM compressed.
        
        Pipeline quantum circuit mô phỏng:
        768D embedding → chia 3 nhóm: a(256), b(256), c(256)
            → Encode thành qubit states
            → ZYZ(c), ZYZ(b) → tensor product → Controlled-U → cb
            → Partial trace → ZYZ(b_cb), ZYZ(a) → tensor product → Controlled-U → ba
            → Partial trace → final state (256 qubits)
        """
        n = self.N_QUBITS  # 256

        # Gates cho round 1: entangle c và b
        self.c_zyz = ZYZ(n)
        self.b_zyz = ZYZ(n)
        self.cb_controlled_zyz = ZYZ(n)  # Controlled-U on cb pair

        # Gates cho round 2: entangle (result of cb) và a
        self.b_cb_zyz = ZYZ(n)
        self.a_zyz = ZYZ(n)
        self.ba_controlled_zyz = ZYZ(n)  # Controlled-U on ba pair

    @property
    def alpha_value(self) -> torch.Tensor:
        """Trả về alpha ∈ (0, 1) qua sigmoid, đảm bảo luôn hợp lệ."""
        return torch.sigmoid(self.raw_alpha)

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

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over non-padding tokens (NO L2 normalization)."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_batch_raw(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode a list of texts into raw (non-normalized) pooled embeddings.
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
            embeddings = self._mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode → L2 normalize. Backward compatible with UniXCoderRetriever.
        Returns: Tensor of shape (len(texts), hidden_dim)
        """
        raw = self.encode_batch_raw(texts, batch_size)
        return F.normalize(raw, p=2, dim=-1)

    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text. Backward compatible."""
        return self.encode_batch([text], batch_size=1)

    # ── Quantum Core Methods ──────────────────────────────────────────────────

    def _amplitude_encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode dense real vectors → qubit states on Bloch sphere.

        Theo QIEPSM paper:
          θ = tanh(emb) * π/2 + π/2        → θ ∈ [0, π]
          φ = (2π - ε) / 2 ≈ π             → φ ≈ π (fixed phase)
          |ψ⟩ = [cos(θ/2), sin(θ/2)·e^(iφ)]

        Args:
            embeddings: (batch, dim) — raw pooled embeddings
        Returns:
            (batch, dim, 2) complex tensor — qubit states
        """
        ae = torch.tanh(embeddings)

        # Magnitudes: θ ∈ [0, π]
        mags = ae * (torch.pi / 2) + (torch.pi / 2)

        # Phases: φ = (2π - ε)/2 ≈ π (matching QIEPSM exactly)
        # QIEPSM: be*(2*pi - 10^-9)/2 + ones*(2*pi - 10^-9)/2
        # be = zeros → phases = (2π - ε)/2 ≈ π
        eps = 1e-9
        phase_val = (2.0 * torch.pi - eps) / 2.0
        phases = torch.full_like(ae, phase_val)

        # Qubit state: |ψ⟩ = [cos(θ/2), sin(θ/2)·e^(iφ)]
        a = torch.cos(mags / 2).unsqueeze(-1)                          # (batch, dim, 1) real
        b = (torch.sin(mags / 2) * torch.exp(1j * phases)).unsqueeze(-1)  # (batch, dim, 1) complex

        return torch.cat((a, b), dim=-1)  # (batch, dim, 2) complex

    def _quantum_transform(self, qubits: torch.Tensor) -> torch.Tensor:
        """
        Apply ZYZ quantum circuit (mô phỏng) theo QIEPSM compressed architecture.

        Input:  (batch, 768, 2) → 768 qubits
        Output: (batch, 256, 2) → 256 qubits (compressed via entanglement)

        Circuit:
          Split 768 qubits → a(256), b(256), c(256)
          Round 1: ZYZ(c) ⊗ ZYZ(b) → tensor product → CU → partial trace → b_cb
          Round 2: ZYZ(b_cb) ⊗ ZYZ(a) → tensor product → CU → partial trace → a_ba
        """
        n = self.N_QUBITS  # 256

        # Split into 3 groups
        a = qubits[:, :n, :]          # (batch, 256, 2)
        b = qubits[:, n:2*n, :]       # (batch, 256, 2)
        c = qubits[:, 2*n:3*n, :]     # (batch, 256, 2)

        # Nếu embedding dim < 768 (chưa đủ 3 nhóm), pad zeros
        if qubits.shape[1] < 3 * n:
            # Fallback: chỉ dùng amplitude encoding, không qua ZYZ
            return qubits[:, :n, :]

        # ── Round 1: Entangle c và b ──
        # Apply ZYZ gates independently
        c_rotated = self.c_zyz(c)    # (batch, 256, 2)
        b_rotated = self.b_zyz(b)    # (batch, 256, 2)

        # Tensor product: |c⟩ ⊗ |b⟩ → 4D state
        # einsum: (batch, n, i) × (batch, n, j) → (batch, n, i, j) → reshape → (batch, n, 4)
        cb_product = torch.einsum('bni,bnj->bnij', c_rotated, b_rotated)
        cb_state = cb_product.reshape(cb_product.shape[0], n, 4)  # (batch, 256, 4)

        # Apply Controlled-Unitary
        cb_state = self.cb_controlled_zyz(cb_state, controlled=True)

        # Partial trace: đo qubit b, giữ qubit c → trả về qubit state 2D
        # P(|0⟩_b) = |ψ_00|² + |ψ_01|²,  P(|1⟩_b) = |ψ_10|² + |ψ_11|²
        # Reduced state amplitudes:
        # Note: conj(x)*x luôn real ≥ 0, nhưng PyTorch giữ complex dtype
        # → .real để tránh complex artifact từ floating point noise trước sqrt()
        b_cb = (
            cb_state[:, :, 0:2].conj() * cb_state[:, :, 0:2]
            + cb_state[:, :, 2:4].conj() * cb_state[:, :, 2:4]
        ).real.sqrt().to(cb_state.dtype)  # (batch, 256, 2) complex

        # ── Round 2: Entangle b_cb và a ──
        b_cb_rotated = self.b_cb_zyz(b_cb)
        a_rotated = self.a_zyz(a)

        ba_product = torch.einsum('bni,bnj->bnij', b_cb_rotated, a_rotated)
        ba_state = ba_product.reshape(ba_product.shape[0], n, 4)

        ba_state = self.ba_controlled_zyz(ba_state, controlled=True)

        # Partial trace again (same .real trick)
        a_ba = (
            ba_state[:, :, 0:2].conj() * ba_state[:, :, 0:2]
            + ba_state[:, :, 2:4].conj() * ba_state[:, :, 2:4]
        ).real.sqrt().to(ba_state.dtype)  # (batch, 256, 2) — final quantum state

        return a_ba

    def _quantum_fidelity(self, query_q: torch.Tensor, doc_q: torch.Tensor) -> torch.Tensor:
        """
        Compute Quantum Fidelity giữa query và documents.

        QIEPSM gốc:  F = |Π_k ⟨ψ_q^k|ψ_d^k⟩|²   (prod trước, abs sau)
        Bản này:      log_F ≈ 2 * Σ_k log|⟨ψ_q^k|ψ_d^k⟩|  (abs trước, sum log)

        Args:
            query_q: (n_qubits, 2) complex — query quantum state
            doc_q:   (N, n_qubits, 2) complex — document quantum states
        Returns:
            (N,) real tensor — log-fidelity scores (NOT true fidelity)
        """
        q_conj = query_q.conj().unsqueeze(0)  # (1, n_qubits, 2)

        # Inner product per qubit: ⟨ψ_q^k|ψ_d^k⟩ (complex scalar per qubit)
        inner = (q_conj[..., 0] * doc_q[..., 0]) + (q_conj[..., 1] * doc_q[..., 1])
        # inner shape: (N, n_qubits) complex

        # Log-space computation: abs per qubit → log → sum
        # Equivalent to: log(Π_k |inner_k|²) = 2 * Σ_k log|inner_k|
        # NOTE: QIEPSM gốc dùng: |Π_k inner_k|² (phase-aware)
        #       Bản này dùng:     Π_k |inner_k|² (phase-blind)
        log_abs_inner = torch.log(inner.abs() + 1e-12)  # (N, n_qubits)
        log_fidelity = 2.0 * log_abs_inner.sum(dim=1)   # (N,)

        return log_fidelity  # log-space scores, suitable for softmax/ranking

    def _compute_quantum_scores(self, query_raw: torch.Tensor, doc_raw: torch.Tensor) -> torch.Tensor:
        """
        Full quantum scoring pipeline:
          raw embeddings → amplitude encode → ZYZ transform → fidelity

        Args:
            query_raw: (1, 768) raw query embedding
            doc_raw:   (N, 768) raw document embeddings
        Returns:
            (N,) quantum similarity scores (log-fidelity)
        """
        # Step 1: Amplitude encode
        query_qubits = self._amplitude_encode(query_raw)  # (1, 768, 2)
        doc_qubits = self._amplitude_encode(doc_raw)       # (N, 768, 2)

        # Step 2: ZYZ quantum transform (compress 768 → 256 qubits)
        if hasattr(self, 'c_zyz'):
            query_qubits = self._quantum_transform(query_qubits)  # (1, 256, 2)
            doc_qubits = self._quantum_transform(doc_qubits)       # (N, 256, 2)

        # Step 3: Quantum fidelity
        scores = self._quantum_fidelity(
            query_qubits.squeeze(0),  # (256, 2)
            doc_qubits,                # (N, 256, 2)
        )

        return scores  # (N,) log-fidelity

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
          3. Dense/Quantum encode → similarity scoring
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

        # Step 1: Get pre-built index
        index = self.get_or_build_index(crossfile_dict, repo_key)

        if len(index) == 0:
            empty_lp = torch.zeros(1, requires_grad=True, device=self.device)
            if return_aux:
                return [], empty_lp, {"all_logprobs": empty_lp, "topk_indices": torch.zeros(1, dtype=torch.long, device=self.device), "filenames": []}
            return [], empty_lp

        # Step 2: BM25 Pre-filter
        filenames, contents = index.bm25_filter(query, top_n=self.bm25_top_n)

        # Step 3: Encode
        safe_contents = [c[:1500] for c in contents]
        query_emb_raw = self.encode_batch_raw([query[-1500:]], batch_size=1)   # (1, 768)
        doc_embs_raw = self.encode_batch_raw(safe_contents, batch_size=32)      # (N, 768)

        # Step 4: Compute scores based on mode
        classical_scores = None
        quantum_scores = None

        if self.mode in ("classical", "hybrid"):
            query_c = F.normalize(query_emb_raw, p=2, dim=-1)
            doc_c = F.normalize(doc_embs_raw, p=2, dim=-1)
            classical_scores = torch.matmul(doc_c, query_c.squeeze(0))  # (N,) ∈ [-1, 1]

        if self.mode in ("quantum", "hybrid"):
            quantum_scores = self._compute_quantum_scores(query_emb_raw, doc_embs_raw)  # (N,) log-fidelity

        # Combine scores
        if self.mode == "classical":
            scores_tensor = classical_scores
        elif self.mode == "quantum":
            scores_tensor = quantum_scores
        else:  # hybrid
            # Z-score normalize cả hai về cùng scale trước khi mix.
            # Classical: cosine ∈ [-1, 1], Quantum: log-fidelity ∈ [-500, 0]
            # Nếu mix trực tiếp, quantum sẽ dominate hoàn toàn do magnitude lớn hơn.
            alpha = self.alpha_value  # sigmoid-clamped ∈ (0, 1)
            c_std = classical_scores.std()
            q_std = quantum_scores.std()
            if c_std < 1e-8 or q_std < 1e-8:
                # Edge case: N ≤ 1 candidate hoặc tất cả scores giống nhau
                # → z-score vô nghĩa, fallback dùng classical scores
                scores_tensor = classical_scores
            else:
                c_norm = (classical_scores - classical_scores.mean()) / c_std
                q_norm = (quantum_scores - quantum_scores.mean()) / q_std
                scores_tensor = alpha * c_norm + (1.0 - alpha) * q_norm

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
            }
            return top_snippets, logprobs_tensor, aux
        return top_snippets, logprobs_tensor
