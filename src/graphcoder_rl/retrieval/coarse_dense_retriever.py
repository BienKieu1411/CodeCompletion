"""
UniXCoder Retriever (Production Implementation)
Pre-indexed chunking + BM25 Pre-filter + Batch Encoding.

Pipeline:
  1. build_index():  AST-based chunking → pre-compute BM25 index (1 lần duy nhất)
  2. retrieve_top_k(): BM25 filter → Dense encoding → cosine similarity → top-k
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import torch
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


# ── UniXCoder Retriever ───────────────────────────────────────────────────────

class CoarseDenseRetriever(torch.nn.Module):
    """
    Production UniXCoder retriever with:
    - Pre-indexed chunking (build once, query many)
    - BM25 pre-filter (reduces candidate set)
    - Batch encoding (single forward pass for all chunks)
    - Mean pooling + L2 normalization
    """

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        device: str = "cuda",
        max_seq_len: int = 512,
        bm25_top_n: int = 50,
    ):
        super().__init__()
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.max_seq_len = max_seq_len
        self.bm25_top_n = bm25_top_n

        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

        # Cache: repo_key → ChunkIndex
        self._index_cache: Dict[str, ChunkIndex] = {}

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
        """Mean pooling over non-padding tokens, then L2 normalize."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_emb = sum_embeddings / sum_mask
        return F.normalize(mean_emb, p=2, dim=-1)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode a list of texts into L2-normalized embeddings.
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
            embeddings = self._mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text. Kept for backward compatibility."""
        return self.encode_batch([text], batch_size=1)

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
          3. Dense encode (batched) → cosine similarity
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
        query_emb = self.encode_batch([query[-1500:]], batch_size=1)    # shape: (1, dim)
        doc_embs = self.encode_batch(safe_contents, batch_size=32)       # shape: (N, dim)

        # Step 4: Cosine similarity (already L2-normalized → dot product)
        scores_tensor = torch.matmul(doc_embs, query_emb.squeeze(0))  # shape: (N,)

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
