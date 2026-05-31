"""
Deterministic Query Probing (DQP)

Thay thế Stochastic Query Augmentation của AlignCoder.

Vấn đề AlignCoder:
  - Sample N completions (T=0.8) → ghép thành query mở rộng
  - Ảo giác tích lũy: ε_n = n × (1 - p_s) tăng tuyến tính

Giải pháp GraphFRL:
  - Greedy decode 1 lần (T=0) → extract identifiers → bổ sung query
  - Không ảo giác: identifiers hoặc đúng hoặc sai, không "nửa đúng"
  - Nhanh hơn: 1 forward pass thay vì N forward passes

Thuật toán:
  Q_enhanced = Q_original + Identifiers(Greedy_Decode(Q_original, LLM))
"""

import re
import logging
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Python/Java keywords to exclude from identifiers
_KEYWORDS = {
    # Python
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
    'try', 'while', 'with', 'yield', 'self', 'cls',
    # Common builtins (noisy, not useful for retrieval)
    'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict',
    'set', 'tuple', 'bool', 'type', 'isinstance', 'enumerate', 'zip',
    'map', 'filter', 'sorted', 'reversed', 'open', 'super',
    # Java
    'public', 'private', 'protected', 'static', 'final', 'void',
    'class', 'interface', 'extends', 'implements', 'new', 'this',
    'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
    'break', 'continue', 'try', 'catch', 'finally', 'throw', 'throws',
    'null', 'true', 'false',
}

_IDENTIFIER_RE = re.compile(r'[_a-zA-Z][_a-zA-Z0-9]{2,}')  # Min 3 chars to filter noise


def extract_identifiers_from_code(code: str) -> List[str]:
    """
    Extract unique meaningful identifiers from code.
    Filters out keywords, builtins, and too-short names.
    """
    # Remove string literals and comments
    code_clean = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code_clean = re.sub(r"'''.*?'''", '', code_clean, flags=re.DOTALL)
    code_clean = re.sub(r'"[^"]*"', '', code_clean)
    code_clean = re.sub(r"'[^']*'", '', code_clean)
    code_clean = re.sub(r'#.*', '', code_clean)
    code_clean = re.sub(r'//.*', '', code_clean)

    tokens = _IDENTIFIER_RE.findall(code_clean)

    seen: Set[str] = set()
    result: List[str] = []
    for t in tokens:
        if t not in _KEYWORDS and t not in seen and not t.startswith('__'):
            seen.add(t)
            result.append(t)

    return result


class DeterministicQueryProber:
    """
    Deterministic Query Probing (DQP).

    Thay thế stochastic sampling của AlignCoder bằng:
    1. Greedy decode (T=0) → dự đoán tất định
    2. Extract identifiers từ dự đoán
    3. Bổ sung identifiers vào query gốc

    So sánh:
        AlignCoder:  Q' = Q + concat(sample_1, sample_2, ..., sample_N)
                     → N forward passes, ảo giác tích lũy
        GraphFRL:    Q' = Q + " " + join(identifiers(greedy(Q)))
                     → 1 forward pass, không ảo giác

    Mệnh đề (Claim):
        Cho p_s là xác suất token đúng:
        - AlignCoder: P(query_sạch) = p_s^(n×L)  (giảm theo hàm mũ)
        - GraphFRL:   P(query_sạch) = 1 - (1-p_id)  (chỉ phụ thuộc identifier accuracy)
        Với p_id >> p_s^n khi n > 1.
    """

    def __init__(self, max_probe_tokens: int = 50):
        """
        Args:
            max_probe_tokens: Số token tối đa để greedy decode
        """
        self.max_probe_tokens = max_probe_tokens

    def probe(
        self,
        query: str,
        llm=None,
        predicted_text: Optional[str] = None,
    ) -> str:
        """
        Tạo enhanced query bằng cách bổ sung identifiers từ dự đoán.

        Args:
            query: Left context gốc
            llm: DeepSeekGenerator instance (nếu predicted_text chưa có)
            predicted_text: Nếu đã có kết quả generate, truyền vào trực tiếp

        Returns:
            Enhanced query string
        """
        # Bước 1: Lấy dự đoán greedy (nếu chưa có)
        if predicted_text is None:
            if llm is None:
                return query
            try:
                # Greedy decode: temperature=0, chỉ cần identifiers
                predicted_text, _, _ = llm.generate_with_attention(
                    query, retrieved_tokens_len=0, max_new_tokens=self.max_probe_tokens
                )
            except Exception as e:
                logger.debug(f"Probe generation failed: {e}")
                return query

        if not predicted_text or not predicted_text.strip():
            return query

        # Bước 2: Extract identifiers từ dự đoán
        probe_identifiers = extract_identifiers_from_code(predicted_text)

        if not probe_identifiers:
            return query

        # Bước 3: Lọc identifiers đã có trong query (chỉ giữ cái mới)
        query_identifiers = set(extract_identifiers_from_code(query))
        new_identifiers = [id_ for id_ in probe_identifiers if id_ not in query_identifiers]

        if not new_identifiers:
            return query

        # Bước 4: Bổ sung vào query (append dưới dạng comment để không phá syntax)
        # Giới hạn số identifiers để tránh query quá dài
        new_identifiers = new_identifiers[:15]
        probe_suffix = "\n# Probed identifiers: " + ", ".join(new_identifiers)

        logger.debug(f"DQP found {len(new_identifiers)} new identifiers: {new_identifiers[:5]}...")
        return query + probe_suffix

    def enhance_query_for_retrieval(
        self,
        query: str,
        llm=None,
        predicted_text: Optional[str] = None,
    ) -> str:
        """
        Giống probe() nhưng trả về dạng phẳng (flat) cho BM25/dense retrieval.
        Không thêm format comment, chỉ append identifiers thuần.
        """
        if predicted_text is None:
            if llm is None:
                return query
            try:
                predicted_text, _, _ = llm.generate_with_attention(
                    query, retrieved_tokens_len=0, max_new_tokens=self.max_probe_tokens
                )
            except Exception:
                return query

        if not predicted_text or not predicted_text.strip():
            return query

        probe_identifiers = extract_identifiers_from_code(predicted_text)
        query_identifiers = set(extract_identifiers_from_code(query))
        new_identifiers = [id_ for id_ in probe_identifiers if id_ not in query_identifiers]

        if not new_identifiers:
            return query

        return query + " " + " ".join(new_identifiers[:15])
