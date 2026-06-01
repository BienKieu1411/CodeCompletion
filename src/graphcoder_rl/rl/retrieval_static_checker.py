"""
Static Checker (Tier 1 Judge)
Lọc các dự đoán sinh mã bị lỗi cú pháp trước khi tốn GPU chạy LLM-as-a-judge.
"""

import ast

class RetrievalStaticChecker:
    """Kiểm tra tĩnh cấu trúc mã nguồn."""
    
    @staticmethod
    def evaluate_syntax(prediction: str, left_context: str, right_context: str) -> int:
        """
        Kiểm tra tính hợp lệ cú pháp của toàn bộ file sau khi chèn code dự đoán.
        Trả về 1 nếu mã hợp lệ, -1 nếu bị lỗi cú pháp (SyntaxError).
        """
        full_code = left_context + prediction + right_context
        try:
            ast.parse(full_code)
            return 1
        except SyntaxError:
            return -1
        except Exception:
            return -1


StaticChecker = RetrievalStaticChecker
