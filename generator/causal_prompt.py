"""
Causal Prompt Generator
Dùng cho chuẩn Repository-Level Code Completion trong môi trường IDE thực tế.
"""

class CausalPromptGenerator:
    """Tạo Prompt theo chuẩn Causal LM (Next-Token Prediction)."""
    def __init__(self, model_name: str = "deepseek-coder", max_length: int = 2048):
        self.max_length = max_length
        self.model_name = model_name

    def construct_prompt(self, retrieved_context: str, left_context: str, file_path: str = "current_file.py") -> str:
        """
        Cấu trúc:
        # file path: <tên file retrieved>
        <nội dung retrieved>
        
        # file path: <tên file hiện tại>
        <left_context>
        """
        # retrieved_context đã được format dạng: ### Snippet: filename ### \n content
        # Ta sẽ đổi format comment tùy theo ngôn ngữ (Python dùng #)
        prompt = ""
        
        if retrieved_context.strip():
            # Dán ngữ cảnh truy xuất lên trên cùng
            prompt += f"{retrieved_context}\n\n"
            
        # Dán đường dẫn file hiện tại để mô hình hiểu bối cảnh
        prompt += f"# file path: {file_path}\n"
        prompt += left_context
        
        return prompt
