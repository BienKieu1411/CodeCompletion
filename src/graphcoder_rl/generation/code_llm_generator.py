"""
DeepSeek Generator (Real Implementation)
Sinh mã và trích xuất Cross-Attention, Logprobs từ HuggingFace Transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CodeLLMGenerator:
    """Wrapper cho DeepSeek-Coder-1.3B."""
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-1.3b-base", device: str = "cuda"):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        print(f"[*] Đang nạp Generator {model_name} lên {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=dtype
        ).to(self.device)
        
        self.model.eval() 

    def generate_with_attention(self, prompt: str, retrieved_tokens_len: int, max_new_tokens: int = 32):
        """
        Thực hiện giải mã (Decoding), trả về:
        - Text dự đoán
        - Logprobs của các token được sinh ra
        - Ma trận Cross-Attention của Layer cuối
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # 1. Bóc tách Text dự đoán
        generated_ids = outputs.sequences[0, input_ids.shape[1]:]
        pred_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 2. Bóc tách Logprobs
        logprobs = []
        for i, scores in enumerate(outputs.scores): 
            # scores shape: (batch=1, vocab_size)
            logprob_dist = torch.nn.functional.log_softmax(scores, dim=-1)
            token_logprob = logprob_dist[0, generated_ids[i]].item()
            logprobs.append(token_logprob)
            
        logprobs_tensor = torch.tensor(logprobs, device=self.device)
        
        # 3. Bóc tách Cross-Attention
        attentions = getattr(outputs, "attentions", None)
        if attentions and len(attentions) > 0:
            # attentions[-1]: tuple(layer_attn) cho token cuối cùng.
            last_step = attentions[-1]
            if isinstance(last_step, (list, tuple)) and len(last_step) > 0:
                last_layer_attn = last_step[-1]
                cross_attn = last_layer_attn.mean(dim=1)
            else:
                cross_attn = torch.zeros((1, 1, input_ids.shape[1]), device=self.device)
        else:
            cross_attn = torch.zeros((1, 1, input_ids.shape[1]), device=self.device)
            
        return pred_text, logprobs_tensor, cross_attn

    def score_sequence(self, prompt: str, target_text: str) -> torch.Tensor:
        """
        Tính toán logprobs của target_text dựa trên prompt (Teacher Forcing).
        Rất cần thiết để đo lường độ tụt giảm Loss khi xóa đi ngữ cảnh (Ablation).
        """
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        target_ids = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            
        # Logits có shape: (1, seq_len, vocab_size). Chỉ chấm điểm phần target
        target_logits = logits[0, prompt_ids.shape[1]-1 : -1, :]
        logprob_dist = torch.nn.functional.log_softmax(target_logits, dim=-1)
        
        logprobs = []
        for i in range(target_ids.shape[1]):
            token_id = target_ids[0, i]
            logprobs.append(logprob_dist[i, token_id].item())
            
        return torch.tensor(logprobs, device=self.device)
