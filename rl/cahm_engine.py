"""
Causal-Attention Hybrid Masking (CAHM) Engine
Tính toán mặt nạ tinh hạt dựa trên Attention ($U_k$) và Ablation ($I_k$).
Giúp giảm phương sai Policy Gradient.
"""

import torch

class CAHMEngine:
    def __init__(self, tau_1: float = 0.05, tau_2: float = 0.01):
        self.tau_1 = tau_1  # Ngưỡng Attention (U_k)
        self.tau_2 = tau_2  # Ngưỡng Causality / Ablation (I_k)

    def compute_attention_mask(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Tính toán $U_k$ cho không gian hành động retriever.
        attention_scores shape: (num_actions,)
        """
        if attention_scores.numel() == 0:
            return torch.tensor([])
            
        mask_U = (attention_scores > self.tau_1).float()
        return mask_U

    def compute_causal_mask(self, influences: torch.Tensor) -> torch.Tensor:
        """
        Tính toán $I_k$: Ablation-based Influence cho từng hành động retriever.
        influences shape: (num_actions,)
        """
        # I_k > 0 nghĩa là hành động đó có đóng góp dương khi sinh mã
        mask_I = (influences > self.tau_2).float()
        return mask_I

    def compute_hybrid_mask(self, mask_U: torch.Tensor, mask_I: torch.Tensor) -> torch.Tensor:
        """
        $M_k = \\mathbb{1}(U_k > \\tau_1 \\lor I_k > \\tau_2)$
        Kết hợp để tạo mặt nạ CAHM cuối cùng.
        """
        # Sử dụng torch.clamp để tạo tính chất OR nhị phân (giới hạn max = 1.0)
        M_k = torch.clamp(mask_U + mask_I, max=1.0)
        return M_k
