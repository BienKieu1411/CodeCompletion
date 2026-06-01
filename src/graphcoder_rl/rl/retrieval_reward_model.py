"""
Composite Reward Model (Production Implementation)

Three reward signals:
1. R_similarity: Edit Similarity (editdistance-based, same metric as benchmarks)
2. R_struct:     AST structural match (node-type overlap between prediction and ground truth)
3. R_judge:      LLM Judge or fast fallback (difflib)

EMA-based Pareto normalization for stable PPO training.
"""

import re
import ast
import logging
import difflib
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Edit Similarity (matches RepoEval/CCEval metric exactly) ─────────────────

def compute_edit_similarity(prediction: str, ground_truth: str) -> float:
    """
    Compute Edit Similarity between prediction and ground truth.
    Uses editdistance (Levenshtein distance) — same as AlignCoder/RepoEval.
    Falls back to difflib if editdistance not installed.
    """
    pred_lines = [l.strip() for l in prediction.splitlines() if l.strip()]
    gt_lines = [l.strip() for l in ground_truth.splitlines() if l.strip()]

    # Truncate prediction to same number of lines as ground truth
    pred_lines = pred_lines[:len(gt_lines)]

    pred_str = "\n".join(pred_lines)
    gt_str = "\n".join(gt_lines)

    if not gt_str and not pred_str:
        return 1.0
    if not gt_str or not pred_str:
        return 0.0

    try:
        import editdistance
        dist = editdistance.eval(gt_str, pred_str)
        max_len = max(len(gt_str), len(pred_str))
        return 1.0 - (dist / max_len) if max_len > 0 else 1.0
    except ImportError:
        return difflib.SequenceMatcher(None, pred_str, gt_str).ratio()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Returns 1.0 if exact match (after whitespace normalization), else 0.0."""
    pred_lines = [l.strip() for l in prediction.splitlines() if l.strip()]
    gt_lines = [l.strip() for l in ground_truth.splitlines() if l.strip()]
    pred_lines = pred_lines[:len(gt_lines)]
    return 1.0 if pred_lines == gt_lines else 0.0


# ── AST Structural Similarity ────────────────────────────────────────────────

def _collect_ast_node_types(code: str) -> list:
    """Parse Python code and collect AST node type names (flattened)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    types = []
    for node in ast.walk(tree):
        types.append(type(node).__name__)
    return types


def compute_structural_similarity(prediction: str, ground_truth: str) -> float:
    """
    Compare AST node-type distributions between prediction and ground truth.
    Returns Jaccard-like overlap score in [0, 1].

    This measures whether the prediction has the same structural
    patterns (loops, assignments, calls, etc.) as the ground truth.
    """
    pred_types = _collect_ast_node_types(prediction)
    gt_types = _collect_ast_node_types(ground_truth)

    if not gt_types:
        return 0.5  # Can't parse ground truth — neutral score

    if not pred_types:
        return 0.0  # Prediction doesn't parse — bad

    pred_set = set(pred_types)
    gt_set = set(gt_types)

    intersection = pred_set & gt_set
    union = pred_set | gt_set

    if not union:
        return 0.5

    # Weighted by frequency overlap
    pred_counts = {}
    for t in pred_types:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gt_counts = {}
    for t in gt_types:
        gt_counts[t] = gt_counts.get(t, 0) + 1

    overlap_score = 0.0
    total_weight = 0.0
    for node_type in union:
        p = pred_counts.get(node_type, 0)
        g = gt_counts.get(node_type, 0)
        total_weight += max(p, g)
        overlap_score += min(p, g)

    return overlap_score / total_weight if total_weight > 0 else 0.0


# ── Composite Reward Model ────────────────────────────────────────────────────

class RetrievalRewardModel:
    """
    Multi-signal reward model with EMA normalization.

    Reward follows novelty design:
      R = alpha * Q_completion
          - beta * token_cost
          - gamma * retrieval_uncertainty
          + lambda * path_relevance
    with EMA normalization for PPO stability.
    """

    def __init__(
        self,
        judge_model_name: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        device: str = "cuda",
        use_llm_judge: bool = False,
        w_similarity: float = 0.5,
        w_struct: float = 0.2,
        w_judge: float = 0.3,
        alpha_completion: float = 1.0,
        beta_token_cost: float = 0.08,
        gamma_uncertainty: float = 0.15,
        lambda_path: float = 0.35,
        lambda_hit_gold: float = 0.15,
        ema_decay: float = 0.99,
    ):
        self.device = device
        self.use_llm_judge = use_llm_judge
        self.w_sim = w_similarity
        self.w_struct = w_struct
        self.w_judge = w_judge
        self.alpha_completion = alpha_completion
        self.beta_token_cost = beta_token_cost
        self.gamma_uncertainty = gamma_uncertainty
        self.lambda_path = lambda_path
        self.lambda_hit_gold = lambda_hit_gold
        self.ema_decay = ema_decay
        self.ema_reward = 0.0

        self.tokenizer = None
        self.model = None
        if self.use_llm_judge:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                logger.info(f"Loading judge model {judge_model_name} ...")
                self.tokenizer = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    judge_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                ).to(device)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Failed to load judge model: {e}. Falling back to difflib.")
                self.use_llm_judge = False

    def _llm_judge(self, prediction: str, context: str, ground_truth: str) -> float:
        """Score prediction using LLM judge or fast fallback."""
        if not self.use_llm_judge or self.model is None:
            return difflib.SequenceMatcher(None, prediction.strip(), ground_truth.strip()).ratio()

        messages = [
            {"role": "system", "content": "You are an expert code reviewer. Score predicted code vs ground truth. Output ONLY a float between 0.0 and 1.0."},
            {"role": "user", "content": (
                f"CONTEXT:\n{context[-500:]}\n\n"
                f"GROUND TRUTH:\n{ground_truth}\n\n"
                f"PREDICTION:\n{prediction}\n\n"
                "Score (0.0-1.0):"
            )},
        ]

        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids, max_new_tokens=10, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            match = re.search(r'[01]\.\d+|1\.0|0\.0|[01]', text)
            if match:
                return max(0.0, min(1.0, float(match.group())))
        except Exception as e:
            logger.debug(f"LLM judge error: {e}")

        return difflib.SequenceMatcher(None, prediction.strip(), ground_truth.strip()).ratio()

    def calculate_reward(
        self,
        prediction: str,
        query: str,
        ground_truth: str,
        repo_ast=None,
        retrieved_context: str = "",
        ppl_no_ctx: Optional[float] = None,
        ppl_with_ctx: Optional[float] = None,
        path_relevance: float = 0.0,
        token_cost: float = 0.0,
        redundancy_penalty: float = 0.0,
        irrelevant_node_penalty: float = 0.0,
        hit_gold: float = 0.0,
    ) -> float:
        """
        Calculate composite reward with EMA normalization.

        Returns:
            Normalized reward (can be negative — centered around 0 by EMA baseline)
        """
        # Q_completion: prefer direct PPL reduction signal when available.
        if ppl_no_ctx is not None and ppl_with_ctx is not None:
            q_completion = max(-10.0, min(10.0, float(ppl_no_ctx - ppl_with_ctx)))
        else:
            r_sim = compute_edit_similarity(prediction, ground_truth)
            r_struct = compute_structural_similarity(prediction, ground_truth)
            r_judge = self._llm_judge(prediction, query, ground_truth)
            q_completion = (
                self.w_sim * r_sim
                + self.w_struct * r_struct
                + self.w_judge * r_judge
            )

        retrieval_uncertainty = max(0.0, float(redundancy_penalty)) + max(0.0, float(irrelevant_node_penalty))
        raw_reward = (
            self.alpha_completion * q_completion
            + self.lambda_path * float(path_relevance)
            + self.lambda_hit_gold * float(hit_gold)
            - self.beta_token_cost * float(token_cost)
            - self.gamma_uncertainty * retrieval_uncertainty
        )

        # EMA-based Pareto normalization
        self.ema_reward = self.ema_decay * self.ema_reward + (1 - self.ema_decay) * raw_reward
        normalized_reward = raw_reward - self.ema_reward

        return normalized_reward

    def calculate_reward_detailed(
        self,
        prediction: str,
        query: str,
        ground_truth: str,
        ppl_no_ctx: Optional[float] = None,
        ppl_with_ctx: Optional[float] = None,
        path_relevance: float = 0.0,
        token_cost: float = 0.0,
        redundancy_penalty: float = 0.0,
        irrelevant_node_penalty: float = 0.0,
        hit_gold: float = 0.0,
    ) -> dict:
        """Return individual reward components for logging/analysis."""
        if ppl_no_ctx is not None and ppl_with_ctx is not None:
            q_completion = max(-10.0, min(10.0, float(ppl_no_ctx - ppl_with_ctx)))
        else:
            r_sim = compute_edit_similarity(prediction, ground_truth)
            r_struct = compute_structural_similarity(prediction, ground_truth)
            r_judge = self._llm_judge(prediction, query, ground_truth)
            q_completion = self.w_sim * r_sim + self.w_struct * r_struct + self.w_judge * r_judge

        retrieval_uncertainty = max(0.0, float(redundancy_penalty)) + max(0.0, float(irrelevant_node_penalty))
        raw = (
            self.alpha_completion * q_completion
            + self.lambda_path * float(path_relevance)
            + self.lambda_hit_gold * float(hit_gold)
            - self.beta_token_cost * float(token_cost)
            - self.gamma_uncertainty * retrieval_uncertainty
        )
        return {
            "q_completion": q_completion,
            "path_relevance": float(path_relevance),
            "token_cost": float(token_cost),
            "redundancy_penalty": float(redundancy_penalty),
            "irrelevant_node_penalty": float(irrelevant_node_penalty),
            "hit_gold": float(hit_gold),
            "retrieval_uncertainty": retrieval_uncertainty,
            "raw_reward": raw,
            "ema_baseline": self.ema_reward,
        }
