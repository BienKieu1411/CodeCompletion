"""
GraphCoderRL Main Training Script (Production)

Full pipeline: Data → Retrieval → Generation → CAHM → Reward → PPO
With: batch processing, validation, logging, checkpointing.
"""

import os
import json
import time
import logging
import math
from datetime import datetime

import torch
import torch.optim as optim

from graphcoder_rl.retrieval.coarse_dense_retriever import CoarseDenseRetriever
from graphcoder_rl.retrieval.multi_hop_graph_retriever import MultiHopGraphRetriever
from graphcoder_rl.data.repository_dataset_loader import DatasetLoader
from graphcoder_rl.data.left_context_anchor_extractor import LeftContextAnchorExtractor
from graphcoder_rl.generation.graphcoder_prompt_builder import GraphCoderPromptBuilder
from graphcoder_rl.generation.code_llm_generator import CodeLLMGenerator
from graphcoder_rl.rl.causal_credit_mask_engine import CausalCreditMaskEngine
from graphcoder_rl.rl.retrieval_reward_model import RetrievalRewardModel
from graphcoder_rl.rl.graph_traversal_ppo_trainer import GraphTraversalPPOTrainer, ValueHead
from graphcoder_rl.rl.graph_traversal_policy import GraphTraversalPolicy
from graphcoder_rl.evaluation.graphcoder_rl_eval import evaluate_predictions

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("GraphCoderRL.train")


def _mean_logprob_to_ppl(logprobs: torch.Tensor) -> float:
    if logprobs.numel() == 0:
        return 1e6
    return float(math.exp(-logprobs.float().mean().item()))


def _build_state_feature_vector(graph_meta: dict, local_graph) -> torch.Tensor:
    """
    Approximate RL state_t from novelty:
      left_context + current scope/imports/local vars + traversal path + semantic states.
    Returns a compact numeric vector in [0, 1] scale.
    """
    path_len = len(graph_meta.get("retrieval_path", []) or [])
    sem_states = graph_meta.get("semantic_state_ids", []) or []
    unique_sem_states = len(set(sem_states))
    token_cost = float(graph_meta.get("token_cost", 0.0))
    path_rel = float(graph_meta.get("path_relevance", 0.0))
    red_pen = float(graph_meta.get("redundancy_penalty", 0.0))
    irr_pen = float(graph_meta.get("irrelevant_node_penalty", 0.0))
    stop_selected = 1.0 if graph_meta.get("stop_selected", False) else 0.0

    imports = list(getattr(local_graph, "imports", []) or [])
    local_vars = list(getattr(local_graph, "local_variables", []) or [])
    has_parent_fn = 1.0 if getattr(local_graph, "parent_function", None) else 0.0
    has_parent_cls = 1.0 if getattr(local_graph, "parent_class", None) else 0.0

    feat = torch.tensor([
        min(1.0, path_len / 8.0),
        min(1.0, unique_sem_states / 8.0),
        min(1.0, token_cost / 8.0),
        min(1.0, max(0.0, path_rel)),
        min(1.0, max(0.0, red_pen)),
        min(1.0, max(0.0, irr_pen)),
        min(1.0, len(imports) / 20.0),
        min(1.0, len(local_vars) / 20.0),
        has_parent_fn,
        has_parent_cls,
        stop_selected,
    ], dtype=torch.float32)
    return feat


# ── Training Logger ───────────────────────────────────────────────────────────

class TrainingLogger:
    """File-based training logger (no WandB dependency)."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "training_log.jsonl")
        self.summary_path = os.path.join(log_dir, "summary.json")
        self._best_es = 0.0

    def log_step(self, epoch: int, step: int, metrics: dict):
        entry = {"epoch": epoch, "step": step, "timestamp": datetime.now().isoformat(), **metrics}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_epoch(self, epoch: int, metrics: dict):
        entry = {"epoch": epoch, "type": "epoch_summary", "timestamp": datetime.now().isoformat(), **metrics}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        # Print formatted summary
        logger.info(
            f"EPOCH {epoch} | "
            f"loss={metrics.get('avg_loss', 0):.4f} | "
            f"reward={metrics.get('avg_reward', 0):.4f} | "
            f"val_EM={metrics.get('val_em', '-')} | "
            f"val_ES={metrics.get('val_es', '-')}"
        )

    def update_best(self, es: float) -> bool:
        if es > self._best_es:
            self._best_es = es
            return True
        return False

    def save_summary(self, summary: dict):
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    data_loader: DatasetLoader,
    dense_retriever: CoarseDenseRetriever,
    graph_retriever: MultiHopGraphRetriever,
    ast_extractor: LeftContextAnchorExtractor,
    prompt_gen: GraphCoderPromptBuilder,
    llm: CodeLLMGenerator,
    graph_policy: GraphTraversalPolicy,
    device: str,
    max_samples: int = 50,
) -> dict:
    """Run validation and return EM/ES metrics."""
    try:
        val_samples = data_loader.load_test_samples(
            dataset_name="repoeval", max_samples=max_samples
        )
    except FileNotFoundError:
        logger.warning("No test set found, skipping validation.")
        return {"val_em": "-", "val_es": "-"}

    predictions = []

    for sample in val_samples:
        query = sample["left_context"]
        file_path = sample["id"]

        # Coarse retrieval first (limits graph action space)
        dense_snippets, _, aux = dense_retriever.retrieve_top_k(
            query, sample["crossfile_context"], top_k=2, return_aux=True
        )

        # Graph retrieval on local subgraph around coarse candidates
        local_graph = ast_extractor.extract_local_graph(
            query, cursor_line=len(query.split("\n")), file_path=file_path
        )
        graph_snippets = graph_retriever.retrieve_paths(
            local_graph=local_graph,
            crossfile_dict=sample["crossfile_context"],
            current_file=file_path,
            left_context=query,
            coarse_candidate_chunks=aux.get("filenames", []),
            policy_model=graph_policy,
            policy_device=device,
        )

        # Generate
        repo_snippet = "\n".join(graph_snippets) + "\n\n" + "\n".join(dense_snippets)
        prompt = prompt_gen.construct_prompt(
            repo_snippet, query, file_path=file_path, local_graph=local_graph
        )
        pred_text, _, _ = llm.generate_with_attention(prompt, retrieved_tokens_len=0)

        predictions.append({
            "task_id": sample["id"],
            "pred": pred_text.strip(),
            "target": sample["ground_truth"],
        })

    results = evaluate_predictions(predictions)
    return {"val_em": results["em"], "val_es": results["es"]}


# ── Main Training ─────────────────────────────────────────────────────────────

def train(config: dict | None = None):
    print("\n" + "=" * 60)
    print("=== GRAPHCODERRL TRAINING (PRODUCTION) ===")
    print("=" * 60)

    # ── Config ────────────────────────────────────────────────────
    cfg = config or {}
    NUM_EPOCHS = int(cfg.get("num_epochs", 5))
    BATCH_SIZE = int(cfg.get("batch_size", 3))
    PPO_UPDATE_STEPS = int(cfg.get("ppo_update_steps", 3))
    TOP_K = int(cfg.get("top_k", 2))
    VALIDATE_EVERY = int(cfg.get("validate_every", 1))
    LR_GRAPH_POLICY = float(cfg.get("lr_graph_policy", 2e-4))
    LR_VALUE = float(cfg.get("lr_value", 1e-4))
    WARM_START_STEPS = int(cfg.get("warm_start_steps", 100))
    CHECKPOINT_DIR = cfg.get("checkpoint_dir", os.path.join(os.path.dirname(__file__), "checkpoints"))
    LOG_DIR = cfg.get("log_dir", os.path.join(os.path.dirname(__file__), "logs"))
    PRETRAIN_CHECKPOINT = cfg.get("pretrain_checkpoint")

    # ── Initialize Pipeline ───────────────────────────────────────
    logger.info("Initializing pipeline...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dense_retriever = CoarseDenseRetriever(
        device=device,
        scoring_mode=cfg.get("coarse_scoring_mode", "dense"),
        quantum_alpha=float(cfg.get("coarse_quantum_alpha", 0.5)),
    )
    graph_retriever = MultiHopGraphRetriever()
    data_loader = DatasetLoader(
        dataset_path=cfg.get("dataset_path", "data/github_repos/python/train.parquet"),
        use_fim=bool(cfg.get("use_fim", False)),
        completion_level=cfg.get("completion_level", "line"),
        fixed_train=bool(cfg.get("fixed_train", True)),
        fixed_train_size=int(cfg.get("fixed_train_size", 2000)),
    )
    ast_extractor = LeftContextAnchorExtractor()
    prompt_gen = GraphCoderPromptBuilder(model_name="deepseek-coder")
    llm = CodeLLMGenerator(device=device)
    graph_policy = GraphTraversalPolicy(input_dim=8).to(device)

    if PRETRAIN_CHECKPOINT and os.path.exists(PRETRAIN_CHECKPOINT):
        ckpt = torch.load(PRETRAIN_CHECKPOINT, map_location=device, weights_only=True)
        if "coarse_retriever" in ckpt:
            dense_retriever.model.load_state_dict(ckpt["coarse_retriever"], strict=False)
            logger.info("Loaded coarse_retriever from pretrain checkpoint: %s", PRETRAIN_CHECKPOINT)

    # RL components
    cahm = CausalCreditMaskEngine()
    reward_model = RetrievalRewardModel(use_llm_judge=False)  # Fast mode

    # Value head
    hidden_dim = dense_retriever.model.config.hidden_size
    value_head = ValueHead(hidden_dim=hidden_dim).to(device)

    # Optimizers
    optimizer = optim.AdamW(graph_policy.parameters(), lr=LR_GRAPH_POLICY, weight_decay=0.01)
    value_optimizer = optim.AdamW(value_head.parameters(), lr=LR_VALUE)

    ppo_trainer = GraphTraversalPPOTrainer(
        model=graph_policy,
        optimizer=optimizer,
        cahm_engine=cahm,
        value_head=value_head,
        value_optimizer=value_optimizer,
    )

    train_logger = TrainingLogger(LOG_DIR)

    # ── Prepare Data ──────────────────────────────────────────────
    data_loader.prepare_dataset()

    # ── Training Loop ─────────────────────────────────────────────
    logger.info("Starting PPO training loop...\n")
    global_step = 0
    best_checkpoint = None

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        epoch_losses = []
        epoch_rewards = []
        epoch_metrics = []

        logger.info(f"--- EPOCH {epoch + 1}/{NUM_EPOCHS} ---")

        for step, batch in enumerate(data_loader.get_epoch_batches(batch_size=BATCH_SIZE)):
            for sample in batch:
                global_step += 1
                query = sample["left_context"]
                file_path = sample["id"]

                # ── Step 1: Coarse retrieval (for action-space limiting) ──
                dense_snippets, _, aux = dense_retriever.retrieve_top_k(
                    query, sample["crossfile_context"], top_k=TOP_K, return_aux=True
                )

                # ── Step 2: Multi-hop graph retrieval ──
                local_graph = ast_extractor.extract_local_graph(
                    query, cursor_line=len(query.split("\n")), file_path=file_path
                )
                warm_start = global_step <= WARM_START_STEPS
                graph_snippets, graph_meta = graph_retriever.retrieve_paths(
                    local_graph=local_graph,
                    crossfile_dict=sample["crossfile_context"],
                    current_file=file_path,
                    left_context=query,
                    return_metadata=True,
                    coarse_candidate_chunks=aux.get("filenames", []),
                    policy_model=None if warm_start else graph_policy,
                    policy_device=device,
                )

                action_features = graph_meta.get("policy_action_features", [])
                selected_action_indices = graph_meta.get("policy_selected_indices", [])
                if not action_features or not selected_action_indices:
                    continue
                action_features_t = torch.tensor(action_features, dtype=torch.float32, device=device)
                selected_idx_t = torch.tensor(selected_action_indices, dtype=torch.long, device=device)
                graph_action_logprobs, _ = graph_policy.selected_logprobs(
                    action_features_t, selected_idx_t
                )

                # Merge context
                repo_snippet = "\n".join(graph_snippets) + "\n\n" + "\n".join(dense_snippets)
                prompt = prompt_gen.construct_prompt(
                    repo_snippet, query, file_path=file_path, local_graph=local_graph
                )
                prompt_no_ctx = prompt_gen.construct_prompt(
                    "", query, file_path=file_path, local_graph=local_graph
                )

                # ── Step 3: Generate + get base logprobs ──
                pred_text, base_logprobs, cross_attn = llm.generate_with_attention(
                    prompt, retrieved_tokens_len=0
                )

                # ── Step 4: CAHM Masking ──
                # U_k: Attention-based mask
                mean_attn = cross_attn.mean()
                mask_U_scalar = cahm.compute_attention_mask(torch.tensor([mean_attn.item()]))
                mask_U = mask_U_scalar.expand_as(graph_action_logprobs).to(graph_action_logprobs.device)

                # I_k: Ablation-based causal influence
                action_influences = []
                n_graph_actions = len(graph_snippets)
                for idx in range(n_graph_actions):
                    graph_without_i = [s for j, s in enumerate(graph_snippets) if j != idx]
                    snippet_without_i = "\n".join(graph_without_i) + "\n\n" + "\n".join(dense_snippets)
                    prompt_without_i = prompt_gen.construct_prompt(
                        snippet_without_i, query, file_path=file_path, local_graph=local_graph
                    )
                    logprobs_without_i = llm.score_sequence(prompt_without_i, pred_text)
                    min_len = min(base_logprobs.size(0), logprobs_without_i.size(0))
                    if min_len == 0:
                        action_influences.append(torch.tensor(0.0, device=device))
                    else:
                        influence_i = (base_logprobs[:min_len] - logprobs_without_i[:min_len]).mean()
                        action_influences.append(influence_i)
                # STOP action influence (neutral baseline)
                if len(selected_action_indices) > n_graph_actions:
                    action_influences.append(torch.tensor(0.0, device=device))

                if action_influences:
                    influences_tensor = torch.stack(action_influences)
                else:
                    influences_tensor = torch.zeros_like(graph_action_logprobs)
                mask_I = cahm.compute_causal_mask(influences_tensor)
                mask_k = cahm.compute_hybrid_mask(mask_U, mask_I)

                # ── Step 5: Compute Reward ──
                with torch.no_grad():
                    gt_lp_with_ctx = llm.score_sequence(prompt, sample["ground_truth"])
                    gt_lp_no_ctx = llm.score_sequence(prompt_no_ctx, sample["ground_truth"])

                ppl_with_ctx = _mean_logprob_to_ppl(gt_lp_with_ctx)
                ppl_no_ctx = _mean_logprob_to_ppl(gt_lp_no_ctx)
                hit_gold = 1.0 if sample["ground_truth"].strip() and sample["ground_truth"].strip() in repo_snippet else 0.0

                final_reward = reward_model.calculate_reward(
                    pred_text,
                    query,
                    sample["ground_truth"],
                    retrieved_context=repo_snippet,
                    ppl_no_ctx=ppl_no_ctx,
                    ppl_with_ctx=ppl_with_ctx,
                    path_relevance=graph_meta.get("path_relevance", 0.0),
                    token_cost=graph_meta.get("token_cost", 0.0),
                    redundancy_penalty=graph_meta.get("redundancy_penalty", 0.0),
                    irrelevant_node_penalty=graph_meta.get("irrelevant_node_penalty", 0.0),
                    hit_gold=hit_gold,
                )
                epoch_rewards.append(final_reward)

                # ── Step 6: Value estimation ──
                with torch.no_grad():
                    query_emb = dense_retriever.encode_batch([query[-1500:]], batch_size=1)
                    state_feat = _build_state_feature_vector(graph_meta, local_graph).to(device)
                    query_emb_state = query_emb.clone()
                    n = min(state_feat.numel(), query_emb_state.shape[1])
                    query_emb_state[0, :n] = query_emb_state[0, :n] + state_feat[:n]
                    values_retriever = value_head(query_emb_state).expand(graph_action_logprobs.shape[0])

                # ── Step 7: PPO Update ──
                old_logprobs = graph_action_logprobs.detach()
                rewards_tensor = torch.full_like(graph_action_logprobs, final_reward)

                step_losses = []
                for _ in range(PPO_UPDATE_STEPS):
                    new_logprobs, _ = graph_policy.selected_logprobs(
                        action_features_t, selected_idx_t
                    )
                    ppo_metrics = ppo_trainer.step(
                        new_logprobs, old_logprobs,
                        values_retriever, rewards_tensor, mask_k,
                    )
                    step_losses.append(ppo_metrics["total_loss"])

                avg_loss = sum(step_losses) / len(step_losses)
                epoch_losses.append(avg_loss)
                epoch_metrics.append(ppo_metrics)

                # Log step
                train_logger.log_step(epoch + 1, global_step, {
                    "loss": avg_loss,
                    "reward": final_reward,
                    "ppl_delta": ppl_no_ctx - ppl_with_ctx,
                    "path_relevance": graph_meta.get("path_relevance", 0.0),
                    "policy_loss": ppo_metrics["policy_loss"],
                    "entropy": ppo_metrics["entropy"],
                    "clip_frac": ppo_metrics["clip_fraction"],
                })

                if global_step % 10 == 0:
                    logger.info(
                        f"  Step {global_step} | "
                        f"reward={final_reward:+.4f} | "
                        f"loss={avg_loss:.4f} | "
                        f"entropy={ppo_metrics['entropy']:.4f}"
                    )

        # ── Epoch Summary ──
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        avg_epoch_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)

        epoch_summary = {
            "avg_loss": avg_epoch_loss,
            "avg_reward": avg_epoch_reward,
            "n_steps": len(epoch_losses),
            "time_seconds": round(epoch_time, 1),
        }

        # ── Validation ──
        if (epoch + 1) % VALIDATE_EVERY == 0:
            logger.info("Running validation...")
            dense_retriever.eval()
            val_results = validate(
                data_loader, dense_retriever, graph_retriever,
                ast_extractor, prompt_gen, llm, graph_policy, device, max_samples=50,
            )
            dense_retriever.train()
            epoch_summary.update(val_results)

            # Check if best model
            val_es = val_results.get("val_es", 0)
            if isinstance(val_es, (int, float)) and train_logger.update_best(val_es):
                best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    "coarse_retriever": dense_retriever.model.state_dict(),
                    "graph_traversal_policy": graph_policy.state_dict(),
                    "value_head": value_head.state_dict(),
                    "epoch": epoch + 1,
                    "val_es": val_es,
                    "meta": {"framework": "GraphCoderRL"},
                }, best_path)
                best_checkpoint = best_path
                logger.info(f"  ★ New best model (ES={val_es}%) saved to {best_path}")

        train_logger.log_epoch(epoch + 1, epoch_summary)

        # ── Save checkpoint ──
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}.pt")
        torch.save({
            "coarse_retriever": dense_retriever.model.state_dict(),
            "graph_traversal_policy": graph_policy.state_dict(),
            "value_head": value_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "meta": {"framework": "GraphCoderRL"},
        }, ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path}")

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    if best_checkpoint:
        print(f"Best model: {best_checkpoint}")
    print("=" * 60)

    train_logger.save_summary({
        "total_epochs": NUM_EPOCHS,
        "best_checkpoint": best_checkpoint,
        "final_avg_loss": avg_epoch_loss,
    })


def main():
    train()


if __name__ == "__main__":
    main()
