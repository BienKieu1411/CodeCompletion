"""
GraphFRL Main Training Script (Production)

Full pipeline: Data → Retrieval → Generation → CAHM → Reward → PPO
With: batch processing, validation, logging, checkpointing.
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime

import torch
import torch.optim as optim

# Module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GraphFRL.retriever.unixcoder_retriever import UniXCoderRetriever
from GraphFRL.retriever.graph_retriever import GraphRetriever
from GraphFRL.data.dataset_loader import GraphFRLDataLoader
from GraphFRL.data.ast_parser import ASTQueryExtractor
from GraphFRL.generator.causal_prompt import CausalPromptGenerator
from GraphFRL.generator.deepseek_generator import DeepSeekGenerator
from GraphFRL.rl.cahm_engine import CAHMEngine
from GraphFRL.rl.reward_model import CompositeRewardModel
from GraphFRL.rl.ppo_trainer import GraphFRLPPOTrainer, ValueHead
from GraphFRL.evaluate import evaluate_predictions

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("GraphFRL.train")


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
    data_loader: GraphFRLDataLoader,
    dense_retriever: UniXCoderRetriever,
    graph_retriever: GraphRetriever,
    ast_extractor: ASTQueryExtractor,
    prompt_gen: CausalPromptGenerator,
    llm: DeepSeekGenerator,
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

        # Graph retrieval
        local_graph = ast_extractor.extract_local_graph(
            query, cursor_line=len(query.split("\n"))
        )
        graph_snippets = graph_retriever.retrieve_paths(
            local_graph=local_graph, crossfile_dict=sample["crossfile_context"]
        )

        # Dense retrieval
        dense_snippets, _ = dense_retriever.retrieve_top_k(
            query, sample["crossfile_context"], top_k=2
        )

        # Generate
        repo_snippet = "\n".join(graph_snippets) + "\n\n" + "\n".join(dense_snippets)
        prompt = prompt_gen.construct_prompt(repo_snippet, query, file_path=file_path)
        pred_text, _, _ = llm.generate_with_attention(prompt, retrieved_tokens_len=0)

        predictions.append({
            "task_id": sample["id"],
            "pred": pred_text.strip(),
            "target": sample["ground_truth"],
        })

    results = evaluate_predictions(predictions)
    return {"val_em": results["em"], "val_es": results["es"]}


# ── Main Training ─────────────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 60)
    print("=== GRAPHFRL TRAINING (PRODUCTION) ===")
    print("=" * 60)

    # ── Config ────────────────────────────────────────────────────
    NUM_EPOCHS = 5
    BATCH_SIZE = 3
    PPO_UPDATE_STEPS = 3
    TOP_K = 2
    VALIDATE_EVERY = 1  # epochs
    LR_RETRIEVER = 1e-5
    LR_VALUE = 1e-4
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
    LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

    # ── Initialize Pipeline ───────────────────────────────────────
    logger.info("Initializing pipeline...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dense_retriever = UniXCoderRetriever(device=device)
    graph_retriever = GraphRetriever()
    data_loader = GraphFRLDataLoader(use_fim=False, completion_level="line")
    ast_extractor = ASTQueryExtractor()
    prompt_gen = CausalPromptGenerator(model_name="deepseek-coder")
    llm = DeepSeekGenerator(device=device)

    # RL components
    cahm = CAHMEngine()
    reward_model = CompositeRewardModel(use_llm_judge=False)  # Fast mode

    # Value head
    hidden_dim = dense_retriever.model.config.hidden_size
    value_head = ValueHead(hidden_dim=hidden_dim).to(device)

    # Optimizers
    optimizer = optim.AdamW(dense_retriever.parameters(), lr=LR_RETRIEVER, weight_decay=0.01)
    value_optimizer = optim.AdamW(value_head.parameters(), lr=LR_VALUE)

    ppo_trainer = GraphFRLPPOTrainer(
        model=dense_retriever,
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

                # ── Step 1: Multi-semantic Retrieval ──
                local_graph = ast_extractor.extract_local_graph(
                    query, cursor_line=len(query.split("\n"))
                )
                graph_snippets = graph_retriever.retrieve_paths(
                    local_graph=local_graph,
                    crossfile_dict=sample["crossfile_context"],
                    current_file=file_path,
                )

                dense_snippets, retriever_logprobs, aux = dense_retriever.retrieve_top_k(
                    query, sample["crossfile_context"], top_k=TOP_K, return_aux=True
                )
                selected_indices = aux["topk_indices"]

                # Merge context
                repo_snippet = "\n".join(graph_snippets) + "\n\n" + "\n".join(dense_snippets)
                prompt = prompt_gen.construct_prompt(repo_snippet, query, file_path=file_path)

                # ── Step 2: Generate + get base logprobs ──
                pred_text, base_logprobs, cross_attn = llm.generate_with_attention(
                    prompt, retrieved_tokens_len=0
                )

                # ── Step 3: CAHM Masking ──
                # U_k: Attention-based mask
                mean_attn = cross_attn.mean()
                mask_U_scalar = cahm.compute_attention_mask(torch.tensor([mean_attn.item()]))
                mask_U = mask_U_scalar.expand_as(retriever_logprobs).to(retriever_logprobs.device)

                # I_k: Ablation-based causal influence
                action_influences = []
                for idx in range(len(dense_snippets)):
                    dense_without_i = [s for j, s in enumerate(dense_snippets) if j != idx]
                    snippet_without_i = "\n".join(graph_snippets) + "\n\n" + "\n".join(dense_without_i)
                    prompt_without_i = prompt_gen.construct_prompt(snippet_without_i, query, file_path=file_path)
                    logprobs_without_i = llm.score_sequence(prompt_without_i, pred_text)
                    min_len = min(base_logprobs.size(0), logprobs_without_i.size(0))
                    if min_len == 0:
                        action_influences.append(torch.tensor(0.0, device=device))
                    else:
                        influence_i = (base_logprobs[:min_len] - logprobs_without_i[:min_len]).mean()
                        action_influences.append(influence_i)

                if action_influences:
                    influences_tensor = torch.stack(action_influences)
                else:
                    influences_tensor = torch.zeros_like(retriever_logprobs)
                mask_I = cahm.compute_causal_mask(influences_tensor)
                mask_k = cahm.compute_hybrid_mask(mask_U, mask_I)

                # ── Step 4: Compute Reward ──
                final_reward = reward_model.calculate_reward(
                    pred_text, query, sample["ground_truth"],
                    retrieved_context=repo_snippet
                )
                epoch_rewards.append(final_reward)

                # ── Step 5: Value estimation ──
                with torch.no_grad():
                    query_emb = dense_retriever.encode_batch([query[-1500:]], batch_size=1)
                    values_retriever = value_head(query_emb).expand(retriever_logprobs.shape[0])

                # ── Step 6: PPO Update ──
                old_logprobs = retriever_logprobs.detach()
                rewards_tensor = torch.full_like(retriever_logprobs, final_reward)

                step_losses = []
                for _ in range(PPO_UPDATE_STEPS):
                    _, new_logprobs = dense_retriever.retrieve_top_k(
                        query, sample["crossfile_context"],
                        top_k=TOP_K, force_indices=selected_indices,
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
                ast_extractor, prompt_gen, llm, max_samples=50,
            )
            dense_retriever.train()
            epoch_summary.update(val_results)

            # Check if best model
            val_es = val_results.get("val_es", 0)
            if isinstance(val_es, (int, float)) and train_logger.update_best(val_es):
                best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    "retriever": dense_retriever.model.state_dict(),
                    "value_head": value_head.state_dict(),
                    "epoch": epoch + 1,
                    "val_es": val_es,
                }, best_path)
                best_checkpoint = best_path
                logger.info(f"  ★ New best model (ES={val_es}%) saved to {best_path}")

        train_logger.log_epoch(epoch + 1, epoch_summary)

        # ── Save checkpoint ──
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}.pt")
        torch.save({
            "retriever": dense_retriever.model.state_dict(),
            "value_head": value_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
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


if __name__ == "__main__":
    train()
