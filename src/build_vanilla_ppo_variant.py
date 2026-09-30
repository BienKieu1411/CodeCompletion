"""Derive a standalone terminal-reward learned-critic PPO Kaggle pair from RRPO.

The data, generator, prompt builder and retriever stay fixed; PPO scores only each
completed slate. The user-requested K=10 / 3,072-token context budget is retained.
"""

import ast
import json
from pathlib import Path

from sync_ast_ppo_notebooks import source_cells


ROOT = Path(__file__).parent
RRPO_TRAIN = ROOT / "ast_ppo_unixcoder_kaggle.py"
RRPO_TRAIN_NOTEBOOK = RRPO_TRAIN.with_suffix(".ipynb")
RRPO_EVAL = ROOT / "ast_ppo_unixcoder_eval_kaggle.py"
RRPO_EVAL_NOTEBOOK = RRPO_EVAL.with_suffix(".ipynb")
PPO_TRAIN = ROOT / "ast_ppo_unixcoder_ppo_kaggle.py"
PPO_TRAIN_NOTEBOOK = ROOT / "repo_level_ast_retrieval_ppo_clip_k10_train_kaggle.ipynb"
PPO_EVAL = ROOT / "ast_ppo_unixcoder_ppo_eval_kaggle.py"


TRAIN_MARKDOWN = '''# %% [markdown]
# # AST retrieval + terminal-reward PPO-Clip with learned critic — Kaggle T4×2
# Standalone PPO counterpart to the RRPO notebook; upload only this `.ipynb`.
# GPU 0 serves frozen DeepSeek-Coder-1.3B via vLLM; GPU 1 trains all UniXcoder-base
# encoder weights and the slate policy/value heads.
# The attached prebuilt Parquet remains unchanged: same Python+Java rows, AST/BM25
# candidate pools, labels, split and tokenizer schema. No data preparation on Kaggle.
# The generator sees up to 3,072 input tokens plus 96 output tokens. Cross-file
# snippets are capped at 2,344 tokens; the path and left context share 728 tokens.
# Each completed slate is scored once with RLCoder-style edit similarity and the
# existing identifier-F1 blend. No partial-prefix or BM25-reference generations.
# Standard PPO-Clip, learned state-value critic, GAE and clipped value regression
# are retained; only the reward timing/aggregation changes to terminal-only.
# Model, data, prompt construction and scalar scorer stay fixed. K/context are expanded
# as requested; a strict algorithm-only ablation needs RRPO rerun with K=10 and
# the same 2,344/3,072-token budget. The RL algorithm uses learned-critic PPO.
# Full mode processes the fixed train split until the 11-hour wall-clock guard,
# validates on the fixed validation split, and evaluates up to 50 rows per parquet.
# AST chunk cap=384, candidate pool=64, slate K<=10, encoder LR=5e-5 and vLLM batch=16.
# Checkpoints/cache are isolated under `/kaggle/working/ast_ppo_terminal_v1_k10_ctx3072_xfb2344`;
# this run cannot resume from a prefix-reward PPO or RRPO checkpoint. It supports safe same-run resume.
# Attach the training dataset and, for held-out evaluation, the Data4AlignCoder
# CCEval/RepoEval parquets. Checkpoint signatures record the PPO algorithm.
'''


ROLLOUT = '''def rollout(row, greedy=False):
    encoder.train()  # Dropout is zero; keep HF gradient checkpointing active in updates.
    heads.eval()
    with torch.no_grad():
        query, candidates = encode_row(row)
        selected, steps = [], []
        remaining = CROSSFILE_TOKEN_BUDGET
        for _ in range(MAX_SLATE_STEPS + 1):
            log_probs, value, valid = state_distribution(
                row, query, candidates, selected, remaining
            )
            action = int(torch.argmax(log_probs).item()) if greedy else int(
                torch.multinomial(log_probs.exp(), 1).item()
            )
            steps.append({"selected": tuple(selected), "remaining": remaining,
                          "valid": valid, "action": action,
                          "old_logp": float(log_probs[action].item()),
                          "old_value": float(value.item()),
                          "old_probs": log_probs.exp().cpu().tolist()})
            if action == len(row["candidate_ids"]):
                break
            selected.append(action)
            remaining -= row["candidate_costs"][action]
            if len(selected) >= MAX_SLATE_STEPS:
                break
    return {"row": row, "selected": selected, "steps": steps}
'''


TERMINAL_STEP_REWARDS = '''def terminal_step_rewards(step_count, terminal_utility):
    """Place the completed-slate score only on the final action transition."""
    if not 1 <= step_count <= MAX_SLATE_STEPS + 1:
        raise ValueError("Terminal reward needs one to max-K-plus-STOP action steps")
    rewards = [0.0] * step_count
    rewards[-1] = float(terminal_utility)
    return rewards
'''


SCORE_EPISODES = '''def score_ppo_episodes(episodes):
    """Generate and score exactly one completion for each completed slate."""
    if not episodes:
        return 0
    prompts = [compose_prompt(episode["row"], episode["selected"])
               for episode in episodes]
    api_batches = math.ceil(len(prompts) / max(1, VLLM_MAX_NUM_SEQS))
    print(f"PPO terminal reward scoring: {len(episodes)} episodes, "
          f"{len(prompts)} final-slate completion prompts, "
          f"{api_batches} vLLM batches.", flush=True)
    outputs = generate_completions(prompts, progress_label="PPO rewards")
    for episode, output in zip(episodes, outputs):
        final_components = synthetic_reward(output, episode["row"])
        final_utility = final_components["utility"]
        step_rewards = terminal_step_rewards(len(episode["steps"]), final_utility)
        if len(step_rewards) != len(episode["steps"]):
            raise AssertionError("Terminal rewards and sampled actions have different lengths")
        episode["final_utility"] = final_utility
        episode["final_es"] = final_components["es"]
        episode["final_id_f1"] = final_components["id_f1"]
        episode["step_rewards"] = step_rewards
        episode["reward"] = final_utility
    return len(prompts)
'''


PREPARE_ADVANTAGES = '''def prepare_advantages(episodes):
    """Standard GAE from the learned V(s); every slate ends at STOP or max-K."""
    all_advantages = []
    for episode in episodes:
        rewards = episode["step_rewards"]
        values = [step["old_value"] for step in episode["steps"]]
        if len(rewards) != len(episode["steps"]) or len(values) != len(rewards):
            raise ValueError("PPO rewards/critic values must match action steps")
        advantages = [0.0] * len(rewards)
        running = 0.0
        for index in reversed(range(len(rewards))):
            next_value = values[index + 1] if index + 1 < len(values) else 0.0
            delta = rewards[index] + PPO_GAMMA * next_value - values[index]
            running = delta + PPO_GAMMA * PPO_GAE_LAMBDA * running
            advantages[index] = running
        episode["raw_advantages"] = advantages
        episode["returns"] = [advantage + value
                              for advantage, value in zip(advantages, values)]
        all_advantages.extend(advantages)
    mean = float(np.mean(all_advantages))
    std = float(np.std(all_advantages))
    for episode in episodes:
        episode["advantages"] = [(value - mean) / max(std, 1e-6)
                                 for value in episode["raw_advantages"]]
    return mean, std
'''


PPO_EPISODE_LOSS = '''def ppo_episode_loss(episode):
    row = episode["row"]
    query, candidates = encode_row(row)
    actor_terms, value_terms, entropy_terms, clips = [], [], [], []
    for step, advantage, target_return in zip(
            episode["steps"], episode["advantages"], episode["returns"]):
        log_probs, value, valid = state_distribution(
            row, query, candidates, step["selected"], step["remaining"]
        )
        if valid != step["valid"]:
            raise AssertionError("PPO action support changed between rollout and update")
        old_logp = log_probs.new_tensor(step["old_logp"])
        log_ratio = log_probs[step["action"]] - old_logp
        if abs(float(log_ratio.detach())) > 20:
            raise FloatingPointError("PPO ratio diverged; lower LR or KL limit")
        ratio = log_ratio.exp()
        advantage_tensor = log_probs.new_tensor(advantage)
        unclipped = ratio * advantage_tensor
        clipped = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * advantage_tensor
        actor_terms.append(torch.minimum(unclipped, clipped))
        old_value = log_probs.new_tensor(step["old_value"])
        target = log_probs.new_tensor(target_return)
        clipped_value = old_value + (value - old_value).clamp(-VALUE_CLIP, VALUE_CLIP)
        value_terms.append(0.5 * torch.maximum((value - target).square(),
                                                (clipped_value - target).square()))
        entropy_terms.append(-(log_probs.exp() * log_probs).sum())
        clips.append(((ratio - 1).abs() > PPO_CLIP).float())
    horizon = MAX_SLATE_STEPS + 1
    actor_loss = -torch.stack(actor_terms).sum() / horizon
    value_loss = torch.stack(value_terms).sum() / horizon
    entropy = torch.stack(entropy_terms).sum() / horizon
    loss = actor_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
    diagnostics = {"actor": float(actor_loss.detach()),
                   "value": float(value_loss.detach()),
                   "entropy": float(entropy.detach()),
                   "clip_fraction": float(torch.stack(clips).mean().detach())}
    return loss, diagnostics
'''


PPO_UPDATE = '''def ppo_update(episodes):
    prepare_advantages(episodes)
    encoder.train()
    heads.train()
    totals = {k: [] for k in ("actor", "value", "entropy", "clip_fraction",
                              "grad_norm", "amp_skipped")}
    passes_completed = 0
    pass_kl = 0.0
    for update_pass in range(PPO_PASSES):
        order = list(range(len(episodes)))
        random.shuffle(order)
        for start in range(0, len(order), ACCUMULATION_STEPS):
            batch = order[start:start + ACCUMULATION_STEPS]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss, diagnostics = ppo_episode_loss(episodes[index])
                scaler.scale(loss / len(batch)).backward()
                for key, value in diagnostics.items():
                    totals[key].append(value)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(heads.parameters()), MAX_GRAD_NORM
            )
            if not torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                totals["amp_skipped"].append(1.0)
                optimizer.zero_grad(set_to_none=True)
                continue
            totals["grad_norm"].append(float(grad_norm))
            totals["amp_skipped"].append(0.0)
            scaler.step(optimizer)
            scaler.update()
        passes_completed += 1
        pass_kl = measure_rollout_kl(episodes)
        if pass_kl > TARGET_KL:
            print(f"PPO early stop after pass {passes_completed}: KL={pass_kl:.5f}")
            break
    return {key: float(np.mean(values)) if values else float("nan")
            for key, values in totals.items()} | {
        "kl": pass_kl,
        "passes": passes_completed,
        "amp_scale": float(scaler.get_scale()),
    }
'''


def replace_function(source, name, replacement):
    tree = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected one top-level function {name}; got {len(nodes)}")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [replacement.rstrip() + "\n"]
    result = "".join(lines)
    ast.parse(result)
    return result


def replace_block(source, begin, end, replacement):
    start = source.index(begin)
    finish = source.index(end, start)
    return source[:start] + replacement + source[finish:]


def build_train_source():
    source = RRPO_TRAIN.read_text(encoding="utf-8")
    marker = "# %%\n"
    source = source[source.index(marker):]
    for old, new in (
        ("ast_rrpo_unixcoder_v6_k5_ctx2048_xfb1344",
         "ast_ppo_terminal_v1_k10_ctx3072_xfb2344"),
        ("ast_rrpo_eval_v6_k5_ctx2048_xfb1344",
         "ast_ppo_terminal_eval_v1_k10_ctx3072_xfb2344"),
        ("RRPO_FINAL_WEIGHT", "FINAL_UTILITY_WEIGHT"),
        ("RRPO_IDENTIFIER_WEIGHT", "IDENTIFIER_F1_WEIGHT"),
        ("RRPO_GAMMA", "PPO_GAMMA"),
        ("RRPO_GAE_LAMBDA", "PPO_GAE_LAMBDA"),
        ("RRPO", "PPO"),
        ("score_rrpo_episodes", "score_ppo_episodes"),
        ("rrpo_step_rewards", "shaped_step_rewards"),
        ("rrpo_unixcoder_ast", "ppo_unixcoder_ast"),
        ("ast_bm25_token_ids_rrpo_v2.pt", "ast_bm25_token_ids_ppo_v1.pt"),
    ):
        source = source.replace(old, new)
    source = source.replace("GENERATOR_INPUT_TOKENS = 2048", "GENERATOR_INPUT_TOKENS = 3072")
    source = source.replace("CROSSFILE_TOKEN_BUDGET = 1344", "CROSSFILE_TOKEN_BUDGET = 2344")
    source = source.replace("MAX_SLATE_STEPS = 5", "MAX_SLATE_STEPS = 10")
    source = source.replace('PPO_REFERENCE = "bm25_ast_from_actual_state_v1"\n', "")
    source = source.replace("FINAL_UTILITY_WEIGHT = 0.7\n", "")
    source = source.replace("Undiscounted final-plus-prefix utility.",
                            "Undiscounted terminal utility.")
    source = source.replace("PPO_GAE_LAMBDA = 0.95\n", """PPO_GAE_LAMBDA = 0.95
VALUE_COEF = 0.5
VALUE_CLIP = 0.2
""".lstrip("+"))
    signature = '''DATA_SIGNATURE = {"mode": RUN_MODE, "seed": SEED, "train": TRAIN_EXAMPLES,
                  "valid": VALID_EXAMPLES, "pool_schema": 6,
                  "data_pipeline_schema": DATA_PIPELINE_SCHEMA,
                  "score_schema": SCORE_SCHEMA,
                  "file_selection_policy": "all_repo_files_then_query_bm25",
                  "sampling": "full_fixed_repo_rows_v1",
                  "dataset_sha256": DATASET_SHA256,
                  "valid_examples_sha256": VALID_SHA256,
                  "max_related_files": MAX_RELATED_FILES,
                  "pool": CANDIDATE_POOL_SIZE, "chunk": AST_CHUNK_TOKENS,
                  "encode_batch_size": ENCODE_BATCH_SIZE,
                  "retriever": RETRIEVER_MODEL, "generator": GENERATOR_MODEL,
                  "retriever_query_length": RETRIEVER_QUERY_LENGTH,
                  "retriever_candidate_length": RETRIEVER_CANDIDATE_LENGTH,
                  "crossfile_budget": CROSSFILE_TOKEN_BUDGET,
                  "max_slate_steps": MAX_SLATE_STEPS,
                  "generator_output_tokens": GENERATOR_OUTPUT_TOKENS,
                  "generator_input_tokens": GENERATOR_INPUT_TOKENS,
                  "generator_model_len": GENERATOR_MAX_MODEL_LEN,
                  "prompt_packer": "ast_relevant_hints_cursor_suffix_v2",
                  "objective": "ppo_clip_terminal_reward_learned_critic_gae_v1",
                  "advantage_estimator": "learned_critic_gae_v1",
                  "identifier_f1_weight": IDENTIFIER_F1_WEIGHT,
                  "ppo_gamma": PPO_GAMMA,
                  "ppo_gae_lambda": PPO_GAE_LAMBDA,
                  "value_coef": VALUE_COEF, "value_clip": VALUE_CLIP,
                  "ppo_clip": PPO_CLIP, "ppo_passes": PPO_PASSES,
                  "accumulation_steps": ACCUMULATION_STEPS,
                  "target_kl": TARGET_KL,
                  "encoder_lr": ENCODER_LR, "head_lr": HEAD_LR,
                  "max_grad_norm": MAX_GRAD_NORM,
                  "entropy_coef": ENTROPY_COEF,
                  "reward": "terminal_rlcoder_es_identifier_f1_v1"}'''
    source = replace_block(source, "DATA_SIGNATURE = {", "\n\nVALIDATION_SIGNATURE_KEYS", signature)
    source = replace_function(source, "rollout", ROLLOUT)
    source = replace_function(source, "shaped_step_rewards", TERMINAL_STEP_REWARDS)
    source = replace_function(source, "score_ppo_episodes", SCORE_EPISODES)
    source = replace_function(source, "prepare_advantages", PREPARE_ADVANTAGES)
    source = replace_function(source, "ppo_episode_loss", PPO_EPISODE_LOSS)
    source = replace_function(source, "ppo_update", PPO_UPDATE)
    source = source.replace('"mean_reference_value": float(np.mean([\n                          value for ep in episodes for value in ep["reference_values"]]))',
                            '"mean_rollout_value": float(np.mean([\n                          step["old_value"] for ep in episodes for step in ep["steps"]]))')
    if '"reference_values"' in source:
        raise AssertionError("PPO training source still refers to RRPO reference values")
    source = TRAIN_MARKDOWN + source
    ast.parse(source)
    return source


def build_eval_source():
    source = RRPO_EVAL.read_text(encoding="utf-8")
    source = source.replace(
        "Evaluate AST/BM25/custom-RRPO retrieval",
        "Evaluate AST/BM25/learned-critic-PPO retrieval",
    )
    for old, new in (
        ("ast_rrpo_unixcoder_v6_k5_ctx2048_xfb1344",
         "ast_ppo_terminal_v1_k10_ctx3072_xfb2344"),
        ("ast_rrpo_eval_v6_k5_ctx2048_xfb1344",
         "ast_ppo_terminal_eval_v1_k10_ctx3072_xfb2344"),
        ("rrpo_best_ast", "ppo_best_ast"),
        ("rrpo_latest_ast", "ppo_latest_ast"),
        ("rrpo_select", "ppo_select"),
        ("RRPO", "PPO"),
    ):
        source = source.replace(old, new)
    source = source.replace("GENERATOR_INPUT_TOKENS = 2048", "GENERATOR_INPUT_TOKENS = 3072")
    source = source.replace("CROSSFILE_TOKEN_BUDGET = 1344", "CROSSFILE_TOKEN_BUDGET = 2344")
    source = source.replace("MAX_SLATE_STEPS = 5", "MAX_SLATE_STEPS = 10")
    source = source.replace("K=5 PPO checkpoints", "K=10 PPO checkpoints")
    source = source.replace(
        '"objective": "rrpo_final_plus_prefix_es_idf1_v2",',
        '"objective": "ppo_clip_terminal_reward_learned_critic_gae_v1",\n'
        '    "advantage_estimator": "learned_critic_gae_v1",\n'
        '    "ppo_gamma": 1.0, "ppo_gae_lambda": 0.95,\n'
        '    "value_coef": 0.5, "value_clip": 0.2,\n'
        '    "identifier_f1_weight": 0.2,\n'
        '    "reward": "terminal_rlcoder_es_identifier_f1_v1",',
    )
    source = source.replace('    "rrpo_final_weight": 0.7,\n', "")
    source = source.replace('    "rrpo_identifier_weight": 0.2,\n', "")
    source = source.replace('    "reward": "rrpo_rlcoder_es_identifier_f1_v2",\n', "")
    source = source.replace('"rrpo_reference": "bm25_ast_from_actual_state_v1",\n', "")
    source = source.replace('        "generator_output_tokens": GENERATOR_OUTPUT_TOKENS,',
                            '    "generator_output_tokens": GENERATOR_OUTPUT_TOKENS,')
    source = source.replace('"reference": expected["rrpo_reference"],',
                            '"reward_mode": "terminal_only",')
    if "expected[\"rrpo_reference\"]" in source:
        raise AssertionError("PPO eval source still expects RRPO reference metadata")
    ast.parse(source)
    return source


def write_notebook(source_path, template_path, output_path):
    notebook = json.loads(template_path.read_text(encoding="utf-8"))
    parsed = source_cells(source_path.read_text(encoding="utf-8"))
    if len(parsed) != len(notebook["cells"]):
        raise ValueError(f"Cell count mismatch: template={len(notebook['cells'])}, "
                         f"source={len(parsed)} for {output_path}")
    cells = []
    for (kind, lines), old in zip(parsed, notebook["cells"]):
        if old["cell_type"] != kind:
            raise ValueError(f"Cell type mismatch while creating {output_path}")
        cell = dict(old)
        cell["source"] = lines
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            compile("".join(lines), str(output_path), "exec")
        cells.append(cell)
    notebook["cells"] = cells
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")


def main():
    train_source = build_train_source()
    eval_source = build_eval_source()
    PPO_TRAIN.write_text(train_source, encoding="utf-8")
    PPO_EVAL.write_text(eval_source, encoding="utf-8")
    write_notebook(PPO_TRAIN, RRPO_TRAIN_NOTEBOOK,
                   PPO_TRAIN_NOTEBOOK)
    write_notebook(PPO_EVAL, RRPO_EVAL_NOTEBOOK,
                   PPO_EVAL.with_suffix(".ipynb"))
    print("Generated learned-critic PPO train/eval source + Kaggle notebooks:")
    for path in (PPO_TRAIN, PPO_TRAIN_NOTEBOOK,
                 PPO_EVAL, PPO_EVAL.with_suffix(".ipynb")):
        print(path)


if __name__ == "__main__":
    main()
