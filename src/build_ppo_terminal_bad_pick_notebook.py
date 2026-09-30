"""Build a standalone Kaggle notebook for terminal PPO with harmful-pick penalty.

The input is the current, self-contained terminal-PPO notebook. This builder
copies only its training cells, switches generation to the dedicated Modal
endpoint, and replaces the scalar terminal reward with leave-one-out utility
penalty. It does not alter the existing terminal-PPO notebook.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).parent
BASE_NOTEBOOK = ROOT / "repo_level_ast_retrieval_ppo_clip_k10_train_kaggle.ipynb"
OUTPUT_NOTEBOOK = ROOT / "ast_ppo_unixcoder_terminal_bad_pick_k10_modal_kaggle.ipynb"


MARKDOWN = '''# AST retrieval + terminal PPO with counterfactual harmful-pick penalty — Kaggle T4×2

Training-only variant of the current terminal-reward PPO notebook. It keeps the
same prebuilt Python+Java train/validation Parquet, AST chunking, 64-candidate
pool, K≤10 slate policy, 2,344 cross-file token budget, 3,072 input-token limit,
RLCoder-style ES/identifier-F1 scorer, learned critic, PPO-Clip, and GAE.

DeepSeek-Coder-1.3B is served remotely by the dedicated Modal A100-80GB endpoint
in `src/modal_deepseek_vllm_ppo_terminal_endpoint.py`; UniXcoder and PPO train on
Kaggle T4 GPU 1. Add the Kaggle secret `MODAL_PROXY_TOKEN` (combined token ID and
secret) and set the deployed endpoint URL in the configuration cell. This
notebook does not prepare data or evaluate held-out test sets.

At the end of each episode, it scores the selected slate once and then scores
leave-one-selected-chunk-out slates. For a selected chunk, marginal gain is
`U(full slate) - U(without chunk)`. A chunk is penalized if its gain is below
0.02: harmful chunks have negative gain, while redundant/low-value chunks fail
the minimum-usefulness threshold. This makes the learned STOP decision trade
off utility against unnecessary selections. The summed shortfall is capped and
subtracted from full-slate utility; this remains one scalar reward at the final
PPO transition. It does not construct pairwise training examples, use a
pairwise ranking loss, or add per-step reward shaping. It adds up to K extra
completion prompts per episode; exact prompts/responses are deduplicated and
persistently cached in SQLite.

Checkpointing is isolated from old PPO/RRPO runs. A training exception writes
the fatal traceback, attempts an atomic latest checkpoint, and re-raises so the
session stops. The checkpoint is also saved at the normal safe interval and
validation improvements. `KAGGLE T4×2`, the prepared dataset, and the Modal
proxy secret are required. The run stops after two complete train epochs. A
seven-hour wall-clock guard remains as a Modal spending safety ceiling; a
partial epoch resumes from its saved cursor rather than counting as complete.
'''


MODAL_SETUP = '''# Modal endpoint: set this to the URL printed by `modal deploy` for the
# dedicated PPO terminal server. Do not paste the proxy credential into code.
MODAL_ENDPOINT_ROOT = (
    "https://bien14112005--bienkieu-ppo-terminal-badpick-deepseek-vll-add36f."
    "us-east.modal.direct"
)
ENDPOINT_READY_TIMEOUT_SECONDS = 15 * 60
ENDPOINT_RECOVERY_TIMEOUT_SECONDS = 5 * 60
GENERATOR_REQUEST_TIMEOUT_SECONDS = 600
GENERATOR_RETRY_BACKOFF_SECONDS = 3
VLLM_MAX_NUM_SEQS = 108
VLLM_SERVER_VERSION = "0.21.0"

try:
    from kaggle_secrets import UserSecretsClient
    _kaggle_secrets = UserSecretsClient()
    MODAL_PROXY_TOKEN = _kaggle_secrets.get_secret("MODAL_PROXY_TOKEN").strip()
except Exception as exc:
    raise RuntimeError(
        "Add Kaggle Secret MODAL_PROXY_TOKEN (combined token ID.secret) before running."
    ) from exc

if (not MODAL_ENDPOINT_ROOT.startswith("https://") or
        not MODAL_PROXY_TOKEN):
    raise ValueError("Set the deployed HTTPS Modal endpoint URL and proxy-token secret.")
MODAL_ENDPOINT_ROOT = MODAL_ENDPOINT_ROOT.rstrip("/")
API_BASE = (MODAL_ENDPOINT_ROOT if MODAL_ENDPOINT_ROOT.endswith("/v1")
            else MODAL_ENDPOINT_ROOT + "/v1")
MODAL_HEALTH_URL = API_BASE[:-3] + "/health"
MODAL_AUTH_HEADERS = {"Authorization": f"Bearer {MODAL_PROXY_TOKEN}"}
del MODAL_PROXY_TOKEN, _kaggle_secrets
'''


MODAL_READINESS = '''def verify_served_model(cards):
    matching = [card for card in cards if card.get("id") == SERVED_MODEL_NAME]
    if len(matching) != 1:
        raise RuntimeError(
            f"Modal endpoint does not serve {SERVED_MODEL_NAME}: "
            f"{[card.get('id') for card in cards]}"
        )
    advertised = matching[0].get("max_model_len")
    if advertised is None or int(advertised) < GENERATOR_MAX_MODEL_LEN:
        raise RuntimeError(
            f"Modal vLLM must advertise max_model_len >= {GENERATOR_MAX_MODEL_LEN}; "
            f"received {advertised}. Redeploy the dedicated endpoint."
        )
    print("vLLM advertised context length:", advertised, flush=True)

def wait_for_modal_vllm(timeout_seconds=ENDPOINT_READY_TIMEOUT_SECONDS):
    deadline = time.monotonic() + timeout_seconds
    last_report = 0.0
    last_error = "endpoint is starting"
    while time.monotonic() < deadline:
        try:
            health = requests.get(MODAL_HEALTH_URL, headers=MODAL_AUTH_HEADERS,
                                  timeout=15)
            if health.status_code in (401, 403):
                raise RuntimeError(
                    f"Modal authentication failed (HTTP {health.status_code}); "
                    "check Kaggle Secret MODAL_PROXY_TOKEN."
                )
            if health.ok:
                models = requests.get(API_BASE + "/models",
                                      headers=MODAL_AUTH_HEADERS, timeout=20)
                if models.status_code in (401, 403):
                    raise RuntimeError(
                        f"Modal authentication failed (HTTP {models.status_code}); "
                        "check Kaggle Secret MODAL_PROXY_TOKEN."
                    )
                if models.ok:
                    verify_served_model(models.json().get("data", []))
                    print("Authenticated Modal vLLM endpoint ready:",
                          MODAL_ENDPOINT_ROOT, "batch:", VLLM_MAX_NUM_SEQS,
                          flush=True)
                    return
                last_error = f"/v1/models returned HTTP {models.status_code}"
            else:
                last_error = f"/health returned HTTP {health.status_code}"
        except RuntimeError:
            raise
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        now = time.monotonic()
        if now - last_report >= 30:
            print("Waiting for Modal vLLM endpoint:", last_error,
                  f"{max(0, int(deadline - now))}s remaining", flush=True)
            last_report = now
        time.sleep(10)
    raise TimeoutError(
        f"Modal vLLM endpoint was not ready in {timeout_seconds}s: {last_error}. "
        "Training has not started; verify the deployment URL and endpoint logs."
    )

wait_for_modal_vllm()
'''


GENERATE_COMPLETIONS = '''def generate_completions(prompts, progress_label=None):
    """Authenticated, bounded-retry Modal calls with deterministic SQLite caching."""
    if not prompts:
        return []
    prompt_by_key = {}
    ordered_keys = []
    for prompt in prompts:
        key = completion_cache_key(prompt)
        ordered_keys.append(key)
        prompt_by_key.setdefault(key, prompt)
    unique_keys = list(prompt_by_key)
    cached = {}
    for offset in range(0, len(unique_keys), 400):
        keys = unique_keys[offset:offset + 400]
        placeholders = ",".join("?" for _ in keys)
        cached.update(GENERATION_CACHE_CONNECTION.execute(
            f"SELECT cache_key, completion FROM generation_cache "
            f"WHERE cache_key IN ({placeholders})", keys
        ).fetchall())

    missing = [(key, prompt_by_key[key]) for key in unique_keys if key not in cached]
    persistent_hits = sum(key in cached for key in ordered_keys)
    duplicate_reuse = len(prompts) - len(unique_keys)
    batch_size = max(1, VLLM_MAX_NUM_SEQS)
    total_batches = math.ceil(len(missing) / batch_size)
    if progress_label or persistent_hits or duplicate_reuse:
        label = f"{progress_label}: " if progress_label else ""
        print(f"{label}cache: {persistent_hits}/{len(prompts)} persistent hits, "
              f"{duplicate_reuse} duplicate reuses, {len(missing)} new prompts",
              flush=True)

    generated = {}
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    for start in range(0, len(missing), batch_size):
        batch_number = start // batch_size + 1
        batch = missing[start:start + batch_size]
        if progress_label:
            print(f"{progress_label}: Modal batch {batch_number}/{total_batches} "
                  f"starting; generated={start}/{len(missing)}", flush=True)
        token_ids = [GEN_TOKENIZER.encode(prompt, add_special_tokens=True)
                     for _, prompt in batch]
        if any(len(ids) > GENERATOR_INPUT_TOKENS for ids in token_ids):
            raise ValueError(
                f"A prompt exceeds {GENERATOR_INPUT_TOKENS} input tokens; "
                "refusing to truncate left context or retrieved chunks."
            )
        request_payload = {
            "model": SERVED_MODEL_NAME, "prompt": token_ids,
            "max_tokens": GENERATOR_OUTPUT_TOKENS,
            "temperature": 0.0, "top_p": 1.0,
        }
        recovery_deadline = time.monotonic() + ENDPOINT_RECOVERY_TIMEOUT_SECONDS
        response = None
        retry_count = 0
        while time.monotonic() < recovery_deadline:
            timeout = min(GENERATOR_REQUEST_TIMEOUT_SECONDS,
                          recovery_deadline - time.monotonic())
            try:
                response = requests.post(
                    API_BASE + "/completions", headers=MODAL_AUTH_HEADERS,
                    json=request_payload, timeout=timeout,
                )
            except requests.RequestException as exc:
                retry_count += 1
                response = None
                error_text = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        f"Modal authentication failed (HTTP {response.status_code}); "
                        "check Kaggle Secret MODAL_PROXY_TOKEN."
                    )
                if response.status_code not in retryable_statuses:
                    break
                retry_count += 1
                error_text = f"HTTP {response.status_code}: {response.text[:500]}"
            remaining = recovery_deadline - time.monotonic()
            if remaining <= 0:
                break
            delay = min(GENERATOR_RETRY_BACKOFF_SECONDS *
                        (2 ** min(retry_count - 1, 4)), 30.0, remaining)
            print(f"Modal batch temporarily failed ({error_text}); retry "
                  f"{retry_count} in {delay:.1f}s.", flush=True)
            time.sleep(delay)
        if response is None or response.status_code in retryable_statuses:
            raise RuntimeError(
                f"Modal batch {batch_number}/{total_batches} failed after "
                f"{retry_count} bounded retries; outer training handler will "
                "checkpoint and stop."
            )
        if not response.ok:
            raise RuntimeError(
                f"Non-retryable vLLM HTTP {response.status_code}: "
                f"{response.text[:2000]}"
            )
        try:
            choices = sorted(response.json()["choices"],
                             key=lambda item: item.get("index", 0))
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Modal vLLM returned malformed completion JSON.") from exc
        if len(choices) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} completions, received {len(choices)}."
            )
        batch_outputs = [str(choice.get("text", "")) for choice in choices]
        generated.update((key, output)
                         for (key, _prompt), output in zip(batch, batch_outputs))
        GENERATION_CACHE_CONNECTION.executemany(
            "INSERT INTO generation_cache (cache_key, completion) VALUES (?, ?) "
            "ON CONFLICT(cache_key) DO UPDATE SET completion=excluded.completion",
            list(zip((key for key, _prompt in batch), batch_outputs)),
        )
        GENERATION_CACHE_CONNECTION.commit()
        if progress_label:
            print(f"{progress_label}: Modal batch {batch_number}/{total_batches} "
                  f"done; generated={start + len(batch)}/{len(missing)}; "
                  f"cache entries={GENERATION_CACHE_CONNECTION.execute(
                      'SELECT COUNT(*) FROM generation_cache').fetchone()[0]}",
                  flush=True)
    outputs_by_key = {**cached, **generated}
    return [outputs_by_key[key] for key in ordered_keys]
'''


SCORE_EPISODES = '''def score_ppo_episodes(episodes):
    """Score the final slate and its one-chunk ablations; emit one terminal scalar."""
    if not episodes:
        return 0
    prompts = []
    prompt_refs = []
    for episode_index, episode in enumerate(episodes):
        selected = list(episode["selected"])
        prompts.append(compose_prompt(episode["row"], selected))
        prompt_refs.append((episode_index, None))
        for removed_position in range(len(selected)):
            ablated = selected[:removed_position] + selected[removed_position + 1:]
            prompts.append(compose_prompt(episode["row"], ablated))
            prompt_refs.append((episode_index, removed_position))
    print(f"PPO counterfactual terminal scoring: {len(episodes)} episodes, "
          f"{len(prompts)} full/leave-one-out prompts, K={MAX_SLATE_STEPS}, "
          f"Modal batch={VLLM_MAX_NUM_SEQS}.", flush=True)
    outputs = generate_completions(prompts, progress_label="PPO counterfactual rewards")
    if len(outputs) != len(prompt_refs):
        raise AssertionError("Completion count differs from full/ablation prompt count")
    full_components = [None] * len(episodes)
    ablation_utilities = [[] for _ in episodes]
    for (episode_index, removed_position), output in zip(prompt_refs, outputs):
        episode = episodes[episode_index]
        component = synthetic_reward(output, episode["row"])
        if removed_position is None:
            full_components[episode_index] = component
        else:
            ablation_utilities[episode_index].append(component["utility"])

    for episode_index, episode in enumerate(episodes):
        components = full_components[episode_index]
        if components is None:
            raise AssertionError("Every episode must have one full-slate score")
        full_utility = float(components["utility"])
        marginal_gains = [full_utility - float(ablated_utility)
                          for ablated_utility in ablation_utilities[episode_index]]
        shortfalls = [max(0.0, WRONG_PICK_MIN_GAIN - gain)
                      for gain in marginal_gains]
        total_shortfall = sum(shortfalls)
        capped_shortfall = min(WRONG_PICK_PENALTY_CAP, total_shortfall)
        penalty = WRONG_PICK_PENALTY_WEIGHT * capped_shortfall
        terminal_utility = full_utility - penalty
        step_rewards = terminal_step_rewards(len(episode["steps"]), terminal_utility)
        if len(step_rewards) != len(episode["steps"]):
            raise AssertionError("Terminal reward and sampled actions differ in length")
        episode["final_utility"] = full_utility
        episode["final_es"] = float(components["es"])
        episode["final_id_f1"] = float(components["id_f1"])
        episode["wrong_pick_penalty"] = float(penalty)
        episode["harmful_selection_count"] = sum(gain < 0.0 for gain in marginal_gains)
        episode["subthreshold_selection_count"] = sum(
            gain < WRONG_PICK_MIN_GAIN for gain in marginal_gains
        )
        episode["reward"] = float(terminal_utility)
        episode["step_rewards"] = step_rewards
    print("Counterfactual penalty:", {
        "mean": float(np.mean([ep["wrong_pick_penalty"] for ep in episodes])),
        "harmful_picks": int(sum(ep["harmful_selection_count"] for ep in episodes)),
        "below_min_gain_picks": int(sum(
            ep["subthreshold_selection_count"] for ep in episodes
        )),
        "min_marginal_gain": WRONG_PICK_MIN_GAIN,
        "weight": WRONG_PICK_PENALTY_WEIGHT,
        "cap": WRONG_PICK_PENALTY_CAP,
    }, flush=True)
    return len(prompts)
'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one {label} marker; found {count}")
    return source.replace(old, new, 1)


def replace_function(source: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected one top-level {name}; found {len(nodes)}")
    lines = source.splitlines(keepends=True)
    node = nodes[0]
    lines[node.lineno - 1:node.end_lineno] = [replacement.rstrip() + "\n"]
    result = "".join(lines)
    ast.parse(result)
    return result


def modify_notebook() -> None:
    notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    if len(notebook["cells"]) != 10:
        raise ValueError("Base terminal PPO notebook changed; inspect before regenerating")
    cells = notebook["cells"][:9]  # Intentionally omit held-out test evaluation.
    cells[0]["source"] = MARKDOWN.splitlines(keepends=True)

    config = "".join(cells[1]["source"])
    config = replace_once(
        config,
        "import time\nimport zlib\nfrom datetime import datetime, timedelta\n",
        "import time\nimport zlib\nimport traceback\nfrom datetime import datetime, timedelta\n",
        "traceback import",
    )
    config = replace_once(
        config,
        'WORK_DIR = Path("/kaggle/working/ast_ppo_terminal_v1_k10_ctx3072_xfb2344")',
        'WORK_DIR = Path("/kaggle/working/ast_ppo_terminal_badpick_v1_k10_ctx3072_xfb2344")',
        "isolated work directory",
    )
    config = replace_once(
        config,
        'API_BASE = "http://127.0.0.1:8000/v1"\n'
        'VLLM_PORT = 8000\n'
        'VLLM_GPU_MEMORY_UTILIZATION = 0.78\n'
        'VLLM_MAX_NUM_SEQS = 16\n',
        'API_BASE = ""  # Filled from the deployed Modal endpoint.\n'
        'VLLM_MAX_NUM_SEQS = 108\n',
        "remote-vLLM configuration",
    )
    config = replace_once(config, "ROLLOUT_QUERIES = 32", "ROLLOUT_QUERIES = 64",
                          "64-episode rollout batch")
    config = replace_once(
        config,
        "IDENTIFIER_F1_WEIGHT = 0.2\n",
        "IDENTIFIER_F1_WEIGHT = 0.2\n"
        "WRONG_PICK_PENALTY_WEIGHT = 0.5\n"
        "WRONG_PICK_PENALTY_CAP = 1.0\n"
        "WRONG_PICK_MIN_GAIN = 0.02\n",
        "bounded harmful-pick penalty configuration",
    )
    config = replace_once(
        config,
        "TRAIN_STOP_HOURS_FROM_KERNEL_START = 11.0",
        "TRAIN_STOP_HOURS_FROM_KERNEL_START = 7.0",
        "Modal-budgeted stop guard",
    )
    config = replace_once(
        config,
        "TRAIN_STOP_HOURS_FROM_KERNEL_START = 7.0\n",
        "TRAIN_STOP_HOURS_FROM_KERNEL_START = 7.0\nMAX_TRAIN_EPOCHS = 2\n",
        "two-epoch limit",
    )
    cells[1]["source"] = config.splitlines(keepends=True)

    runtime = "".join(cells[2]["source"])
    runtime = replace_once(
        runtime,
        "import requests\n",
        "import requests\n\n" + MODAL_SETUP + "\n",
        "Modal authentication setup",
    )
    cells[2]["source"] = runtime.splitlines(keepends=True)

    signature_source = "".join(cells[5]["source"])
    signature_source = replace_once(
        signature_source,
        'RESUMABLE_TUNING_KEYS = {"encode_batch_size", "accumulation_steps", "encoder_lr"}',
        'RESUMABLE_TUNING_KEYS = {"encode_batch_size", "accumulation_steps", '
        '"encoder_lr", "rollout_queries"}',
        "safe rollout-batch resume compatibility",
    )
    signature_source = replace_once(
        signature_source,
        '"objective": "ppo_clip_terminal_reward_learned_critic_gae_v1",',
        '"objective": "ppo_clip_terminal_leave_one_out_bad_pick_v1",',
        "checkpoint objective signature",
    )
    signature_source = replace_once(
        signature_source,
        '"reward": "terminal_rlcoder_es_identifier_f1_v1"}',
        '"reward": "terminal_es_idf1_minus_capped_loo_harm_v1",\n'
        '                  "wrong_pick_penalty_weight": WRONG_PICK_PENALTY_WEIGHT,\n'
        '                  "wrong_pick_penalty_cap": WRONG_PICK_PENALTY_CAP,\n'
        '                  "wrong_pick_min_gain": WRONG_PICK_MIN_GAIN,\n'
        '                  "rollout_queries": ROLLOUT_QUERIES,\n'
        '                  "max_train_epochs": MAX_TRAIN_EPOCHS}',
        "reward/checkpoint signature",
    )
    cells[5]["source"] = signature_source.splitlines(keepends=True)

    generator_cell = "".join(cells[6]["source"])
    local_setup = generator_cell.split("def compose_prompt(row, selected):", 1)
    if len(local_setup) != 2 or "kaggle-vllm runtime" not in local_setup[0]:
        raise ValueError("Could not locate local-vLLM bootstrap in terminal-PPO cell")
    generator_cell = MODAL_READINESS + "\ndef compose_prompt(row, selected):" + local_setup[1]
    generator_cell = replace_function(
        generator_cell, "generate_completions", GENERATE_COMPLETIONS
    )
    cells[6]["source"] = generator_cell.splitlines(keepends=True)

    policy_cell = "".join(cells[7]["source"])
    policy_cell = replace_once(
        policy_cell,
        'if hasattr(encoder, "gradient_checkpointing_enable"):\n'
        '    try:\n'
        '        encoder.gradient_checkpointing_enable(\n'
        '            gradient_checkpointing_kwargs={"use_reentrant": False}\n'
        '        )\n'
        '    except TypeError:\n'
        '        encoder.gradient_checkpointing_enable()\n',
        'if hasattr(encoder, "gradient_checkpointing_disable"):\n'
        '    encoder.gradient_checkpointing_disable()\n',
        "disable gradient checkpointing for faster PPO updates",
    )
    policy_cell = replace_once(
        policy_cell,
        'encoder.train()  # Dropout is zero; keep HF gradient checkpointing active in updates.\n',
        'encoder.train()  # Dropout is zero; gradient checkpointing is disabled for faster updates.\n',
        "gradient-checkpointing status comment",
    )
    policy_cell = replace_function(policy_cell, "score_ppo_episodes", SCORE_EPISODES)
    cells[7]["source"] = policy_cell.splitlines(keepends=True)

    # The training cell is wrapped so errors in rollout, reward scoring, PPO,
    # validation, or finalization all checkpoint and then stop the Kaggle run.
    training_cell = "".join(cells[8]["source"])
    training_cell = replace_once(
        training_cell,
        '                      "mean_final_id_f1": float(np.mean([\n'
        '                          ep["final_id_f1"] for ep in episodes])),\n'
        '                      "mean_rollout_value": float(np.mean([\n',
        '                      "mean_final_id_f1": float(np.mean([\n'
        '                          ep["final_id_f1"] for ep in episodes])),\n'
        '                      "mean_wrong_pick_penalty": float(np.mean([\n'
        '                          ep["wrong_pick_penalty"] for ep in episodes])),\n'
        '                      "mean_harmful_picks": float(np.mean([\n'
        '                          ep["harmful_selection_count"] for ep in episodes])),\n'
        '                      "mean_below_min_gain_picks": float(np.mean([\n'
        '                          ep["subthreshold_selection_count"] for ep in episodes])),\n'
        '                      "mean_rollout_value": float(np.mean([\n',
        "counterfactual reward telemetry",
    )
    training_cell = replace_once(
        training_cell,
        '                      "epoch": STATE["epoch"],\n'
        '                      "epoch_progress_pct": round(100 * STATE["cursor"] /\n',
        '                      "epoch": STATE["epoch"],\n'
        '                      "max_epochs": MAX_TRAIN_EPOCHS,\n'
        '                      "epoch_progress_pct": round(100 * STATE["cursor"] /\n',
        "epoch-cap progress telemetry",
    )
    training_cell = replace_once(
        training_cell,
        "    except Exception:\n        save_checkpoint(CHECKPOINT_PATH)\n        raise\n",
        "    except Exception:\n        raise\n",
        "inner checkpoint handler (outer handler owns recovery)",
    )
    training_cell = replace_once(
        training_cell,
        "train_deadline = START_MONOTONIC + TRAIN_STOP_HOURS_FROM_KERNEL_START * 3600\n",
        "train_deadline = START_MONOTONIC + TRAIN_STOP_HOURS_FROM_KERNEL_START * 3600\n"
        "STOP_NEW_BATCH_RESERVE_SECONDS = 600\n",
        "safe Modal shutdown reserve",
    )
    training_cell = replace_once(
        training_cell,
        "while time.monotonic() < train_deadline:",
        "def max_epoch_run_complete():\n"
        "    return (STATE['epoch'] >= MAX_TRAIN_EPOCHS and\n"
        "            STATE['cursor'] >= TRAIN_EXAMPLES)\n\n"
        "while (time.monotonic() < train_deadline - STOP_NEW_BATCH_RESERVE_SECONDS\n"
        "       and not max_epoch_run_complete()):",
        "two-complete-epoch and Modal safety-stop guards",
    )
    training_cell = replace_once(
        training_cell,
        'print("Training stopped; elapsed hours:",\n'
        '      round((time.monotonic() - START_MONOTONIC) / 3600, 2))',
        'if max_epoch_run_complete():\n'
        '    print(f"Reached epoch cap: {STATE[\'epoch\']}/{MAX_TRAIN_EPOCHS} complete.")\n'
        'elif RUN_MODE == "smoke":\n'
        '    print("Smoke episode cap reached; this was not a full two-epoch run.")\n'
        'else:\n'
        '    print("Stopped by the wall-clock safety guard before both epochs completed; "\n'
        '          "resume from latest.pt to finish.")\n'
        'print("Training stopped; elapsed hours:",\n'
        '      round((time.monotonic() - START_MONOTONIC) / 3600, 2))',
        "epoch completion report",
    )
    wrapped = "try:\n" + "".join(
        "    " + line if line.strip() else line
        for line in training_cell.splitlines(keepends=True)
    )
    wrapped += '''except BaseException:
    failure_traceback = traceback.format_exc()
    FATAL_LOG_STREAM.write(failure_traceback + "\\n")
    FATAL_LOG_STREAM.flush()
    print("Training failed/interrupted; stopping after emergency checkpoint.",
          flush=True)
    try:
        GENERATION_CACHE_CONNECTION.commit()
    except Exception as cache_error:
        print("Generation-cache commit failed:", repr(cache_error), flush=True)
    try:
        save_checkpoint(CHECKPOINT_PATH)
    except BaseException as checkpoint_error:
        FATAL_LOG_STREAM.write(
            "Emergency checkpoint failed: " + repr(checkpoint_error) + "\\n"
        )
        FATAL_LOG_STREAM.flush()
        print("Emergency checkpoint failed; inspect python_fatal.log.", flush=True)
    raise
finally:
    try:
        GENERATION_CACHE_CONNECTION.commit()
        GENERATION_CACHE_CONNECTION.close()
    finally:
        FATAL_LOG_STREAM.flush()
        FATAL_LOG_STREAM.close()
'''
    ast.parse(wrapped)
    cells[8]["source"] = wrapped.splitlines(keepends=True)

    for index, cell in enumerate(cells):
        cell["id"] = f"ppo-badpick-{index:02d}"
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"{OUTPUT_NOTEBOOK}#cell-{index}", "exec")
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["cells"] = cells
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print("Built", OUTPUT_NOTEBOOK)
    print("Training cells:", len(cells), "| K=10 | episode batch=64 | Modal seqs=108")


if __name__ == "__main__":
    modify_notebook()
