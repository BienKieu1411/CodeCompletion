"""One-shot A100 job: co-resident vLLM + online CUR, persistent recovery.

Importing this module does not launch a GPU. Use `modal run -m ...` explicitly.
The localhost server is not deployed as a public endpoint.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import modal

APP_NAME = "codecompletion-cur-online"
VOLUME_NAME = "codecompletion-cur-online"
MOUNT = Path("/cur")
HARD_TIMEOUT = 11 * 3600
MAX_SOFT_HOURS = 10.75
SOURCE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL = SOURCE_ROOT / "dataset" / "data_prepared" / "train.parquet"
DEFAULT_VALID = SOURCE_ROOT / "dataset" / "data_prepared" / "valid.parquet"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgomp1")
         .pip_install("vllm==0.11.0", "torch==2.8.0", "transformers==4.57.1",
                      "numpy==2.2.6", "pyarrow==21.0.0", "requests==2.32.5",
                      "tree-sitter==0.25.2", "tree-sitter-python==0.25.0",
                      "tree-sitter-java==0.23.5", "fuzzywuzzy==0.18.0",
                      "python-Levenshtein==0.27.1", "modal==1.6.0")
         .pip_install("rank-bm25==0.2.2")
         .run_commands("python -c 'import rank_bm25, pyarrow, tree_sitter, tree_sitter_python, tree_sitter_java, fuzzywuzzy, Levenshtein'")
         .env({"HF_HOME": "/cur/hf-cache", "PYTHONPATH": "/workspace",
               "TOKENIZERS_PARALLELISM": "true", "PYTHONUNBUFFERED": "1",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "RAYON_NUM_THREADS": "2"})
         .workdir("/workspace")
         .add_local_dir(SOURCE_ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**", "**/*.pyc"]))


def check_arguments(run_id, epochs, soft_hours):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must be a simple 1–80 character name (letters/digits/_/-)")
    if epochs < 1 or not 0 < soft_hours <= MAX_SOFT_HOURS:
        raise ValueError(f"epochs must be positive; soft-hours must be in (0, {MAX_SOFT_HOURS}]")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_pool(path):
    """Read metadata only; never prepare or rewrite user's dataset."""
    import pyarrow.parquet as pq
    from src.data.ast_training_data import DataConfig
    from src.data.audit_completion_data import TOKENIZER_REVISION
    from src.data.build_ast_repo_pool import RET_REVISION
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    expected_columns = {"split", "language", "repo_id", "repo_uid", "payload"}
    if (not expected_columns.issubset(parquet.schema_arrow.names)
            or metadata.get(b"artifact_schema") != b"ast_repo_pool_v3"
            or b"preparation_contract" not in metadata or parquet.metadata.num_rows == 0):
        raise ValueError("Need the new ast_repo_pool_v3 train.parquet, not old PPO task rows")
    contract = json.loads(metadata[b"preparation_contract"])
    DataConfig(**contract["config"])
    if (contract["generator_tokenizer_revision"] != TOKENIZER_REVISION
            or contract["retriever_tokenizer_revision"] != RET_REVISION):
        raise ValueError("Prepared tokenizer revisions differ from training")
    return {"rows": parquet.metadata.num_rows, "sha256": sha256_file(path)}


def validate_valid(path):
    """Validate the held-out benchmark artifact; it is never used as train input."""
    from src.train.benchmark_validation import validate_file
    result = validate_file(path)
    return {**result, "sha256": sha256_file(path)}


def vllm_command(algorithm="potential"):
    from src.data.audit_completion_data import TOKENIZER_REVISION
    from src.train.build_utility_bundles import MODEL
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL, "--revision", TOKENIZER_REVISION,
            "--tokenizer-revision", TOKENIZER_REVISION, "--host", "127.0.0.1",
            "--port", "8000", "--tensor-parallel-size", "1", "--dtype", "bfloat16",
            "--max-model-len", "3264", "--gpu-memory-utilization", "0.55",
            "--max-num-seqs", "64", "--max-num-batched-tokens", "8192",
            "--enable-prefix-caching", "--enable-chunked-prefill"]
    if algorithm == "gain24":
        # Prompt-logprob workloads score prefill tokens, not long generations.
        args.remove("--enable-prefix-caching")
        args.append("--no-enable-prefix-caching")
        args[args.index("--max-num-batched-tokens") + 1] = "8192"
        args[args.index("--max-num-seqs") + 1] = "128"
        args[args.index("--gpu-memory-utilization") + 1] = "0.35"
    return args


def trainer_command(pool, output, epochs, deadline, stop_file, resume, valid=None, algorithm="potential"):
    args = [sys.executable, "-u", "-m", "src.train.train_online_utility",
            "--pool", str(pool), "--base-url", "http://127.0.0.1:8000",
            "--output", str(output), "--epochs", str(epochs),
            "--batch-size", "16", "--generator-batch-size", "128",
            "--encoder-microbatch", "64", "--encoder-backward", "direct",
            "--candidate-pool-size", "100",
            "--encoder-lr", "5e-5", "--head-lr", "2e-4", "--precision", "bf16",
            "--save-every", "5", "--max-minutes", str(MAX_SOFT_HOURS * 60),
            "--deadline-unix", str(deadline),
            "--stop-file", str(stop_file), "--checkpoint-volume", VOLUME_NAME]
    if algorithm == "gain24":
        args += ["--algorithm", "gain24", "--schedule", "warmup_constant", "--warmup-updates", "20",
                 "--valid-every-epochs", "0"]
    if valid is not None:
        args += ["--valid", str(valid)]
    if resume:
        args += ["--resume", str(output / "latest.pt")]
    return args


def start_logged(command, log_path, label):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)

    def drain():
        with Path(log_path).open("a", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{label}] {line}", end="", flush=True)
        process.stdout.close()

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return process, thread


def stop_group(process):
    """Stop only a process group created by this launcher, including its workers."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Workers may outlive an already-exited parent.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else "nvidia-smi unavailable"


@app.function(image=image, gpu="A100-80GB", cpu=(2, 4), memory=(16384, 24576),
              volumes={str(MOUNT): volume}, timeout=HARD_TIMEOUT, startup_timeout=900,
              retries=0, min_containers=0, max_containers=1, buffer_containers=0,
              scaledown_window=5, single_use_containers=True)
def train(run_id: str, pool_sha256: str, epochs: int = 5, resume: bool = False,
          soft_hours: float = MAX_SOFT_HOURS, valid_sha256: str = "", algorithm: str = "potential"):
    import requests
    check_arguments(run_id, epochs, soft_hours)
    if algorithm not in {"potential", "gain24"}:
        raise ValueError("Unknown training algorithm")
    if not re.fullmatch(r"[0-9a-f]{64}", pool_sha256):
        raise ValueError("Expected SHA-256 of the prepared pool")
    started = time.time()
    deadline = started + soft_hours * 3600
    kill_deadline = min(started + HARD_TIMEOUT - 300, deadline + (240 if algorithm == "gain24" else 600))
    pool = MOUNT / "inputs" / pool_sha256 / "train.parquet"
    if validate_pool(pool)["sha256"] != pool_sha256:
        raise ValueError("Uploaded pool digest mismatch")
    valid = None
    if valid_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", valid_sha256):
            raise ValueError("Expected SHA-256 of prepared validation parquet")
        valid = MOUNT / "inputs" / valid_sha256 / "valid.parquet"
        if validate_valid(valid)["sha256"] != valid_sha256:
            raise ValueError("Uploaded validation digest mismatch")
    output = MOUNT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=resume)
    if resume and not (output / "latest.pt").is_file():
        raise ValueError("Resume requested but this run has no latest.pt")
    # Unique per invocation: never remove an old stop marker on user's behalf.
    session_id = str(time.time_ns())
    stop_file = output / f"stop-{session_id}.request"
    server, worker, threads = None, None, []
    result = {"status": "failed", "run_id": run_id, "session_id": session_id}
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (output / f"environment-{session_id}.txt").write_text(freeze.stdout)
        volume.commit()
        print(json.dumps({"phase": "start", "gpu": gpu_snapshot(), "deadline_unix": deadline,
                          "epochs": epochs, "pool_sha256": pool_sha256,
                          "valid_sha256": valid_sha256 or None}), flush=True)
        server, thread = start_logged(vllm_command(algorithm), output / "vllm.log", "vllm")
        threads.append(thread)
        startup_deadline = min(deadline, time.time() + 900)
        with requests.Session() as client:
            while True:
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited during startup; see vllm.log")
                if time.time() >= startup_deadline:
                    raise TimeoutError("vLLM did not become healthy within the startup allowance")
                try:
                    if client.get("http://127.0.0.1:8000/health", timeout=3).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(3)
        # Load trainer only after vLLM profiling completes. Both stay resident.
        worker, thread = start_logged(trainer_command(pool, output, epochs, deadline, stop_file, resume, valid, algorithm),
                                      output / "train.log", "train")
        threads.append(thread)
        next_snapshot, server_failed = 0., False
        while worker.poll() is None:
            now = time.time()
            if now >= deadline or server.poll() is not None:
                if not stop_file.exists():
                    stop_file.write_text("server-exit" if server.poll() is not None else "time-budget")
                if server.poll() is not None and not server_failed:
                    server_failed = True
                    kill_deadline = min(kill_deadline, now + 240)
            if now >= kill_deadline:
                result["status"] = "forced_stop"
                break
            if now >= next_snapshot:
                print(json.dumps({"phase": "gpu", "snapshot": gpu_snapshot(),
                                  "remaining_minutes": max(0, (deadline-now)/60)}), flush=True)
                next_snapshot = now + 60
            time.sleep(2)
        if worker.poll() == 0 and not server_failed:
            status_path = output / "train_status.json"
            result.update(json.loads(status_path.read_text()) if status_path.exists() else {"status": "unknown"})
        elif result["status"] != "forced_stop":
            result["error"] = "vLLM exited" if server_failed else f"Trainer exit code {worker.poll()}"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        # Not a deployed endpoint: always tear down both child process groups.
        for process in (worker, server):
            try:
                stop_group(process)
            except Exception as error:
                result.setdefault("cleanup_errors", []).append(type(error).__name__)
        for thread in threads:
            thread.join(timeout=5)
        latest = output / "latest.pt"
        result.update(elapsed_seconds=time.time()-started,
                      checkpoint_sha256=sha256_file(latest) if latest.is_file() else None)
        (output / "session_result.json").write_text(json.dumps(result, indent=2) + "\n")
        volume.commit()
    return result


def download_checkpoint(run_id, local_dir, expected_sha256):
    """Stream download without loading a multi-GB checkpoint into local RAM."""
    root = Path(local_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"latest-{expected_sha256[:12]}.pt"
    if final.exists():
        if sha256_file(final) != expected_sha256:
            raise ValueError("Existing local checkpoint has the wrong digest; refusing overwrite")
        return final
    temporary = final.with_suffix(".pt.partial")
    with temporary.open("wb") as handle:
        for block in volume.read_file(f"/runs/{run_id}/latest.pt"):
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    if sha256_file(temporary) != expected_sha256:
        raise ValueError("Downloaded checkpoint hash mismatch; incomplete .partial retained")
    os.replace(temporary, final)
    return final


@app.local_entrypoint()
def main(run_id: str, pool_file: str = str(DEFAULT_POOL), epochs: int = 5, resume: bool = False,
         soft_hours: float = MAX_SOFT_HOURS, download_dir: str = "checkpoints/modal-cur",
         valid_file: str = str(DEFAULT_VALID), algorithm: str = "potential"):
    check_arguments(run_id, epochs, soft_hours)
    pool_path = Path(pool_file).expanduser().resolve()
    if not pool_path.is_file():
        raise FileNotFoundError(f"Prepared pool not found: {pool_path}; pass --pool-file with its new location")
    checked = validate_pool(pool_path)
    valid_path = Path(valid_file).expanduser().resolve()
    if not valid_path.is_file():
        raise FileNotFoundError(f"Prepared validation file not found: {valid_path}; pass --valid-file")
    checked_valid = validate_valid(valid_path)
    print(json.dumps({"phase": "local_preflight", "pool": checked,
                      "valid": checked_valid}), flush=True)
    # Only the explicitly supplied Parquet is uploaded, into a content-addressed
    # location; source files are separately mounted by the image definition.
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(pool_path), f"/inputs/{checked['sha256']}/train.parquet")
        batch.put_file(str(valid_path), f"/inputs/{checked_valid['sha256']}/valid.parquet")
    try:
        # Spawn the remote input explicitly.  Calling train.remote() from a
        # local entrypoint ties the GPU input lifecycle to this client process;
        # a lost terminal/session can then send Modal a cancellation signal.
        # FunctionCall.spawn() leaves the input independent while preserving
        # the convenient wait-and-download behavior when this process remains.
        options = {"algorithm": algorithm} if algorithm != "potential" else {}
        call = train.spawn(run_id, checked["sha256"], epochs, resume, soft_hours,
                           checked_valid["sha256"], **options)
        print(json.dumps({"phase": "spawned", "function_call_id": call.object_id,
                          "run_id": run_id}), flush=True)
        result = call.get()
    except Exception:
        print(f"Remote call interrupted. Recovery: modal volume get {VOLUME_NAME} /runs/{run_id}/latest.pt <local-path.pt>",
              flush=True)
        raise
    print(json.dumps(result, indent=2))
    if result.get("checkpoint_sha256"):
        path = download_checkpoint(run_id, download_dir, result["checkpoint_sha256"])
        print(f"Downloaded verified recovery checkpoint: {path.resolve()}")
    if result["status"] not in {"complete", "stopped"}:
        raise RuntimeError("Modal run failed/stopped forcibly; recovery files remain on the Volume")
