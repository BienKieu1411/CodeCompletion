"""Kaggle setup + short adaptive training run for AlignCoder.

Run this file in one Kaggle Python cell (Internet must be enabled), or upload it
as a Kaggle script.  It freezes DeepSeek-Coder and only updates AlignRetriever.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# Four-epoch adaptation run from the paper's AlignRetriever checkpoint.
# A later session can restore model, optimizer, scheduler, and RNG state.
REPO_DIR = Path("/kaggle/working/AlignCoder")
DATA_DIR = Path("/kaggle/working/aligncoder_data")
SESSION_NAME = "n4_dpe2000_e4"
OUTPUT_DIR = Path("/kaggle/working") / f"align_adapt_deepseekcoder_1p3b_{SESSION_NAME}"
TOTAL_EPOCHS = 4
EPOCHS_THIS_SESSION = 4
DATA_PER_EPOCH = 2000
SAMPLE_NUMBER = 8       # candidates scored by the retriever / PPL feedback
NUMBER_SAMPLE = 4       # sampled generator completions for query enhancement

# Leave None so this run starts from AlignCoder/AlignRetriever (paper).
# To continue later, set this to an attached `retriever_cpkt/result_N` folder.
RESUME_FROM_CHECKPOINT = None


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(list(cmd), check=True)


def resolve_checkpoint(path: Path | None) -> Path | None:
    """Accept either a result_N directory or an attached Dataset root."""
    if path is None:
        return None
    if (path / "config.json").is_file() and ((path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()):
        return path
    weights = list(path.rglob("model.safetensors")) + list(path.rglob("pytorch_model.bin"))
    if not weights:
        raise FileNotFoundError(f"Could not find a Hugging Face retriever checkpoint under {path}")
    preferred = [weight.parent for weight in weights if weight.parent.name == "result_4"]
    return (preferred or [weight.parent for weight in weights])[0]


def patch_source() -> None:
    """Add --skip_training_eval and lower multiprocessing oversubscription."""
    main = REPO_DIR / "main.py"
    text = main.read_text(encoding="utf-8")

    if "--skip_training_eval" not in text:
        needle = '    parser.add_argument("--eval", action="store_true", help="Perform evaluation")\n'
        replacement = needle + (
            '    parser.add_argument("--skip_training_eval", action="store_true", '
            'help="Skip initial and per-epoch completion evaluation during training")\n'
        )
        if needle not in text:
            raise RuntimeError("AlignCoder main.py layout changed: cannot add --skip_training_eval")
        text = text.replace(needle, replacement, 1)

    if "--resume_trainer_state" not in text:
        needle = '    parser.add_argument("--skip_training_eval", action="store_true", help="Skip initial and per-epoch completion evaluation during training")\n'
        replacement = needle + (
            '    parser.add_argument("--resume_trainer_state", default=None, type=str, '
            'help="Path to trainer_state.pt; restore optimizer/scheduler/RNG when available")\n'
            '    parser.add_argument("--max_epochs_this_run", default=None, type=int, '
            'help="Stop after this many epochs while preserving continuation state")\n'
        )
        if needle not in text:
            raise RuntimeError("AlignCoder main.py layout changed: cannot add resume arguments")
        text = text.replace(needle, replacement, 1)

    # Emptying this mapping disables every initial/final/per-epoch evaluation
    # loop while leaving the training examples and checkpoint saves untouched.
    marker = "    global generator\n"
    injection = (
        "    if args.skip_training_eval:\n"
        "        print('Skipping completion evaluation (training checkpoints will still be saved).')\n"
        "        all_eval_examples = {}\n\n"
    )
    if injection not in text:
        if marker not in text:
            raise RuntimeError("AlignCoder main.py layout changed: cannot disable evaluation loops")
        text = text.replace(marker, injection + marker, 1)

    # The public training release contains github_repos but not every completion
    # benchmark.  Upstream loads all benchmarks before it gets to its training
    # branch, even when --eval is absent.  Give the later (empty) eval mapping
    # harmless placeholders instead of requiring those unrelated files.
    eval_loads = (
        '    cceval_python_examples = load_test_dataset(args, "cceval", "python")\n'
        '    cceval_java_examples = load_test_dataset(args, "cceval", "java")\n'
        '    # codereval_python_examples = load_test_dataset(args, "codereval", "python")\n'
        '    # codereval_java_examples = load_test_dataset(args, "codereval", "java")\n'
        '    repoeval_line_examples = load_test_dataset(args, "repoeval", "line_level")\n'
        '    repoeval_api_examples = load_test_dataset(args, "repoeval", "api_level")\n'
    )
    eval_loads_replacement = (
        '    if args.skip_training_eval:\n'
        '        cceval_python_examples = cceval_java_examples = []\n'
        '        repoeval_line_examples = repoeval_api_examples = []\n'
        '    else:\n'
        '        cceval_python_examples = load_test_dataset(args, "cceval", "python")\n'
        '        cceval_java_examples = load_test_dataset(args, "cceval", "java")\n'
        '        repoeval_line_examples = load_test_dataset(args, "repoeval", "line_level")\n'
        '        repoeval_api_examples = load_test_dataset(args, "repoeval", "api_level")\n'
    )
    if eval_loads in text:
        text = text.replace(eval_loads, eval_loads_replacement, 1)
    main.write_text(text, encoding="utf-8")

    generator = REPO_DIR / "generator.py"
    text = generator.read_text(encoding="utf-8")
    # vLLM is only used by optional generator classes.  Avoid installing it on
    # Kaggle; HF_Generator below is the class used by main.py.
    legacy_vllm_import = "from vllm import LLM, SamplingParams\n"
    optional_vllm_import = (
        "try:\n"
        "    from vllm import LLM, SamplingParams\n"
        "except ImportError:\n"
        "    LLM = SamplingParams = None\n"
    )
    if legacy_vllm_import in text and optional_vllm_import not in text:
        text = text.replace(legacy_vllm_import, optional_vllm_import, 1)
    # The upstream code makes pools of 64 processes for short post-processing;
    # that is counterproductive on Kaggle's small CPU allocation.
    text = text.replace("mp.Pool(processes=64)", "mp.Pool(processes=min(4, os.cpu_count() or 1))")
    # The upstream default is vLLM_online_Generator, which is an OpenAI-client
    # wrapper aimed at a separately hosted vLLM server.  Kaggle has no such
    # server.  Append an equivalent local Transformers generator instead.
    if "class HF_Generator:" not in text:
        text += '''\n\nclass HF_Generator:\n    """Local Hugging Face generator; drop-in replacement for vLLM_online_Generator."""\n    def __init__(self, args):\n        self.tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path)\n        self.tokenizer.model_max_length = int(1e10)\n        if self.tokenizer.pad_token_id is None:\n            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id\n        self.model = AutoModelForCausalLM.from_pretrained(\n            args.generator_model_path, torch_dtype=torch.float16\n        ).cuda().eval()\n        self.args = args\n\n    @torch.no_grad()\n    def generate(self, examples, retrieved_codeblocks, temperature=0, top_p=1.0,\n                 sample_number=0, deduplicated=False):\n        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks, generation=True)\n        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset),\n                                batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)\n        outputs = []\n        torch.manual_seed(123)\n        torch.cuda.manual_seed_all(123)\n        for input_ids in tqdm(dataloader, disable=not self.args.enable_tqdm, desc="Generating"):\n            input_ids = input_ids.cuda()\n            kwargs = {"max_new_tokens": self.args.generator_max_generation_length,\n                      "pad_token_id": self.tokenizer.pad_token_id}\n            if temperature > 0:\n                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)\n            else:\n                kwargs.update(do_sample=False)\n            if sample_number:\n                input_ids = input_ids.repeat_interleave(sample_number, dim=0)\n            generated = self.model.generate(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), **kwargs)\n            outputs.extend(self.tokenizer.decode(x, skip_special_tokens=True)\n                           for x in generated[:, input_ids.size(1):])\n        if not sample_number:\n            return outputs\n        repeated_examples = [x for x in examples for _ in range(sample_number)]\n        lang = examples[0].language\n        ts_lib = "utils/build/python-lang-parser.so" if lang == "python" else "utils/build/java-lang-parser.so"\n        with mp.Pool(processes=max(1, min(4, self.args.num_workers))) as pool:\n            outputs = pool.starmap(partial(process_single_item, language_name=lang, ts_lib=ts_lib),\n                                   zip(repeated_examples, outputs))\n        if not deduplicated:\n            return outputs\n        unique, counts = [], []\n        for i in range(0, len(outputs), sample_number):\n            batch, seen = outputs[i:i + sample_number], set()\n            keep = [x for x in batch if x.strip() and not (x in seen or seen.add(x))]\n            unique.extend(keep)\n            counts.append(len(keep))\n        return unique, counts\n'''
    generator.write_text(text, encoding="utf-8")

    # Make main.py instantiate the local class, not the upstream API/vLLM
    # client.  This is idempotent, so it also fixes a partially patched clone.
    text = main.read_text(encoding="utf-8")
    old_import = "from generator import Generator, vLLM_online_Generator, vLLM_offline_Generator, process_single_item"
    new_import = "from generator import Generator, vLLM_online_Generator, vLLM_offline_Generator, HF_Generator, process_single_item"
    if old_import in text:
        text = text.replace(old_import, new_import, 1)
    text = text.replace("generator = vLLM_online_Generator(args)", "generator = HF_Generator(args)")

    # The original source saves only model weights.  Save and restore the
    # optimizer/scheduler/RNG too, so a new Kaggle session is a true resume.
    scheduler_line = (
        '        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch * 0.2, num_training_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch)\n'
    )
    restore_block = scheduler_line + '''        start_epoch = 0
        if args.resume_trainer_state:
            if os.path.isfile(args.resume_trainer_state):
                state = torch.load(args.resume_trainer_state, map_location="cpu", weights_only=False)
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                for optimizer_state in optimizer.state.values():
                    for key, value in optimizer_state.items():
                        if torch.is_tensor(value):
                            optimizer_state[key] = value.cuda()
                start_epoch = int(state["next_epoch"])
                if "python_rng" in state:
                    random.setstate(state["python_rng"])
                    np.random.set_state(state["numpy_rng"])
                    torch.set_rng_state(state["torch_rng"])
                    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
                        torch.cuda.set_rng_state_all(state["cuda_rng"])
                print(f"Resumed trainer state at epoch {start_epoch} from {args.resume_trainer_state}", flush=True)
            else:
                print(f"No trainer_state at {args.resume_trainer_state}; warm-starting model with fresh optimizer.", flush=True)
        end_epoch = args.epoch if args.max_epochs_this_run is None else min(args.epoch, start_epoch + args.max_epochs_this_run)
        print(f"Training epochs [{start_epoch}, {end_epoch}) of total schedule {args.epoch}", flush=True)
'''
    if restore_block not in text:
        if scheduler_line not in text:
            raise RuntimeError("AlignCoder main.py layout changed: cannot inject continuation restore")
        text = text.replace(scheduler_line, restore_block, 1)
    text = text.replace('        for epoch in range(args.epoch):\n', '        for epoch in range(start_epoch, end_epoch):\n', 1)

    tokenizer_save_line = '            retriever.tokenizer.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")\n'
    state_save = tokenizer_save_line + '''            trainer_state_path = f"{args.output_dir}/retriever_cpkt/result_{epoch}/trainer_state.pt"
            torch.save({
                "next_epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }, trainer_state_path)
            print(f"[epoch {epoch}] saved continuation state: {trainer_state_path}", flush=True)
'''
    if state_save not in text:
        if tokenizer_save_line not in text:
            raise RuntimeError("AlignCoder main.py layout changed: cannot save continuation state")
        text = text.replace(tokenizer_save_line, state_save, 1)

    # Compact print-based progress logs: Kaggle's captured output is easier to
    # inspect this way than with nested tqdm bars.
    retrieval_line = (
        '            queries, retrieved_codeblocks = retrieve_codeblocks(args, training_examples, bm25, retriever, "github_training_{}".format(epoch), True)\n'
    )
    retrieval_logged = (
        '            print(f"[epoch {epoch}] building {len(training_examples)} tasks -> retrieve + generator-enhanced queries", flush=True)\n'
        + retrieval_line +
        '            print(f"[epoch {epoch}] retrieval complete; scoring {len(training_examples) * args.sample_number} candidate contexts with PPL", flush=True)\n'
    )
    if retrieval_line in text and retrieval_logged not in text:
        text = text.replace(retrieval_line, retrieval_logged, 1)

    ppl_create = '                generator1 = Generator(args)\n'
    ppl_create_logged = '                print(f"[epoch {epoch}] loading frozen PPL scorer", flush=True)\n' + ppl_create
    if ppl_create in text and ppl_create_logged not in text:
        text = text.replace(ppl_create, ppl_create_logged, 1)

    batch_loop = '                for batch in dataloader:\n'
    batch_loop_logged = (
        '                print(f"[epoch {epoch}] updating retriever: {len(dataloader)} batches", flush=True)\n'
        '                for batch_index, batch in enumerate(dataloader, 1):\n'
    )
    if batch_loop in text and batch_loop_logged not in text:
        text = text.replace(batch_loop, batch_loop_logged, 1)
        anchor = '                    total_loss += loss.item()\n'
        logged_anchor = (
            anchor +
            '                    if batch_index == 1 or batch_index == len(dataloader) or batch_index % 25 == 0:\n'
            '                        print(f"[epoch {epoch}] train batch {batch_index}/{len(dataloader)} | loss={loss.item():.4f}", flush=True)\n'
        )
        text = text.replace(anchor, logged_anchor, 1)

    save_line = '            retriever.model.module.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")\n'
    save_logged = '            print(f"[epoch {epoch}] saving retriever checkpoint", flush=True)\n' + save_line
    if save_line in text and save_logged not in text:
        text = text.replace(save_line, save_logged, 1)
    main.write_text(text, encoding="utf-8")


# Do not install AlignCoder's pinned requirements.txt: it pins Python-3.12-
# incompatible old Torch/Pandas and an unnecessary vLLM dependency.
run(sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.46.3", "accelerate>=1.1.0", "datasets>=3.0.0",
    "huggingface_hub>=0.26.0", "openai>=1.0.0", "tree-sitter==0.20.4",
    "func-timeout", "fuzzywuzzy", "timeout-decorator", "prettytable",
    "editdistance", "sacrebleu", "python-Levenshtein", "rank-bm25")

if not REPO_DIR.exists():
    run("git", "clone", "--depth", "1", "https://github.com/DeepSoftwareAnalytics/AlignCoder.git", str(REPO_DIR))

patch_source()

# AlignCoder expects `data/github_repos/...`; the Hub archive itself has a
# top-level `data/` folder, so link that *inner* folder into the source tree.
from huggingface_hub import snapshot_download

if not DATA_DIR.exists():
    snapshot_download(
        repo_id="AlignCoder/Data4AlignCoder",
        repo_type="dataset",
        local_dir=str(DATA_DIR),
    )

payload_data = DATA_DIR / "data"
if not (payload_data / "github_repos" / "python" / "train.parquet").is_file():
    raise RuntimeError(f"Dataset download is incomplete; expected {payload_data / 'github_repos/python/train.parquet'}")

repo_data = REPO_DIR / "data"
if repo_data.exists() or repo_data.is_symlink():
    if repo_data.resolve() != payload_data.resolve():
        # This only removes the old symlink created by an earlier version of
        # this script; never delete a real source/data directory.
        if not repo_data.is_symlink():
            raise RuntimeError(f"{repo_data} is a real directory; remove it manually before continuing")
        repo_data.unlink()
        repo_data.symlink_to(payload_data, target_is_directory=True)
else:
    repo_data.symlink_to(payload_data, target_is_directory=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(REPO_DIR)
RESUME_CHECKPOINT = resolve_checkpoint(RESUME_FROM_CHECKPOINT)
if RESUME_CHECKPOINT:
    print("Warm-start/resume checkpoint:", RESUME_CHECKPOINT)

run(
    sys.executable, "main.py",
    "--weighted_keywords",
    "--enable_generation", "--enable_prediction", "--add_api_blocks",
    "--inference_type", "unixcoder_with_rl",
    "--skip_training_eval",
    "--retriever_model_path", str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else "AlignCoder/AlignRetriever",
    # Same generator family and the paper's released default (1.3B, not 1.5B).
    "--generator_model_path", "deepseek-ai/deepseek-coder-1.3b-base",
    "--generator_batch_size_per_gpu", "2",
    "--generator_max_crossfile_length", "512",
    "--generator_max_context_length", "1024",
    "--generator_max_generation_length", "64",
    "--retriever_batch_size_per_gpu", "64",
    "--batch_size", "4",
    "--number_sample", str(NUMBER_SAMPLE),
    "--sample_number", str(SAMPLE_NUMBER),
    "--data_per_epoch", str(DATA_PER_EPOCH),
    "--epoch", str(TOTAL_EPOCHS), "--max_epochs_this_run", str(EPOCHS_THIS_SESSION), "--inner_epoch", "1",
    "--lr", "1e-5",
    "--num_workers", "4",
    "--output_dir", str(OUTPUT_DIR),
    "--resume_trainer_state", str(RESUME_CHECKPOINT / "trainer_state.pt") if RESUME_CHECKPOINT else "",
)

print("\nFinished. Retriever checkpoints:", OUTPUT_DIR / "retriever_cpkt")
