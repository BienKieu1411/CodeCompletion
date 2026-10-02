# CUR training: online Stage-1 reference

## Active Modal run: gain24 v2 (2026-10-02)

`--algorithm gain24` supersedes the legacy potential baseline described below.
It fully tunes UniXcoder and a conditional candidate-to-set attention head using
up to24 teacher-forced contexts per task (four base states, five additions each).
Teacher utility is mean target-token log probability. For each measured addition,
`g=U(S+c)-U(S)`, `z=g/(1+abs(g))`. Two-hot interpolation on257 fixed bins in[-1,1]
gives nonnegative labels with unit mass and mean z, optimized by cross entropy.
No first-batch support calibration, no raw-gain clipping, no persistent label cache.
The selector uses expected **transformed score**, NOT expected raw logprob gain;
this changes magnitude weighting but preserves individual target sign/order/zero.
The model's greedy positive-score/STOP policy still needs downstream evaluation;
large gradient norms do not establish completion quality.

Proposal growth ranks candidates first, checks exact token budgets lazily, and
caches only within-task current-weight rankings. Final probes retain exact full
eligibility checks. No snippet text/target/prepared data changes. The random
growth stream now orders the full outside pool before feasibility filtering, so
sampled contexts need not be bit-identical to v1.

Modal profile: A10080GB, CPU2request/4max,16–24GiB RAM,8 preparation threads,
16tasks/update, encoder microbatch64, checkpointing off, generator batch128.
Peak LR5e-5encoder/2e-4head,20update warmup then constant. Initial baseline only;
no post-epoch validation. latest.pt is the trained checkpoint; best.pt remains the
untrained baseline. Deadline-controlled continuation uses1000 as a safety ceiling.
v1 one-update checkpoint is retained but not resumed with the changed loss.

## Legacy potential baseline

Implementation: 2026-10-01. Full fine-tuning of pretrained
`microsoft/unixcoder-base`; frozen `deepseek-ai/deepseek-coder-1.3b-base` provides
completion scores. No model-quality result has been measured for this implementation.

The active entry point is `src/train/train_online_utility.py`. Both frozen vLLM
and trainable UniXcoder remain resident. **No offline label-preparation phase is
required.** Every optimizer batch proposes contexts using the current model,
scores them, and updates the full encoder/heads before starting the next batch.

This implements the **global-embedding conditional-potential baseline** from
`conditional_utility_retriever_plan.md`. Model-dependent conditional proposals
are implemented for online training; the role-aware branch is not. First verify this baseline
before attributing improvements to the larger architecture.

## Learning objective

Encode query and candidates with shared UniXcoder, using its `<encoder-only>`
token format and normalized mean pooling. Its encoder weights are trainable;
encoder dropout is disabled for deterministic representation replay.

The head predicts a signed set potential:

`F(x,S) = sum(u(x,c)) + psi(q, sum(v(x,c)), |S|/10, cost(S)/budget) - psi(q,0,0,0)`.

`F(x,empty)=0`. Positive/negative effects, redundancy and complementary snippets
are representable; monotonicity is not imposed.

Frozen generator completions give utility `U(S)=ES(completion,target)`. Each query
has scored sets and measured add/remove/joint/swap edges. Loss is:

`L_gain = mean_group mean_edge ((F(T)-F(S)) - (U(T)-U(S)))^2`

`L_set = mean_nonempty_set (F(S) - (U(S)-U(empty)))^2`

`L = L_gain + 0.25 L_set`.

Scale is fixed at 1.0 in this reference, preserving zero and the ES units.
Group ownership deduplicates edges: member > add > joint > swap. Gradients
reach the shared encoder and both new heads. These are measured scalar-gain
regression targets; there is no preference-pair mining, policy ratio, critic,
or backpropagation through the generator.

## Data and annotation

The input is the prepared **repository-pool** `train.parquet`, with its
`preparation_contract` metadata. It is not the old PPO task-row file. There is
no repository-level validation split: every quality-filtered repository in the
pool is eligible for training, while benchmark validation/test files remain outside
the repository pool. Each epoch samples 2,000 repository instances: 800 Python and
1,200 Java. The fixed validation artifact contains 100 rows each from CCEval Python,
CCEval Java, RepoEval line and RepoEval API (400 rows total).
The active online trainer does not silently cap this epoch size. The older
offline builder's `--limit` is only an explicit development-subset option.

The query/candidate encoder inputs are allowlisted. Target code and dependency
evidence never enter model features. The target is used only to score completion.

The annotation recipe has at most **20 scored sets/query**. Its derivation and
research limitations are in [CUR-20 design](research/cur20/design.md).

- One empty baseline.
- Three neighborhoods at small/medium/large base sizes (1–2, 4–5, 7–8).
  Each has at most six sets: S, S+a, S+b, S+a+b, S-i, S-i+a.
- One singleton for an extra empty-to-singleton observation.

Thus the ceiling is `1 + 3*6 + 1 = 20`, not twenty per neighborhood. Lexical,
random and path-cluster order are randomly assigned to the three size strata.
Each neighborhood uses one lexical and one random outside probe plus one random
member removal. Feasibility/deduplication can reduce actual counts and base sizes;
the annotation records requested/actual size and complete four-set diamonds.
This is sparse supervision, not exhaustive attribution of every candidate.
Prompt and chunk-text budgets are unchanged; the online candidate pool is now 100.

Online mode replaces path-cluster ordering with current-model conditional
ranking, keeping lexical and random exploration. Its outside probes include a
current-model-ranked feasible addition and a random alternative. Candidates are
drawn from fresh **top 100 BM25 cross-file chunks** for each sampled target:
all repository chunks remain in the prepared artifact, and the current file is
excluded before BM25 scoring. Smaller repositories supply fewer than 100 candidates.
This is model-dependent
set proposal, **not full-repository dense mining**. Within the model stratum,
base construction ranks `F(S+i)` conditional on the current partial set.

The online trainer pools sixteen tasks before issuing HTTP batches (cap 128).
Each task still has at most 20 contexts;
pooling changes scheduling, not its labels. Partial batches and in-call duplicates can
reduce actual request sizes. 2,000 queries have a ceiling of **40,000
completions** per data epoch. On the same
number of queries, 20 versus 123 reduces the maximum count by about **83.7%**;
it is not a measured runtime reduction.
Online mode calls the generator for each optimizer batch. It does not reuse
fixed labels over supervised passes or precompute a whole epoch. Online startup
does not by itself reduce total decoding work. No RLCoder-like runtime is claimed.

RLCoder's checked-in `main.py:350` repeats each query `sample_number` times
(default 10), then `generator.evaluate` batches teacher-forced target-loss
scoring. It does not use one context per training query, nor ten autoregressive
completions. CUR still decodes completions for ES; fewer contexts do not establish
equal or lower runtime. Teacher-forced scoring would change the utility target
and is not silently substituted here. Old 8/123-context annotations need a new
artifact path; bundle version/source hashes prevent accidental mixing.

Online mode keeps only the current batch's labels in memory and deduplicates
identical prompts within that call; there is no persistent label/response cache.
A frozen generator's score for an unchanged prompt is not mathematically made
invalid by a retriever update. The reason for the online loop is to refresh
model-dependent context proposals, not to assert that all old ES labels expire.

The older `build_utility_bundles.py` and `train_conditional_utility.py` remain
available as an offline reference, not the active workflow. Their SQLite
artifacts/checkpoints must not be passed to the online trainer.

## Prompt and metric contract

- Pure prefix completion, no right context; generator input cap 3,072 tokens,
  cross-file cap 2,344, K≤10, output cap 192.
- Fixed cursor-near prefix per query, approximately 650–700 tokens after header
  and reserve. Prepared data may store a longer prefix. Changing this allocation
  requires new annotations.
- Canonical source ordering; complete snippet text and exact joined-token
  accounting. An infeasible set is not scored. Other candidate chunks remain
  in the pool. Online `--candidate-pool-size 100` overrides the older prepared
  metadata value of 64 without rewriting the Parquet. This is distinct from K≤10.
- `line`/`member_suffix`: first nonempty completion line; `fuzzywuzzy.fuzz.ratio`
  divided by 100, matching the checked-in RLCoder `cal_edit_sim` arithmetic.
- `api_statement`: nonempty stripped lines, trim prediction to target line count,
  normalized Levenshtein ES and EM following checked-in RepoEval arithmetic.

The training adapter deliberately does **not** claim full CCEval statement
postprocessor parity. RLCoder's Java statement clipping and regex comment
removal differ from this line adapter. Future benchmark evaluation must use the
benchmark-specific official adapters; do not report these training scores as
official CCEval scores. No test labels are modified or used for calibration.

## Commands

On this Mac use `/Users/kieugiangbien/bienkieu_env/bin/python`; on the training
server use its Python environment. Run from the repository root. Examples below
assume the prepared file is `dataset/data_prepared/train.parquet` and a vLLM endpoint exists
on the same GPU with the co-resident settings below. No command here deploys a
paid endpoint. Keep vLLM running throughout training.

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000

python -m src.train.train_online_utility \
  --pool dataset/data_prepared/train.parquet --base-url "$VLLM_BASE_URL" \
  --output runs/cur_online --epochs 5 \
  --batch-size 16 --generator-batch-size 128 \
  --encoder-backward direct --encoder-microbatch 64 \
  --candidate-pool-size 100 \
  --encoder-lr 5e-5 --head-lr 2e-4 --precision bf16 \
  --save-every 5 --max-minutes 645
```

The endpoint must serve the default model name. Deployment must pin its weights
to the revision in the run contract; `/v1/models` checks model
name but cannot prove the weights revision. Use environment variables
`MODAL_KEY` + `MODAL_SECRET` or `VLLM_API_KEY` for authentication. No credentials
are stored in manifests or source code.

Each epoch selects a fresh 2,000-instance order and samples file/target per
instance. Seeds depend on epoch and position. With sixteen tasks/update there
are 125 updates/epoch. Five epochs mean 625 updates, not three reuse passes per
epoch. All proposals/labels in one batch precede its single optimizer step;
there is no stale next-batch prefetch and no gradient through generator outputs.

Online defaults: AdamW, encoder peak LR 5e-5, heads 2e-4, decay .01, warmup 5%,
max gradient norm 2, sixteen queries/update. Use `bf16` on supported A100-class CUDA
GPUs; use `fp32` on CPU/T4 for this correctness reference. FP16+GradScaler and
multi-GPU execution are not implemented. Dependencies are the data-preparation
environment plus PyTorch, transformers, requests and fuzzywuzzy.

## Memory and recovery

Representation-gradient replay is an optional memory fallback: first encode without saved graphs,
backprop the set loss into embeddings, then replay small encoder microbatches
with their exact representation gradients. Weights stay fixed until all query
gradients for an optimizer update are accumulated. This reduces activation
memory, but adds forward work; CUDA throughput is unmeasured.

The online A100 preset selects `--encoder-backward direct`: encode with gradients once,
then backpropagate the same loss. This avoids replay's additional encoder forward.
Its sequence microbatch controls individual kernel sizes, **not** total saved
activations: all candidate/query graphs for one task remain until its backward.
Tasks accumulate gradients sequentially, so sixteen queries/update does not retain
sixteen whole tasks' graphs. No encoder activation checkpointing is enabled by this
implementation (online mode explicitly disables it). If one task does not fit,
use replay with microbatch 16 or 32 in a new
run; changing the backward mode changes the strict resume contract.

`latest.pt` contains weights, optimizer, scheduler, RNG state, epoch/cursor,
data hashes and model/generator contracts. It is saved atomically every 5 updates,
on epoch completion and at the next optimizer boundary after the time budget is
reached. With repository validation disabled, `latest.pt` is the authoritative
checkpoint; no `best.pt` is selected from benchmark test data. Default time
budget is 645 minutes from before loading; it is checked between updates, not an exact GPU-provider
shutdown timer.

Unexpected failure preserves the last durable valid checkpoint and stops. An
incomplete gradient or corrupted optimizer is never written over it. At most
5 updates may be replayed after failure; `--save-every 1` reduces that to one
at additional disk-I/O cost. An interrupted batch is rescored after resume.

```bash
# Repeat the same training arguments, plus:
# --resume runs/cur_online/latest.pt
```

Resume requires the same code, data, objective, optimizer, precision and pass
schedule. The per-session time limit may change. It can move between devices
using the same precision, though GPU bitwise reproducibility is not promised.

## Selection and tests

`select_context` in `src/train/select_conditional_context.py` encodes the pool
once, then searches add/remove/joint/swap moves using `F(S)-0.01*cost/budget`.
It stops when no predicted improvement exists in the examined neighborhood.
It examines a bounded pair/swap shortlist; this is not global optimality. A
30-move cap is reported as censored search, not learned STOP. Deployment uses
no labels or generator calls to choose snippets, then generates one completion.

```bash
/Users/kieugiangbien/bienkieu_env/bin/python -m unittest tests.test_cur_training -v
/Users/kieugiangbien/bienkieu_env/bin/python -m unittest tests.test_cur_online -v
```

Tests cover prompt/gold invariance, exact token admission, signed loss,
empty-set identity, encoder gradients, replay equivalence (including a small
real RoBERTa attention stack), tiny-bundle overfit, complementary-pair search,
response-cache reuse, and checkpoint resume matching uninterrupted updates.
Online tests additionally cover model-dependent/gold-independent proposals,
one update after all batch labels, next-batch proposals observing new encoder
weights, no update on generation failure, and exact tiny-model resume.
Tiny synthetic results validate implementation, not generalization or superiority
over AlignCoder. Full generator/GPU training and benchmark evaluation remain to run.

## Modal profile: $30 planning budget

Provisional choice: **one A100 80GB**, hosting vLLM and UniXcoder together.
Generation and backward alternate due to their data dependency; both models
remain resident throughout. Do not stop/reload vLLM between batches. Persist
checkpoints in a mounted Modal Volume and commit them after checkpoint saves.

`src/train/modal_online_utility.py` implements the one-shot job and shutdown
supervisor. It has **not been launched**. Importing it or running unit tests
does not start a GPU. The standalone trainer alone still cannot shut down an
unrelated, separately deployed endpoint. No paid GPU benchmark has been run.

### Resources and serving

The launcher requests one GPU, 4 CPU cores (limit 4), 24 GiB host RAM (hard limit
32 GiB), one single-use container, retries zero. No selected region or
non-preemptible premium. RAM above the request is still billable. A RAM-limit
kill is not a graceful save; the last committed checkpoint remains the recovery
point. [Modal resource limits](https://modal.com/docs/guide/resources).

```bash
vllm serve deepseek-ai/deepseek-coder-1.3b-base \
  --revision c919139c3a9b4070729c8b2cca4847ab29ca8d94 \
  --tokenizer-revision c919139c3a9b4070729c8b2cca4847ab29ca8d94 \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 \
  --dtype bfloat16 --max-model-len 3264 \
  --gpu-memory-utilization 0.55 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching --enable-chunked-prefill
```

The image pins vLLM 0.11.0, its required torch 2.8.0, transformers 4.57.1 and
data/metric dependencies (including rank-bm25 0.2.2); `pip freeze` is saved per
invocation. The Modal function uses 8 vCPU and enables parallel fast-tokenizer
batching for context preparation. On 2026-10-01 the 70% vLLM profile completed
one online update but
OOMed during the next backward with 100-candidate tasks. The current profile
reserves 55% for vLLM, leaving more headroom for direct UniXcoder backward;
batch 16 and sequence microbatch 64 are unchanged. `3264=3072+192` includes
both prompt and generation. HTTP batch 128 is not 128 active sequences: vLLM
queues/schedules up to 64 per iteration. The 8,192 prefill/decode token scheduling
limit is not the total KV-cache capacity. These are initial settings, not an
OOM guarantee; profile before increasing concurrency. See the official
[serving options](https://docs.vllm.ai/en/latest/cli/serve/) and
[optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/).

From the pinned generator's [configuration](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base/blob/c919139c3a9b4070729c8b2cca4847ab29ca8d94/config.json),
BF16 KV per token is `2 * 24 layers * 16 KV heads * 128 head_dim * 2 bytes`
= 192 KiB. At full sequence length, 64 sequences need **38.25 GiB KV alone**;
128 need 76.5 GiB, before weights, activations and CUDA graphs. Shared prefixes
may save space, but should not be assumed in a worst-case budget. `.55` is an
executor memory target, not a hard isolation boundary. On a nominal 80 GiB
device it allocates about 44 GiB to vLLM, leaving approximately 36 GiB for
UniXcoder/runtime headroom; target encoder peak below 28 GiB with at least
4 GiB slack. Actual device capacity/process overhead must be measured. Start
vLLM and let its memory profiling finish before loading the trainable encoder.
Do not launch another workload during profiling. No OOM-free guarantee is made.

Startup verification of the 60% profile (2026-10-01): two real optimizer updates
completed. The second batch included four tasks with exactly 100 candidates;
encoder peak allocated memory was 20.95 GiB, backward/update took 3.44 s,
generator scoring 21.30 s, and proposal construction 143.42 s. The full batch
took 169.70 s. This verifies those batches, not full-run memory safety or quality.

The follow-up optimization batches exact cross-file/prompt token counting and
canonical-set feasibility checks, and batches fast-tokenizer calls. A local
100-candidate graph benchmark took 0.64 s for 1,258 cached set checks; the
remote run remains the authority because UniXcoder forward and Modal CPU/GPU
contention are not represented by that microbenchmark.

### Fine-tuning settings

Use the A100 commands above: BF16, direct backward, sequence microbatch 64,
16 tasks/update, encoder peak LR `5e-5`, new-head peak LR `2e-4`, warmup 5%,
linear decay, weight decay 0.01, max gradient norm 2. These
learning rates are starting values, not empirically optimized hyperparameters.

Increasing vLLM batch does not affect the training optimizer batch. Increasing
tasks/update reduces optimizer-step frequency but does not parallelize the
current per-task forward/backward loop. Direct mode only removes the extra
forward; do not claim a 2x end-to-end training speedup.

### Cost and measured go/no-go

Rates checked 2026-10-01. Nominal costs below include 8 CPU cores + 24 GiB RAM
at requested usage, exclude premiums/overages, and assume the full $30 remains.
Keep $3 in reserve. [Modal pricing](https://modal.com/pricing).

| GPU | GPU $/hour | Nominal total $/hour | Hours for $27 |
|---|---:|---:|---:|
| L40S 48GB | 1.95 | 2.52 | 10.71 |
| A100 80GB | 2.50 | 3.07 | 8.79 |
| H100 80GB | 3.95 | 4.52 | 5.97 |

A100's memory headroom motivates this choice; its price/performance is unmeasured.
It must be about 1.235x faster than L40S end-to-end to be cheaper for the same
work. H100 must be about 1.50x faster than A100. A lower-cost co-resident L40S
fallback would need a smaller generator memory allocation/concurrency and encoder
replay; it is not the selected profile and has not been benchmarked.

Implemented A100 wrapper limits: soft save/exit at **10h45**, provider function timeout
at **11h**, retries zero, following the user's revised 11–11.5 hour request.
Eleven nominal hours cost about $31.69 at 24 GiB host RAM, or $32.40
at its 32 GiB limit, before extra startup/build costs. **This does not guarantee
completion within $30 credit.** Existing workspace billing caps can stop it sooner;
this launcher does not raise those caps. Choose `--soft-hours 8.75` for the
earlier budget-oriented duration.
These are planning estimates, not guaranteed invoice caps. Startup/retries must not reset a shared
job deadline. The online trainer counts proposals, generation and updates toward
its one local timer; an in-flight update/HTTP request can exceed that timer.
The supervisor requests a cooperative stop at the shared deadline, allows up to
10 minutes to finish/save, then kills its child process groups if stuck, leaving
five minutes before provider timeout. A failed server shortens that grace to
four minutes. Trainer HTTP requests have a 180-second read timeout and stop
checks between requests. An abrupt kill/preemption can still lose unsaved updates.
The function's provider startup allowance is separate (15 minutes); image build
and provider cold-start are outside the function-body clock and can add cost.
This is not a billing cap. [Modal timeouts](https://modal.com/docs/guide/timeouts).

Protect the account separately: cumulative workspace usage limit $30 and net
spend limit $0, adjusted for existing billing-cycle usage/remaining credit.
These settings have not been changed by this work. A usage cap may interrupt a
job before its next save. [Modal budget semantics](https://modal.com/docs/guide/budgets).

Before a full allocation, use the first real online batches as a throughput
gate (suggested initial measurement budget $1). No test data is used. Logs expose
end-to-end tasks/second, proposal/generator/update time and encoder peak allocated
VRAM (not total server+trainer VRAM). Use `nvidia-smi` for combined use. Exclude cold-start
windows from steady-state rates, but add startup separately to the projection.

For `R` fresh data epochs, `N=2000` tasks/epoch:

`T_seconds = startup + R*N / online_tasks_per_second + shutdown_and_saves`.

Add at least 20% contingency. The five-data-epoch target entails at most 200,000
completion contexts and 10,000 training tasks (625 optimizer updates at
16 tasks/update). **This is a workload definition, not a promise that $30 completes
it.** Proceed only when the measured projection fits the shared soft deadline;
otherwise stop at a safe boundary and retain checkpoints for resume. Do not
silently drop samples or alter labels/token budgets.

### Run and resume from this Mac

The following command **starts billable compute when you execute it**. It first
validates the supplied local Parquet and uploads it to a content-addressed path
on Volume `codecompletion-cur-online`. It does not use the older HF task dataset,
Kaggle credentials, or proxy tokens. Only `src/` and the chosen Parquet are sent;
models download from their pinned public HF revisions inside the container.
Local Modal authentication must already be configured. No credentials in source.

```bash
# From the project root, using bienkieu_env. Keep this command running for
# automatic download when the remote call finishes.
/Users/kieugiangbien/bienkieu_env/bin/modal run -m src.train.modal_online_utility \
  --pool-file dataset/data_prepared/train.parquet \
  --run-id cur-online-001 --epochs 5
```

The launcher defaults to `dataset/data_prepared/train.parquet`, resolved relative
to this project (not the terminal's working directory). `--pool-file` can override
it; an explicitly supplied relative path is resolved from the terminal directory.
The run id must be new; an existing output directory is not silently overwritten.
On normal completion, safe stop, or a handled training failure, the latest
checkpoint is downloaded in a stream to
`checkpoints/modal-cur/cur-online-001/latest-<sha256-prefix>.pt`; its full SHA-256
must match the remote artifact before the local `.pt` is published. On abrupt
remote cancellation/disconnection, automatic download is not guaranteed. The
Volume keeps committed recovery files.

```bash
# Resume: unchanged dataset/code/LR/epochs, same run id, new session timer.
# Recheck remaining credit FIRST; resuming does not reset your billing budget.
/Users/kieugiangbien/bienkieu_env/bin/modal run -m src.train.modal_online_utility \
  --pool-file dataset/data_prepared/train.parquet \
  --run-id cur-online-001 --epochs 5 --resume --soft-hours 2

# Manual recovery if your local process disconnected; use a new destination.
/Users/kieugiangbien/bienkieu_env/bin/modal volume get codecompletion-cur-online \
  /runs/cur-online-001/latest.pt checkpoints/recovered-cur-online-001.pt
```

Checkpoint saves explicitly commit the mounted Volume every five updates and
at epoch/stop boundaries. The supervisor commits logs/status after closing both
process groups. Files include `latest.pt`, `contract.json`, `train.log`,
`vllm.log`, `train_status.json`, `session_result.json`, and environment versions.
[Volume persistence semantics](https://modal.com/docs/guide/volumes).

The GPU log every 60 seconds reports combined VRAM/use; training logs separately
report its own peak allocated VRAM. A high total allocation alone is not evidence
of high GPU compute utilization. No automatic LR/batch tuning or silent OOM retry
changes the training contract. No test-set evaluation is run by this launcher.
