# Hướng dẫn triển khai và chạy Co-Retrieval

Tài liệu này mô tả trạng thái implementation hiện tại, cách chuẩn bị môi trường và các lệnh chạy pipeline. Các ví dụ dùng PowerShell và được chạy từ thư mục gốc của repository.

## 1. Những thay đổi chính

### Repository-disjoint evaluation

Dataset loader giữ lại `repo_id` từ ranh giới repository trong file Parquet. Neural pipeline chia train/eval theo repository thay vì shuffle từng sample, tránh một repository xuất hiện ở cả hai tập.

Nếu dataset chỉ có một repository hoặc sample không có `repo_id`, toàn bộ sample được dùng cho train và eval để trống. Pipeline không tự động quay lại random sample split.

### Gate supervision không còn oracle leakage

Gate chỉ học utility của strategy sẽ được dùng khi inference:

| Experiment | Strategy tạo gate label |
|---|---|
| `intent_main`, `raw_query_main`, `retriever_only`, sequential modes | `current` |
| `bm25` | `bm25` |
| `dense_frozen` | `dense_frozen` |

`oracle` và `hard_neg` vẫn có thể tạo preference data cho retriever, nhưng không được dùng để quyết định nhãn retrieve/skip.

### LiPO là retriever loss mặc định

LiPO tối ưu phân phối listwise của toàn bộ context strategies theo utility:

```text
U(C) = NLL(stop) - NLL(C)
target_distribution = softmax(U / tau)
```

Candidate `stop` được đưa vào distribution với utility và retriever score bằng `0`. Các strategy trả về cùng một context set được deduplicate. DPO vẫn được giữ làm ablation bằng `--retriever-loss dpo`.

### Adapter treatment nhất quán

Evaluation, retrieve/skip comparison và leave-one-out analysis đều giữ cùng giá trị `use_soft_prompt`. Improvement vì vậy không còn trộn lẫn ảnh hưởng của retrieved context với việc bật/tắt adapter.

### Cost-aware query enhancement

Mode `cost_aware` chạy theo hai đường:

1. Tạo static intent query và retrieval lần đầu.
2. Chuẩn hóa entropy của top-k scores về `[0, 1]`.
3. Nếu entropy thấp, dùng luôn kết quả retrieval đầu tiên.
4. Nếu entropy cao, sinh completion drafts, lấy identifier mới để tăng cường query rồi retrieval lại.

Nếu không đủ scores, draft generation lỗi hoặc draft không thêm identifier mới, pipeline dùng kết quả static ban đầu. Các mode `static` và `raw` không bị thay đổi hành vi.

## 2. Yêu cầu môi trường

- Python 3.10 trở lên.
- Proxy mode không cần GPU.
- Neural mode cần tải model Hugging Face và nên dùng NVIDIA GPU có VRAM phù hợp.
- Model mặc định khá lớn: Jina code embedding 1.5B và Qwen Coder 7B. DPO cần thêm reference encoder; LiPO nhẹ hơn DPO ở điểm này nhưng pipeline vẫn giữ frozen initial encoder cho baseline.

Tạo virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài proxy mode và test:

```powershell
python -m pip install -e ".[test]"
```

Cài đầy đủ neural pipeline, parser và test:

```powershell
python -m pip install -e ".[neural,parsers,test]"
```

Kiểm tra cài đặt:

```powershell
python co_retrieval_cli.py --help
python co_retrieval_cli.py train --help
python -m pytest -q
```

## 3. Chuẩn bị dataset

Train loader đọc file Parquet. Dataset mặc định:

```text
data/github_repos/python/train.parquet
```

Các cột bắt buộc:

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `path` | string | Đường dẫn file tương đối trong repository |
| `content` | string | Nội dung source code |
| `first` | boolean | `true` tại file đầu tiên của mỗi repository |

Các hàng của cùng repository phải nằm liên tiếp. Mỗi repository cần ít nhất hai file để có cross-file context. `repo_id` được loader tạo ổn định từ thứ tự các nhóm `first`; không cần có sẵn trong Parquet.

Mặc định loader chỉ ưu tiên file đủ dài:

- `--min-file-lines 200`
- `--min-file-chars 2000`
- `--min-left-context-lines 30`

Khi smoke test bằng dataset nhỏ, có thể hạ các ngưỡng này.

## 4. Chạy test và proxy mode

Chạy toàn bộ test:

```powershell
python -m pytest -q
python -m compileall -q src tests
git diff --check
```

Proxy mode kiểm tra data loading, chunking và training flow mà không tải neural model:

```powershell
python co_retrieval_cli.py train `
  --dataset-path data/github_repos/python/train.parquet `
  --max-samples 50 `
  --num-epochs 2 `
  --checkpoint-dir checkpoints/proxy `
  --log-dir logs/proxy
```

Smoke test với dataset nhỏ:

```powershell
python co_retrieval_cli.py train `
  --dataset-path data/github_repos/python/train.parquet `
  --max-samples 5 `
  --min-file-lines 10 `
  --min-file-chars 100 `
  --min-left-context-lines 3
```

Proxy output chính:

```text
checkpoints/proxy/co_retrieval_checkpoint.json
logs/proxy/training_history.jsonl
```

## 5. Chạy neural pipeline

### LiPO + static intent

Đây là cấu hình chính và cũng là default:

```powershell
python co_retrieval_cli.py train `
  --use-neural `
  --dataset-path data/github_repos/python/train.parquet `
  --experiment-mode intent_main `
  --intent-mode static `
  --retriever-loss lipo `
  --lipo-tau 1.0 `
  --max-samples 50 `
  --eval-ratio 0.1 `
  --max-eval-samples 20 `
  --warmup-steps 20 `
  --num-rounds 2 `
  --steps-per-round-prompt 20 `
  --steps-per-round-retriever 20 `
  --device cuda `
  --checkpoint-dir checkpoints/lipo_static `
  --log-dir logs/lipo_static
```

`--steps-per-round-dpo` là alias tương thích cũ. Nếu truyền `--steps-per-round-retriever`, giá trị mới được ưu tiên.

### DPO ablation

```powershell
python co_retrieval_cli.py train `
  --use-neural `
  --dataset-path data/github_repos/python/train.parquet `
  --retriever-loss dpo `
  --dpo-beta 0.1 `
  --steps-per-round-retriever 20 `
  --max-samples 50 `
  --device cuda `
  --checkpoint-dir checkpoints/dpo_ablation
```

DPO tạo frozen reference encoder ở Phase 3 nên cần nhiều memory hơn LiPO.

### Cost-aware intent

```powershell
python co_retrieval_cli.py train `
  --use-neural `
  --dataset-path data/github_repos/python/train.parquet `
  --intent-mode cost_aware `
  --retriever-loss lipo `
  --query-entropy-threshold 0.8 `
  --query-num-drafts 2 `
  --query-draft-max-tokens 32 `
  --query-draft-temperature 0.8 `
  --query-draft-top-p 0.95 `
  --max-samples 50 `
  --device cuda `
  --checkpoint-dir checkpoints/lipo_cost_aware
```

Threshold thấp làm draft sampling thường xuyên hơn và tốn thêm latency. Threshold cao ưu tiên static path.

### Dùng model khác

```powershell
python co_retrieval_cli.py train `
  --use-neural `
  --encoder-name <huggingface-encoder> `
  --generator-name <huggingface-causal-lm> `
  --generator-dtype bfloat16 `
  --device cuda
```

Encoder phải trả về `last_hidden_state`; generator phải là causal LM và có input embedding API tương thích Transformers.

## 6. Experiment và ablation modes

| Mode | Mục đích |
|---|---|
| `intent_main` | Full pipeline với intent query |
| `raw_query_main` | So sánh raw left context với intent query |
| `retriever_only` | Tắt soft prompt adapter |
| `always_retrieve` | Bỏ learned gate, luôn retrieve |
| `always_skip` | Không retrieval |
| `bm25` | Lexical retrieval baseline |
| `dense_frozen` | Frozen dense retriever baseline |
| `sequential_adapter_first` | Train adapter trước, retriever/gate sau |
| `sequential_retriever_first` | Train retriever/gate trước, adapter sau |

Ví dụ gate ablation:

```powershell
python co_retrieval_cli.py train --use-neural --experiment-mode always_retrieve
python co_retrieval_cli.py train --use-neural --experiment-mode always_skip
```

Nên dùng dataset, model, seed và tổng gradient steps giống nhau khi so sánh các mode.

## 7. Output và metrics

Neural checkpoint directory chứa:

```text
retriever/
gate.pt
soft_prompt.pt
meta.json
```

`meta.json` lưu config và history, bao gồm loss type, LiPO temperature và cost-aware parameters.

Các metrics quan trọng:

- `exact_match`, `edit_similarity`, `identifier_f1`: chất lượng completion.
- `retrieval_rate`: tỷ lệ gate chọn retrieval.
- `nll_improvement`: NLL(no context) trừ NLL(context).
- `phase3_retriever_loss` và `phase3_retriever_loss_type`: loss retriever và loại LiPO/DPO.
- `gate_label_metrics`: precision, recall, F1, AUC và confusion matrix.
- `query_enhancement.mean_normalized_entropy`: entropy trung bình.
- `query_enhancement.sampling_rate`: tỷ lệ query phải sinh drafts.
- `query_enhancement.query_changed_rate`: tỷ lệ draft thực sự thay đổi query.
- `leave_one_out_analysis`: đóng góp của từng chunk trong top-k context.

Gate chỉ được xem là có ích khi giảm retrieval compute mà không vượt quality tolerance, hoặc cải thiện quality so với `always_retrieve`.

## 8. Troubleshooting

### `No training samples found`

- Kiểm tra `--dataset-path`.
- Kiểm tra ba cột `path`, `content`, `first`.
- Hạ `--min-file-lines`, `--min-file-chars` và `--min-left-context-lines` khi debug.
- Đảm bảo mỗi repository có ít nhất hai file.

### Eval rỗng

Repository-disjoint evaluation cần ít nhất hai repository có `repo_id`. Với một repository, đây là hành vi an toàn có chủ đích.

### CUDA out of memory

- Giảm `--max-samples`, `--top-k`, `--max-context-tokens` và `--batch-encode-size`.
- Dùng LiPO thay cho DPO để tránh reference snapshot trong Phase 3.
- Chọn encoder/generator nhỏ hơn hoặc `bfloat16` nếu GPU hỗ trợ.
- Cost-aware mode làm tăng generation work; dùng `static` trong lần smoke test đầu.

### Không tìm thấy `pytest`, `torch` hoặc `transformers`

```powershell
python -m pip install -e ".[neural,parsers,test]"
```

### Hugging Face download hoặc authentication lỗi

Xác nhận máy có network, đủ disk và đã đăng nhập nếu model bị giới hạn:

```powershell
huggingface-cli login
```

### PowerShell không cho activate virtual environment

Chỉ nới policy cho process hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 9. Cấu hình chạy khuyến nghị

Thứ tự an toàn để bắt đầu experiment:

1. Chạy `pytest` và proxy smoke test.
2. Neural smoke test với `static`, LiPO và số sample/steps nhỏ.
3. Chạy đủ `intent_main` và các baseline `raw`, `always_retrieve`, `always_skip`.
4. Chạy DPO ablation với cùng budget.
5. Cuối cùng bật `cost_aware` và so sánh quality, sampling rate và latency.

Không nên claim hiệu quả của co-training, learned gate hoặc cost-aware enhancement nếu chưa có ablation tương ứng trên cùng data/model/budget.
