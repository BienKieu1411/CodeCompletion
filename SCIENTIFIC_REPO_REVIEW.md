# Scientific Review: ICAR Co-Retrieval

## 1. Tài liệu này dùng để làm gì?

Tài liệu này giải thích repository theo góc nhìn của một dự án nghiên cứu khoa
học. Mục tiêu là giúp một người mới đọc có thể trả lời nhanh năm câu hỏi:

1. Repo đang giải quyết bài toán gì?
2. Phương pháp hoạt động như thế nào?
3. Đâu là đóng góp có tiềm năng trở thành novelty của paper?
4. Những vấn đề nào có thể khiến kết quả thực nghiệm không đáng tin?
5. Cần làm gì tiếp theo trước khi chạy thí nghiệm lớn?

Đánh giá ngắn gọn: repo đã vượt qua mức scaffold và có một pipeline nghiên cứu
tương đối hoàn chỉnh. Tuy nhiên, nó hiện vẫn là **research prototype**, chưa phải
một artifact đủ bằng chứng để khẳng định vượt AlignCoder hoặc sẵn sàng nộp paper.

---

## 2. Bài toán nghiên cứu

Trong repository-level code completion, mô hình phải hoàn thành đoạn code tại
con trỏ. Code cần thiết để hoàn thành có thể nằm trong một file khác của cùng
repository, ví dụ:

- định nghĩa class hoặc function;
- chữ ký API;
- kiểu dữ liệu nội bộ;
- convention đặt tên của project;
- helper method được sử dụng ở nhiều nơi.

Một pipeline retrieval thông thường luôn tìm vài đoạn code rồi đưa chúng cho
generator. Cách này có hai vấn đề:

1. Query là code chưa hoàn chỉnh nên có thể không thể hiện đúng ý định của target.
2. Retrieved context không phải lúc nào cũng hữu ích; context nhiễu có thể làm
   generator dự đoán tệ hơn.

Repo này cố gắng giải quyết cả hai vấn đề bằng một framework có tên tạm thời là
**Intent-Conditioned Adaptive Co-Retrieval (ICAR)**.

---

## 3. Ý tưởng cốt lõi

Pipeline có thể hiểu bằng chuỗi quyết định sau:

```text
Code trước con trỏ
        |
        v
Tạo retrieval query từ intent và left context
        |
        v
Dense retriever xếp hạng các AST code chunks
        |
        v
Gate quyết định: RETRIEVE hay SKIP?
        |
        +--------------------+
        |                    |
        v                    v
  Dùng context          Không dùng context
        |                    |
        +----------+---------+
                   v
             Generator sinh code
```

Trong lúc huấn luyện, generator còn đóng vai trò evaluator. Với target thật
`Y`, repo đo:

```text
U(C) = NLL(Y | left context, không retrieval)
     - NLL(Y | left context, context C)
```

Ý nghĩa:

- `U(C) > 0`: context giúp generator dự đoán target tốt hơn.
- `U(C) < 0`: context gây nhiễu.
- `U(C) = 0`: tương đương hành động `stop`, tức không retrieval.

Đây là phần hợp lý và có giá trị nhất của phương pháp: chất lượng context được
neo vào một no-retrieval baseline cụ thể, thay vì mặc định xem mọi retrieval là
có ích.

---

## 4. Các thành phần trong implementation

### 4.1 Chuẩn bị dữ liệu

`DatasetLoader` đọc repository từ parquet, giữ lại ranh giới repository và tạo
completion sample bằng cách cắt code tại AST boundary. Mỗi sample gồm:

- `left_context`;
- target code;
- file hiện tại;
- các file khác trong cùng repository làm candidate context;
- `repo_id` để tránh train/eval leakage.

Code chính: `src/co_retrieval/data/repository_dataset_loader.py`.

### 4.2 AST chunking

Các file khác trong repository được chia thành chunk theo class, function hoặc
entity boundary. Nếu parser không hoạt động, code có fallback sang text chunks.

Code chính: `src/co_retrieval/chunking.py`.

AST chunking là hạ tầng tốt nhưng không nên được claim là novelty chính.

### 4.3 Intent query

`IntentSketcher` lấy các tín hiệu rẻ từ code trước con trỏ:

- identifier gần cursor;
- member access như `client.ref...`;
- import;
- class/type hints;
- đoạn cuối của left context.

Chế độ `cost_aware` có thể sinh draft completions khi retriever có entropy cao,
sau đó đưa identifier mới từ draft vào query.

Code chính: `src/co_retrieval/intent.py`.

### 4.4 Dense retriever

Retriever là bi-encoder, mặc định dùng UniXcoder. Query và code chunks được mã
hóa thành vector chuẩn hóa L2, sau đó xếp hạng bằng cosine similarity.

Code chính: `src/co_retrieval/dense_retriever.py`.

### 4.5 Utility supervision

Generator được giữ frozen và dùng teacher forcing để tính target NLL cho từng
context strategy. Các strategy có thể gồm:

- `stop`;
- BM25;
- frozen dense retriever;
- retriever hiện tại;
- hard negative;
- oracle context chỉ dùng lúc train.

Code chính: `src/co_retrieval/context_utility.py` và
`src/co_retrieval/neural_training.py`.

### 4.6 Retriever loss

Mặc định hiện tại là listwise loss: tạo một soft target distribution từ utility
và ép phân phối retriever scores tiến gần phân phối đó bằng KL divergence.

DPO-style pairwise loss vẫn được giữ làm ablation.

Điểm cần thận trọng: tên **LiPO** đã tồn tại trong literature. Công thức hiện tại
nên được mô tả chính xác là `utility-softmax listwise loss` hoặc một softmax
ranking instance của LiPO framework, không nên claim rằng repo phát minh ra tên
LiPO.

### 4.7 Adaptive gate

Gate là MLP nhận query embedding và một số retrieval features, rồi dự đoán xác
suất retrieval có ích. Gate được học từ nhãn:

```text
retrieve_is_better = adjusted_utility > utility_margin
```

Adjusted utility có thể trừ thêm chi phí context, nhờ đó gate có khả năng tối ưu
quality-cost trade-off thay vì chỉ tối ưu chất lượng.

Code chính: `src/co_retrieval/neural_gate.py`.

### 4.8 Generator adapter

Generator backbone được frozen. Repo hỗ trợ soft prompt như một adapter nhẹ,
nhưng launcher chạy server hiện mặc định `adapter_type=none` để giảm thời gian.

Code chính: `src/co_retrieval/soft_prompt.py`.

---

## 5. Novelty nào có khả năng bảo vệ được?

Các đóng góp nên được xếp theo thứ tự sau.

### Đóng góp mạnh nhất

**Adaptive retrieval được hiệu chỉnh bằng downstream generator utility.**

Repo không chỉ hỏi “retrieve đoạn nào?” mà còn hỏi “retrieval có thực sự giúp
completion này không?”. Utility được đo so với hành động stop và có thể tính cả
context cost.

### Đóng góp có tiềm năng

**Utility-aware listwise retriever training.**

Thay vì chỉ chọn một positive và một negative, loss sử dụng toàn bộ utility
spectrum. Ý tưởng này cần được so sánh công bằng với cross-entropy, InfoNCE,
pairwise ranking và DPO.

### Đóng góp chỉ được giữ nếu ablation xác nhận

- Cost-aware hybrid query enhancement.
- Alternating co-training giữa retriever, gate và adapter.
- Soft prompt adaptation.
- Static intent sketch.

Nếu các thành phần này không thắng baseline tương ứng, chúng nên được trình bày
như engineering choices hoặc analysis, không phải core novelty.

---

## 6. Các paper blocker hiện tại

### Blocker 1: query dài có thể mất phần gần cursor

Dense tokenizer dùng `truncation=True` với `max_length=512` nhưng không đặt rõ
`truncation_side`. Raw left context và phần `Left context tail` có thể bị cắt mất
nếu tokenizer giữ phần đầu chuỗi.

Tác động: retriever có thể không nhìn thấy code ngay trước cursor, làm sai cả
retrieval, preference data và gate labels.

Cách sửa:

- đặt left truncation cho raw context; hoặc
- tự phân bổ token budget để luôn giữ intent fields và cursor-near tail;
- thêm unit test cho query dài hơn 512 token.

### Blocker 2: gate labels và features có thể bị stale

Pipeline xây gate label/features, sau đó update retriever, rồi mới train gate.
Gate vì vậy có thể học utility và uncertainty của retriever cũ trong khi
inference sử dụng retriever mới.

Cách sửa:

- train retriever trước;
- refresh retrieval results, utilities và gate features;
- sau đó freeze retriever và train gate.

Alternating schedule cũng phải refresh gate data sau lần update retriever cuối.

### Blocker 3: ablation schedule đang chạy không adapter

Launcher chính mặc định `ADAPTER_TYPE=none`. Script reviewer ablation chỉ thay
schedule mà không bật soft prompt. Vì vậy `sequential_adapter_first` mặc định
không thực sự có adapter để train.

Cách sửa: khi so sánh alternating với sequential, tất cả schedule phải dùng
cùng `ADAPTER_TYPE=soft_prompt`, cùng seed, data và tổng gradient steps. Thí
nghiệm `adapter_type=none` phải là một ablation riêng.

### Blocker 4: RepoEval comparison đang đọc sai metric

AlignCoder có thể ghi kết quả dạng `normal_metric(repoeval_metric)`. Script
summary hiện lấy phần trước dấu ngoặc cho mọi benchmark. Với RepoEval, điều này
không phải repository-averaged metric cần báo cáo.

Cách sửa:

- CCEval dùng `em`, `es`;
- RepoEval dùng `repoeval_em`, `repoeval_es`;
- chạy toàn bộ predictions qua cùng evaluator chính thức trước khi lập bảng.

### Blocker 5: chưa có experimental evidence

Repo chưa chứa checkpoint hoặc result table đủ để xác nhận các claim. Unit test
pass chỉ chứng minh các hàm nhỏ hoạt động theo thiết kế, không chứng minh phương
pháp tốt hơn baseline.

Paper chưa nên dùng các cụm từ như:

- “outperforms AlignCoder”;
- “LiPO improves retrieval”;
- “adaptive gate reduces cost without hurting quality”;
- “co-training is superior to sequential training”.

Các câu này chỉ hợp lệ sau ablation nhiều seed và kiểm định thống kê.

---

## 7. Thiết kế thí nghiệm tối thiểu

### Research questions

**RQ1 — Overall effectiveness**  
Full method có cải thiện EM, Edit Similarity và Identifier F1 so với no-retrieval,
BM25, frozen dense, RLCoder và AlignCoder không?

**RQ2 — Retrieval quality**  
Utility-aware listwise training có cải thiện Recall@k, MRR hoặc nDCG so với
cross-entropy, InfoNCE và DPO không?

**RQ3 — Adaptive retrieval**  
Learned gate có giữ chất lượng trong một tolerance định trước đồng thời giảm
retrieval rate, token count và latency không?

**RQ4 — Query enhancement**  
Static intent và cost-aware drafts có tạo quality-cost Pareto frontier tốt hơn
raw query và always-sample query không?

**RQ5 — Co-training**  
Alternating training có tốt hơn cả adapter-first và retriever-first khi tổng số
gradient steps bằng nhau không?

### Baselines bắt buộc

1. No retrieval.
2. BM25.
3. Frozen dense retriever.
4. Raw-query trained dense retriever.
5. RLCoder.
6. AlignCoder chính thức.
7. Full ICAR.

### Ablations bắt buộc

1. Raw query / static intent / cost-aware query / always-sample query.
2. Always skip / always retrieve / rule gate / learned gate.
3. Cross-entropy / InfoNCE / pairwise / DPO / utility-softmax listwise.
4. No adapter / soft prompt.
5. Alternating / adapter-first / retriever-first.
6. Có và không có context-cost penalty.

### Metrics

Chất lượng output:

- Exact Match;
- Edit Similarity;
- Identifier EM/F1;
- nếu RepoEval hỗ trợ, functional hoặc test-based correctness.

Chất lượng retrieval:

- Recall@k;
- MRR;
- nDCG;
- oracle recall của candidate pool.

Hiệu quả:

- retrieval rate;
- số context tokens trung bình;
- draft sampling rate;
- latency p50/p95;
- peak GPU memory;
- train GPU-hours.

Độ tin cậy:

- ít nhất ba random seeds;
- paired bootstrap confidence interval;
- báo cả mean, standard deviation và effect size;
- không chọn checkpoint dựa trên test set.

---

## 8. Reproducibility checklist

Mỗi experiment cần lưu:

- Git commit SHA;
- toàn bộ CLI config;
- random seed;
- model name và exact model revision;
- dataset version và checksum;
- package lock hoặc Docker image digest;
- CUDA, driver, PyTorch và Transformers versions;
- GPU model;
- train/eval duration;
- raw per-sample predictions;
- raw per-sample metrics;
- checkpoint selection rule.

Nên tạo một manifest dạng:

```json
{
  "git_sha": "...",
  "seed": 13,
  "encoder": {
    "name": "microsoft/unixcoder-base",
    "revision": "..."
  },
  "generator": {
    "name": "deepseek-ai/deepseek-coder-6.7b-base",
    "revision": "..."
  },
  "dataset_sha256": "...",
  "experiment_mode": "intent_main",
  "adapter_type": "soft_prompt",
  "retriever_loss": "utility_softmax"
}
```

---

## 9. Trạng thái kiểm thử hiện tại

Kết quả kiểm tra local tại thời điểm review:

```text
50 passed, 1 skipped
```

Test bị skip là Java AST chunking vì môi trường local thiếu
`tree_sitter_languages`.

Điểm tích cực:

- repository-disjoint split có test;
- gate không dùng oracle khi inference có test;
- listwise stop action có test;
- adapter consistency có test;
- AlignCoder-style artifact writer có test;
- schedule budget có test.

Phần còn thiếu:

- integration test với Hugging Face model thật;
- GPU smoke test;
- query truncation test;
- gate refresh/staleness test;
- parity test với evaluator AlignCoder chính thức;
- end-to-end test trên một benchmark subset cố định.

---

## 10. Roadmap đề xuất

### Giai đoạn A — Sửa correctness trước khi train

- [ ] Sửa token truncation của retrieval query.
- [ ] Refresh gate labels/features sau retriever update.
- [ ] Sửa RepoEval metric comparison.
- [ ] Sửa ablation launcher để adapter schedule có ý nghĩa.
- [ ] Thêm các regression test tương ứng.

### Giai đoạn B — Smoke experiments

- [ ] Chạy 100–500 samples với encoder/generator nhỏ.
- [ ] Kiểm tra loss có giảm và không có NaN.
- [ ] Đo phân phối utility và tỷ lệ gate-positive.
- [ ] Đo entropy và draft sampling rate thực tế.
- [ ] Xác nhận output bằng evaluator chính thức.

### Giai đoạn C — Main experiments

- [ ] Khóa dataset, model revisions và environment.
- [ ] Chạy ba seed cho mỗi baseline chính.
- [ ] Chạy full CCEval Python/Java và RepoEval line/API.
- [ ] Thu retrieval, output và efficiency metrics.
- [ ] Chạy paired bootstrap CI.

### Giai đoạn D — Viết paper theo kết quả

- Nếu gate giữ chất lượng và giảm retrieval rõ rệt: đặt adaptive utility gate
  làm core contribution.
- Nếu listwise loss thắng các objective khác: giữ utility-aware listwise
  optimization như contribution thứ hai.
- Nếu alternating không thắng sequential: bỏ co-training khỏi title và core
  claim.
- Nếu static intent thua generated query: chỉ mô tả nó như cheap first stage.
- Nếu cost-aware query không giảm compute: không claim adaptive computation.

---

## 11. Kết luận cuối

Repo có một thesis nghiên cứu đáng theo đuổi:

> Không phải completion nào cũng cần retrieval, và không phải retrieved context
> nào cũng hữu ích. Retrieval nên được học và kích hoạt dựa trên utility thật mà
> context mang lại cho downstream generator, đồng thời tính đến chi phí sử dụng
> context.

Đây là framing mạnh hơn việc chỉ nói repo có một retriever mới, soft prompt mới
hoặc DPO/LiPO mới. Sau khi sửa các paper blocker và chạy ablation nghiêm ngặt,
repo có thể trở thành một artifact nghiên cứu tốt về **utility-calibrated
adaptive retrieval for repository-level code completion**.
