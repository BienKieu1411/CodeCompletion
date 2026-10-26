# Báo cáo tổng hợp đánh giá ICAR trên CCEval

**Ngày:** 2026-08-06  
**Phạm vi:** CCEval Python/Java, setting train ICAR-LiPO hiện tại và các nhận định insight sau training.

## 1. Kết luận điều hành

Kết quả hiện tại chưa đủ cơ sở để kết luận novelty của ICAR yếu. Có ít nhất hai vấn đề implementation/evaluation ảnh hưởng trực tiếp đến kết quả:

1. Run được sử dụng có `ADAPTER_TYPE=none`, nên soft prompt không được train và không được dùng trong pipeline chính.
2. Đường generate của ICAR không ép tokenizer giữ phần tail của `left_context`. Nếu tokenizer DeepSeek runtime dùng `truncation_side="right"`, phần gần cursor có thể bị cắt khi context dài.

Vì vậy, mức giảm EM hiện tại nhiều khả năng phản ánh một phần đáng kể sự khác biệt trong implementation và protocol đánh giá, đặc biệt ở nhóm long-context. Tuy nhiên, chưa thể quy toàn bộ chênh lệch cho truncation nếu chưa phân nhóm theo retrieval/no-retrieval và kiểm tra prompt thực tế sau khi pack.

Nhận định “LiPO group hoặc label chắc chắn được reuse từ epoch trước” không được xác nhận trong code hiện tại sau khi pull repository. Logic hiện tại đã tạo lại `epoch_samples` và preference data theo từng epoch.

## 2. Cách tính metric trong repository

Implementation AlignCoder-style nằm ở [`src/co_retrieval/aligncoder_metrics.py`](src/co_retrieval/aligncoder_metrics.py).

### Exact Match

EM được tính theo từng mẫu, sau khi:

- postprocess prediction theo ngôn ngữ;
- Java được cắt về statement đầu tiên;
- Python được cắt về một statement hợp lệ;
- comment và dòng rỗng được loại bỏ;
- prediction và target phải giống nhau hoàn toàn theo từng dòng không rỗng.

Do đó, EM là metric nhị phân ở cấp mẫu. Một ký tự, token hoặc identifier sai có thể làm cả mẫu nhận điểm `0`; nó không bị trừ dần theo số ký tự sai. Tuy nhiên, whitespace, comment và phần output dư có thể được xử lý trước khi tính nên không phải mọi khác biệt hình thức đều làm EM bằng 0.

### Edit Similarity

Edit Similarity dùng `SequenceMatcher` và cho điểm liên tục từ 0 đến 100. Vì vậy prediction có thể có EM bằng 0 nhưng vẫn có Edit Similarity cao nếu giữ lại nhiều chuỗi đúng hoặc có cấu trúc tương đồng với target.

### Identifier F1

Identifier F1 lấy tập identifier trong prediction và target rồi tính precision, recall và F1. Metric này không yêu cầu đúng thứ tự, cú pháp hoặc toàn bộ nội dung. Vì vậy một prediction sai statement nhưng vẫn chứa đúng các tên biến/hàm/class có thể đạt Identifier F1 tương đối cao.

### Prefix EM phụ và Full-output EM

Prefix EM của AlignCoder là metric mềm hơn: so sánh prefix ngắn hơn sau khi loại whitespace. Nó không tương đương với full-output exact match. Full-output EM bằng 0 ở cả hai model cho thấy metric này quá nghiêm ngặt hoặc output generation chưa được crop theo protocol chính; nó không đủ khả năng phân biệt chất lượng hai model trong bảng hiện tại.

## 3. Kiểm tra setting train hiện tại

Command đã sử dụng có các tham số chính:

```text
RUN_TRAIN=1
RUN_EVAL=0
MAX_TRAIN_SAMPLES=4000
TRAIN_EPOCHS=10
RESAMPLE_TRAIN_EACH_EPOCH=1
EXPERIMENT_MODE=intent_main
ADAPTER_TYPE=none
MAX_CONTEXT_TOKENS=4096
MAX_NEW_TOKENS=128
```

Script train dùng dataset list Python và Java. Vì vậy `MAX_TRAIN_SAMPLES=4000` là cap cho tập train đã cấu hình, không nên mặc định hiểu là 4000 mẫu cho mỗi ngôn ngữ nếu chưa kiểm tra log phân bổ thực tế.

### 3.1. Soft prompt không được train

Trong [`src/co_retrieval/neural_training.py`](src/co_retrieval/neural_training.py), adapter chỉ được bật khi:

```python
self.use_adapter = config.adapter_type == "soft_prompt"
```

Với `ADAPTER_TYPE=none`, `self.use_adapter=False` và Phase 1 soft-prompt trả về ngay với loss bằng 0. Generator backbone DeepSeek vẫn bị đóng băng.

Do đó run hiện tại nên được mô tả chính xác là:

> ICAR retriever + gate với frozen generator, không có soft-prompt adapter.

Không nên dùng kết quả của run này để kết luận trực tiếp về phiên bản ICAR đầy đủ có soft prompt.

### 3.2. LiPO group và label theo epoch

Trong code hiện tại:

- `RESAMPLE_TRAIN_EACH_EPOCH=1` tạo sample provider mới;
- mỗi epoch xây dựng lại `epoch_samples`;
- Phase 2 gọi lại `phase2_build_preference_data(epoch_samples)`;
- utility và preference groups được tính lại từ dữ liệu epoch hiện tại.

Vì vậy chưa có bằng chứng cho nhận định label bị dùng lại từ epoch trước. Nhận định này có thể từng đúng ở commit cũ, nhưng không khớp với logic của version hiện tại.

Cần phân biệt vấn đề này với train/eval mismatch: preference utility dùng teacher forcing và có cơ chế lấy tail context, còn final prediction dùng đường generate khác. Hai đường này không hoàn toàn chịu cùng một cách truncate.

### 3.3. Lệnh train không tạo metric eval

Command đặt `RUN_EVAL=0` và script train truyền `--skip-train-eval`. Vì vậy các metric CCEval phải đến từ một job eval riêng, không phải trực tiếp từ job train này.

## 4. Lỗi truncation context

### 4.1. Bằng chứng từ code

Trong [`src/co_retrieval/soft_prompt.py`](src/co_retrieval/soft_prompt.py), tokenizer được load nhưng không có lệnh ép:

```python
self.tokenizer.truncation_side = "left"
```

Các đường tokenize generation sử dụng:

```python
truncation=True
max_length=...
```

Nếu tokenizer runtime có `truncation_side="right"`, context dài sẽ giữ đầu file và bỏ phần cuối gần cursor.

AlignCoder xử lý rõ ràng bằng cách lấy tail của `left_context`:

```python
infile_context = tokenizer.encode(left_context)[-allowed_prompt_length:]
```

### 4.2. Ngưỡng 3968

Với setting train:

```text
max_context_tokens = 4096
max_new_tokens = 128
ADAPTER_TYPE = none
```

input budget của nhánh không soft prompt là:

```text
4096 - 128 = 3968 tokens
```

Vì vậy nhận định “các mẫu dài hơn 3968 token có nguy cơ bị cắt mất cursor” là đúng về mặt kỹ thuật.

Nếu dùng soft prompt với 64 prompt tokens thì ngưỡng thực tế là:

```text
4096 - 64 - 128 = 3904 tokens
```

### 4.3. Giới hạn của kết luận 110/73 mẫu

`TokenBudgetManager.pack()` của ICAR đã giữ tail của `left_context` tối đa 1500 token khi có retrieval. Tổng prompt cũng được dành headroom cho generation. Vì vậy lỗi right-truncation ảnh hưởng rõ nhất tới:

- mẫu bị gate chọn `no retrieval`;
- đường generate nhận raw `left_context`;
- prompt vẫn vượt budget sau khi pack;
- các đường batch/draft chưa được tail-slice trước khi tokenizer truncate.

Do đó, việc ICAR EM bằng 0 trên 110 mẫu Python và 73 mẫu Java là bằng chứng đáng chú ý, nhưng chưa chứng minh rằng cả nhóm đều bị lỗi truncation. Cần biết từng mẫu có retrieval hay không và prompt thực tế sau pack dài bao nhiêu.

### 4.4. Chênh lệch giữa context train và eval

Script eval mặc định đặt `EVAL_MAX_CONTEXT_TOKENS=3072`, trong khi script train dùng 4096. Với `max_new_tokens=128` và `ADAPTER_TYPE=none`, ngưỡng input trong eval mặc định là:

```text
3072 - 128 = 2944 tokens
```

Vì vậy con số 3968 chỉ đúng nếu job eval thực sự được chạy với `--max-context-tokens 4096`. Cần kiểm tra `result.json`, log eval hoặc command thực tế trước khi thống kê nhóm long-context.

### 4.5. Các mẫu Java sinh lại method trước

Hiện tượng các mẫu `project_cc_java/3842`, `3843`, `3900` sinh lại fragment của method trước phù hợp với chẩn đoán mất vùng cursor:

1. model thấy phần đầu file hoặc method trước;
2. vùng code ngay trước cursor bị bỏ;
3. model dự đoán tiếp tục từ ngữ cảnh sai;
4. prediction có thể vẫn chứa code hợp lệ và identifier đúng, nhưng không thể khớp target.

Đây là một dấu hiệu mạnh của lỗi prompt construction, nhưng vẫn cần xác nhận bằng input token thực tế.

## 5. Các nhận định logic bổ sung từ rà code và test

Rà code bổ sung đã chạy được **55 tests passed, 1 skipped**. Kết quả này xác nhận các test hiện tại không bị regression, nhưng pass test không đồng nghĩa với việc prompt protocol, train/eval distribution hoặc objective LiPO đã đúng với mục tiêu nghiên cứu. Các rủi ro bổ sung dưới đây cần được xem là những giả thuyết implementation có mức ưu tiên khác nhau.

### 5.1. Truncation sai hướng ở DenseRetriever query

`DenseRetriever._encode_batch()` tokenize text với `truncation=True`, `max_length=512` nhưng không ép giữ suffix. Xem [`src/co_retrieval/dense_retriever.py`](src/co_retrieval/dense_retriever.py).

Trong khi đó, `IntentSketcher` đặt các hint trước rồi mới đặt `### Left context tail` và phần tail của `left_context` vào cuối query. Nếu query vượt 512 token và tokenizer dùng right truncation, phần query bị cắt đúng ở phía cuối — nơi chứa context gần cursor.

Đây là lỗi nghiêm trọng hơn lỗi truncation ở generator vì nó làm hỏng retrieval trước khi gate/generator nhận context. Hệ quả có thể là:

1. query embedding không còn chứa prefix/cursor context quan trọng;
2. retriever chọn chunk không liên quan;
3. gate học utility trên retrieval sai;
4. generator nhận context retrieval kém dù left context ban đầu vẫn đúng.

Vì vậy cần kiểm tra truncation ở cả hai tầng: **query encoder** và **generator input**. Chỉ sửa generator là chưa đủ.

### 5.2. Mismatch giữa teacher-forcing utility và generation thật

Đây là một điểm confounder quan trọng:

- đường teacher-forcing dùng trong utility/LiPO đã chủ động lấy tail của context;
- đường `generate()`/`generate_batch()` vẫn dùng tokenizer truncation mà không chỉ định left truncation.

Như vậy, utility có thể đánh giá một prompt còn vùng cursor, trong khi final generation lại nhận prompt đã mất vùng cursor. LiPO và gate sẽ tối ưu trên một điều kiện khác với điều kiện dùng để tính EM/ES.

Đây là lý do kết quả utility/NLL có thể trông hợp lý nhưng prediction cuối cùng vẫn rất kém trên long-context.

### 5.3. Mismatch trong cost-aware gate

Ở `intent_mode=cost_aware`, retrieval query có thể được thay đổi bằng draft completion trong [`src/co_retrieval/neural_training.py`](src/co_retrieval/neural_training.py):

```text
base query -> cheap retrieval -> draft enhancement -> enhanced retrieval
```

Tuy nhiên, gate example vẫn dùng `_gate_query(sample)` và static gate features. Như vậy label utility có thể phản ánh retrieval được tạo bởi enhanced query, còn input mà gate nhìn thấy lại mô tả static query.

Đây là train/inference feature mismatch: gate học dự đoán utility của một query khác với query được dùng làm input cho gate. Cần kiểm tra riêng bằng `intent_mode=static` và `intent_mode=cost_aware`, đồng thời log `base_query`, `enhanced_query`, `gate_query` và candidate chunks.

### 5.4. LiPO đang chấm điểm cả set bằng mean similarity

Trong [`src/co_retrieval/dense_retriever.py`](src/co_retrieval/dense_retriever.py), retrieval score của một context set là:

```python
mean(sim(query, chunk_i))
```

Với `top_k=3`, nếu một chunk rất tốt nhưng hai chunk còn lại nhiễu, score set bị trung bình hóa. Objective không có tín hiệu đủ rõ để phân biệt chunk nào làm hại set; retriever có thể học nâng hoặc hạ cả nhóm cùng nhau.

Đây là rủi ro thiết kế objective, chưa phải bug chắc chắn. Cần đo:

- utility của từng chunk riêng lẻ;
- utility của từng top-k set;
- correlation giữa từng chunk similarity và NLL improvement;
- kết quả với `top_k=1`, `top_k=3` và một objective aggregation khác.

### 5.5. Retriever được fine-tune trực tiếp ở BF16

Trên CUDA, `DenseRetriever` load encoder bằng BF16 rồi đưa trực tiếp parameters vào AdamW. Đây là thay đổi cần kiểm tra vì full fine-tuning encoder với parameters và optimizer state ở BF16 có thể làm update kém ổn định hơn FP32 master weights hoặc mixed-precision training chuẩn.

Không nên kết luận ngay rằng BF16 là nguyên nhân. Cần log:

- dtype của encoder parameters;
- dtype của AdamW state (`exp_avg`, `exp_avg_sq`);
- gradient norm và số lần gradient underflow/overflow;
- loss/recall theo epoch.

Nên có ablation `dense_frozen` và `learned_retriever` để tách ảnh hưởng của retriever fine-tuning khỏi các lỗi prompt.

### 5.6. Các confounder về baseline và experiment mode

Hai thay đổi có thể làm kết quả không còn là phép so sánh cùng điều kiện:

- encoder mặc định đã đổi từ Jina 1.5B sang UniXcoder;
- generator đã đổi từ Qwen Coder Instruct sang DeepSeek Coder Base.

Ngoài ra, script hiện tại mặc định dùng `sequential_retriever_first` và `ADAPTER_TYPE=none`; setting đang chạy là `intent_main` nhưng vẫn không bật adapter. Đây không phải full alternating co-training với soft prompt.

Vì vậy cần ghi rõ generator, encoder, adapter, experiment mode và context budget trong mọi bảng so sánh. Nếu không, metric drop có thể đến từ model/config confounder thay vì novelty.

## 6. Phân loại nguyên nhân: novelty hay implementation?

### Đã có cơ sở mạnh

- Soft prompt không được train trong run hiện tại.
- Tokenizer không được ép left truncation.
- Train và eval có thể dùng context budget khác nhau.
- EM là metric rất nhạy và sẽ phạt toàn bộ mẫu khi mất cursor hoặc sai một token.

### Chưa được xác nhận

- Toàn bộ 110 mẫu Python và 73 mẫu Java đều bị right-truncation.
- LiPO label bị reuse giữa các epoch trong version hiện tại.
- Toàn bộ khoảng cách metric giữa ICAR và AlignCoder chỉ do truncation.

### Đánh giá tổng thể

Đối với nhóm long-context, implementation/evaluation protocol có khả năng là nguyên nhân chính hơn novelty. Đối với toàn bộ dataset, vẫn có thể tồn tại thêm các nguyên nhân khác như:

- retriever chưa học đủ tốt;
- gate chọn sai retrieval/no-retrieval;
- prompt packing khác AlignCoder;
- generator bị đóng băng;
- soft prompt chưa được sử dụng;
- context budget và output postprocessing không đồng nhất.

Vì vậy, kết quả hiện tại chưa phải một phép so sánh công bằng giữa AlignCoder và full ICAR.

## 7. Thí nghiệm xác nhận cần thực hiện

### Thứ tự ưu tiên kiểm tra

1. Sửa hoặc kiểm chứng giữ suffix ở cả DenseRetriever query và generator input.
2. Eval lại cùng checkpoint với `max_context_tokens=4096`, batch size 1 và log context thực tế.
3. So sánh `raw_query_main` với `intent_main`.
4. So sánh `always_retrieve` với `learned` gate.
5. So sánh `dense_frozen` với retriever sau LiPO.
6. Sau cùng mới bật soft prompt và chạy full alternating co-training.

### Thí nghiệm A/B không retrain

Giữ nguyên checkpoint, dataset, index và decoding; chỉ thay đổi tokenizer sang left truncation:

```python
self.tokenizer.truncation_side = "left"
```

Nên áp dụng và kiểm tra tất cả đường generation scalar, batch và draft.

### Log bắt buộc cho mỗi mẫu

- `tokenizer.truncation_side`;
- `max_context_tokens`;
- `max_new_tokens`;
- raw context token count;
- packed context token count;
- token count sau truncation;
- `retrieval_used`;
- số chunk được pack;
- prefix cuối cùng của input ngay trước generation.

### Báo cáo metric theo nhóm

Tách riêng:

1. context ngắn dưới budget;
2. context dài + retrieval;
3. context dài + no retrieval;
4. context vượt ngưỡng eval thực tế;
5. Python và Java.

Nếu sau khi sửa left truncation, EM của nhóm dài tăng mạnh, hiện tượng lặp method biến mất và nhóm ngắn gần như không đổi, thì có thể kết luận phần sụt giảm đó chủ yếu do implementation.

## 8. Hướng giải quyết các vấn đề tồn đọng

Mục tiêu là bảo đảm retriever, gate và generator cùng nhìn thấy một query/prompt nhất quán. Nên triển khai theo thứ tự P0 → P1 → P2 để tránh retrain tốn kém trước khi sửa xong protocol.

### 8.1. P0 — Thống nhất truncation và context contract

Đây là nhóm sửa có khả năng cải thiện metric trực tiếp nhất và có thể kiểm chứng mà không cần retrain.

#### Query của retriever

Không nên chỉ đặt global `tokenizer.truncation_side="left"`, vì intent hint nằm ở đầu query còn cursor context nằm ở cuối. Left truncation thuần túy có thể giữ cursor nhưng làm mất hint. Nên cấp ngân sách riêng:

```text
[intent hints đã rút gọn] + [left-context tail]
```

Ví dụ với budget 512 token:

- dành 96–128 token cho intent hints;
- phần còn lại cho suffix gần cursor;
- luôn kiểm tra token length sau khi ghép;
- không cắt raw string theo số ký tự.

Nên tạo một hàm `pack_retrieval_query()` và dùng chung ở train, Phase 2, gate feature extraction và inference. Hàm này trả cả query text lẫn metadata: token hint, token tail và trạng thái truncation.

#### Input của generator

Tạo một hàm `pack_generation_context()` dùng chung cho teacher forcing, `generate()`, `generate_batch()`, draft generation và evaluation. Contract cần:

1. reserve token cho output;
2. reserve soft-prompt token nếu adapter bật;
3. luôn giữ suffix gần cursor;
4. chỉ dùng budget còn lại cho retrieved chunks;
5. scalar và batch generation tạo cùng input IDs cho cùng một mẫu.

Sau thay đổi, teacher-forcing utility và generation thật phải nhận cùng phần context; target chỉ xuất hiện thêm ở teacher forcing.

#### Test cần thêm

- Query dài hơn 512 token vẫn chứa marker ngay trước cursor.
- Generation context dài hơn budget vẫn giữ marker cuối file.
- Scalar và batch generation có cùng input IDs trước padding.
- Nhánh retrieval/no-retrieval đều giữ cursor suffix.
- `ADAPTER_TYPE=none` và `soft_prompt` đều không vượt context window.

### 8.2. P0 — Thống nhất train/eval protocol

Đưa `max_context_tokens`, `max_new_tokens`, tokenizer, truncation policy và postprocessing vào một config lưu cùng checkpoint. Eval mặc định phải đọc config này; mọi override phải được ghi vào `result.json` và tên experiment.

Trước mắt nên eval lại cùng checkpoint với:

```text
max_context_tokens=4096
max_new_tokens=128
batch_size=1
```

Sau đó chạy 3072 như một ablation về memory, không dùng 3072 làm kết quả chính nếu baseline dùng 4096.

Mỗi prediction row nên lưu raw/packed/final token count, số token bị cắt, query đã pack, selected chunk IDs và scores, gate probability/decision và context budget thực tế.

### 8.3. P1 — Dùng tín hiệu retriever để hiệu chỉnh gate

Gate hiện có chín feature inference-safe:

| Nhóm | Feature | Ý nghĩa |
|---|---|---|
| Độ tin cậy | `top1_score`, `top1_margin` | Top-1 có mạnh và tách biệt top-2 không |
| Độ bất định | `score_entropy`, `score_std` | Retriever có phân vân không |
| Quy mô | `selected_count_ratio`, `candidate_pool_ratio` | Quy mô context và candidate pool |
| Chất lượng query | `query_identifier_ratio` | Query có đủ identifier cụ thể không |
| Chi phí | `context_token_ratio` | Retrieval chiếm bao nhiêu context budget |
| Liên quan lexical | `identifier_overlap_ratio` | Query và chunk có chung identifier không |

Các feature này cần được hiệu chỉnh theo tín hiệu đích:

```text
utility_delta = NLL(stop) - NLL(retrieved_context)
```

Trên validation set, đo correlation/AUROC của từng feature với nhãn `utility_delta > margin`. Sau đó:

- bỏ feature không có tín hiệu ổn định;
- chuẩn hóa score theo encoder/checkpoint;
- calibration threshold gate trên validation, không cố định 0.5;
- báo cáo precision/recall retrieve cùng retrieval rate.

Một policy khởi đầu có thể dùng ba vùng:

```text
Tin cậy cao: margin cao, entropy thấp, overlap cao -> retrieve
Tin cậy thấp: margin thấp, entropy cao, overlap thấp -> skip
Không chắc chắn -> query enhancement hoặc giảm xuống top_k=1
```

Threshold phải được calibration, không hard-code theo trực giác. Entropy hiện chỉ dựa trên top-k selected scores, nên cần log thêm margin/entropy trên candidate pool lớn hơn để có tín hiệu bất định đáng tin hơn.

### 8.4. P1 — Sửa mismatch cost-aware query và gate

Chọn một trong hai contract:

1. **Gate-before-enhancement:** gate nhìn base query; label cũng tính từ retrieval bằng base query. Chỉ enhancement sau khi gate quyết định retrieve.
2. **Gate-after-enhancement:** gate nhìn enhanced query, enhanced retrieval scores và đúng features của enhanced retrieval; label cũng tính từ context đó.

Phương án 1 rẻ và đơn giản hơn. Phương án 2 sát deployed retrieval hơn nhưng luôn tốn chi phí draft trước quyết định gate. Không tiếp tục dùng label từ enhanced retrieval nhưng feature từ static query.

Log `base_query`, `enhanced_query`, `query_changed`, score trước/sau và utility delta. Chỉ giữ enhancement nếu cải thiện Recall@k hoặc NLL utility, đồng thời không làm giảm EM/ES/Identifier F1.

### 8.5. P1 — Cải thiện LiPO bằng tín hiệu theo chunk

Mean similarity làm mờ contribution của từng chunk. Trước tiên tính leave-one-out utility:

```text
contribution(chunk_i) = NLL(C_without_i) - NLL(C_full)
```

- dương: chunk có ích;
- âm: chunk gây hại;
- gần 0: chunk dư thừa.

So sánh contribution với similarity rank. Nếu chunk score cao thường có contribution âm, retriever đang tối ưu sai proxy hoặc query bị lỗi.

Thử lần lượt:

1. `top_k=1` làm baseline attribution rõ ràng;
2. softmax-weighted aggregation thay mean đều;
3. auxiliary per-chunk ranking loss theo leave-one-out utility;
4. kết hợp set-level LiPO với per-chunk loss;
5. dynamic top-k, chỉ thêm chunk nếu margin/estimated utility đủ tốt.

Không đổi objective trước khi sửa query truncation, vì attribution trên query sai sẽ tiếp tục nhiễu.

### 8.6. P1 — Kiểm chứng retriever có thực sự học tốt hơn

So sánh BM25, `dense_frozen` và `learned_retriever` trên cùng validation/index. Không chỉ nhìn EM cuối:

| Tầng | Tín hiệu cần theo dõi |
|---|---|
| Query | tỷ lệ truncate, identifier retention, query changed rate |
| Retrieval | Recall@1/3/5, MRR, top-1 score, margin, entropy |
| Utility | NLL delta, tỷ lệ utility dương, leave-one-out contribution |
| Gate | AUROC, precision/recall retrieve, retrieval rate, calibration error |
| Generation | EM, ES, Identifier F1 theo retrieval/no-retrieval và context length |

Diễn giải:

- similarity/margin tăng nhưng Recall@k và utility giảm → LiPO đang tối ưu sai proxy;
- Recall@k tăng nhưng generation không tăng → lỗi ở packing/generator/gate;
- `always_retrieve` tốt hơn learned gate → gate/label/calibration có vấn đề;
- `dense_frozen` tốt hơn learned retriever → fine-tuning hoặc LiPO gây degradation.

### 8.7. P1 — Ổn định fine-tuning retriever

Chạy ablation BF16 với mixed precision có FP32 master weights hoặc encoder FP32, giữ nguyên seed và LiPO groups. Log retriever loss, gradient norm, parameter/update norm ratio, NaN/Inf, embedding norm, cosine drift so với frozen encoder và Recall@k theo epoch.

Nếu BF16 có update gần 0, loss bất ổn hoặc retrieval recall giảm rõ rệt, chuyển sang mixed precision/FP32 optimizer state. Nếu không khác đáng kể thì loại BF16 khỏi nhóm nguyên nhân chính.

### 8.8. P2 — Đánh giá đúng full ICAR và novelty

Chỉ sau P0/P1 mới chạy full setting:

```text
intent_main
ADAPTER_TYPE=soft_prompt
RESAMPLE_TRAIN_EACH_EPOCH=1
```

Ablation matrix tối thiểu:

| Run | Retriever | Gate | Soft prompt | Query mode |
|---|---|---|---|---|
| A | frozen | always retrieve | off | raw/static |
| B | LiPO | always retrieve | off | raw/static |
| C | LiPO | learned | off | raw/static |
| D | LiPO | learned | off | intent |
| E | LiPO | learned | on | intent |
| F | LiPO | learned | on | cost-aware |

Ma trận này tách được đóng góp của LiPO, gate, intent query, soft prompt và cost-aware enhancement.

### 8.9. Tiêu chí hoàn thành

Chỉ coi các lỗi implementation chính đã được xử lý khi:

1. query/generation luôn giữ marker gần cursor trong test long-context;
2. teacher forcing và generation dùng cùng packed context IDs;
3. train/eval dùng cùng context contract;
4. gate label/query/features đến từ cùng retrieval path;
5. learned retriever cải thiện Recall@k hoặc utility so với `dense_frozen`;
6. A/B left-truncation và 4096-context được báo cáo riêng;
7. metric được phân tầng theo context length, retrieval decision và query mode.

## 9. Kết luận cuối

Nhận định ban đầu của partner rằng “có thể train/eval đang xử lý sai left context” là có cơ sở và cần ưu tiên kiểm tra ở cả query retriever lẫn generator. Rà code bổ sung còn cho thấy mismatch giữa utility teacher-forcing và generation thật, rủi ro cost-aware gate dùng query khác với query đầu vào, cùng một số confounder về objective LiPO, BF16 và model configuration.

Nhận định về soft prompt cũng đúng: setting hiện tại không train soft prompt. Ngược lại, nhận định LiPO label bị reuse giữa các epoch không được xác nhận trong repository hiện tại.

Do đó, trước khi đánh giá novelty của ICAR, cần sửa và kiểm soát ba điểm:

1. bật đúng adapter nếu muốn đánh giá full ICAR;
2. thống nhất context budget và truncation protocol với baseline;
3. chạy A/B left-truncation trên cùng checkpoint.

Chỉ sau các bước này mới có thể kết luận phần chênh lệch còn lại đến từ chất lượng phương pháp ICAR thay vì lỗi implementation hoặc protocol đánh giá.
