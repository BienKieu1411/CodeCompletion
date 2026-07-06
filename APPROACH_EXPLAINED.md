# Co-Retrieval trong repository này đang làm gì?

## 1. Tóm tắt trong một câu

Repository này đang xây dựng một phương pháp **retrieval-augmented code completion thích ứng**: từ đoạn code đang viết dở, hệ thống tạo một truy vấn biểu diễn ý định, tìm các đoạn code liên quan trong repository, dùng mức cải thiện xác suất sinh target để học retriever, và học thêm một gate quyết định khi nào nên retrieve hoặc bỏ qua retrieval.

Tên hiện tại trong proposal là:

> **Intent-Conditioned Adaptive Co-Retrieval for Repository-Level Code Completion**

Phương pháp được định vị là một hướng phát triển từ **AlignCoder**, nhưng thay đổi cả cách tạo query, tín hiệu huấn luyện retriever và cách quyết định có nên retrieval hay không.

---

## 2. Bài toán đang giải quyết

Giả sử lập trình viên đang viết:

```python
client = PaymentClient(...)
client.ref
```

Model cần đoán phần tiếp theo, ví dụ:

```python
client.refund_payment(request)
```

Thông tin cần thiết có thể không nằm trong file hiện tại mà nằm ở file khác:

```python
class PaymentClient:
    def refund_payment(self, request: RefundRequest):
        ...
```

Nếu dùng trực tiếp `client.ref` làm retrieval query, retriever có rất ít thông tin. Đây là **semantic gap** giữa:

- code chưa hoàn chỉnh ở cursor;
- code target chưa được nhìn thấy;
- evidence cần tìm trong repository.

Repo này tìm cách xử lý semantic gap, retrieval noise và trường hợp không cần cross-file context.

---

## 3. Pipeline hiện tại

### 3.1. Tạo completion sample

Mỗi sample gồm:

- `left_context`: code trước cursor;
- `target`: đoạn code cần sinh;
- `candidate_chunks`: các đoạn code có thể retrieve từ repository.

Code liên quan:

- `src/co_retrieval/data/repository_dataset_loader.py`
- `src/co_retrieval/data/left_context_anchor_extractor.py`
- `src/co_retrieval/chunking.py`

### 3.2. Chia repository thành AST chunks

Thay vì cắt code thành cửa sổ cố định, hệ thống cố gắng giữ các đơn vị có nghĩa như:

- function;
- class header;
- method;
- global/import block;
- fallback line block nếu parse thất bại.

Trực giác của AST chunking giống việc chia một cuốn sách theo đoạn/chương thay vì cứ đủ 100 từ là cắt. Retriever nhận được các đơn vị ít bị đứt gãy về ngữ nghĩa hơn.

AST chunking là hạ tầng tốt, nhưng không nên được xem là novelty chính.

### 3.3. Tạo intent-conditioned query

`IntentSketcher` không đưa raw left context trực tiếp cho retriever. Nó trích các tín hiệu như:

- incomplete prefix;
- object và prefix trong member access;
- imports;
- class/type hints;
- local identifiers;
- phần code gần cursor.

Ví dụ trực giác:

```text
Raw query:
    client.ref

Intent sketch:
    member_owner: client
    member_prefix: ref
    imported/type hints: PaymentClient, RefundRequest
    local identifiers: client, request
    tail context: client.ref
```

Nếu coi dense retrieval là tìm các vector gần nhau, intent sketch cố gắng tạo một đầu vào chứa nhiều “móc ngữ nghĩa” hơn để query embedding nằm gần các code chunk hữu ích.

Code liên quan:

- `src/co_retrieval/intent.py`
- `NeuralCoTrainer._retrieval_query`

### 3.4. Dense retrieval

Retriever encode query và code chunk thành vector chuẩn hóa:

\[
q=f_\theta(X),\qquad d_i=f_\theta(C_i)
\]

Sau đó tính cosine similarity:

\[
s(q,d_i)=q^\top d_i
\]

Chunk có similarity cao hơn được xếp hạng cao hơn.

Đây là phần gần với **contrastive representation learning**:

- query và chunk hữu ích cần có embedding gần nhau;
- query và chunk gây nhiễu cần có embedding xa nhau.

Điểm khác là repo không xác định positive/negative chỉ bằng lexical match. Positive và negative được suy ra từ tác động của chunk lên generator.

Code liên quan:

- `src/co_retrieval/dense_retriever.py`
- `src/co_retrieval/embedding_cache.py`

### 3.5. Đo context utility bằng generator

Với target vàng `Y`, hệ thống đo:

\[
U(C)=NLL(Y\mid X)-NLL(Y\mid X,C)
\]

Trong đó:

- `NLL(Y|X)` là độ khó khi generator dự đoán target mà không retrieve;
- `NLL(Y|X,C)` là độ khó khi có retrieved context `C`;
- NLL càng thấp thì model gán xác suất cho target càng cao.

Diễn giải:

- `U(C) > 0`: context giúp generator;
- `U(C) = 0`: không tốt hơn stop;
- `U(C) < 0`: context làm generator dự đoán target kém đi.

Ví dụ:

```text
Không retrieve: NLL = 2.8
Retrieve chunk A: NLL = 1.9  -> utility = +0.9
Retrieve chunk B: NLL = 3.1  -> utility = -0.3
```

Chunk A là positive tốt hơn chunk B, dù B có thể nhìn giống query về mặt từ vựng.

Đây là dạng **learning from downstream feedback**: retriever không chỉ học “đoạn nào giống query”, mà học “đoạn nào thực sự giúp generator”.

Code liên quan:

- `src/co_retrieval/context_utility.py`
- `NeuralCoTrainer.phase2_build_preference_data`

### 3.6. Tạo preference pairs và huấn luyện retriever

Các strategy được chấm utility rồi ghép thành pair:

```text
chosen  = context có utility cao hơn
rejected = context có utility thấp hơn
```

Retriever được tối ưu để score `chosen` cao hơn `rejected`, đồng thời so với một reference encoder đã freeze.

Trực giác giống preference learning:

```text
Query q
  chunk A giúp generator       -> kéo score lên
  chunk B làm generator tệ đi  -> đẩy score xuống
```

Code gọi objective này là DPO-style loss. Chính xác hơn về mặt học thuật, nó có thể được mô tả là:

> **reference-regularized pairwise preference optimization for retrieval**

Lý do nên dùng tên thận trọng: policy ở đây là cosine scoring model, không phải autoregressive policy như DPO truyền thống.

Code liên quan:

- `DenseRetriever.dpo_loss`
- `NeuralCoTrainer.phase3_dpo_training`

### 3.7. Học retrieve/skip gate

Không phải sample nào cũng cần retrieval. Gate học nhãn:

\[
y_{gate}=\mathbb{1}\left[\max_C U(C)>\delta\right]
\]

Trong đó `δ` là `utility_margin`.

Trực giác của gate giống một classifier nhị phân:

```text
left context -> P(retrieve)

P(retrieve) cao: gọi retriever
P(retrieve) thấp: sinh trực tiếp
```

Mục tiêu là giảm:

- context noise;
- retrieval latency;
- số token cross-file đưa vào generator.

Code liên quan:

- `src/co_retrieval/neural_gate.py`
- `GateTrainingExample`
- `NeuralCoTrainer._should_retrieve`

### 3.8. Soft prompt và alternating co-training

Generator chính được freeze. Repo chỉ học một số `soft prompt embeddings` đặt vào input của generator.

Soft prompt có thể hình dung là một chuỗi “virtual tokens” có vector học được:

```text
[virtual token 1 ... virtual token n] + retrieved context + left context
```

Nó không sửa toàn bộ LLM mà học cách hướng dẫn LLM sử dụng retrieved context.

Training luân phiên:

1. cập nhật soft prompt để đọc context tốt hơn;
2. dùng generator hiện tại đo lại utility;
3. cập nhật retriever theo preference;
4. cập nhật gate theo utility label;
5. rebuild embedding index;
6. lặp lại.

Ý tưởng co-adaptation là:

- retriever học lấy loại context generator hiện tại dùng được;
- adapter học đọc loại context retriever hiện tại thường trả về.

Điều này chỉ trở thành contribution nếu alternating training thắng sequential training với cùng compute budget.

---

## 4. AlignCoder thực sự làm gì?

Theo [paper AlignCoder](https://arxiv.org/abs/2601.19697) và [source code chính thức](https://github.com/DeepSoftwareAnalytics/AlignCoder), pipeline có ba giai đoạn.

### 4.1. Xây retrieval codebase

AlignCoder tạo hai loại snippet:

1. **Base snippets**: code được split theo blank line rồi aggregate tới giới hạn độ dài.
2. **Dependency snippets**: signature/class/method information trích từ intra-repository imports bằng tree-sitter.

### 4.2. Query enhancement bằng sampled completions

AlignCoder thực hiện:

```text
unfinished code
    -> coarse BM25 retrieval
    -> sampler LLM sinh k candidate completions
    -> nối unfinished code với tất cả candidate completions
    -> enhanced query
    -> fine-grained dense retrieval
```

Trực giác:

```python
client.ref
```

có thể chưa chứa token `refund_payment`. Nếu một trong bốn sampled completions đoán ra token này, enhanced query sẽ chứa tín hiệu giúp retrieve đúng definition.

AlignCoder dùng nhiều sample vì một sample sai không nhất thiết phá toàn bộ query. Paper báo cáo hiệu quả tốt nhất thường quanh bốn samples; quá nhiều sample bắt đầu thêm noise.

### 4.3. Huấn luyện AlignRetriever

Với mỗi enhanced query, AlignCoder retrieve một tập candidate chunks. Evaluator LLM tính target loss/PPL khi ghép từng chunk. Chunk có PPL thấp nhất được chọn làm positive:

\[
c^*=\arg\min_{c_i}\operatorname{PPL}(Y\mid q,c_i)
\]

Retriever sau đó tăng softmax probability của `c*` trong candidate set.

Paper gọi đây là reinforcement learning. Tuy nhiên, code công khai thực hiện gần với classification/ranking hơn policy-gradient RL:

```python
labels = losses.reshape(batch, num_candidates).argmin(-1)
logits = query_embeddings @ document_embeddings.T
loss = CrossEntropyLoss(logits, labels)
```

Do đó, cách hiểu đơn giản nhất là:

> Generator dùng PPL để chọn positive chunk; retriever học kéo query embedding gần positive chunk và xa các chunk còn lại.

Đây là một điểm quan trọng khi định vị novelty: repo hiện tại không chỉ “thay PPO bằng DPO”, vì implementation AlignCoder công khai vốn đã gần preference/contrastive ranking hơn RL policy optimization theo nghĩa chặt.

---

## 5. Repo này khác AlignCoder ở đâu?

| Thành phần | AlignCoder | Repo hiện tại |
|---|---|---|
| Query enhancement | LLM sinh nhiều candidate completions | Static intent sketch từ code hiện có |
| Chi phí query | Cao hơn vì phải gọi sampler | Rẻ, deterministic |
| Rủi ro query | Hallucinated/noisy completion | Thiếu suy luận semantic sâu |
| Chunking | Base + dependency snippets | AST entity chunks |
| Feedback | Chọn chunk có PPL thấp nhất trong retrieved set | Utility tương đối với no-retrieval baseline |
| Retriever objective | Cross-entropy trên positive chunk | DPO-style pairwise preference loss |
| Không cần retrieval | Không có learned utility gate rõ ràng | Separate retrieve/skip gate |
| Generator adaptation | Generator dùng retrieved context | Frozen generator + trainable soft prompt |
| Joint adaptation | Chủ yếu train retriever | Alternating retriever/gate/adapter |

Phần improvement có triển vọng nhất không phải static intent sketch đơn lẻ, mà là:

1. **utility được neo vào stop baseline**;
2. **adaptive retrieve/skip gate**;
3. **retriever–consumer co-adaptation**.

---

## 6. Những vấn đề trong implementation hiện tại

### 6.1. Gate label đang nhìn thấy oracle strategy

Trong `phase2_build_preference_data`, strategy pool mặc định chứa `oracle`. Sau đó gate label được tạo từ context có utility cao nhất trong toàn bộ pool.

Hệ quả:

```text
Oracle biết một context tốt tồn tại -> gate label = retrieve
Retriever thật không tìm được context đó -> inference vẫn retrieve noise
```

Gate đang học “trong repository có context hữu ích không”, thay vì “deployed retriever hiện tại có lấy được context hữu ích không”.

#### Hướng sửa

Tách hai pool:

```text
preference_pool = {current, bm25, dense_frozen, hard_neg, oracle}
gate_pool       = {current} hoặc đúng policy sẽ deploy
```

Gate label nên là:

\[
y_{gate}=\mathbb{1}[U(R_\theta(X))>\delta]
\]

Nếu inference retrieve top-k thành một set, utility cũng phải chấm đúng top-k set đó.

### 6.2. Split ngẫu nhiên theo sample

`runner._split_train_eval` shuffle completion samples rồi chia train/eval. Với repository-level completion, sample từ cùng repo hoặc cùng file có thể xuất hiện ở cả hai tập.

Đây là data leakage nghiêm trọng vì model có thể ghi nhớ:

- tên API;
- identifier;
- class hierarchy;
- gần như chính code chunk cần retrieve.

#### Hướng sửa

Split theo `repository_id` trước, sau đó mới tạo samples:

```text
repositories -> train repositories / validation repositories / test repositories
             -> generate completion samples độc lập trong từng split
```

Benchmark chính nên là CrossCodeEval và RepoEval; internal random split chỉ dùng để debug.

### 6.3. Nhánh skip và retrieve dùng adapter khác nhau

Evaluation hiện dùng soft prompt khi retrieve nhưng tắt soft prompt khi skip. Vì vậy comparison đồng thời thay đổi cả context và adapter.

Không thể kết luận improvement đến từ gate hay soft prompt.

#### Hướng sửa

Thực hiện factorial ablation:

| Context | Adapter |
|---|---|
| Không context | Tắt |
| Không context | Bật |
| Có context | Tắt |
| Có context | Bật |

Trong gate comparison chính, giữ adapter treatment giống nhau giữa retrieve và skip.

### 6.4. Static intent sketch có thể chưa đủ mạnh để vượt AlignCoder

Static sketch rẻ nhưng phần lớn là feature extraction. AlignCoder dùng generator samples để đưa vào query token chưa xuất hiện trong left context.

Static sketch không thể tự tạo `refund_payment` nếu token đó không xuất hiện trong scope.

#### Hướng sửa khả thi

Thiết kế **hybrid adaptive query enhancement**:

```text
Stage 1: static intent sketch cho mọi sample
Stage 2: query-confidence gate
    - confidence cao -> retrieve bằng static sketch
    - confidence thấp -> gọi lightweight sampler sinh 2-4 intent candidates
Stage 3: uncertainty-aware aggregation và retrieval
```

Cách này có thể vừa rẻ hơn AlignCoder vừa giữ khả năng suy luận token mới.

### 6.5. Utility phụ thuộc generator và có thể bị stale

Context tốt với một generator chưa chắc tốt với generator khác. Khi soft prompt thay đổi, utility ranking cũng có thể đổi.

#### Hướng sửa

Đo:

- Spearman correlation của utility ranks giữa các rounds;
- utility transfer giữa ít nhất hai generators;
- correlation giữa NLL improvement và EM/Edit Similarity/Identifier-F1;
- tỷ lệ preference pair đổi chiều sau mỗi adapter update.

Nếu ranking quá bất ổn, dùng EMA teacher hoặc chỉ rebuild preference data khi rank drift vượt threshold.

### 6.6. DPO cần baseline objective

Vì AlignCoder code đã dùng cross-entropy ranking, paper mới phải chứng minh DPO-style loss tốt hơn objective đơn giản.

Các baseline bắt buộc:

- AlignCoder-style listwise cross-entropy;
- pairwise logistic loss;
- margin ranking loss;
- InfoNCE;
- DPO-style loss hiện tại.

Nếu DPO không thắng, nên bỏ “DPO” khỏi core claim và giữ đóng góp utility/gating.

---

## 7. Hướng phương pháp đề xuất

Một phiên bản chặt hơn có thể gọi là:

> **Utility-Calibrated Adaptive Query and Context Retrieval for Repository-Level Code Completion**

### 7.1. Hai quyết định thích ứng riêng biệt

Học hai gate:

1. `query_enhancement_gate`: có cần gọi sampler để enrich query không?
2. `retrieval_gate`: retrieved top-k hiện tại có đáng đưa vào generator không?

Pipeline:

```text
left context
   -> static intent sketch
   -> query uncertainty estimator
       -> đủ chắc: dùng static query
       -> không chắc: sinh candidate intent/completions
   -> retriever
   -> context utility gate
       -> skip context
       -> use top-k context
   -> generator/adapter
```

Điểm mới so với AlignCoder:

- không phải sample nào cũng trả chi phí multiple sampling;
- không phải retrieved result nào cũng được đưa vào generator;
- hai quyết định được supervision bằng downstream utility/cost.

### 7.2. Utility có tính cả chất lượng và chi phí

Thay vì chỉ dùng NLL:

\[
U(C)=NLL_{stop}-NLL_C-\lambda_{tok}\operatorname{Tokens}(C)-\lambda_{lat}\operatorname{Latency}(C)
\]

Với query enhancement:

\[
U_q(q')=U(R(q'))-\lambda_{sample}\operatorname{SamplingCost}(q')
\]

Như vậy “adaptive” có ý nghĩa thực tế: hệ thống chỉ dùng computation bổ sung khi expected quality gain đủ lớn.

### 7.3. Utility-aware listwise retrieval

Mỗi query có nhiều candidate chunks và utility liên tục. Thay vì ép thành binary chosen/rejected, có thể xây target distribution:

\[
p^*(c_i\mid q)=\operatorname{softmax}(U(c_i)/\tau)
\]

Retriever tạo:

\[
p_\theta(c_i\mid q)=\operatorname{softmax}(s_\theta(q,c_i))
\]

Sau đó tối ưu KL divergence hoặc cross-entropy giữa hai distribution.

Ưu điểm:

- dùng được toàn bộ độ lớn utility;
- không cần chọn preference margin tùy ý;
- so sánh trực tiếp và công bằng với AlignCoder-style cross-entropy.

Pairwise DPO vẫn có thể giữ làm ablation.

### 7.4. Counterfactual gate supervision

Gate nên được học bằng hai outcome đúng với policy deploy:

```text
Outcome 0: generate với no retrieved context
Outcome 1: generate với top-k do current retriever trả về
```

Label:

\[
y=\mathbb{1}[U(C_{top-k}^{current})>\delta]
\]

Không dùng oracle cho gate. Oracle chỉ dùng:

- upper bound;
- retriever training analysis;
- đo headroom còn lại.

---

## 8. Thí nghiệm cần có để bảo vệ claim “cải thiện AlignCoder”

### 8.1. Baselines

- no retrieval;
- BM25;
- frozen dense retriever;
- RepoCoder;
- RLCoder;
- AlignCoder chính thức;
- AlignCoder + cùng AST chunk pool để kiểm soát ảnh hưởng chunking.

### 8.2. Ablations

- raw query vs static intent vs sampled query vs hybrid adaptive query;
- always retrieve vs learned gate vs oracle gate;
- absolute PPL ranking vs stop-relative utility;
- CE/InfoNCE/pairwise/DPO objectives;
- fixed generator vs soft prompt;
- sequential vs alternating cùng gradient-step budget;
- gate labels có oracle vs deployed-retriever-only.

### 8.3. Metrics

Generation:

- Exact Match;
- Edit Similarity;
- Identifier EM/F1;
- target NLL.

Retrieval:

- Recall@k;
- MRR;
- oracle-hit@k;
- utility@k;
- harmful retrieval rate: `P(U(C)<0)`.

Adaptive efficiency:

- retrieval rate;
- sampler invocation rate;
- latency;
- retrieved tokens;
- quality–cost Pareto curve.

### 8.4. Protocol

- repository-disjoint train/validation/test;
- không index future/current target code trái quy định benchmark;
- cùng generator, context budget và decoding parameters;
- ít nhất ba random seeds;
- bootstrap confidence interval hoặc paired significance test;
- báo cáo cả absolute và relative improvement.

---

## 9. Claim nên dùng ở từng mức kết quả

### Trường hợp A: hybrid query + utility gate thắng AlignCoder

Có thể claim:

> An adaptive framework that selectively invokes query enhancement and retrieval based on generator-side utility, improving the quality–cost trade-off over AlignCoder.

### Trường hợp B: chỉ utility gate thắng

Nên claim:

> Utility-Calibrated Adaptive Retrieval for Repository-Level Code Completion.

Không nên đặt intent/query enhancement làm core novelty.

### Trường hợp C: alternating không thắng sequential

Bỏ “co-training” khỏi core contribution. Giữ nó như negative result hoặc implementation variant.

### Trường hợp D: static intent không thắng sampled query

Không xem static intent là replacement cho AlignCoder. Định vị nó là cheap first-stage query trong adaptive hybrid pipeline.

---

## 10. Kết luận

Repo hiện tại có ba ý tưởng đáng giữ:

1. đo context bằng utility tương đối với no-retrieval;
2. học gate riêng để tránh harmful retrieval;
3. cho retriever và context consumer thích nghi với nhau.

Điểm cần thay đổi ngay là:

1. gate label chỉ được tạo từ deployed retriever, không từ oracle;
2. split theo repository;
3. giữ adapter treatment công bằng giữa retrieve và skip;
4. so sánh DPO-style loss với AlignCoder-style cross-entropy;
5. cân nhắc hybrid static/sampled query thay vì kỳ vọng static intent luôn thay thế được AlignCoder.

Nếu thực nghiệm xác nhận được quality–cost trade-off tốt hơn AlignCoder, đóng góp mạnh nhất của bài sẽ là **adaptive computation and retrieval calibrated by downstream utility**, không đơn thuần là một retriever mới.

## Tài liệu đối chiếu

- [AlignCoder paper (arXiv)](https://arxiv.org/abs/2601.19697)
- [AlignCoder official implementation](https://github.com/DeepSoftwareAnalytics/AlignCoder)
- [AlignRetriever checkpoint](https://huggingface.co/AlignCoder/AlignRetriever)
- [AlignCoder dataset](https://huggingface.co/datasets/AlignCoder/Data4AlignCoder)
> Implementation update (2026-07-06): the neural path now defaults to LiPO including the no-retrieval `stop` action. Gate supervision follows the deployed strategy and evaluation uses repository-disjoint data. Static intent remains the default; `cost_aware` intent optionally samples completion drafts only when normalized retrieval entropy is high.

