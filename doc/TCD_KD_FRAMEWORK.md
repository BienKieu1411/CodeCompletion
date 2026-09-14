# TCD-KD: Target-Conditioned Candidate-Distribution Knowledge Distillation

**Cập nhật và kiểm chứng:** 14-09-2026.  
**Trạng thái:** report phương pháp đã sửa qua đối chiếu nguồn và kiểm tra toán học; **chưa triển khai, chưa có kết quả training/benchmark**. Đây là tài liệu canonical duy nhất của hướng hiện tại.

## 1. Kết luận sau verification

Giữ một hướng: **distill độ hữu ích của AST context đối với code generation vào UniXcoder**. Loss chính là KD trên phân phối candidate do frozen code generator tạo từ target NLL; Jina là semantic teacher phụ cần kiểm chứng, không phải thành phần chắc chắn có lợi.

Không có LiPO, chosen/rejected, pairwise training hay policy-gradient trong phương pháp đề xuất. Candidate pool cố định; teacher score được cache; chỉ student được update. KD có thể tự là loss chính, không cần ghép với một loss RL để trở thành objective hợp lệ.

**Chưa được claim “loss mới”, “framework tốt nhất” hoặc “chắc chắn vượt AlignCoder”.** REPLUG-LSR đã huấn luyện retriever bằng phân phối likelihood của frozen LM và KL. TCD-KD là tên làm việc cho một thiết kế và giả thuyết thực nghiệm, không phải xác nhận novelty. Phải vượt các baseline kiểm soát ở mục 8 trước khi viết contribution.

### 1.1 Các chỉnh sửa quan trọng so với bản trước

1. **Sửa stop decision:** `p_S(empty)` phụ thuộc số candidate; không phải xác suất tuyệt đối “không cần retrieval”. Dùng chênh lệch logit candidate–null, chi tiết mục 5.
2. **Sửa novelty:** bổ sung REPLUG-LSR; phép trừ empty NLL không tự tạo objective mới.
3. **Sửa mô tả baseline:** local AlignCoder/RLCoder dùng winner index và hard-label cross-entropy. Không có bằng chứng từ code đó để kết luận lợi ích sẽ đến từ “tránh policy-gradient không ổn định”.
4. **Sửa temperature:** teacher NLL và student cosine khác đơn vị; bỏ ép nhiệt độ chung và hệ số temperature-squared không cần thiết.
5. **Sửa Jina:** tên prefix API trước đó không chính xác; code-to-completion không mặc nhiên tương ứng dependency retrieval. Semantic loss vẫn có thể tác động gián tiếp đến stop.
6. **Sửa fairness:** baseline local dùng left context, không phải FIM mặc định. Teacher 6.7B lớn hơn evaluator 1.3B của baseline local phải được kiểm soát.
7. **Sửa cache và budget:** cache teacher, không cache embedding student qua optimizer updates; giảm pilot từ 64 xuống 16 real candidates; kiểm tra budget bằng cả tokenizer student lẫn generator.
8. **Bổ sung plan triển khai có test và điều kiện dừng**, thay cho danh sách giai đoạn chung chung.

### 1.2 Model và thành phần chốt

- **Task teacher G và final code generator:** `deepseek-ai/deepseek-coder-6.7b-base`, freeze. Teacher dùng teacher forcing; serving dùng một lần generation.
- **Semantic teacher J:** `jinaai/jina-code-embeddings-1.5b`, freeze. Chỉ dùng embeddings/similarity; không làm code generator.
- **Student S:** `microsoft/unixcoder-base`, fine-tune toàn bộ encoder và các head nhỏ ở mục 4.3.
- **Dữ liệu:** giữ cải tiến AST cho GR training và chunking của người dùng; phải nối vào interface và kiểm tra leakage trước khi sử dụng. Chưa xác minh implementation riêng của phần AST-GR trong lượt audit này.
- **Index:** AST evidence units; dense index từ student cuối cùng. BM25 và exact symbol/import lookup bổ sung candidate. SCIP không phải dependency bắt buộc của bản đầu.

Reference cho checkpoint generator xuất hiện trong [README RLCoder local](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/RLCoder/README.md); tokenizer và model revision thực tế phải được pin trong manifest trước khi chạy.

### 1.3 Cơ chế trong một hình

```text
Visible prefix + repository snapshot
  → AST units → fixed candidate IDs
                   ├─ G(prefix, candidate, target) → target NLL ─┐
                   └─ J(prefix, candidate) → similarity ────────┤
                                                               ↓ cache
                          UniXcoder → candidate logits → KD loss
                                                               ↓
Serving: visible prefix → student retrieval → null gate → context pack
                                                   → frozen G → completion
```

Chỉ mũi tên từ logits/loss về UniXcoder có gradient. Target thật không đi vào retrieval query hoặc serving.

## 2. Nguồn và kết quả đối chiếu

### 2.1 Phạm vi đọc và extraction

Đã đọc toàn bộ bản report trước sửa, các đoạn thuật toán/thí nghiệm liên quan trong nguồn primary, và mã local về label generation, retriever, prompt formatting, dataset construction, AST chunking. PDF AlignCoder có 14 trang, REPLUG có 12 trang; phần công thức cần kiểm tra đọc được từ PDF/HTML, không cần OCR cho những phần này.

Đây là **audit những claim của report**, không phải systematic literature review toàn bộ lĩnh vực. Không suy ra độ mới tuyệt đối từ việc chưa thấy một paper trong tập nguồn này. Code local được xem là snapshot đang có; chưa xác nhận nó giống hoàn toàn release gốc hoặc mọi nhánh thực nghiệm.

### 2.2 AlignCoder và RLCoder: phân biệt paper với code chạy

Paper AlignCoder dùng Split-Aggregate cho base snippets nhưng **đã có dependency parsing dùng tree-sitter**; không đúng nếu nói toàn bộ hệ thống không dùng AST. Paper mô tả evaluator DeepSeekCoder-1B khi train và nhiều generator khi đánh giá. Vì vậy AST chunking nhắm vào nhánh tạo base snippets, không chứng minh AlignCoder hoàn toàn thiếu cấu trúc. [AlignCoder, mục III.A và IV.D, PDF trang 5–7](https://arxiv.org/pdf/2601.19697).

Trong code local:

- PPL feedback → `argmin` candidate → `CrossEntropyLoss` trên cosine logits nhân 20. [AlignCoder main.py](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/AlignCoder/main.py:446).
- RLCoder có cùng dạng hard-winner update ở đoạn train được đọc. [RLCoder main.py](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/RLCoder/main.py:350).
- Query enhancement có thể gọi generator lấy draft; cần tính cả chi phí này khi so latency. Không được coi bỏ draft chắc chắn tăng accuracy. [AlignCoder main.py](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/AlignCoder/main.py:84).
- Retriever dùng mean pooling có mask và L2 normalization. [retriever.py](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/AlignCoder/retriever.py:91).
- Generator formatter chỉ lấy `left_context`, số token prefix giữ lại phụ thuộc độ dài cross-file context. [generator.py](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/AlignCoder/generator.py:58).
- Builder chọn span 16–96 trên kết quả `split(" ")`, **không phải 16–96 token của LM tokenizer**. [datasets.py](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/AlignCoder/datasets.py:128).

Hệ quả: so sánh cốt lõi là **hard-label so với soft-label cùng tín hiệu teacher**, cộng với tác động của fixed pool và representation; không nên dựng đối lập đơn giản “RL nhiễu” với “supervised KD ổn định”.

### 2.3 REPLUG-LSR là prior art gần nhất phải có

REPLUG-LSR chấm retrieved documents bằng likelihood của ground-truth continuation từ frozen LM, chuẩn hóa thành phân phối, rồi update retriever bằng KL; paper viết chiều `KL(P_R || Q_LM)` và có refresh index. [REPLUG, mục 4.1–4.4](https://arxiv.org/html/2301.12652v3).

TCD-KD đề xuất forward KL `KL(p_T || p_S)`, score âm mean NLL, fixed cache và AST/null handling cho repo completion. Những khác biệt này phải được đo; không đủ để tự kết luận có đóng góp thuật toán mới. Đặc biệt, công thức REPLUG in trong paper softmax likelihood, không được trích thành “chính xác softmax âm mean NLL” của report này.

Cần một baseline **REPLUG-style code adaptation** có tài liệu hóa phần thay đổi. Nếu chỉ đổi tên LM-supervised retrieval thành KD mà không tạo lợi ích mới, novelty chưa đạt.

### 2.4 Các nguồn liên quan khác và giới hạn suy luận

- **RocketQAv2:** listwise distribution distillation, nhưng joint update retriever–reranker và có supervised signal cho reranker. Không phải bằng chứng rằng mọi KD đều cần auxiliary supervised loss hoặc phải dùng frozen teacher. [Mục 3.2–3.4](https://ar5iv.labs.arxiv.org/html/2110.07367).
- **SPENCER:** distillation trong code retrieval, với hệ thống dual-encoder rồi cross-encoder. Không chuyển trực tiếp kết quả latency/accuracy của họ sang pipeline completion chỉ dùng student. [Paper](https://arxiv.org/abs/2508.00546).
- **cAST:** AST recursive split/merge là prior art cho structural chunking. Biên cú pháp không đảm bảo mọi dependency ngữ nghĩa đã nằm đủ trong chunk. [Paper](https://arxiv.org/abs/2506.15655).
- **UniXcoder:** có AST trong pretraining; điều đó không đảm bảo thêm AST non-terminals vào downstream input sẽ tốt hơn. Giữ source text được AST chọn biên trước; full AST serialization là ablation, không prerequisite. [Paper, mục 3.3](https://arxiv.org/abs/2203.03850).
- **Repoformer:** selective retrieval là prior art cho việc không phải query nào cũng cần cross-file evidence; không xác nhận công thức stop của bản report cũ. [Paper](https://arxiv.org/abs/2403.10059).
- **Jina technical report:** code embeddings từ backbone sinh code, không phải chứng cứ rằng semantic similarity đo trực tiếp ích lợi sinh target. [Efficient Code Embeddings from Code Generation Models](https://arxiv.org/abs/2508.21290).

## 3. Data contract: AST và candidate pool không leakage

### 3.1 Một example chứa gì?

Đặt `x` là prefix code mà IDE thật sự nhìn thấy; `y` là target continuation; `F` là các file cross-file hợp lệ trong cùng repository snapshot.

`q = R(x, visible_metadata)` là input retriever, khác với prompt code thô của generator. `R` lấy source text theo biên AST và metadata gọn như path/scope/import hiện hữu. Không dùng ký hiệu vòng `AST(q)` để định nghĩa chính q.

Bản so sánh chính là **left-only completion** theo local baseline. Suffix/FIM chỉ được bật trong một track riêng khi mọi baseline cũng có cùng thông tin.

Quy tắc bắt buộc:

1. Split theo repository trước khi tạo examples; kiểm tra fork/near-duplicate giữa các split.
2. Cố định commit/snapshot; loại target file khỏi cross-file retrieval.
3. AST của query chỉ parse phần visible. Không parse file hoàn chỉnh chứa target/suffix rồi lấy features từ cây đó.
4. Có thể dùng AST file training hoàn chỉnh để chọn biên mask, nhưng sau khi mask phải tái tạo mọi input/feature từ phần được phép nhìn thấy.
5. `y` chỉ đi vào teacher scorer. Không dùng target symbols để tạo pool, resolve query dependency, lọc candidate hoặc chọn “positive”.
6. Không dùng `str(Example)` làm query: implementation local chứa cả target và right context. [datasets.py](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/AlignCoder/datasets.py:50).
7. Nếu dữ liệu thô thiếu repo identity đáng tin cậy, chưa được tuyên bố repo-disjoint split; phải bổ sung mapping provenance trước full run.

### 3.2 AST evidence unit

Một candidate là source code cộng metadata tối thiểu, có ID ổn định theo repo commit, file, byte span và hash nội dung.

- Giữ trọn function/class khi vừa budget.
- Node quá dài: split đệ quy theo child nodes, merge siblings liên tiếp; mang theo signature/ancestor context cần thiết, đánh dấu phần được mở rộng.
- Một bundle signature/import/definition chỉ được tạo từ dependency lookup hợp lệ và phải có source spans cho từng phần.
- Không mặc định AST giải được dynamic dispatch, alias hoặc import phức tạp. Ghi unresolved edges thay vì tự đoán.
- Loại duplicate và kiểm soát nested/overlapping chunks để không lặp cùng evidence trong pack.

Budget cuối phải kiểm tra bằng **cả UniXcoder tokenizer lẫn generator tokenizer**. Non-whitespace size của astchunk chỉ là heuristic chia sơ bộ; expansion có thể làm chunk vượt budget. [astchunk builder local](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/astchunk/src/astchunk/astchunk_builder.py:63).

Nếu leaf node vẫn quá dài: log lý do và dùng fallback được khai báo hoặc loại candidate; không được âm thầm bỏ đoạn code mà vẫn claim coverage đầy đủ. Test byte-span coverage phải bao gồm comment, Unicode, node lỗi và oversized leaf.

Bản đầu dùng **AST-bounded source text**, chưa thêm full tree serialization hoặc SCIP mới. Điều này giữ cải tiến AST của người dùng nhưng tránh làm UniXcoder hết 512-token budget vì ký hiệu cây.

### 3.3 Fixed candidate pool

Pilot: **K = 16 real candidates + một null**. Tạo bốn ranked lists không nhìn target:

- exact import/symbol/AST dependency hits;
- BM25;
- UniXcoder pretrained, chưa fine-tune;
- Jina pretrained.

Lấy tối đa 4 ID chưa trùng từ mỗi nguồn theo thứ tự cố định; khi thiếu, backfill từ union còn lại bằng reciprocal-rank fusion `sum 1/(60 + rank)`, tie-break theo candidate ID. Ghi source rank, nguồn đóng góp và K thực tế; không cộng trực tiếp BM25 score với cosine.

Sau pilot mới mở K=32 rồi K=64 nếu evidence coverage và chi phí cho phép. Cố định pool cho cả run, không mining lại sau mỗi student update. Nếu thay K hoặc candidate builder, đó là cấu hình/cache version khác.

Jina dùng để mining được pin instruction `code2completion` trong pilot này. Việc thử instruction khác cho auxiliary scoring không tự thay candidate IDs; muốn đổi mining instruction phải tạo pool version riêng và chạy lại các đối chứng dùng pool đó.

Để đo riêng lợi ích semantic KD, **task-only và task+Jina dùng cùng candidate IDs**, kể cả khi pool có Jina contributions. “Không Jina auxiliary” khác với “không Jina trong toàn bộ pipeline”; ablation bỏ Jina candidate mining phải được đặt tên riêng.

Fixed pool không đảm bảo recall cao và không biến student thành reranker-only: validation/test phải truy hồi trên **toàn bộ repository index**, không chỉ trên list được teacher chấm.

## 4. Loss, teacher và student

### 4.1 Teacher NLL và prompt control

Với mỗi `c_i`, gồm `c_0 = empty`:

\[
\ell_i = -\frac{1}{T}\sum_{t=1}^{T}
\log G(y_t \mid P(x,c_i),y_{<t}).
\]

`G` freeze, `eval`, không gradient, không sampling. Chỉ target tokens đóng góp NLL. Shift causal logits/labels đúng một vị trí; mask prompt và padding theo vị trí, không suy ra padding chỉ từ token ID nếu pad trùng EOS.

Target token IDs và số T phải giống nhau giữa mọi candidate của một example. Lưu span mask gốc và scored target IDs; nếu benchmark giới hạn completion length thì áp đúng giới hạn đó và log truncation. Không tự cắt một AST target lớn rồi gọi phần còn lại là full target.

**Quan trọng:** giữ nguyên current-file token IDs cho tất cả candidates, kể cả empty. Nếu empty nhận thêm prefix còn real candidate làm prefix bị cắt, `ell_0 - ell_i` trộn hai hiệu ứng: thêm evidence và mất local context.

Controlled formatter dự kiến:

- total prompt tối đa 2048 token generator;
- cố định local prefix tối đa 512 token;
- path/header/separator/special tokens chiếm H token;
- cross-file allowance `B_C = min(1536, 2048 - len(local_ids) - H)`;
- phần budget trống không được dùng để kéo thêm prefix riêng cho một candidate;
- scorer và final generator dùng cùng formatter version và rendering source code.

Baseline native formatter được giữ trong reproduction track riêng ở mục 8. Không lặng lẽ sửa baseline rồi so với số paper.

Main dùng unweighted mean NLL. Weighted identifier/first-token NLL chỉ là ablation. Local weighted implementation dùng index trên labels đã flatten, nên điều kiện `i < 1` không chỉ đúng first valid target token của từng example; không copy nguyên xi sang scorer mới. [generator.py](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/AlignCoder/generator.py:118).

### 4.2 Soft target trên candidate list

\[
g_0=0,\quad g_i=\ell_0-\ell_i,\qquad
p_T(i)=\operatorname{softmax}_{0:K}(g_i/\tau_G).
\]

`g_i > 0` nghĩa là candidate giảm mean target NLL trong prompt đã cố định. Đây là proxy generation utility, không phải nhãn correctness.

Phép trừ baseline là một hằng số trên cả list:

\[
\operatorname{softmax}((\ell_0-\ell_i)/\tau_G)
=\operatorname{softmax}(-\ell_i/\tau_G).
\]

Vì vậy **gain không tạo thêm thông tin ranking**. Giá trị thực tế của empty là thêm một action tham chiếu vào list; gain hữu ích cho diagnostic và quyết định candidate–null.

Không clip gain trong cấu hình đầu. Nếu clip về sau phải khai báo vì clipping thay đổi score ratios. Mean NLL giúp scale giữa target lengths; trong một example, tổng NLL và mean NLL cho cùng thứ tự nhưng khác độ sắc của phân phối.

`tau_G` điều chỉnh độ mềm, không biến `p_T` thành calibrated probability. Khởi đầu `tau_G=0.5`; thử `{0.1, 0.25, 0.5, 1.0}` trên validation. Lưu raw NLL để đổi nhiệt độ không cần chạy G lại.

### 4.3 UniXcoder student và score scale

Dùng một backbone UniXcoder shared weights cho query/candidate, masked mean pooling như baseline. Giữ hai projection heads 768→768 của thiết kế trước, khởi tạo identity, không bias:

\[
h_q=\operatorname{norm}(W_q f_\theta(q)),\qquad
h_i=\operatorname{norm}(W_c f_\theta(c_i)).
\]

Real-candidate logits:

\[
z_i=\beta h_q^\top h_i,\quad i\ge1,\qquad \beta=20.
\]

Null head:

\[
z_0=\beta(u^\top h_q+b_0).
\]

`u` và `b_0` train cùng encoder và projections. Khởi tạo `u=0`; `b_0` bằng median cosine của các real candidates trong một minibatch training cố định, không dùng test hoặc target. Mục đích là tránh null logit lệch scale ngay đầu train, không phải một prior được học từ test.

\[
p_S=\operatorname{softmax}(z_0,\ldots,z_K).
\]

`beta=20` là starting scale theo local baseline; không ép `tau_G` bằng nghịch đảo beta vì NLL và cosine khác đơn vị. Chỉ tune beta nếu pilot cho thấy không fit được teacher distribution. Projection/no-projection được kiểm soát bằng ablation nhỏ, không claim đây là contribution.

Input cap ban đầu: query 256, candidate 512 **token UniXcoder gồm special tokens và metadata**. Dùng mode/tokenization của local UniXcoder adapter; không vô tình encode AST candidate rồi cắt bỏ signature cuối cùng. Checkpoint, tokenizer và heads phải cùng version với index.

Teacher có target y và có thể thấy nhiều prefix token hơn student. Đây là privileged supervision, không phải khả năng student nhìn target khi serving. Nếu utility phụ thuộc thông tin đã bị cắt khỏi query student, KL có thể không fit hết; phải log độ phủ input và kiểm tra ablation query budget, không coi mọi residual KL là lỗi optimizer.

### 4.4 Loss chính thực sự được backprop

\[
\boxed{\mathcal L_{\mathrm{task}}
=\operatorname{KL}(p_T\parallel p_S)}
\]

Trong implementation có thể dùng soft-label cross-entropy:

\[
\mathcal L_{\mathrm{task,CE}}
=-\sum_{i=0}^{K}p_T(i)\log p_S(i).
\]

Hai loss có cùng gradient vì khác nhau một hằng số entropy của teacher:

\[
\frac{\partial\mathcal L_{\mathrm{task,CE}}}{\partial z_i}
=p_S(i)-p_T(i).
\]

Ví dụ: teacher cho candidate A 0.6, student chỉ 0.2 → gradient logit A là −0.4; optimizer tăng score A. Không cần chosen/rejected record, hard relevance label hay RL reward.

**NLL của G tạo nhãn; KL/soft CE train S.** Không backprop qua G. Khi nhiệt độ teacher tiến về 0 và winner duy nhất, soft target tiến về one-hot winner; vì thế hard-label baseline là đối chứng trực tiếp cần có.

Không cần hệ số `tau²` trong công thức này. Nó là quy ước scale gradient hữu ích trong một số thiết lập logit KD, không phải điều kiện để loss được gọi là distillation.

### 4.5 Jina auxiliary: interface đúng và giả thuyết còn mở

Jina 1.5B dùng last-token pooling, embeddings mặc định 1536 chiều. Model card có task instructions; không có cơ sở dùng chuỗi `code2completion.query` như tên API. Lấy chuỗi từ `INSTRUCTION_CONFIG["code2completion"]["query"]` và `["passage"]` trong ví dụ chính thức, hoặc map sang wrapper đã được kiểm tra. [Model card, mục Usage](https://huggingface.co/jinaai/jina-code-embeddings-1.5b).

Hai literal tương ứng:

```text
query:   Find the most relevant completion given the following start of code snippet:\n
passage: Candidate completion:\n
```

Đây là khởi điểm thử nghiệm, **không phải xác nhận phù hợp nhất**: dependency/helper definition không phải đoạn continuation. Đối chiếu cấu hình `code2code` trong pilot trên cùng pool; task này cũng thiên về code tương đương, không bảo đảm dependency utility. Chốt instruction từ validation trước full run; lưu literal và hash, không chỉ tên task.

Trong interface sử dụng ở đây, Jina cung cấp embeddings; không thay cosine bằng target token likelihood của G. Đây là phân biệt interface/task, không phải khẳng định backbone Jina không có lịch sử pretraining sinh code.

\[
r_i^J=\cos(J(q),J(c_i)),\qquad
p_J=\operatorname{softmax}_{1:K}(r_i^J/\tau_J).
\]

\[
\bar p_S=\operatorname{softmax}_{1:K}(z_i),\qquad
\mathcal L_{\mathrm{sem}}=\operatorname{KL}(p_J\parallel\bar p_S).
\]

Tính `bar_pS` bằng log-softmax trực tiếp trên real logits, không chia `pS` cho `1-pS(0)` vì có thể mất ổn định số.

\[
\boxed{\mathcal L=\mathcal L_{\mathrm{task}}
+\lambda_J\mathcal L_{\mathrm{sem}}}
\]

Main pilot đầu: `lambda_J=0`. Sau khi task-only hoạt động, kiểm tra `lambda_J ∈ {0,0.05,0.1,0.2}` và `tau_J ∈ {0.02,0.05,0.1}` theo stage nhỏ, không full Cartesian search. Đây là một framework với auxiliary có hệ số validation-selected, không phải nhiều hướng nghiên cứu độc lập.

**Không có bảo đảm Jina không ảnh hưởng stop.** Conditional real-only KL có đạo hàm trực tiếp theo `z_0` bằng 0, nhưng đổi real logits và shared encoder, từ đó đổi khoảng cách candidate–null. Nếu nó làm xấu EM hoặc harmful-retrieval rate, chọn `lambda_J=0`; không giữ chỉ vì teacher lớn.

Model card đang ghi `cc-by-nc-4.0`; lưu license/version và xem xét điều khoản phù hợp trước sử dụng ngoài phạm vi nghiên cứu. [Jina model card](https://huggingface.co/jinaai/jina-code-embeddings-1.5b).

### 4.6 Online, offline và cache: không nhập nhằng thuật ngữ

Có hai trục khác nhau:

- **Teacher update:** freeze teacher hay joint/co-learning với teacher thay đổi.
- **Teacher execution:** score trước rồi cache, hay forward teacher on-the-fly mỗi batch.

Frozen pretrained teacher chạy mỗi batch vẫn thường thuộc offline KD theo cách phân loại teacher–student; “on-the-fly” nói về thời điểm tính score. Nó **không bắt buộc** dynamic retrieval, pairwise hoặc RL.

Chọn **frozen teachers + cached scores + fixed candidate IDs** làm run chính. On-the-fly cho cùng input, checkpoint, preprocessing, eval mode và candidate IDs cho cùng objective tới sai số số học; augmentation/truncation khác thì cache cũ không còn tương đương.

Được đổi `tau_G, tau_J, lambda_J` từ raw cached scores. Không được tái dùng cache nếu target IDs, prompt, teacher, candidate text hoặc rendering đã thay đổi.

## 5. Null gate đúng về toán và serving

### 5.1 Vì sao công thức cũ sai?

Giả sử K candidates đều có `g_i = -0.1`, `tau_G=0.5`, empty gain bằng 0:

\[
p_T(0)=\frac{1}{1+K e^{-0.2}}.
\]

- K=4: `p_T(0) ≈ 0.2339`.
- K=64: `p_T(0) ≈ 0.01873`.

Empty vẫn là **candidate riêng lẻ tốt nhất**, nhưng tổng real mass lớn vì có nhiều candidate. Vì vậy gate `p_S(0)>0.5` hoặc so với tổng real mass không tương ứng “mọi candidate đều không có ích”.

### 5.2 Policy thay thế

Lấy logit difference:

\[
d_i=z_i-z_0=\log\frac{p_S(i)}{p_S(0)}.
\]

Chọn candidates có `d_i > delta`; nếu không có thì dùng no-retrieval prompt. Ban đầu `delta=0`, sau đó chốt trên validation từ grid `{0,0.5,1.0,2.0}` trước test.

Nếu student fit teacher chính xác trên list:

\[
d_i=\log\frac{p_T(i)}{p_T(0)}=\frac{g_i}{\tau_G}.
\]

Vì vậy delta=0 tương ứng candidate thắng empty theo NLL proxy. **Không thêm loss pairwise**: đây là phép so score khi inference; training vẫn chỉ một soft vector trên list.

Thêm candidate yếu hơn threshold không đổi `d_i` của candidate cũ khi model và inputs cố định. Thêm một candidate tốt thật sự có thể và nên đổi quyết định. Bất biến này không bảo đảm chống ANN miss, distribution shift hoặc tăng false positives khi truy hồi kho lớn.

### 5.3 Serving path chốt

1. Cập nhật AST units cho repository snapshot.
2. Encode/index units bằng **student checkpoint cuối cùng**.
3. Encode visible query; union ANN top-20, BM25 top-20 và tối đa 20 exact dependency hits; deduplicate.
4. Tính student logits cho union và null; filter `d_i>delta`.
5. Sort real logits giảm dần, tie-break ID; pack dưới `B_C`. Loại overlapping source evidence, không cắt tùy tiện unit để nhét nốt budget.
6. Không còn unit hợp lệ → empty; còn → fixed-format retrieved context.
7. Gọi frozen G đúng một lần để sinh completion; không teacher scoring, Jina forward hay draft generation ở serving.

Gate này **không tránh chi phí query encoding/ANN** vì phải có candidate scores trước. Nó tránh đưa cross-file context không được chọn vào generator; không được quảng cáo là “skip mọi retrieval cost”.

### 5.4 Giới hạn single-candidate teacher và multi-candidate pack

Teacher chấm mỗi evidence unit riêng; serving pack nhiều units. Không thể từ đó suy ra utility của cả pack là tổng utility từng unit.

Pilot phải chạy **top-1 unit** trước để kiểm tra đường label–inference khớp nhau, rồi đo greedy pack trên cùng validation. Nếu pack tệ hơn, dùng top-1 cho cấu hình được chốt và báo rõ hạn chế. Dependency bundle chỉ giúp các quan hệ đã được gói sẵn; không giải quyết mọi multi-file synergy.

Không tạo pairwise preferences hoặc quét tổ hợp candidate để che hạn chế này.

## 6. Cache, training loop và chi phí

### 6.1 Cache schema cần triển khai

Schema sau là contract, không phải file dữ liệu đã được tạo:

```text
Example:
  task_id, repo_id, repo_commit, split
  visible_query_hash, retriever_input_hash, target_token_ids_hash
  target_span, scored_target_token_count, truncation_flag
  candidate_ids_in_order, candidate_set_hash, candidate_content_hashes

TeacherManifest:
  G_model_id + immutable_revision, G_tokenizer_revision
  J_model_id + immutable_revision, J_tokenizer_revision
  prompt_version, exact_input_ids_hashes, budget_config
  AST_builder_version, J_instruction_literals + hash
  precision, attention_backend, score_implementation_version

Scores:
  nll[0:K+1]       # null ở index 0; K+1 phần tử
  jina_cosine[0:K] # chỉ K real candidates
  candidate_valid_mask, token_counts, scoring_status
```

Lưu raw scores là source of truth. Softmax được dựng lại khi train; nếu cache thêm probabilities thì phải ghi temperature và assert tổng xác suất bằng 1 trên valid support. Padding class không được mang probability; K=0 → chỉ null, semantic loss bằng 0.

Không cache target text trong index phục vụ retrieval. Scorer nhận target qua interface riêng; query/index builder không được import hoặc truy cập trường đó.

### 6.2 Loop train

```text
freeze G, J; validate cached-score manifest
initialize UniXcoder, Wq, Wc, null head
for minibatch in fixed-candidate examples:
    encode current query and candidate text with current student weights
    logits = real_scores_plus_null
    log_pS = masked_log_softmax(logits)
    pT = stop_gradient(masked_softmax(-cached_nll / tau_G))
    task_ce = mean_over_queries(-sum_over_candidates(pT * log_pS))

    if lambda_J > 0:
        pJ = stop_gradient(masked_softmax(cached_jina_cosine / tau_J))
        log_real = masked_log_softmax(real_logits)
        sem_ce = mean_over_eligible_queries(-sum(pJ * log_real))
    else:
        sem_ce = 0

    loss = task_ce + lambda_J * sem_ce
    backward → clip_grad_norm(1.0) → optimizer.step → zero_grad
```

Với mask, tránh phép `0 * (-inf)` tạo NaN bằng cách chỉ sum valid entries. Average theo query, không vô tình chia thêm cho số candidate; nếu báo KL, trừ teacher entropy khỏi CE để log đúng tên metric.

Starting optimizer: AdamW, lr `2e-5`, weight decay `0.01`, warmup 5%, effective batch 16 queries, 3 epochs, clip 1.0. Microbatch theo GPU memory; accumulation phải giữ đúng query averaging. Đây là cấu hình pilot, không khẳng định hyperparameters tối ưu.

**Không dùng cached student embeddings qua optimizer steps.** Cả encoder và projections thay đổi nên vectors cũ bị stale. Có thể deduplicate candidate encodings trong cùng step. Rebuild index cho validation checkpoint và deployment; fixed pool chỉ cố định ID/text, không cố định embedding.

### 6.3 Ước lượng chi phí trước khi scale

Số sequence cần teacher scoring xấp xỉ `N × (K+1)`, không phải số batch GPU:

- Pilot 1.000 train examples, K=16: **17.000 sequences**, chưa tính validation.
- Run giả định 60.000 unique examples, K=64: **3.900.000 sequences**.
- Nếu padded sequence length trung bình 1.024 thì run thứ hai xử lý khoảng **3,99 tỷ token-position**; đây là phép tính budget, không phải benchmark throughput.

Không gọi đó là “rẻ” chỉ vì teacher forcing nhanh hơn autoregressive sampling. Đo sequences/s, GPU-hours, peak memory và token counts trên pilot rồi mới dự toán full run.

Riêng weights BF16 xấp xỉ 13,4 GB cho 6.7B và 3 GB cho 1.5B theo 2 byte/parameter, chưa gồm activations, logits, optimizer hoặc framework overhead. Chạy G và J tuần tự khi tạo cache; không cần cùng nằm trên GPU lúc student train.

Lượt audit này không tải model, không cài thư viện nặng, không chạy GPU training. Python trên Mac nếu cần phải dùng `/Users/kieugiangbien/bienkieu_env/bin/python`; GPU environment sẽ được pin riêng khi triển khai.

## 7. Kế hoạch triển khai theo dependency và acceptance test

Các module bên dưới **là cấu trúc đề xuất, chưa tồn tại**. Tạo package mới `src/tcd_kd/`, giữ `RepoClone/` làm reference, không sửa code baseline để tiện cho KD. Không khôi phục code/docs đã bị xóa ở các lượt trước.

### P0 — Khóa baseline và manifest

**Làm:** inventory data/checkpoints và phần AST-GR hiện có; pin repo/model/tokenizer revisions; chốt split, language, benchmark, prompt mode, scoring và decoding.

**Dự kiến:** `config.py`, `manifest.py`, `baseline_adapter.py`.

**Pass khi:**

- Có 20 example fixtures render được bằng native formatter lẫn controlled formatter.
- Ghi rõ left-only; target không nằm trong query.
- Repo identity và snapshot đủ để kiểm tra split leakage.
- Phân biệt được checkpoint released AlignRetriever, reproduction run và controlled hard-label run.
- Không chạy tải weights hoặc benchmark lớn khi chưa biết compute budget.

### P1 — AST dataset và evidence units

**Làm:** adapter vào AST-GR/chunker của người dùng, kiểm soát tokenizer budgets, tạo source spans và deterministic IDs.

**Dự kiến:** `data.py`, `ast_units.py`, `rendering.py`.

**Test bắt buộc:** incomplete prefix; Python/Java; Unicode; oversized function/leaf; nested classes; overlapping bundle; query có target field nhưng encoder không được đọc; target/suffix thay đổi không đổi visible-query features khi mask boundary cố định.

**Pass khi:** mọi unit hoặc có coverage/budget hợp lệ hoặc có reason code loại bỏ; không silent truncation; target-file exclusion và split checks pass.

Không giả định tree-sitter version đồng nhất: astchunk local và legacy parser loading của baseline dùng API khác nhau. Cô lập adapter/environment nếu cần, không nâng dependency baseline hàng loạt.

### P2 — Fixed pool builder

**Làm:** union bốn nguồn ở mục 3.3, deterministic dedup/RRF, lưu manifest. Freeze pretrained mining encoders trước student training.

**Dự kiến:** `candidate_pool.py`.

**Test:** ID/order tái lập; K thiếu; duplicate/nested span; pool không đọc y; tất cả ablations dùng đúng cùng pool khi đang đo loss.

**Pass khi:** báo được K thực tế, coverage có provenance, tỷ lệ nguồn đóng góp và cost. Nếu thiếu gold evidence annotations, ghi rõ coverage thủ công/symbol proxy, không đặt tên nó là ground-truth Recall@K.

### P3 — Frozen teacher scorers và pilot cache

**Làm:** G scorer với fixed current-file tokens; J scorer với instructions được kiểm tra; cache raw NLL/cosine.

**Dự kiến:** `teacher_scorer.py`, `semantic_scorer.py`, `cache.py`.

**Test bắt buộc:**

- NLL vectorized trùng tính tay trên tiny causal-LM fixture.
- First valid target token có loss; prompt/padding không có loss.
- Batched/unbatched score tương đương trong tolerance theo precision đã đo.
- Empty và real có cùng local/target token IDs.
- Candidate permutation chỉ hoán vị scores, không đổi score từng candidate.
- Cache stale do đổi tokenizer/prompt/chunk bị từ chối.
- Cached/on-the-fly cùng fixture cho cùng phân phối tới tolerance số học.

**Pilot:** 1.000 train + khoảng 200 validation examples từ repo disjoint; số lượng validation chỉ để debug, không đủ tự động xác nhận superiority.

**Go/no-go:** chạy completion diagnostic với teacher-ranked candidate và empty. Nếu NLL-selected context không có dấu hiệu cải thiện EM/ES, điều tra scorer/prompt/proxy trước full training. Nó không phải “oracle upper bound” của EM.

### P4 — Student task-KD trước, semantic KD sau

**Làm:** shared UniXcoder, identity heads, scaled null head, masked soft CE; chưa bật Jina auxiliary ở run đầu.

**Dự kiến:** `student.py`, `losses.py`, `train.py`.

**Test:** finite-difference gradient; không gradient vào teachers; gradient vào cả query/candidate/student heads; K=0/K=1/padded list; finite loss khi teacher rất nhọn; permutation invariance; batch duplication không đổi mean loss.

**Pass khi:** overfit được một tiny fixed dataset, loss giảm, score ordering có thay đổi đúng chiều; save/load checkpoint giữ cùng logits và null score. Training tiny chưa chứng minh generalization.

Sau đó thử Jina auxiliary trên cùng examples/pool; log task CE/KL, semantic CE/KL, entropy, gradient norms, candidate–null margins và completion validation. Chọn lambda=0 nếu auxiliary không có lợi.

### P5 — Retrieval, null gate và end-to-end evaluation

**Dự kiến:** `index.py`, `retrieve.py`, `packing.py`, `evaluate.py`.

**Làm:** index đúng checkpoint, union serving candidates, exact score/null filter, top-1 diagnostic rồi budgeted pack.

**Test:** ANN so exact scan trên repo nhỏ; append distractors không đổi margin cũ; invalid/overlap candidates không vào pack; pack không vượt token budget; cache/index sai checkpoint bị từ chối; serving trace không có Jina/teacher scoring/draft generation.

**Pass khi:** chạy được toàn pipeline trên held-out repositories và lưu per-task prediction/metric/latency. Không đánh giá bằng cách đưa sẵn teacher list rồi gọi đó là repository retrieval.

### P6 — Controlled experiments và quyết định scale/paper

Chạy ma trận ở mục 8, chọn hyperparameters bằng validation, khóa cấu hình trước final test.

**Dự kiến:** `experiment_runner.py`, `analysis.py`; run manifests và results có schema thống nhất.

**Chưa triển khai CLI:** hiện không có lệnh `python -m tcd_kd ...` chạy được. Khi code được viết, CLI phải có các stage độc lập build-units → build-pool → score-teachers → train → build-index → evaluate, mỗi stage kiểm manifest đầu vào.

Không cần thêm một vòng “update retriever → tạo pairwise → train lại”. Index rebuild là yêu cầu vectors khớp checkpoint, không phải tạo preference labels.

## 8. Thiết kế thí nghiệm để xác nhận hoặc bác bỏ

### 8.1 Hai track so sánh không được trộn

**Track A — Reproduction:** AlignCoder/RLCoder native code, released checkpoint hoặc retraining được ghi rõ, native query enhancement và formatter. Dùng cùng final G, benchmark evaluator và decoding. Báo sai lệch so với số paper trước khi kết luận.

**Track B — Controlled attribution:** cùng train examples, teacher size, candidate pool, allowed information, fixed formatter, student architecture/budget và evaluation. Chỉ thay yếu tố đang nghiên cứu.

Lý do cần cả hai: dùng G6.7B làm task teacher có thể thắng chỉ vì teacher lớn hơn evaluator 1.3B của baseline; dùng AST mới có thể thắng chỉ vì chunking. Không được gộp hai hiệu ứng đó thành “KD tốt hơn”.

### 8.2 Bộ đối chứng tối thiểu

1. Frozen UniXcoder + line chunks, rồi frozen UniXcoder + AST chunks.
2. AlignCoder reproduction và RLCoder reproduction ở Track A.
3. AST + **hard-winner CE** từ đúng G6.7B và fixed candidate pool của TCD-KD.
4. AST + task-only soft KD, cùng mọi điều kiện với (3).
5. AST + task KD + Jina auxiliary, cùng candidate IDs với (4).
6. REPLUG-style reverse-KL control: dùng cùng AST/pool/null/student và cùng `p_T` từ mean NLL, nhưng train `KL(p_S || p_T)`. Đây là adaptation để cô lập chiều KL, không phải reproduction nguyên bản: original paper khác score transform và có refresh index. Báo riêng ablation bỏ null nếu muốn đo đóng góp null; không gộp nó với đổi chiều KL.
7. No retrieval; top-1 so với packed-context; có/không null gate.

Jina-only là diagnostic real-candidate ranking, không tự có task-utility stop label. Khi so nó, báo policy always-retrieve hoặc calibrator riêng; không dùng untrained null head rồi kết luận semantic teacher kém.

Sau khi core chạy tốt mới thêm: G1.3B/G6.7B teacher-size control, weighted/unweighted NLL, line/AST GR sampling, full AST serialization, bỏ Jina mining, transfer final generator. Không cần dynamic re-mining ablation trong vòng đầu vì đi ngược yêu cầu đơn giản hóa.

Nếu không muốn train thêm hard-winner baseline, có thể chạy checkpoint reproduction để so end-to-end, nhưng phải thừa nhận chưa cô lập được giá trị riêng của soft KD.

### 8.3 Metrics và cách hiểu đúng

**Primary:** EM theo evaluator benchmark. Báo từng subset CrossCodeEval Python/Java và RepoEval line/API; không chỉ cherry-pick một tập. ES là secondary. Identifier metric hoặc execution/pass@1 chỉ dùng khi có protocol/data hỗ trợ.

**Retrieval diagnostics:** gold Recall/MRR khi có annotation; nếu dùng symbol heuristic phải ghi proxy. Đo cả fixed-pool coverage, toàn-repo retrieval và ANN-vs-exact recall.

**Teacher/gate diagnostics:**

- NLL gain và empirical generation delta trên validation.
- Tỷ lệ gate chọn retrieval nhưng completion kém hơn empty; tỷ lệ gate bỏ context thực sự giúp.
- Candidate–null margin distribution; không vẽ reliability của p(empty) rồi gọi đó là no-retrieval probability.
- Teacher entropy phụ thuộc K và temperature; không so entropy thô giữa hai K như cùng uncertainty.
- Teacher–student KL giảm không đồng nghĩa EM tăng.

**Engineering:** end-to-end P50/P95 và breakdown mining/ANN/packing/generation; số model forwards; index footprint; offline teacher GPU-hours; cùng hardware/batch/concurrency. Một final generation không tự đảm bảo latency thấp hơn baseline mọi setting.

### 8.4 Acceptance và thống kê

Chốt trước primary dataset/aggregate, seed list, delta và selection rule. Tối thiểu ba student seeds nếu compute cho phép; báo biến động giữa seeds tách khỏi uncertainty do test examples.

So paired predictions trên cùng test set; bootstrap theo **repository cluster**, không giả định các completions trong cùng repo độc lập. Báo absolute percentage-point delta, confidence interval và số repos/tasks.

Chỉ kết luận “vượt AlignCoder trong setting X” khi EM tốt hơn với uncertainty được báo, ES không có suy giảm vượt tolerance định trước, và không che regression lớn ở subset quan trọng. Nếu cần nhiều primary comparisons, nêu multiple-testing plan trước; không chọn subset thắng sau khi nhìn test.

NLL-based teacher reranking **không phải upper bound EM/ES**. Oracle chọn candidate theo actual completion correctness chỉ là diagnostic trên tập candidate hữu hạn và cần generation/evaluation riêng; không được dùng ở serving hoặc tuning test.

### 8.5 Novelty gate trước khi viết contribution

Cần trả lời bằng số liệu:

- Soft KD hơn hard CE cùng teacher/AST/pool không?
- Có lợi ích ngoài AST + REPLUG-style adaptation không?
- Null gate giảm harmful retrieval trên unseen repositories không?
- Jina thêm ích lợi ngoài việc nó đã tham gia tạo pool không?
- Bỏ draft sampling giữ/tăng EM trong khi giảm chi phí end-to-end không?

Nếu chỉ AST giúp còn soft KD/Jina không giúp, báo đúng kết quả; không claim loss mới. Nếu chỉ teacher size tạo gain, đó là compute/teacher effect. Nếu không hơn REPLUG-style adaptation, hướng này có thể hữu ích engineering nhưng chưa đủ luận điểm phương pháp cho paper.

## 9. Cây lập luận và evidence ledger

### 9.1 Central claim và argument tree

**Claim C0:** distilling generation utility trên AST evidence units là một hướng khả thi để thử cải thiện repo completion bằng student rẻ, không cần pairwise pipeline.

- **Reason R1 — supports C0:** teacher có thể tạo soft target từ target likelihood.
  - **Evidence E1 — supports R1:** local baseline đã chấm target để train retriever; REPLUG có LM-supervised distribution learning.
- **Reason R2 — supports C0:** AST units có thể giảm lỗi cắt ngang cấu trúc.
  - **Evidence E2 — supports R2:** cAST và implementation structural splitting.
  - **Limitation L2 — qualifies R2:** syntax boundary không bảo đảm đủ semantic dependency hoặc vừa student budget.
- **Reason R3 — supports C0:** cached scores bỏ việc phải chấm lại pool sau mỗi retriever update.
  - **Assumption A3 — qualifies R3:** pool coverage và train–serving distribution shift vẫn chấp nhận được.
- **Counterargument X1 — qualifies C0:** NLL proxy có thể không tương quan EM; teacher lớn có thể là nguồn gain thật.
- **Counterargument X2 — contradicts claim “loss mới”:** REPLUG-LSR đã có cơ chế frozen-LM likelihood → distribution → retriever KL.
- **Unverified U1 — qualifies C0:** chưa có kết quả cho superiority hoặc novelty của tổ hợp này.

### 9.2 Evidence ledger

**E1 — Local baselines có hard-winner supervision**

- Claim: update được audit dùng hard CE, không phải soft candidate labels.
- Evidence: argmin/argmax index và CrossEntropyLoss.
- Location: [AlignCoder main.py:446](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/AlignCoder/main.py:446), [RLCoder main.py:356](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/RLCoder/main.py:356).
- Relationship: supports chọn hard-label control; contradicts mô tả “KL thay một optimizer hoàn toàn khác”.
- Confidence: **Source fact or data**.
- Caveat: giới hạn ở snapshot/nhánh code đã đọc.

**E2 — LM-supervised retriever KL đã có prior art**

- Claim: loss family cốt lõi không mới chỉ vì đổi teacher sang code LM.
- Evidence: frozen LM scoring, distribution matching và KL của REPLUG.
- Location: [REPLUG, mục 4](https://arxiv.org/html/2301.12652v3).
- Relationship: qualifies novelty của TCD-KD.
- Confidence: **Reasoned inference**.
- Caveat: chưa phải tìm kiếm prior art toàn diện; khác chiều KL và score transform phải ghi rõ.

**E3 — AST chunking có cơ sở nhưng không phải bảo đảm**

- Claim: structural splitting là cơ chế tác giả cAST đề xuất để cải thiện code RAG.
- Evidence: recursive split/merge và thí nghiệm được paper báo cáo.
- Location: [cAST, abstract và method](https://arxiv.org/abs/2506.15655).
- Relationship: supports thử AST units.
- Confidence: **Author's stated position**.
- Caveat: không suy ra mọi chunk của implementation hiện tại đều self-contained.

**E4 — Empty softmax mass không phải absolute abstention probability**

- Claim: p(empty) có thể nhỏ dù mọi real candidate có NLL tệ hơn.
- Evidence: counterexample mục 5.1 và kiểm tra số mục 10.
- Location: công thức softmax của report, mục 4.2 và 5.1.
- Relationship: contradicts stop rule bản cũ.
- Confidence: **Reasoned inference**.
- Caveat: logit-gap gate sửa lỗi denominator, chưa đảm bảo calibration/correctness.

**E5 — Jina similarity không phải target utility**

- Claim: semantic KD có thể lệch mục tiêu completion.
- Evidence: score ở interface này là embedding cosine, không condition vào target y.
- Location: [Jina Usage](https://huggingface.co/jinaai/jina-code-embeddings-1.5b), công thức mục 4.5.
- Relationship: qualifies việc dùng semantic auxiliary.
- Confidence: **Reasoned inference**.
- Caveat: cần validation; chưa chứng minh Jina có lợi hay hại trong task này.

**E6 — Semantic auxiliary có thể ảnh hưởng null gate**

- Claim: bỏ null khỏi semantic softmax không cô lập hoàn toàn stop behavior.
- Evidence: real logits/shared encoder thay đổi, d_i=z_i−z_0 thay đổi.
- Location: mục 4.5; finite-difference check mục 10.
- Relationship: contradicts bảo đảm “semantic teacher không thể override utility”.
- Confidence: **Reasoned inference**.
- Caveat: mức ảnh hưởng thực tế tùy lambda và dữ liệu.

**E7 — Formatter local tạo một confound cho NLL gain**

- Claim: thêm context có thể đồng thời cắt bớt prefix.
- Evidence: allowed_prompt_length phụ thuộc crossfile length.
- Location: [generator.py:70](/Users/kieugiangbien/Downloads/Project/CodeCompletion/RepoClone/AlignCoder/generator.py:70).
- Relationship: supports fixed-prefix controlled scorer.
- Confidence: **Source fact or data**.
- Caveat: native reproduction vẫn phải giữ behavior gốc và báo riêng.

**E8 — TCD-KD vượt AlignCoder hoặc đủ novelty cho paper**

- Claim: framework cải thiện end-to-end và có đóng góp riêng.
- Evidence: **not supplied**; chưa chạy thí nghiệm.
- Location: giả thuyết mục 1, protocol mục 8.
- Relationship: qualifies toàn bộ conclusion.
- Confidence: **Unverified**.
- Caveat: không được thay kết quả bằng trực giác “teacher lớn hơn” hoặc “KD mượt hơn”.

## 10. Kiểm chứng đã thực hiện trong lượt audit

Đã chạy các phép tính số độc lập bằng JavaScript trên toy logits, không dùng model weights:

1. **Softmax shift invariance:** gain-softmax và negative-NLL-softmax sai khác tối đa 0 trong fixture.
2. **Null denominator counterexample:** K=4 → 0.233922; K=64 → 0.018727, dù mọi real gain đều −0.1.
3. **Gradient soft CE:** finite differences khớp `p_S−p_T`, max error khoảng `1.82e-11`.
4. **Một bước update đúng chiều:** toy CE giảm từ 1.350342 xuống 1.349914 với learning rate 0.1.
5. **Candidate–null odds:** giữ nguyên 1.648721 khi thêm 60 distractors mà logits cũ không đổi.
6. **Semantic conditional loss:** derivative trực tiếp theo null logit bằng 0 nhưng derivative theo real logits khác 0; không có bảo đảm gap giữ nguyên.

Các test này chỉ xác nhận toán học của objective/gate. **Chưa chạy:** PyTorch implementation tests, AST-GR của người dùng, teacher cache thật, training, memory benchmark hoặc EM/ES evaluation. Các việc đó nằm trong P0–P6.

## 11. Tóm tắt dùng để triển khai và tự kiểm tra

Hướng duy nhất: **AST data/candidates → frozen-G utility labels → UniXcoder candidate-distribution KD → student retrieval và candidate–null gate → frozen-G completion**. Jina auxiliary được giữ khi validation chứng minh có lợi. Cache/on-the-fly là lựa chọn tính toán; main run dùng cache, không pairwise.

Ba câu hỏi phải trả lời rõ trước khi code full run:

1. Loss backprop lấy từ đâu? — Từ soft CE/KL giữa phân phối candidate của G và S; G NLL chỉ tạo soft label.
2. Tại sao không threshold p(empty)? — Vì denominator thay đổi theo candidate list; dùng logit gap, vẫn phải validate downstream.
3. Điều gì còn thiếu để claim paper? — Controlled improvement ngoài AST, hard-winner teacher supervision và REPLUG-style adaptation; hiện chưa có số liệu đó.

**Bước tiếp theo cụ thể:** P0–P3 trước, tạo pilot 1.000 examples + validation và xác minh teacher signal. Chỉ mở full student training khi data, prompt và cache contract pass; chỉ chốt superiority/novelty sau P6.
