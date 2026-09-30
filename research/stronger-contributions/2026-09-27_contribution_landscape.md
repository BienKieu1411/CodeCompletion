# Hướng contribution mạnh hơn cho retrieval trong repository-level code completion

> **Đã supersede (2026-09-28):** bản slate-RL dùng target-likelihood và PAISR/support-label đều không còn là hướng được khuyến nghị. Hướng hiện tại là online outcome-RL, xem [report](2026-09-28_online_ast_gr_outcome_rl.md) và [research plan](online_ast_gr_rl_plan.md). File này chỉ giữ lịch sử khảo sát prior art.

**Ngày research:** 2026-09-27  
**Câu hỏi:** Có contribution nào thuyết phục hơn việc chỉ thêm quantum-inspired scoring, nhưng vẫn nhắm trực tiếp đến việc vượt AlignCoder?  
**Ràng buộc đã giữ:** Không knowledge distillation; không tạo positive/rejected pair; không pairwise/listwise ranking loss; không sửa notebook hoặc implementation.

## Kết luận trước

Có một hướng đáng thử hơn quantum scorer đơn lẻ:

> **AST-GR-constrained slate policy:** học cách chọn *cả tập và thứ tự* các AST evidence unit cho prompt completion, có giới hạn token và action STOP; tối ưu bằng reward là mức tăng gold-continuation log-likelihood của **toàn prompt đã ghép** so với prompt không retrieval.

Đây không phải một lời hứa sẽ vượt AlignCoder. Các thành phần riêng lẻ đã có prior art: [RLCoder](https://arxiv.org/html/2407.19487)/[AlignCoder](https://arxiv.org/html/2601.19697) dùng target likelihood để train retriever; [RepoShapley](https://aclanthology.org/2026.findings-acl.505/) đã chiếm broad claim về interaction và coalition utility; [AIRCoder](https://aclanthology.org/2026.acl-long.1166/) đã chiếm AST chunks cùng hybrid/adaptive fusion; [REVELA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0343104ddbfc48f35f06aaae88980e48-Abstract-Conference.html) đã dùng next-token loss để học retriever ngoài bài toán completion. Khoảng trống hẹp có thể bảo vệ là **AST-GR-relation-constrained action space + trực tiếp tối ưu slate dùng ở inference bằng scalar completion reward**, không Shapley-label pipeline, không KD và không pairwise data. Novelty vẫn ở mức **medium–high risk** cho tới khi ablation chứng minh AST-GR constraints thật sự thêm giá trị.

Nếu dự án có dữ liệu lịch sử nhiều commit hoặc index thường stale, hướng version/freshness-aware có thể mới hơn; nhưng nó lệch khỏi mục tiêu thắng trên benchmark snapshot tĩnh.

**Extension có tiềm năng tăng contribution:** sau khi single-generator pilot có tín hiệu, train policy bằng scalar reward trung bình từ hai frozen code LMs rồi đánh giá thêm một generator holdout. AlignCoder đã báo cáo kết quả trên nhiều backbone và Repoformer nói có thể làm việc với nhiều generator/retriever, nên không được claim “cross-model generalization” chung là mới. Khoảng trống cần kiểm tra là **robust training reward across generators**, không chỉ multi-backbone evaluation. Nó tăng khoảng gấp đôi evaluator cost và vẫn cần search sâu hơn trước khi gọi novelty.

**Khuyến nghị thực dụng:** dùng single-generator chỉ làm feasibility pilot. Nếu contribution cần mạnh hơn để viết paper, bản cần kiểm chứng là **AST-GR slate policy với multi-generator scalar reward**; đây mới là phần có thể nhắm vào evaluator coupling. Nếu compute không đủ cho hai frozen evaluators và một held-out generator, giữ quantum như ablation và không thổi phồng claim single-generator.

## Vì sao các contribution cũ còn yếu

| Ý tưởng đứng riêng | Tình trạng prior art | Kết luận |
|---|---|---|
| Dùng AST chunking | [cAST/CAST](https://aclanthology.org/2025.findings-emnlp.430/), [GRACE](https://arxiv.org/abs/2509.05980) và AIRCoder đã dùng cấu trúc AST/graph; [nghiên cứu chunking 2026](https://arxiv.org/abs/2605.04763) cảnh báo context length có thể quan trọng hơn ranh giới chunk | Là nền tảng dữ liệu tốt, không đủ làm contribution chính |
| Hybrid lexical+dense+AST hoặc query-conditioned fusion | AIRCoder đã kết hợp nhiều chiều và học trọng số theo query; loss của họ là pairwise MSE, không dùng ở phương pháp đề xuất | Đổi loss đơn thuần thành BCE/RL chưa tạo novelty |
| Train retriever bằng target PPL/log-likelihood | RLCoder và AlignCoder đã làm, dùng code LM frozen đánh giá target và thưởng candidate tốt nhất | Không thể claim “task-aligned retrieval bằng generator reward” là mới |
| Chọn context theo utility/tương tác | RepoShapley đo marginal utility, coalition và interference; xác minh tập context bằng generator rồi distill KEEP/DROP | “Joint context selection” hoặc “chunk interactions” chung chung đã bị chiếm |
| Retrieval thích nghi khi generation | ACToR dùng critical-token triggers; RepoCoder/AlignCoder đã có query generation/refinement | Cần cơ chế hẹp hơn “adaptive retrieval” |
| Cross-backbone robustness | AlignCoder đánh giá năm backbone; Repoformer nói có thể dùng nhiều generation models/retrievers; OpenCoder báo source interaction phụ thuộc backend | “Works across generators” không mới. Có thể thử objective huấn luyện bằng scalar reward trung bình nhiều frozen G, nhưng cost cao và novelty chưa cleared |
| Quantum phase/fidelity scorer | Có prior art ở text/general IR và language modeling nhưng chưa có bằng chứng trực tiếp rằng nó cải thiện repository code completion | Giữ làm ablation; phải so với control real-valued có cùng tham số |

Các phân biệt quan trọng được trích trong [F01 — prior-art map](findings/F01_occupied_space.md). AIRCoder là ACL 2026 peer-reviewed; RepoShapley là Findings of ACL 2026. ACToR và OpenCoder còn là preprint trong nguồn đã kiểm tra nên chỉ dùng như prior-art warning. Thông tin citation/source status nằm trong [sources.csv](sources.csv).

## Contribution nên đem đi thử

### Câu hỏi nghiên cứu

Với cùng candidate pool, cùng generator và cùng cross-file token budget, một policy chọn **slate AST-GR có cấu trúc** có cải thiện completion hơn independent top-K, AlignCoder và RepoShapley không?

### Cơ chế

1. Tạo candidate pool cố định từ BM25/identifier retrieval, dense UniXcoder retrieval và AST-GR neighborhood expansion. Deduplicate theo source span/node ID. Candidate generation chỉ được dùng prefix/repository nhìn thấy tại inference.
2. Policy bắt đầu từ UniXcoder-base, cộng một head nhỏ. Ở mỗi bước nó chọn AST evidence unit hoặc action mở rộng relation-derived support evidence; action STOP luôn có. Tổng slate không vượt hard token budget.
3. Các action theo thứ tự tạo thành prompt đúng như deployment. Không giả định từng chunk có ích độc lập.
4. Trong training, frozen code generator teacher-force gold continuation trên prompt không retrieval và từng sampled slate. Target chỉ dùng để tính scalar reward, không đi vào query, candidate pool, feature hay inference.
5. Tối ưu policy trực tiếp bằng policy gradient. Inference chỉ chạy retriever/policy đã train rồi gọi final generator; evaluator không chạy lúc phục vụ.

### Model nào làm gì

| Thành phần | Model khởi đầu | Có update? | Trách nhiệm |
|---|---|---:|---|
| Retriever/policy | **microsoft/unixcoder-base** + compact slate/action head | Có: policy head; encoder freeze lúc pilot | Đề xuất/chọn AST evidence theo thứ tự, tôn trọng token budget và STOP |
| Reward evaluator | **deepseek-ai/deepseek-coder-1.3b-base** | Không | Teacher-force gold continuation và trả scalar log-likelihood lúc training |
| Final generator | **Cùng deepseek-ai/deepseek-coder-1.3b-base trong primary pilot**, để reward khớp với model được đo; sau đó chạy backbone lớn hơn làm secondary generalization test | Không | Sinh completion cuối lúc evaluation/inference |

Notebook hiện tại dùng deepseek-ai/deepseek-coder-1.3b-base làm reward evaluator; đây là lựa chọn cụ thể cho pilot. AlignCoder/RLCoder gọi evaluator là DeepSeek-Coder-1B. Trước khi tuyên bố reproduction, cần xác minh checkpoint/revision chính xác trong artifact của paper; trong mọi so sánh của dự án phải dùng cùng exact checkpoint, tokenizer, prompt format và context budget cho reward lẫn final generator.

Ở stretch run để tăng contribution: giữ DeepSeek-Coder-1.3B-base và thêm **bigcode/starcoder2-7b** làm frozen reward evaluator thứ hai; dùng **CodeLlama-7B** làm một generator holdout nếu resource cho phép. Đánh giá AlignCoder và các baseline trên đúng các backbone đó. Nếu không đủ GPU để chấm likelihood trên hai G, chỉ báo cáo feasibility pilot và không claim robust reward.

### Loss chính — viết rõ từng đại lượng

Gọi q là visible prefix, y là gold continuation, S là slate đã chọn, G là frozen generator. Reward:

~~~text
R(S) = [ log P_G(y | q,S) - log P_G(y | q,∅) ] / |y|
        - λ · token_cost(S)
~~~

Primary comparison nên áp hard token cap giống nhau nên token-cost penalty có thể tắt ở main run; giữ lại làm ablation. Reward cao khi slate giúp **chính generator** dự đoán gold continuation tốt hơn baseline không retrieval.

Policy loss:

~~~text
L_policy(θ) =
  - E_{S ~ πθ}[ (R(S) - b(q)) · Σ_t log πθ(a_t | q, a_<t) ]
  - β H(πθ)
~~~

b(q) là baseline/value estimate giảm variance; entropy term nhỏ giữ exploration. Nếu có critic, critic MSE chỉ là loss phụ. **Loss chính cập nhật retriever/policy là policy-gradient loss trên; log-likelihood của G là reward, không backprop vào G.**

Điều này không phải distillation: không matching token distributions, không KL, không hidden states hay action labels từ teacher. Cũng không có pair records hoặc chosen/rejected comparisons: mỗi sampled slate nhận **một scalar outcome**. Đây là on-policy RL cho retriever/policy, không phải online distillation.

## Pipeline train và serve

### Training

1. **Audit data/AST-GR.** Kiểm tra node/edge types, nhãn, parser/lang support, source spans và split leakage. Trong workspace tôi chỉ xác nhận được notebook quantum; chưa tìm thấy AST-GR schema/implementation tách riêng để audit. Vì vậy không giả định loại edge/label cụ thể trước bước này.
2. **Khóa split theo repository.** Gold continuation chỉ dùng để tính training reward. Loại target span và mọi overlapping target-file fragment khỏi candidate pool theo benchmark protocol.
3. **Đóng băng proposal pool cho pilot.** Dùng cùng BM25 + UniXcoder + AST-GR candidates cho tất cả phương pháp; freeze UniXcoder encoder và cache candidate embeddings trong feasibility pilot; train policy head trước. Báo candidate-pool recall riêng để cô lập selector khỏi candidate coverage.
4. **Sample complete slates.** Current policy lấy 1–m slates mỗi query, tuần tự chọn evidence và STOP theo token cap. Bắt đầu ít rollouts để đo cost; cache exact prompt/reward trùng nhau. Chỉ khi pilot có signal mới unfreeze encoder và re-index theo lịch cố định trong Stage B.
5. **Tính scalar reward.** Frozen evaluator teacher-force y cho từng slate và no-context baseline. Không sinh sample completion, không tạo pairwise dataset.
6. **Update policy.** Dùng REINFORCE với baseline; giữ lại checkpoint dựa trên dev repositories, không đụng test.
7. **Evaluation.** Dùng retriever/policy frozen để lấy context; evaluator không chạy trong inference; final generator thực hiện code completion.

### Inference

~~~text
visible prefix → fixed proposal pool → AST-GR-aware slate policy → dedup + hard token cap → frozen code generator
~~~

Không gọi reward model, không chạy target PPL, không gọi teacher ở inference. Có thể giữ lexical/dense retrieval làm proposal generation, nhưng phải dùng proposal pool chung khi so selector.

## Thí nghiệm tối thiểu để biết hướng này có đáng theo không

### Baselines và controls

- No retrieval; BM25; UniXcoder cosine; RLCoder/AlignCoder; RepoShapley; AIRCoder nếu có reproduction và điều kiện so sánh phù hợp.
- Cùng fixed proposal pool: (a) independent top-K, (b) unconstrained slate policy, (c) AST-GR-constrained slate policy. Ba hàng này cô lập đóng góp AST structure.
- Quantum scorer là ablation riêng trong cùng candidate pool, không thay baseline mạnh.
- Giữ generator checkpoint/prompt/decoding giống nhau; khóa số token cross-file, số generator calls khi train/eval, dedup policy và local-prefix budget.

### Metrics

- **Primary:** Exact Match (EM) của completion trên untouched test, với paired bootstrap theo task/repository; báo confidence interval.
- **Secondary:** Edit Similarity (ES), identifier/API match hoặc syntax/test pass trên subset có oracle phù hợp.
- **Mechanism:** candidate-pool recall; evidence recall@K; reward delta; tỉ lệ AST-GR edge bundles được chọn; utility theo số token.
- **Cost:** train evaluator forward count/compute, retriever latency, final prompt tokens. Retrieval recall hay NLL riêng không thay được completion result.

### Stretch: reward bền vững qua nhiều generator

Chỉ chạy sau pilot đơn generator. Với frozen generators G₁...Gₘ, định nghĩa:

~~~text
Δ_m(S) = [log P_Gm(y | q,S) - log P_Gm(y | q,∅)] / T_m(y)
R_multi(S) = (1/M) · Σ_m Δ_m(S)
~~~

Đây vẫn là **một scalar reward cho mỗi slate**, không distill output distribution và không so sánh candidate theo pairwise loss. Dùng hai G khi train reward, đánh giá trên cả hai và thêm ít nhất một G chưa dùng khi training. So sánh với cùng slate policy chỉ train bằng một G. Nếu không tăng average/holdout completion hoặc lợi ích chỉ chuyển từ generator này sang generator khác, bỏ extension. Chi phí likelihood evaluation xấp xỉ nhân theo số evaluator; giữ no-context scores cache theo query/model.

### Go/no-go

**Tiếp tục** nếu (i) policy AST-GR cải thiện slate reward/coverage ở held-out repositories, (ii) end-to-end EM/ES tăng so với AlignCoder và unconstrained policy với paired CI không chứa 0 hoặc có kiểm định phù hợp, và (iii) tăng không đến từ context dài hơn hay nhiều generator calls hơn.

**Dừng hoặc hạ claim** nếu chỉ retrieval recall/NLL tăng mà EM không tăng; nếu AST-GR policy không vượt unconstrained slate policy; hoặc nếu hiệu quả chỉ có trên một generator/evaluator. Khi đó không gọi đây là framework vượt AlignCoder.

## Các hướng khác, xếp hạng

| Hướng | Novelty tương đối | Hợp mục tiêu thắng AlignCoder tĩnh? | Lý do |
|---|---:|---:|---|
| **AST-GR slate policy, reward toàn prompt** | Trung bình; rủi ro overlap cao | **Cao nhất trong các hướng đã kiểm tra** | Giữ benchmark/mục tiêu, nhưng phải chứng minh AST-constrained actions thêm giá trị so với RepoShapley và RLCoder |
| **Version/freshness-aware evidence** | Cao hơn nếu làm được commit-paired mitigation | Thấp–trung bình | Hợp khi index stale/multi-version là deployment thật; lệch benchmark snapshot |
| **AST-adapted NTP retriever theo REVELA** | Chưa rõ | Trung bình | REVELA đã có NTP retriever, nhưng không completion-specific; chi phí joint training/inference cao và in-batch cross-document interactions có thể bị xem là pairwise |
| **Quantum phase/fidelity** | Chưa được chứng minh trong code completion | Trung bình/thấp | Đáng làm ablation; phải thắng cosine, bilinear/MLP và scorer parameter-matched |

## Adversarial review

1. **“Đây có phải RepoShapley đổi thuật toán?”** Có overlap thật về set utility. Trả lời chỉ bằng “không distill” là yếu; cần chứng minh typed AST-GR action constraints tạo gain mới so với unconstrained slate selector và RepoShapley.
2. **“Target-PPL/RL có mới không?”** Không. AlignCoder và RLCoder đã dùng target likelihood để train retriever. Claim chỉ có thể là reward trên complete ordered prompt + AST relation action space, không phải RL hay PPL.
3. **“AST-GR có chắc là available và không leak?”** Chưa xác minh trong workspace. Stage 0 bắt buộc audit artifact/schema trước khi khẳng định.
4. **“Likelihood tăng có chứng minh code đúng hơn?”** Không. Nó là training reward. Kết luận phải dựa trên exact match/edit similarity và test-backed outcomes, kèm nhiều generator khi đủ tài nguyên.

## Cách viết contribution nếu pilot qua

> We formulate repository-level completion retrieval as budgeted selection of AST-relation-constrained evidence slates. Unlike candidate-independent retriever rewards or offline Shapley-controller distillation, our policy is optimized from the frozen completion model’s scalar likelihood improvement on the assembled prompt. We show through matched-pool and matched-budget ablations whether AST-GR structure contributes to end-to-end completion.

Không dùng “first”, “novel quantum advantage”, hoặc “surpasses AlignCoder” trước khi có search/citation audit và held-out test evidence.

## Nguồn trọng yếu

- AlignCoder: [paper](https://arxiv.org/html/2601.19697), [local DeepRead](../../doc/papers/ALIGNCODER_DEEPREAD.md).
- RLCoder: [paper](https://arxiv.org/html/2407.19487), [local DeepRead](../../doc/papers/RLCODER_DEEPREAD.md).
- RepoShapley: [Findings of ACL 2026 paper](https://aclanthology.org/2026.findings-acl.505/).
- AIRCoder: [ACL 2026 paper](https://aclanthology.org/2026.acl-long.1166/).
- REVELA: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0343104ddbfc48f35f06aaae88980e48-Abstract-Conference.html), [local DeepRead](../../doc/papers/REVELA_DEEPREAD.md).
- ACToR: [September 2026 preprint](https://arxiv.org/abs/2609.01601).
- Freshness diagnostic: [arXiv:2605.14478](https://arxiv.org/abs/2605.14478).
- Adjacent RL evidence selection: [Context-Picker](https://arxiv.org/abs/2512.14465).
- Selective retrieval across models: [Repoformer](https://arxiv.org/abs/2403.10059).

Per-paper notes and confidence/limits are recorded in [sources/](sources/) and [sources.csv](sources.csv). See [F02](findings/F02_ast_slate_policy.md) for the exact method and [refresh_targets.md](refresh_targets.md) before a submission or expensive run.
