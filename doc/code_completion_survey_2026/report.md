# Repository-Level Code Completion: literature survey through 2026

**Cutoff: 24 October 2026.** Focus: cross-file retrieval for left-context-only fill-in-the-middle completion, with particular attention to what is genuinely new in 2026 and what transfers to a UniXcoder retriever plus DeepSeek-Coder-1.3B generator. This is a bounded, source-tracked survey, not an exhaustive systematic review.

## Kết luận ngắn

Năm 2026 làm rõ rằng bài toán không chỉ là “đoạn code nào giống query”. Các hướng mạnh nhất tách ít nhất ba thứ: (1) bằng chứng phụ thuộc thật như symbol/API/call/type, (2) ví dụ tương tự để học pattern, và (3) cấu trúc/ngữ cảnh cần thiết để model hiểu chunk đã lấy. Đồng thời, các bài mới không chứng minh “cứ dựng graph là thắng”: AIRCoder, ACAR và RepoFuse đều kết hợp cấu trúc với retrieval lexical/similarity; RepoShapley cho thấy giá trị của nhiều snippet phụ thuộc tương tác; GrepRAG cho thấy exact identifier search vẫn rất mạnh. 

Với hệ thống của bạn, hướng đáng thử nhất là **single-pass, dependency-role-aware retrieval**: dùng left context để xác định anchor symbols và trạng thái cú pháp, lấy một candidate pool cố định bằng lexical + UniXcoder, rồi xếp hạng bằng loại quan hệ repo có kiểm chứng (định nghĩa/import/call/type/same-scope/analogy). Đây là giả thuyết nghiên cứu hợp lý, **chưa phải novelty đã được xác nhận và chắc chắn không thể hứa trước là sẽ vượt AlignCoder**.

## 1. Năm xuất bản: phân biệt venue với arXiv

RLCoder là bài ICSE 2025; AlignCoder là ASE 2025. Bản arXiv AlignCoder v1 được đăng ngày 27-01-2026, nhưng năm xuất bản hội nghị vẫn là 2025. Đối chiếu theo năm venue là quan trọng vì nếu chỉ lọc arXiv theo năm, survey sẽ xếp nhầm AlignCoder vào nhóm paper publish 2026. [RLCoder — ICSE 2025](https://conf.researchr.org/details/icse-2025/icse-2025-research-track/24/RLCoder-Reinforcement-Learning-for-Repository-Level-Code-Completion), [AlignCoder — ASE 2025](https://conf.researchr.org/details/ase-2025/ase-2025-papers/34/AlignCoder-Aligning-Retrieval-with-Target-Intent-for-Repository-Level-Code-Comple), [arXiv timestamp](https://arxiv.org/abs/2601.19697).

Trong các bài 2026, những nguồn trực tiếp nhất là LC² (ACL), AIRCoder (ACL), RepoShapley (Findings ACL), ACAR (EMNLP), RepoFuse (FSE Industry), GrepRAG (ISSTA/PACMSE), SpecAgent (ACL) và ArkRepoBench (Findings ACL). RepoScope/AllianceCoder/InlineCoder cũng hữu ích nhưng phần lớn làm **full-function repository generation**, không phải completion từ prefix hiện có; vì thế cần để riêng, không xếp số Pass@1 của chúng cạnh EM/ES của CCEval. [LC²](https://aclanthology.org/2026.acl-short.64/), [AIRCoder](https://aclanthology.org/2026.acl-long.1166/), [RepoShapley](https://aclanthology.org/2026.findings-acl.505/), [ACAR](https://aclanthology.org/2026.emnlp-main.81/), [RepoScope](https://conf.researchr.org/details/icse-2026/icse-2026-research-track/169/RepoScope-Leveraging-Call-Chain-Aware-Multi-View-Context-for-Repository-Level-Code-G).

## 2. Các paper 2026 sát nhất với hệ thống hiện tại

| Paper | Đóng góp chính | Mức khớp với bài toán của bạn | Giới hạn khi dùng làm đối chứng |
|---|---|---|---|
| **LC², ACL 2026** | Tách “retrieval context” khỏi context được mở rộng để generator hiểu; query bất đối xứng, bổ sung signature/docstring có chọn lọc. | **Cao nhất về cấu hình model**: DeepSeek-Coder-1.3B + UniXcoder được đánh giá trực tiếp. | Top-5, token budget/prompt và retrieval setup khác; không nên so số thẳng với run của bạn. |
| **AIRCoder, ACL 2026** | 8 tín hiệu: lexical/dense, dependency-existence, path/file/class/function/structural similarity; RRF hoặc MLP fusion thích ứng. | Cao: có benchmark CCEval/RepoEval và DeepSeek-Coder-1.3B; gần trực tiếp với retrieval context. | MLP dùng **pairwise MSE** trên chênh lệch ES; RRF mới là bản không cần train. Dependency extraction dựa Qwen3-8B. |
| **ACAR, EMNLP 2026** | Lexical + AST/LSP structural retrieval, line-level reciprocal-rank fusion, syntax-aware hậu xử lý. | Rất cao về task: paper gọi rõ là interactive repository-level FIM. | Kết quả nổi bật dùng Qwen2.5-Coder-3B; overhead 87–95 ms là warm-cache và không gồm decoding. |
| **RepoShapley, Findings ACL 2026** | Ước lượng signed/context coalition utility; phát hiện synergy, redundancy, conflict; distill lựa chọn keep/drop. | Cao về câu hỏi “chunk đúng/xấu và tương tác ra sao”. | Labeling nhiều forward/decode; exact subset cost tăng 2^K; train control/generator, không chỉ UniXcoder. |
| **GrepRAG, ISSTA 2026** | LLM-assisted grep/identifier search; identifier-weighted rerank + structure-aware dedup. | Cao như lexical baseline/candidate source một lần, nhất là import, API và symbol xuất hiện trong left prefix. | Có thể mất semantic analogue khi không khớp tên; model/query generation trong paper lớn hơn nhiều. |
| **RepoFuse, FSE 2026 Industry** | Tách structural “semantic context” và similarity-based “similar context”, chọn theo relevance/latency. | Cao về cách định nghĩa hai vai trò evidence; có production evidence. | Industry study tại một công ty, A/B một tuần; không phải cùng generator/protocol. |
| **SpecAgent, ACL 2026** | Làm exploration ở thời điểm indexing, chuẩn bị speculative context cho chỉnh sửa tương lai. | Vừa: hữu ích nếu sau này làm IDE/indexing latency. | Không phải one-shot left-prefix retrieval thuần; paper cũng cảnh báo future-context leakage. |
| **ArkRepoBench, Findings ACL 2026** | Chia Single-File / Cross-File Independent / Cross-File Dependent. | Cao về thiết kế evaluation: đo xem dependency có thật sự cần không. | Benchmark ArkTS/HarmonyOS; taxonomy chuyển được, leaderboard thì không. |

### Chi tiết quan trọng

**LC² là paper nên đọc/đối chiếu đầu tiên cho setup của bạn.** Bảng chính dùng DeepSeek-Coder-1.3B với UniXcoder, top-5 candidate, tối đa 4,096 token cross-file context và 8,192 token toàn sequence. Query retrieval lấy 64 token cuối của unfinished code. Ở CCEval Python, Fixed-Token đạt 22.01 EM còn LC² đạt 26.35 — chênh 4.34 điểm tuyệt đối, tương đương 19.7% tương đối. Java tăng 23.38 lên 24.22. Tác giả cũng báo context augmentation không phải lúc nào cũng có ích: signature/function không liên quan có thể gây noise. Vì vậy, bài này củng cố việc tách **candidate representation** khỏi **prompt context sau khi chọn**, và lọc kỹ phần mở rộng thay vì nhét cả function. [Paper + bảng thực nghiệm LC²](https://aclanthology.org/2026.acl-short.64/).

**AIRCoder cần được đọc thành hai hệ thống khác nhau.** Bản RRF tích hợp các thứ hạng nhưng training-free. Bản MLP tạo query-dependent weights, nhưng huấn luyện bằng pairwise MSE trên chênh lệch Edit Similarity giữa candidate dương/âm. Tức là AIRCoder *không* cho bạn một loss mới pairwise-free; nó cho một bộ feature tốt và baseline RRF rất hợp để thử. Với DeepSeek-Coder-1.3B, tác giả báo CCEval Python 31.26 EM cho RRF, CodeRAG 30.92 và DraCo 30.28 trong cùng bảng của paper. Đây là một đối chứng cần reproduce trước khi tuyên bố vượt prior art, nhưng vẫn khác chunking/context/decoding hiện tại. [AIRCoder, method và kết quả](https://aclanthology.org/2026.acl-long.1166/).

**ACAR là baseline FIM mới nhất, còn GrepRAG là baseline retrieval đáng làm.** ACAR kết hợp lexical và AST/LSP structural evidence; ablation của họ cho thấy hai loại tín hiệu bổ sung nhau. GrepRAG nhấn mạnh exact identifier match và deduplication để giữ nguyên vẹn code flow. Kết hợp hai kết quả này gợi ý cần đo riêng “đúng symbol/dependency” với “đoạn code trông giống”, thay vì để dense similarity gánh cả hai vai trò. [ACAR](https://aclanthology.org/2026.emnlp-main.81/), [GrepRAG](https://doi.org/10.1145/3832136).

**RepoShapley chỉ ra một vấn đề mà điểm candidate độc lập có thể bỏ sót.** Hai snippet đều ít giá trị riêng nhưng có thể hữu ích khi ghép (ví dụ signature/type + call-site usage), hoặc từng snippet có ích nhưng cùng đưa vào lại nhiễu. Method của họ giải bằng teacher-forced probes, surrogate interaction game, Shapley labels và distillation. Đây là bằng chứng mạnh cho tương tác slate, nhưng chi phí offline/độ phức tạp cao và không khớp yêu cầu hiện tại là chỉ train retriever. Hãy lấy đây làm **cảnh báo cho objective**, không nhất thiết bê nguyên pipeline. [RepoShapley, method và limitations](https://aclanthology.org/2026.findings-acl.505/).

## 3. Quan hệ repo: graph còn sống, nhưng “có graph” không đủ

Đồ thị cho code completion đã có từ trước: DraCo dùng dataflow-guided repository context; GraphCoder tạo Code Context Graph cho coarse-to-fine retrieval; RepoHyper tìm–mở rộng–tinh chỉnh trên semantic graph. Đến 2026, RepoScope dùng Repository Structural Semantic Graph và call-chain-aware multi-view context, còn AIRCoder/ACAR/RepoFuse đưa structural/dependency evidence vào hybrid retrieval. Vậy nên nhận định “graph cho code mấy năm nay yếu nên không ai làm nữa” không chính xác. Nhưng cũng **không thể claim novelty chỉ bằng cách dựng repo graph**. [DraCo](https://aclanthology.org/2024.acl-long.431/), [GraphCoder](https://conf.researchr.org/details/ase-2024/ase-2024-research/46/GraphCoder-Enhancing-Repository-Level-Code-Completion-via-Coarse-to-fine-Retrieval-B), [RepoHyper](https://doi.org/10.1109/Forge66646.2025.00009), [RepoScope](https://conf.researchr.org/details/icse-2026/icse-2026-research-track/169/RepoScope-Leveraging-Call-Chain-Aware-Multi-View-Context-for-Repository-Level-Code-G).

Hướng khác nhau nằm ở *dùng* quan hệ nào để trả lời câu hỏi nào:

- **Symbol/data dependency:** biến, type, import, API nào trong prefix cần định nghĩa hoặc usage từ file khác? DraCo/GraphCoder nhấn vào data/control relation.
- **Call relation:** callee nào cung cấp behavior/signature cần cho target, và caller nào thể hiện contract/expected usage? RepoScope/InlineCoder dùng call chains/callers-callees, nhưng InlineCoder là multi-stage full-function generation.
- **Structural proximity:** cùng package, class, module, file tree, enclosing scope; AIRCoder dùng path/scope signals.
- **Analogy:** các đoạn cùng pattern nhưng không nhất thiết có dependency thật; RepoFuse/GrepRAG và ACAR đều cho thấy similarity/lexical evidence có ích, nhưng cần tránh noise.

AST chunking phẳng giải quyết **biên cú pháp cục bộ** tốt hơn line chunking, nhưng không tự tạo quan hệ giữa chunk A và chunk B, cũng không cho biết dependency đó cần cho completion hay chỉ tình cờ nằm cùng repo. Nên xem chunking và repo structure là hai layer riêng: mỗi chunk giữ code + symbol/scope metadata; index quan hệ nối định nghĩa, import, call, type và containment.

## 4. RLCoder / AlignCoder khác gì nhánh structural 2026?

RLCoder và AlignCoder là **target-aware retrieval training**. RLCoder lấy feedback từ completion để học retriever; AlignCoder tạo candidate completions để làm query giàu target intent hơn và dùng RL cho AlignRetriever. Cả hai trả lời “retrieval nào giúp continuation”, nhưng trọng tâm không phải mã hóa tường minh quan hệ `defines/imports/calls/types/caller/callee` trong repo. [RLCoder](https://conf.researchr.org/details/icse-2025/icse-2025-research-track/24/RLCoder-Reinforcement-Learning-for-Repository-Level-Code-Completion), [AlignCoder](https://arxiv.org/abs/2601.19697).

Nhánh 2026 không thay thế target-aware retrieval; nó bổ sung **evidence structure**: AIRCoder/ACAR/RepoFuse fuse structural với lexical/similarity, LC² sửa thiếu ngữ nghĩa khi chunk được retrieve, RepoShapley đo tương tác giữa context, RepoScope đưa call-chain structure vào prompt. Đóng góp mới có thể là nối hai mặt này: target utility có điều kiện theo **vai trò quan hệ của candidate**, không chỉ scalar similarity/target reward không giải thích được vì sao candidate liên quan.

## 5. Framework nên thử cho hệ thống của bạn

### Tên làm việc

**Left-Context-Anchored Dependency-Role Retrieval (LADR)** — tên mô tả cho thiết kế thử nghiệm, chưa phải claim novelty.

### Một lượt retrieval, không cần sinh query bằng generator

1. **Đọc query đúng theo task:** lấy left context theo prompt hiện tại; parse vị trí cursor, enclosing function/class, recent identifiers, import aliases, receiver/method/type hints, unresolved symbols. Không sử dụng right context hay target continuation làm query/inference feature.
2. **Tạo một candidate pool cố định** (ví dụ 100 như hiện tại): lấy union từ sparse exact identifier/BM25 và UniXcoder dense. Đây là một logical retrieval pass; nhiều index lookup cùng một query vẫn không phải retrieve–generate–retrieve nhiều vòng.
3. **Gắn mỗi candidate một evidence-role vector:** direct definition/API/type; import/alias; call-site/example usage; same-file/scope; lexical match; semantic analogue. Ghi loại edge, confidence, số hop, path/scope và khoảng cách. Không dùng graph để loại cứng candidate — edge thiếu/parse fail thì fallback về sparse/dense.
4. **Fusion/ranking:** bắt đầu bằng RRF/feature ablation để xác nhận tín hiệu trước; sau đó thử một ranker query-conditioned/listwise trên danh sách candidates, không cần tạo các cặp chosen/rejected thủ công. Label utility có thể từ cùng frozen generator/đúng ES-EM scorer; phần này là đề xuất thực nghiệm, chưa phải một kết quả đã được paper chứng minh.
5. **Slate selection:** lấy top-10 theo score dưới cross-file token budget; deduplicate/merge overlap và giữ nguyên vùng code cần thiết quanh AST chunk. Đánh giá cả retrieval recall lẫn end-to-end EM/ES. Nếu top-10 fixed vượt top-K adaptive thì bỏ STOP; không mặc định model phải học số lượng trước khi ranker ổn.

### Vì sao đây là contribution khá hơn “thêm graph”

Claim có thể kiểm chứng: *a single-pass retriever that explicitly distinguishes dependency evidence from analogous code, anchors relation scoring in the target's left-context symbols, and selects context under a fixed prompt budget*. Điểm mới không phải graph hay RRF riêng lẻ; phải chứng minh relation-role supervision làm tăng utility ở các task dependency-required, không làm giảm ở single-file/independent tasks, và vẫn giữ độ trễ một lượt.

Không khuyến nghị bắt đầu bằng GNN hoặc full graph traversal. Trước hết graph có thể là symbol/edge table và vài typed path features; nếu những feature đó không tăng chỉ số so với sparse+dense thì một GNN nặng sẽ khó có lý do. Chỉ sau khi thấy tín hiệu relation rõ mới cần learned message passing.

## 6. Experiment tối thiểu để biết hướng có thật sự hiệu quả

Giữ cố định generator, prompt/left context, fixed chunking, candidate pool, top-10/context token cap, decoding, ES/EM implementation và tập mẫu. Chia test theo **single-file**, **cross-file có mặt nhưng không bắt buộc**, **cross-file dependency bắt buộc**; ArkRepoBench cho thấy ba nhóm này khác nhau. Với CCEval Python/Java và RepoEval line/API, có thể tạo nhãn dependency audit bằng static analysis trên repo snapshot, sau đó manual audit một mẫu để ước lượng độ chính xác.

Baseline/ablation theo thứ tự: (A) no retrieval, (B) current UniXcoder, (C) BM25/ripgrep identifier, (D) RRF sparse+dense, (E) E + path/scope features, (F) E + typed dependency edges, (G) relation-aware listwise ranker. Không đổi chunking trong thí nghiệm đầu tiên; thêm LC²-style context expansion thành một trục riêng sau. Báo EM, ES, ID-EM/ID-F1 nếu scorer hỗ trợ, dependency Recall@100/10, precision@10, candidate count, selected count/token, latency, parser/edge coverage, và kết quả theo từng dependency stratum.

Để claim vượt AlignCoder/RLCoder, cần chạy baseline của họ hoặc pipeline paper trên **cùng checkpoint generator, prompt, data split/sample set, candidate budget và evaluator**. Không thể so trực tiếp số EM trong paper với run local của bạn, đặc biệt khi model, chunking, output normalization hay top-K khác.

## 7. Tóm lại

- Nếu chỉ chọn một paper để bê đúng setup: **LC²**, vì nó có DeepSeek-Coder-1.3B + UniXcoder và nghiên cứu đúng chunk/comprehension.
- Nếu chọn một 2026 baseline retrieval: **AIRCoder RRF**, lưu ý learned loss của họ là pairwise.
- Nếu chọn 2026 system FIM mạnh để tái lập: **ACAR**, nhưng cần cùng backbone và kiểm soát warm cache/post-processing.
- Nếu chọn insight cho contribution: kết hợp **dependency roles + one-pass target-aware ranking**; không gọi AST chunking là repo graph.
- Nếu chọn cảnh báo: similar snippets có thể gây noise; chunk utility có tương tác; cross-file context không đồng nghĩa dependency-required; future-context leakage phải được loại bỏ.

## Research artifacts

- [Research plan](plan.md)
- [Atomic findings](findings.md)
- [Source index](sources.csv)
- [Adversarial review](adversarial_review.md)
- [Refresh targets](refresh_targets.md)
- [Per-paper source notes](source_notes/)
