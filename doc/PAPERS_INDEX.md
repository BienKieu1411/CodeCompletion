# DeepRead index: toàn bộ paper trong Paper/

## 0. Phạm vi và trạng thái

- Đã xử lý: 15 PDF trong [Paper](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper).
- Tổng số trang: 219.
- Kiểm tra extraction: cả 15 PDF đều có text layer sử dụng được; không có trang dưới ngưỡng kiểm tra text.
- Mỗi paper có một report deep-read riêng trong thư mục [doc/papers](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers).
- Report cũ [REPOSHAPLEY_DEEPREAD.md](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/REPOSHAPLEY_DEEPREAD.md) được giữ nguyên và tính vào bộ report này.
- [TCD_KD_FRAMEWORK.md](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/TCD_KD_FRAMEWORK.md) là tài liệu framework trước đó; không bị xoá.

Các report dùng cùng hệ nhãn bằng chứng:

- **Author's stated position**
- **Source fact or data**
- **Reasoned inference**
- **Unverified**

## 1. Danh mục 15 paper

| PDF | Chủ đề chính | Report |
|---|---|---|
| 2026.acl-short.64.pdf | Late Code Chunking: retrieve ngắn, expand/augment muộn | [LATE_CODE_CHUNKING_DEEPREAD.md](papers/LATE_CODE_CHUNKING_DEEPREAD.md) |
| 2026.findings-acl.505.pdf | REPOSHAPLEY: Shapley/context interaction và filtering | [REPOSHAPLEY_DEEPREAD.md](REPOSHAPLEY_DEEPREAD.md) |
| 2512.21332v1.pdf | C2LLM: PMA code embedding | [C2LLM_DEEPREAD.md](papers/C2LLM_DEEPREAD.md) |
| AST_Chunking.pdf | CAST: split-then-merge trên AST | [CAST_DEEPREAD.md](papers/CAST_DEEPREAD.md) |
| AlignCoder.pdf | Query enhancement + RL retriever | [ALIGNCODER_DEEPREAD.md](papers/ALIGNCODER_DEEPREAD.md) |
| CodeRAG.pdf | Query probing + multi-path + BESTFIT KD | [CODERAG_DEEPREAD.md](papers/CODERAG_DEEPREAD.md) |
| Grace.pdf | Hierarchical graph + hybrid retrieval + graph fusion | [GRACE_DEEPREAD.md](papers/GRACE_DEEPREAD.md) |
| GrepRAG.pdf | LLM grep + BM25 rerank + structural dedup + KD | [GREPRAG_DEEPREAD.md](papers/GREPRAG_DEEPREAD.md) |
| ICLR-2026-revela...pdf | REVELA: NTP để học dense retriever | [REVELA_DEEPREAD.md](papers/REVELA_DEEPREAD.md) |
| LiPO.pdf | Listwise preference/LambdaLoss | [LIPO_DEEPREAD.md](papers/LIPO_DEEPREAD.md) |
| RLCoder.pdf | RL retriever từ weighted target PPL | [RLCODER_DEEPREAD.md](papers/RLCODER_DEEPREAD.md) |
| ReACC.pdf | Hybrid BM25+dense retriever | [REACC_DEEPREAD.md](papers/REACC_DEEPREAD.md) |
| RepoCoder.pdf | Iterative retrieval–generation | [REPOCODER_DEEPREAD.md](papers/REPOCODER_DEEPREAD.md) |
| RepoHyper.pdf | Semantic graph Search–Expand–Refine | [REPOHYPER_DEEPREAD.md](papers/REPOHYPER_DEEPREAD.md) |
| StepCoder.pdf | AST curriculum + execution-aware RL | [STEPCODER_DEEPREAD.md](papers/STEPCODER_DEEPREAD.md) |

## 2. Bản đồ phương pháp

### 2.1. Chunking và context construction

| Hướng | Paper | Giá trị | Điểm yếu cần khắc phục |
|---|---|---|---|
| Fixed line/token | ReACC, RepoCoder, AlignCoder | Rẻ, dễ index | Cắt function/class/AST; mất parent và dependency semantics |
| Split-Aggregate blank lines | RLCoder, AlignCoder | Giữ block tự nhiên hơn fixed window | Vẫn không bảo toàn syntax/AST |
| AST split-then-merge | CAST | Giữ syntactic integrity, density, reconstruction | Chưa target-aware |
| Late expansion/augmentation | Late Code Chunking | Retrieval precision và comprehension completeness tách biệt | Expansion còn heuristic |
| Graph fusion | GRACE, RepoHyper | Giữ structural relation và interaction | Chi phí/parser complexity cao |
| Coalition-aware filtering | REPOSHAPLEY | Mô hình complementarity/conflict giữa chunks | Offline verification tốn kém, vẫn phụ thuộc candidate pool |

Kết luận từ nhóm này: AST chunking nên là tầng nền. Nhưng AST chunk đúng chỉ giải boundary loss; nó chưa nói chunk nào giúp target và chưa xử lý context interaction. Hai tầng sau cần target utility và late expansion/selection.

### 2.2. Retrieval signal

| Signal | Paper | Quan sát |
|---|---|---|
| Exact lexical/API | ReACC, GrepRAG | BM25/grep rất mạnh vì completion phụ thuộc identifier |
| Dense semantic | ReACC, C2LLM, REVELA | Tốt cho paraphrase/semantic relation nhưng có thể bỏ exact identifier |
| Iterative generated query | RepoCoder | Target signal tốt nhưng latency/error propagation |
| Candidate completion query | AlignCoder | Bridge query-target gap, nhưng sampling/noise và hard reward |
| Dataflow/dependency graph | CodeRAG, RepoHyper, GRACE | Tìm relation không giống text |
| Target PPL/LM utility | RLCoder, AlignCoder | Đo trực tiếp tác động với target nhưng phụ thuộc evaluator |
| Structural/coalition utility | REPOSHAPLEY, GRACE | Xử lý interaction giữa context units |
| NTP cross-document | REVELA | Không cần labeled pairs, nhưng relatedness chưa chắc là completion utility |

Kết luận từ nhóm này: framework mạnh nhất cần hybrid lexical + dense + structural candidate generation, sau đó học target utility thay vì để một similarity score quyết định toàn bộ.

### 2.3. Training objective

| Objective | Paper | Có pairwise không? | Nhận xét |
|---|---|---:|---|
| InfoNCE/in-batch contrastive | ReACC, C2LLM | Không tạo explicit pair list, nhưng có negatives trong batch | Hợp pretraining representation, chưa target-aware |
| Iterative inference, không train | RepoCoder | Không | Tăng latency thay vì học student |
| Winner-take-all policy reward | RLCoder, AlignCoder | Không pairwise tuple, nhưng candidate competition hard | Sparse signal, evaluator coupling |
| LambdaLoss | LiPO | Có phân rã theo cặp | Không dùng cho hướng hiện tại |
| Offline teacher CE | CodeRAG, GrepRAG | Không cần pairwise training | Đã có precedent tốt cho distill reranker/query generator |
| NTP cross-document | REVELA | Không | Có thể làm loss chính self-supervised |
| PPO/compiler feedback | StepCoder | Không phải retrieval objective | Hữu ích như auxiliary execution/coverage signal |

## 3. AlignCoder: phần đã tốt và phần cần vượt

### AlignCoder đã giải quyết tốt

1. Query–target semantic misalignment bằng multiple candidate completions.
2. Dependency context từ import/signature/class/method.
3. Target-aware retriever training bằng target PPL.
4. Ablation cho thấy dependency context, query enhancement và retriever training đều cần.
5. Trên bảng của paper, gain lớn nhất là 18.1% relative EM trên CrossCodeEval Python so với RLCoder.

### Các điểm yếu có bằng chứng trực tiếp hoặc suy luận mạnh

1. **Chunking:** base chunks theo blank lines, trong khi CAST cho thấy AST split-then-merge tốt hơn line/fixed chunk.
2. **Sampling cost:** k = 4 thường tốt, nhưng cần nhiều generator samples và k > 4 sinh noise.
3. **Hard reward:** chỉ candidate PPL thấp nhất nhận tín hiệu; candidate gần tốt hoặc cùng bổ trợ không được phân biệt.
4. **Evaluator coupling:** DeepSeekCoder-1B vừa cung cấp PPL reward vừa định hướng retriever; bias của evaluator có thể được học vào retriever.
5. **No explicit soft teacher distribution:** paper không distill score distribution/embedding của một teacher lớn.
6. **No learned context interaction:** dependency/base candidates được retrieve, nhưng complementarity/conflict chưa được mô hình hóa như REPOSHAPLEY/GRACE.
7. **No explicit no-retrieval/expand policy trong cùng objective:** stop signal mạnh của RLCoder và late expansion của LC2 chưa được tích hợp.
8. **Pipeline latency:** coarse BM25 → sampling k completions → fine retrieval → final generation.

## 4. Những gì mỗi paper đóng góp cho framework mới

### Tầng candidate và chunk

- CAST: AST split-then-merge.
- ReACC/GrepRAG: lexical exact-match branch.
- RepoHyper/GRACE: dependency/structural edges.
- Late Code Chunking: late expansion thay vì index context quá dài.

### Tầng target signal

- RepoCoder: generated completion chứa target-like tokens.
- RLCoder/AlignCoder: target teacher-forced PPL.
- CodeRAG: query probing bằng log-probability.
- StepCoder: execution coverage có thể weight code segment.

### Tầng learning

- REVELA: NTP có thể là loss chính không pairwise.
- CodeRAG/GrepRAG: offline teacher → student CE/KD khả thi và giảm latency.
- C2LLM: PMA pooling là ứng viên encoder/pooling mạnh.
- REPOSHAPLEY/GRACE: context interaction cần được đo hoặc approximated.
- LiPO: label magnitude và metric-aware weighting đáng giữ dưới dạng ý tưởng, nhưng không dùng pairwise loss.

## 5. Khoảng trống nghiên cứu còn lại

Khoảng trống có tính khả thi nhất sau khi đọc toàn bộ 15 paper là:

1. AST chunks làm đơn vị context chuẩn.
2. Candidate pool hybrid gồm lexical, dense và structural neighbors.
3. Một teacher lớn hoặc code generator chấm soft target utility cho từng AST chunk và/hoặc expanded view.
4. Student retriever/reranker học pointwise utility hoặc distributional KD, không tạo pairwise examples.
5. REVELA-style NTP có thể dùng làm pretraining/loss chính; KD utility là auxiliary target-alignment loss.
6. Context selection có no-retrieval và expand/keep decisions.
7. Generator cuối được cố định khi training retriever, sau đó đánh giá bằng generator held-out để tránh evaluator overfit.

Đây là synthesis từ paper, chưa phải bằng chứng rằng framework này đã vượt AlignCoder. Claim vượt chỉ được phép đưa ra sau controlled ablation trên cùng generator, candidate budget, repository split và benchmark.

## 6. Checklist thực nghiệm để claim vượt AlignCoder

### Bắt buộc giữ công bằng

- Cùng backbone generator và decoding.
- Cùng repository split, context token budget và top-K.
- Cùng candidate pool khi so sánh chunking/loss.
- Không để teacher đọc target ở inference.
- Báo cả retrieval recall, target PPL, code EM/ES, identifier EM/F1 và latency.

### Ablation tối thiểu

1. AlignCoder reproduction.
2. AlignCoder + AST chunking.
3. AST + dense/lexical baseline.
4. AST + soft KD utility.
5. AST + KD + no-retrieval.
6. AST + KD + late expansion.
7. AST + KD + structural/lexical hybrid.
8. Teacher model swap: DeepSeekCoder, Jina/C2LLM/Revela-style embedding.
9. Held-out evaluator swap.
10. Noise/duplicate/implicit-dependency subsets.

### Failure analysis

- AST boundary lỗi.
- Exact identifier bị miss.
- Candidate cần complement nhưng bị lọc.
- Candidate conflict làm generator sai.
- Teacher utility lệch với execution/compile.
- Retrieval có ích nhưng context ordering làm completion tệ.

## 7. Kết luận ngắn

Không có paper đơn lẻ nào đã giải trọn bài toán. AlignCoder hiện là baseline target-aware mạnh nhất trong folder, nhưng các paper còn lại chỉ ra một hướng nâng cấp có cơ sở:

**AST-preserving hybrid retrieval → offline soft teacher utility distillation → no-retrieval/late expansion policy → fixed generator evaluation.**

Trong hướng này:

- AST/CAST sửa mất cấu trúc của AlignCoder.
- GrepRAG/ReACC sửa mất exact identifier.
- RepoHyper/GRACE sửa mất dependency relation.
- RepoCoder/AlignCoder cung cấp target-like signal.
- RLCoder/AlignCoder cung cấp target utility.
- CodeRAG/GrepRAG chứng minh offline KD thực dụng.
- REVELA cung cấp ứng viên cho loss chính non-pairwise.
- REPOSHAPLEY cảnh báo cần xét interaction giữa context chunks.

Phần trên là bản đồ nghiên cứu; triển khai cụ thể cần được ghi trong một plan/technical design riêng sau khi chốt model teacher, student và generator.

