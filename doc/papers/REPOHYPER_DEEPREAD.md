# DeepRead report: RepoHyper

## 0. Nguồn và trạng thái trích xuất

- Paper: RepoHYPER: Search-Expand-Refine on Semantic Graphs for Repository-Level Code Completion.
- File nguồn: [RepoHyper.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/RepoHyper.pdf).
- Phạm vi: toàn bộ 13 trang; phân tích chính ở trang 1–10.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

RepoHYPER lập semantic graph của repository và tách retrieval thành Search–Expand–Refine. Search dùng embedding để lấy các anchor gần query; Expand đi theo graph bằng exhausted BFS hoặc pattern search; Refine dùng GraphSAGE/link prediction để xếp hạng node. Ý tưởng cốt lõi là code có thể liên quan về program semantics dù không text-similar. Kết quả retrieval và end-to-end tăng rõ trên RepoBench, nhưng graph construction nặng, pattern phụ thuộc language/domain và link predictor vẫn cần pseudo-label từ gold snippets.

## 2. Luận điểm trung tâm

**Author's stated position:** Semantic graph expansion kết hợp link prediction có thể tìm context repository liên quan mà kNN similarity thuần túy bỏ sót.

## 3. Cây lập luận

1. Similarity-only retrieval ưu tiên code giống bề mặt.
2. Repository chứa quan hệ import, call, ownership, inheritance.
3. Semantic graph cho phép mở rộng từ anchor sang node liên quan.
4. Pattern search giảm không gian BFS bằng path type thường gặp.
5. Link predictor xếp hạng node quan trọng hơn cosine.
6. Context tốt hơn giúp LLM completion tốt hơn.

## 4. Phương pháp

### 4.1. Repository Semantic Graph

Node types:

- function/method;
- class;
- Script node cho phần còn lại của file.

Edge types:

- import/imported-by;
- invoke/caller-callee;
- ownership;
- encapsulation;
- class hierarchy.

Graph được tạo bằng tree-sitter, Python AST, PyCG và chuyển Python 2 sang Python 3 bằng 2to3 khi cần.

### 4.2. Search

- Query encode bằng UniXcoder hoặc CodeT5+.
- kNN lấy K = 3 anchor nodes.

### 4.3. Expand

- Exhausted BFS: duyệt đến depth D và giới hạn M node.
- Pattern Search: học frequent path types từ đường anchor-to-gold trong training, sau đó chỉ đi những pattern phù hợp, ví dụ ownership–encapsulate–import.

### 4.4. Refine

- Thêm query node Q vào graph.
- GraphSAGE chạy L layers, paper dùng L = 3.
- Score node i bằng linear layer trên concatenation của embedding node i và Q.
- Link prediction dùng BCE; công thức trong paper được viết rút gọn và không hiển thị đầy đủ positive/negative complement term, nên cần cẩn trọng khi tái triển khai.
- Search top N1, rồi rerank top N2; N2 điều chỉnh theo context budget.

### 4.5. Training và inference

- Pseudo-label lấy bằng Jaccard text matching giữa gold snippets và RSG nodes.
- GraphSAGE: Adam, learning rate 0.01, 10 epochs, 2 A100, khoảng 6 giờ.
- Thiết lập chính: D = 4, M = 1000, K = 3.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| Graph expansion tăng retrieval | UniXcoder Easy acc@3 59.69 → RepoHyper 68.91; Hard acc@3 39.02 → 49.51 | p. 6, Table 1 | Support cho Search–Expand–Refine | **Source fact or data** | RepoBench-R và Jaccard labels |
| CodeT5+ cũng hưởng lợi | Easy acc@3 53.31 → 67.23; Hard 25.24 → 46.70 | p. 6, Table 1 | Không phụ thuộc một retriever | **Source fact or data** | Backbone/metric khác UniXcoder |
| End-to-end tốt hơn RepoCoder | GPT-3.5 XF-F EM 48.73 → 52.76; XF-R 59.55 → 64.06 | pp. 7–8, Table 2 | Retrieval gain truyền sang generation | **Source fact or data** | Có hai ordering L2H/H2L |
| Pattern search và link prediction bổ sung | Easy acc@3 kNN 60.15, pattern 64.23, pattern+link 69.12; Hard 40.74, 44.50, 47.83 | p. 9, Table 4 | Cả graph traversal và refinement đều cần | **Source fact or data** | Pattern được chọn thủ công/học từ train |
| Pattern search đổi coverage lấy tốc độ | Pattern hit 73%, graph coverage 28%; exhausted hit 80%, coverage 40% | p. 9 | Trade-off recall–cost | **Source fact or data** | Không có latency tổng hợp đầy đủ |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Model hóa quan hệ chương trình rõ hơn embedding similarity.
- Search/expand/refine là decomposition dễ ablate.
- Có thể tận dụng import/call/inheritance không hiện diện trong query text.
- Cho thấy thứ tự context cũng ảnh hưởng end-to-end.

### Giới hạn

- Graph extraction phụ thuộc parser, static analysis và chất lượng repository.
- Pattern types cần lựa chọn theo ngôn ngữ/task.
- Pseudo-label từ Jaccard có thể đánh đồng lexical overlap với relevance.
- GraphSAGE training không hoàn toàn label-free.
- M memory/cost tăng nhanh trên repository lớn.

## 7. Hàm ý cho AST/KD

1. AST chunk metadata có thể làm node nhẹ hơn full RSG: node type, parent, import, call, signature.
2. Teacher embedding có thể cung cấp soft node utility; GraphSAGE chỉ cần học propagation/compatibility.
3. Không nên dùng graph-only retrieval: kết hợp exact identifier/BM25 để tránh mất API match.
4. Distill graph-aware teacher scores vào student retriever giúp inference nhẹ hơn, nhưng cần evaluate unseen repositories.
5. Context ordering nên theo dependency/topological order, không chỉ score giảm dần.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** semantic graph + link prediction vượt kNN retrieval.
- **Source fact or data:** retrieval và end-to-end metrics đều có gain trong bảng RepoBench.
- **Reasoned inference:** graph priors là bổ sung tốt cho AST chunk KD, nhất là dependency context không text-similar.
- **Unverified:** student dense retriever distill từ teacher graph có giữ được gain khi bỏ GraphSAGE hay không.

## 9. Câu hỏi recall/transfer

1. RepoHyper khác AlignCoder ở đâu về nguồn target signal?
2. Vì sao pattern search có thể nhanh hơn exhausted BFS nhưng giảm coverage?
3. Pseudo-label Jaccard có thể gây bias gì?
4. AST chunk graph tối giản nào đủ để giữ benefit của RSG?

