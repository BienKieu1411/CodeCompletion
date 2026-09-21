# DeepRead report: GRACE

## 0. Nguồn và trạng thái trích xuất

- Paper: GRACE: Graph-Guided Repository-Aware Code Completion through Hierarchical Code Fusion.
- File nguồn: [Grace.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/Grace.pdf).
- Phạm vi: toàn bộ 13 trang; phân tích chính ở trang 1–10.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

GRACE xây dựng heterogeneous graph ở ba mức repository, module và function. Nó dùng semantic retrieval để tìm node gần query, structural retrieval để tìm node cùng cấu trúc, hybrid scoring để trộn hai nguồn, sau đó graph fusion nối query AST với retrieved subgraphs bằng cross-attention và type-compatible edges. Graph fused được serialize thành triples/markers rồi đưa cho LLM. Đây là hướng mạnh về cấu trúc và context interaction, nhưng hệ thống phức tạp, chi phí graph fusion cao, prompt JSON đặc thù và chưa chứng minh rõ trên compile/unit-test metrics.

## 2. Luận điểm trung tâm

**Author's stated position:** Giữ các quan hệ cấu trúc qua heterogeneous graph và fusion trước khi serialize giúp LLM hiểu repository context tốt hơn RAG phẳng.

## 3. Cây lập luận

1. Semantic similarity bỏ sót quan hệ structural.
2. Structural similarity bỏ sót code có chức năng tương tự nhưng khác topology.
3. Hybrid retriever lấy được cả hai.
4. Chỉ nối context dạng text làm mất edge/ownership/call relation.
5. Graph fusion trước serialization bảo toàn association.
6. Generator nhận graph triples + code và hoàn thành tốt hơn.

## 4. Phương pháp

### 4.1. Hierarchical graph

- Repository level: folder structure và cross-file dependency tree.
- Module level: call graph, type dependency, class inheritance.
- Function level: AST, control-flow, data-flow.
- Node có UUID, code/path/line/signature/complexity/nesting.
- Edge biểu diễn calls, inheritance, defines, uses và cross-level relations.

### 4.2. Hybrid retrieval

- Semantic path: CodeT5p-110m embedding và HNSW.
- Structural path: node embeddings và Laplacian positional encoding; similarity trên graph.
- Score = alpha * SemSim + (1-alpha) * StructSim.
- alpha adaptive qua sigmoid trên query/graph features.
- MMR dùng để giảm duplicate context.

### 4.3. Graph fusion

- Parse incomplete query thành AST graph Gq.
- Lấy top-k retrieved subgraphs.
- GNN tạo node embeddings Hq và Hi.
- Weighted aggregation tạo Hr.
- Cross-attention A = softmax(Hq Hr transpose / sqrt(d)).
- Nếu attention > theta và node type compatible, thêm cross-graph edge.
- Giữ original edges, merge node semantically equivalent.
- Serialize graph thành explicit node/edge markers trước khi prompt LLM.

### 4.4. Setup

- Backbone: Qwen2.5-Coder-14B, GPT-4o mini, DeepSeek-V3.
- k tối đa 10 trong mô tả, k = 3 là default tốt nhất.
- theta = 0.4; input 2048 tokens chia gần nửa local/retrieved; max generation 100.
- Prompt yêu cầu JSON gồm completed_code, explanation, confidence_score và referenced_nodes; cần điều chỉnh nếu benchmark yêu cầu raw completion.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| GRACE có gain trên nhiều tổ hợp | Tác giả báo cáo GRACE đứng đầu 35/48 setting, second 9 | pp. 6–8, Table 2 | Support cho hybrid/fusion | **Source fact or data** | Bảng có nhiều backbone/model và cách tính khác nhau |
| Fusion là thành phần quan trọng | CCEval Python full EM/F1 31.38/82.76; w/o fusion 25.98/76.59; AST-only 28.22/79.48 | p. 9, Table 3 | Ablation trực tiếp | **Source fact or data** | Tên metric/bảng cần giữ đúng khi tái hiện |
| k=3 tốt | Scale/ablation cho thấy tăng k quá 3 plateau hoặc giảm | p. 10 | Context diversity có saturation | **Source fact or data** | Phụ thuộc context budget |
| Graph fusion là bottleneck | Cross-attention O(nq nr d) là chi phí chính | pp. 8–10 | Giới hạn deployment | **Author's stated position** | Chưa có profiling trên million-LOC |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Kết hợp AST, data-flow, control-flow, inheritance và file hierarchy.
- Học alpha adaptive thay vì weight cố định.
- Fusion mô hình hóa interaction giữa query và retrieved graph, gần vấn đề complementarity.
- Có ablation cho semantic/structural/fusion.

### Giới hạn

- Graph pipeline khó triển khai và dễ lỗi parser/static analysis.
- Fusion cross-attention có chi phí lớn.
- JSON/explanation prompt có thể tạo lợi thế không thuần retrieval.
- Benchmark chủ yếu Python/Java; metrics chưa đo compile/pass sâu.
- Hệ thống chưa cho biết phần gain đến từ model backbone hay graph retrieval.

## 7. Hàm ý cho AST/KD

1. Không cần sao chép full GRACE; có thể distill hybrid semantic + structural score thành student embedding.
2. Dùng AST parent/child/type và dependency edges làm auxiliary features cho student.
3. Distill attention/edge compatibility ở node-level sẽ hữu ích hơn chỉ distill final embedding.
4. Fusion có thể được approximated bằng context expansion: lấy parent/children/signature của top AST node.
5. Cần ablation graph metadata vs teacher KD vs AST chunking độc lập.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** hierarchical graph fusion làm RAG hiểu code structure tốt hơn.
- **Source fact or data:** bỏ fusion làm metrics giảm mạnh trong ablation.
- **Reasoned inference:** teacher graph encoder có thể tạo soft supervision cho UniXcoder student mà không cần chạy graph fusion ở inference.
- **Unverified:** distillation có giữ được interaction gain của GRACE hay cần giữ một graph module nhỏ.

## 9. Câu hỏi recall/transfer

1. GRACE semantic path và structural path khác nhau thế nào?
2. Khi nào graph fusion thêm edge giữa hai node?
3. Tại sao AST-only chưa đạt full GRACE?
4. Distill edge/attention nào vào student retriever?

