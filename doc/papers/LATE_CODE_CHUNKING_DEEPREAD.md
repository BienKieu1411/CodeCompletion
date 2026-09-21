# DeepRead report: Late Code Chunking

## 0. Nguồn và trạng thái trích xuất

- Paper: Late Code Chunking: A Code Chunking Strategy for Repository-Level Code Completion.
- File nguồn: [2026.acl-short.64.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/2026.acl-short.64.pdf).
- Phạm vi: toàn bộ 7 trang.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

Late Code Chunking (LC2) tách retrieval context và comprehension context. Retrieval dùng chunk token cố định để index/rank ổn định; sau khi chọn chunk, hệ thống mở rộng về phía trước để lấy semantic antecedents và augment bằng signatures/docstrings của các function được gọi. Query dùng 64 token cuối của unfinished code, trong khi retrieved chunks mặc định 512 token. Kết quả cho thấy việc trì hoãn mở rộng context đến sau retrieval tốt hơn việc index các chunk quá dài ngay từ đầu.

## 2. Luận điểm trung tâm

**Author's stated position:** Retrieval nên dùng đơn vị ngắn và bất đối xứng để match chính xác; comprehension mới nên mở rộng/augment retrieved chunks để khôi phục semantics.

## 3. Cây lập luận

1. Chunk lớn giúp hiểu code nhưng làm retrieval nhiễu.
2. Chunk nhỏ match query tốt nhưng thiếu antecedent/definition.
3. Tách retrieval và comprehension giải quyết trade-off.
4. Query gần cursor chứa identifier/API quan trọng.
5. Context expansion và function augmentation khôi phục missing semantics.
6. Generator nhận context giàu hơn với cùng budget.

## 4. Phương pháp

### 4.1. Retrieval context

- Token-based chunks, default 512 tokens.
- Query là 64 token cuối của unfinished code.
- Kết quả top-5, tổng retrieved budget 4096; generator input tối đa 8192.

### 4.2. Context expansion

- Sau khi retrieval, lấy thêm preceding content tối đa 512 tokens.
- Mục tiêu là nối initialization/semantic antecedent bị cắt khỏi chunk.

### 4.3. Context augmentation

- Phát hiện function calls trong retrieved chunk.
- Retrieve corresponding definition.
- Chỉ append function signature + docstring, tối đa 3 functions.
- Đây là augmentation nhẹ hơn append full implementation.

### 4.4. Setup

- Retrievers: BM25, UniXcoder và CodeRank variants.
- Generator: DeepSeekCoder-1.3B trong bảng chính.
- Baselines: in-file, fixed-window 10 lines, Split-Aggregate, function-level, fixed-token 512.
- Decoding nucleus p = 0.95, max 50 tokens.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| LC2 vượt các chunk baseline | RepoEval line LC2 EM/ES 45.12/72.87 vs fixed-token 44.88/72.36; API 45.00/74.48 vs 44.62/74.39 | p. 5, Table 2 | Tách retrieval/comprehension có lợi | **Source fact or data** | Gain nhỏ trên RepoEval |
| CCEval gain rõ hơn | Python LC2 26.35/71.92 vs fixed-token 22.01/69.05; Java 24.22/66.79 vs 23.38/66.10; C# 21.72/68.19 vs 19.98/66.95 | p. 5 | Generalization đa ngôn ngữ | **Source fact or data** | Generator 1.3B |
| Retrieval model vẫn ảnh hưởng | CodeRankLLM-7B Python 29.47/72.95 cao hơn BM25 26.20/72.29 và UniXcoder 26.35/71.92 | p. 6, Table 3 | Có thể thay retriever | **Source fact or data** | CodeRank teacher lớn hơn |
| Expansion/augmentation bổ sung gain | Ablation cho thấy retrieval context + asymmetric sizing đã tốt, expansion/augmentation thêm gain; augmentation có thể hại C# | pp. 5–6 | Context không phải cứ thêm là tốt | **Author's stated position** | Selective augmentation còn future work |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Giải trade-off retrieval precision và semantic completeness.
- Chi phí augmentation bị giới hạn bằng signature/docstring.
- Tương thích với BM25/dense retriever.
- Có connection trực tiếp với AST parent/child expansion.

### Giới hạn

- Function-call detection và definition retrieval vẫn heuristic.
- Fixed token retrieval chunk chưa chắc bảo toàn AST.
- Augment không chọn theo target utility, có thể thêm noise.
- Chưa có learned stopping/selection.

## 7. Hàm ý cho AST/KD

1. Dùng AST node nhỏ làm retrieval unit, rồi late-expand parent/sibling/definition.
2. Teacher KD có thể chấm cả retrieval unit và expansion edges.
3. Không index context mở rộng; chỉ mở rộng sau khi student chọn node.
4. Tạo auxiliary loss để student dự đoán có nên expand/augment hay không.
5. Kết hợp với stop head của RLCoder để tránh expansion khi candidate không có ích.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** late expansion/augmentation giải quyết retrieval–comprehension trade-off.
- **Source fact or data:** LC2 vượt fixed-token và line/function baselines trên các benchmark được báo cáo.
- **Reasoned inference:** đây là cách rẻ để kết hợp AST chunking với context completeness, phù hợp hơn append full graph.
- **Unverified:** KD teacher có thể học expansion policy tốt hơn heuristic hay không.

## 9. Câu hỏi recall/transfer

1. Vì sao retrieval chunk và comprehension chunk nên khác nhau?
2. LC2 augment phần nào của function definition?
3. AST parent expansion khác late token expansion ra sao?
4. Loss nào có thể dạy student quyết định expand?

