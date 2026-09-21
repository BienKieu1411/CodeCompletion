# DeepRead report: GrepRAG

## 0. Nguồn và trạng thái trích xuất

- Paper: GrepRAG: An Empirical Study and Optimization of Grep-Like Retrieval for Code Completion.
- File nguồn: [GrepRAG.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/GrepRAG.pdf).
- Phạm vi: toàn bộ 22 trang; phân tích chính ở trang 1–18.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

GrepRAG cho thấy retrieval index-free bằng ripgrep có thể cạnh tranh hoặc vượt graph/dense retrievers trong repository code completion vì identifier exact match rất quan trọng. Naive GrepRAG để LLM sinh nhiều ripgrep commands, chạy exact string matching rồi Jaccard-rerank. GrepRAG hoàn thiện pipeline bằng BM25 reranking trong candidate pool và structure-aware deduplication/fusion theo line intervals. Kết quả cải thiện CrossCodeEval và RepoEval_Updated với latency thấp. Paper cũng dùng knowledge distillation: Claude Opus teacher tạo dữ liệu cho Qwen3-0.6B chỉ dự đoán retrieval keywords; command template được lấp cố định. Đây là offline KD đáng chú ý cho deployment, nhưng không giải quyết implicit dependencies và vẫn phụ thuộc năng lực query generator.

## 2. Luận điểm trung tâm

**Author's stated position:** Exact lexical retrieval bằng grep, nếu được rerank theo identifier frequency và fuse các chunk chồng lấn, là một baseline mạnh và thực dụng cho repository-level code completion.

## 3. Cây lập luận

1. Graph/vector index có latency và maintenance cost.
2. Code completion thường chứa identifier cụ thể.
3. LLM có thể tự sinh nhiều truy vấn ripgrep từ local context.
4. Exact match có recall tốt cho class/method/variable definitions.
5. Jaccard không xử lý keyword ambiguity và redundancy.
6. BM25 reranking + interval fusion khắc phục hai lỗi này.
7. Student distilled keyword generator giảm overhead LLM lớn.

## 4. Phương pháp

### 4.1. Naive GrepRAG

1. LLM đọc local context trước cursor và sinh khoảng 10 ripgrep commands.
2. Commands thường truy vấn class name, method name, variable name hoặc wildcard pattern.
3. ripgrep chạy trong repository, tạo candidate snippets.
4. Jaccard similarity với local context được dùng để chọn top-K.

### 4.2. GrepRAG post-processing

- Identifier-weighted reranking: xem mỗi grep chunk là document và local context là query; dùng BM25 để IDF giảm trọng số generic identifiers và tăng rare identifiers.
- Structure-aware deduplication:
  - parse line intervals;
  - merge chunks overlap hoặc adjacent;
  - khôi phục contiguous semantic block;
  - chỉ xử lý top-N% candidate ranked list, mặc định N = 50%;
  - lấy top-K sau fusion trong tổng context 4096 tokens.

### 4.3. Offline distillation

- Teacher: claude-opus-4-5-20251101.
- Student: Qwen3-0.6B, chỉ dự đoán core retrieval keywords.
- Command syntax/flags dùng static template, nên student không phải sinh full command.
- Mục tiêu là giảm token generation và latency; không phải distill completion model.

### 4.4. Evaluation

- CrossCodeEval Python/Java và RepoEval_Updated Python/Java.
- Backbones: DeepSeek-V3.2-EXP và Qwen3-Coder-Plus.
- Top-K = 10, context 4096 tokens, temperature 0.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| Naive grep đã mạnh | CrossCodeEval DeepSeek Python EM 38.61 vs Vanilla RAG 24.99 và RLCoder 36.59; Java 41.70 | pp. 8–9, Table 3 | Exact lexical recall quan trọng | **Source fact or data** | LLM sinh commands vẫn được dùng |
| Full GrepRAG tăng thêm | CrossCodeEval DeepSeek Python EM 42.29/ES 79.66; Java 43.15/80.07 | pp. 15–16, Table 5 | BM25 + dedup bổ sung | **Source fact or data** | So sánh cùng backbone |
| GrepRAG thắng trên repository lớn | RepoEval Python line EM 44.90; API 40.70; Java line 43.65; API 45.67 trong cấu hình DeepSeek | p. 15, Table 5 | Khả năng chống noise | **Source fact or data** | Dataset có cấu hình riêng |
| Dedup quan trọng hơn rerank đơn thuần | Python DeepSeek: naive 38.61, w/o dedup 39.12, w/o BM25 41.93, full 42.29 | p. 16, Table 6 | Information density là bottleneck | **Source fact or data** | BM25 và dedup có interaction |
| Student KD gần teacher và nhanh | Distilled Qwen3-0.6B: Python line EM 44.95, time 1.93s; Java 44.95, 1.69s | p. 18, Table 8 | KD cho retrieval command generator | **Source fact or data** | So sánh latency phụ thuộc môi trường |
| Query command generation là overhead | ripgrep vật lý cỡ milliseconds, nhưng LLM generation có thể tốn đáng kể | pp. 16–18 | Lý do cần distill | **Author's stated position** | Student vẫn cần benchmark CPU/GPU riêng |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Đơn giản, interpretable và repository thay đổi không cần index rebuild.
- Exact identifier retrieval bổ sung tốt cho dense/graph retrieval.
- Dedup theo physical interval giải quyết redundancy và definition-before-use order.
- Có minh chứng rõ ràng cho offline KD vào model nhỏ.

### Giới hạn

- Implicit dependency như inheritance không có lexical identifier thì khó recall.
- Generic identifiers như init/config/run gây ambiguity.
- LLM command generator có thể bỏ sót keyword hoặc hallucinate query.
- Rerank/fusion hiện theo line interval, chưa theo AST units.
- Benchmark contamination không thể loại trừ hoàn toàn.

## 7. Hàm ý cho framework AST/KD

1. Giữ lexical branch như recall guard, không phụ thuộc dense teacher duy nhất.
2. Thay line-interval fusion bằng AST interval/parent-aware fusion.
3. Distill command/query proposal bằng teacher lớn theo kiểu CodeRAG: offline, consensus-filtered, student nhỏ.
4. Dùng student query generator hoặc keyword head để tạo enhanced query cho AST retriever.
5. Explicitly test implicit dependency subset riêng; grep không thể là full solution.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** GrepRAG là retrieval hiệu quả và latency thấp.
- **Source fact or data:** post-processing và distilled 0.6B đều có bảng ablation/latency.
- **Reasoned inference:** lexical AST node retrieval + dense KD teacher có thể bổ sung lẫn nhau tốt hơn dense-only AlignCoder.
- **Unverified:** một student UniXcoder có thể hấp thụ lợi thế exact identifier nếu chỉ distill embedding hay không.

## 9. Câu hỏi recall/transfer

1. Vì sao BM25 không tốt bằng grep ở coarse retrieval nhưng tốt ở candidate reranking?
2. Dedup của GrepRAG merge theo thông tin nào?
3. Student distilled học core keyword hay full command?
4. Hướng AST tương ứng của dedup nên dùng node span hay dependency graph?

