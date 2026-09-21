# DeepRead report: CAST / AST Chunking

## 0. Nguồn và trạng thái trích xuất

- Paper: CAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree.
- File nguồn: [AST_Chunking.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/AST_Chunking.pdf).
- Phạm vi: toàn bộ 11 trang, gồm main paper và appendix.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

CAST đề xuất split-then-merge chunking trên AST. Nếu node vừa budget thì giữ nguyên; nếu quá lớn thì đệ quy xuống children; sau đó greedily merge adjacent siblings khi còn vừa budget. Budget tính theo non-whitespace characters, không theo line. Cách này giữ syntactic integrity, density và khả năng reconstruction. Paper không giải quyết target alignment hay retriever training, nhưng cung cấp bằng chứng trực tiếp rằng chunking theo AST tốt hơn fixed line/token trên RepoEval, CrossCodeEval và SWE-bench.

## 2. Luận điểm trung tâm

**Author's stated position:** Chunking code theo cấu trúc AST với split-then-merge cho context giàu thông tin hơn và giúp retrieval/generation tốt hơn chunking fixed-size.

## 3. Cây lập luận

1. Fixed line/token chunk có thể cắt function, class hoặc control block.
2. Chunk syntactically coherent giúp retriever và generator đọc đúng đơn vị.
3. Split-only tạo mảnh vụn, density thấp.
4. Merge adjacent siblings khôi phục context dày nhưng vẫn hợp syntax.
5. Structure-preserving chunks cải thiện cả retrieval metrics và pass rate.

## 4. Phương pháp

### 4.1. Recursive split-then-merge

1. Parse file bằng tree-sitter.
2. Nếu node không vượt max chunk size, giữ toàn node.
3. Nếu vượt budget, đệ quy split children.
4. Greedily merge các sibling liên tiếp nếu tổng non-whitespace characters chưa vượt budget.
5. Lưu đủ thông tin để reconstruct verbatim.

Mục tiêu thiết kế:

- syntactic integrity;
- high density;
- language invariance;
- plug-and-play với retriever/generator.

### 4.2. Evaluation

- Datasets: RepoEval, CrossCodeEval, SWE-bench.
- Retrievers: BGE, GIST, CodeSage, Jina-v2-code.
- Generators: StarCoder2-7B, CodeLlama-7B, Claude 3.7, Gemini 2.5.
- Default max chunk size 2000 chars; max context 4000 cho RepoEval/SWE-bench và 10000 cho CrossCodeEval; top-5.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| CAST hơn fixed chunk trên RepoEval | CodeSage nDCG khoảng 85.1 vs 83.0; recall 83.9 vs 82.1; StarCoder2 Pass@1 73.2 vs 67.6; CodeLlama 72.1 vs 66.5 | pp. 5–6, Table 1 | Retrieval gain truyền sang generation | **Source fact or data** | Cấu hình exact phụ thuộc bảng |
| Jina-v2-code cũng gain | nDCG@5 87.1 vs 86.8; recall 87.9 vs 84.9; StarCoder2 pass 80.7 vs 75.1 | appendix | Không chỉ CodeSage | **Source fact or data** | Dense retriever chưa được train cho AST chunks |
| CCEval gain đa ngôn ngữ | StarCoder2 Python EM 29.1 vs 24.8; Java 30.9 vs 28.1; C# 28.3 vs 25.5; TypeScript 13.7 vs 11.9 | appendix | AST chunking generalizes | **Source fact or data** | Trích từ appendix table |
| Merge là cần thiết | Split-only làm nDCG/Pass@1 giảm mạnh so với split-then-merge | pp. 7–8 | Không đủ chỉ parse rồi split | **Source fact or data** | Merge heuristic vẫn dựa budget |
| Chunk size có optimum | 1000/1500/2000/2500/3000 cho nDCG 69.0/68.4/71.1/72.3/69.4 và pass 43.4/45.8/51.7/50.1/51.2 trong ablation | p. 8 | Budget cần tune | **Source fact or data** | Optimum phụ thuộc parser/retriever |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Khắc phục trực tiếp line chunk boundary loss.
- Chunks có thể reconstruct, tiện cache và debug.
- Không cần retrain generator.
- Có validation trên nhiều ngôn ngữ/task.

### Giới hạn

- AST parser không hoàn hảo và cần grammar cho từng ngôn ngữ.
- Max chunk size vẫn là heuristic.
- Không có target-aware selection hoặc dependency expansion.
- Node lớn có thể bị split thành children mất ngữ cảnh parent.
- Structure không đồng nghĩa relevance; chunk hợp syntax vẫn có thể không cần cho target.

## 7. Hàm ý cho framework người dùng

1. CAST là bằng chứng nền cho AST chunking, nhưng cần thêm target-aligned supervision để vượt AlignCoder.
2. Dùng recursive split-then-merge cho base chunks; lưu parent chain để late expansion.
3. Tạo hai view của một AST unit: retrieval view ngắn và comprehension view mở rộng.
4. Teacher KD nên chấm AST chunks trên query-target utility; không chỉ cosine embedding.
5. Rất cần ablation: line vs token vs AST, AST-only vs AST+KD, AST+KD vs AlignCoder.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** AST split-then-merge tốt hơn fixed chunking.
- **Source fact or data:** paper có gain trên RepoEval/CrossCodeEval/SWE-bench và split-only ablation.
- **Reasoned inference:** AST chunking là phần nên giữ chắc chắn trong framework mới; phần còn thiếu là learning target alignment.
- **Unverified:** AST + KD có thể vượt AlignCoder trên repository benchmark hiện tại.

## 9. Câu hỏi recall/transfer

1. Vì sao split-only kém split-then-merge?
2. Budget của CAST tính theo gì?
3. AST chunking khắc phục điểm yếu nào của AlignCoder nhưng không khắc phục điểm nào?
4. Parent chain nên được dùng ở retrieval hay comprehension stage?

