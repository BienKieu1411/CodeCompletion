# DeepRead report: CodeRAG

## 0. Nguồn và trạng thái trích xuất

- Paper: CodeRAG: Finding Relevant and Necessary Knowledge for Retrieval-Augmented Repository-Level Code Completion.
- File nguồn: [CodeRAG.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/CodeRAG.pdf).
- Phạm vi: toàn bộ 11 trang, gồm main paper và appendix.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

CodeRAG gồm ba ý: query probing từ target file, multi-path retrieval và BESTFIT reranking. Query probing nối các fine-grained chunks của current file vào target rồi dùng CodeT5p-220m log-probability để chọn các chunks có khả năng giúp target nhất. Retrieval sau đó kết hợp TF-IDF, dense CodeT5p và dataflow graph. BESTFIT dùng Qwen3-8B chọn snippet hữu ích theo cửa sổ 3 candidates; paper còn distill BESTFIT thành Qwen3-0.6B bằng supervised token-level cross-entropy trên các list có consensus. Đây là ví dụ trực tiếp của offline teacher-to-student distillation cho reranker, không cần pairwise training student, nhưng teacher vẫn là LLM reranker chứ chưa phải final code generator.

## 2. Luận điểm trung tâm

**Author's stated position:** Query có thể được xây dựng bằng log-probability probe, sau đó multi-path retrieval và preference-aligned reranking sẽ tìm context vừa relevant vừa necessary cho repository-level completion.

## 3. Cây lập luận

1. Query unfinished code chưa đủ để biểu diễn context cần thiết.
2. Các chunk hiện tại có thể được probe bằng khả năng dự đoán target của một code LM.
3. Một retriever đơn lẻ bỏ sót lexical, semantic hoặc dataflow evidence.
4. Reranker LLM có thể chọn context hữu ích hơn score similarity.
5. Distill reranker lớn thành model nhỏ để giảm latency.

## 4. Phương pháp

### 4.1. AST knowledge base

Knowledge base gồm function, global variable, class variable và class function units. Paper dùng AST/dataflow nhưng query construction vẫn có tham số fine-grained f lines.

### 4.2. Query probing

- Chia current file thành chunks f lines, bỏ target chunk.
- Nối từng chunk với target.
- CodeT5p-220m sinh m tokens và tính tổng maximum token log-probability.
- Chọn top-g chunks rồi nối với target làm retrieval query.

Thiết lập chính: f = 3, g = 1.

### 4.3. Multi-path retrieval

- TF-IDF sparse.
- CodeT5p-220m dense.
- Dataflow-guided retrieval nếu graph có thông tin.
- Lấy j candidate từ sparse/dense và dataflow; n = 2j + 1 trước rerank.

### 4.4. BESTFIT và distillation

- BESTFIT dùng Qwen3-8B prompt để chọn snippet hữu ích.
- Rerank bằng sliding windows size 3, overlap, heap-sort top-u.
- Single forward cho mỗi comparison/list window.
- Distillation:
  - lấy các list ngẫu nhiên kích thước 2–7;
  - chạy BESTFIT 5 lần;
  - chỉ giữ label nếu consensus ít nhất 4/5;
  - fine-tune Qwen3-0.6B bằng LoRA và token-level cross-entropy.
- Đây là offline distillation: teacher chạy trước để tạo dataset; inference student không cần teacher 8B.

### 4.5. Generator

- Generator completion là CodeGen-350M, SantaCoder-1.1B, StarCoder2-3B hoặc Qwen2.5-Coder-7B.
- Pipeline: query construction → multi-path retrieve → BESTFIT/distilled rerank → generator.
- Reranker không được joint train với generator.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| Full multi-path + rerank tốt nhất | ReccEval Qwen2.5-Coder-7B EM 47.48/ES 70.82; identifier EM/F1 55.47/68.68 | pp. 5–6, Table 1 | End-to-end evidence | **Source fact or data** | Tên benchmark khác RepoEval |
| Dataflow bổ sung sparse+dense | Ablation Qwen EM: sparse 39.89, dense 39.66, d+s 41.95, df+s 44.21, df+s+d44.61, full 47.48 | p. 7, Table 3 | Nhiều evidence path hữu ích | **Source fact or data** | Query/rerank cùng thay đổi |
| Distilled reranker giữ phần lớn quality | Trên 30% eval: Qwen distilled EM/ES 44.34/68.42 so với full teacher reranker 47.48/70.82; StarCoder2 39.88/66.21 | p. 8, Table 4 | KD giảm chi phí với degradation hữu hạn | **Source fact or data** | Chỉ 30% evaluation |
| Query construction là chi phí chính | Pipeline cost 0.23s; query construction 0.14s, sparse 0.002, dense 0.015, dataflow 0.03, distilled rerank 0.06 | p. 9 | Tối ưu student chỉ giải một phần latency | **Source fact or data** | Hardware/setup cụ thể |
| Distillation là offline | Teacher BESTFIT tạo labels trước; student train bằng cross-entropy | pp. 4–5 | Không cần online teacher ở inference | **Source fact or data** | Teacher vẫn cần chạy lúc build data |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Có thiết kế distillation rõ ràng: teacher lớn → student nhỏ.
- Consensus filter giảm noisy teacher labels.
- Kết hợp lexical, dense và dataflow.
- Đo cả quality và latency.

### Giới hạn

- BESTFIT teacher chỉ rerank snippets, chưa trực tiếp đo final completion.
- Query probing dùng line chunks, có thể mất AST boundary.
- Student học imitation của reranker chứ không distill full probability distribution của code generator.
- Teacher inference/offline data generation vẫn đắt.
- Pipeline nhiều stage và nhiều hyperparameters.

## 7. Hàm ý cho framework AST/KD

1. Có thể dùng cấu trúc offline teacher dataset này nhưng thay BESTFIT bằng teacher utility của final generator.
2. Distill score liên tục hoặc xác suất KEEP/EXPAND, không cần tạo pairwise examples.
3. Consensus filtering là bắt buộc để tránh distill hallucination/noisy labels.
4. AST chunks nên thay line probe; query probing có thể chạy trên sibling/parent units.
5. Generator cuối có thể là DeepSeekCoder-1.3B/7B cố định; student retriever/reranker được tối ưu độc lập.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** multi-path retrieval + BESTFIT giúp tìm context cần thiết.
- **Source fact or data:** Qwen3-0.6B distilled reranker giảm latency và giữ quality tương đối gần teacher.
- **Reasoned inference:** offline KD là hướng phù hợp với yêu cầu không pairwise, không online teacher ở inference.
- **Unverified:** distill trực tiếp từ Jina/C2LLM teacher embedding hoặc code generator sẽ tốt hơn distill BESTFIT hay không.

## 9. Câu hỏi recall/transfer

1. Query probing của CodeRAG dùng signal nào?
2. Distilled reranker học bằng loss gì?
3. Consensus 4/5 có tác dụng gì?
4. Nếu distill final completion utility, label student nên là scalar score hay token distribution?

