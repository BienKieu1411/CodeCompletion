# DeepRead report: ReACC

## 0. Nguồn và trạng thái trích xuất

- Paper: ReACC: A Retrieval-Augmented Code Completion Framework.
- File nguồn: [ReACC.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/ReACC.pdf).
- Phạm vi: toàn bộ 14 trang; phân tích chính ở trang 1–9 và appendix.
- Trạng thái: PDF có text layer; không cần OCR.

Nhãn bằng chứng:

- **Author's stated position** — luận điểm của tác giả.
- **Source fact or data** — công thức, thiết lập hoặc số liệu có thể truy về paper.
- **Reasoned inference** — suy luận phục vụ nghiên cứu hiện tại.
- **Unverified** — chưa được paper kiểm chứng.

## 1. Tổng hợp

ReACC là framework RAG sớm cho repository-level code completion. Nó tách hệ thống thành hybrid retriever và decoder-only generator. Retriever kết hợp dense similarity với BM25, được train bằng contrastive learning trên các fragment đã biến đổi; generator được fine-tune để dùng fragment retrieved cùng in-file context. Đóng góp quan trọng nhất không phải một loss mới mà là nhận diện ba lệch pha: query unfinished code ngắn, fragment retrieved không cùng dạng với target, và API/identifier quan trọng dễ bị mất nếu chỉ dùng token-level augmentation.

## 2. Luận điểm trung tâm

**Author's stated position:** Có thể cải thiện code completion bằng cách huấn luyện retriever dense với positive code fragment được tạo qua identifier renaming, dead-code insertion và query truncation, rồi kết hợp dense retrieval với BM25 để đưa context repository vào code LM.

## 3. Cây lập luận

1. Repository context giúp completion, nhưng không thể đưa toàn bộ repository vào LM.
   - Cần retriever để chọn fragment.
2. Dense retriever hiểu semantic nhưng có thể bỏ lỡ lexical/API match.
   - BM25 bổ sung matching theo identifier.
3. Query unfinished code và target/full fragment khác nhau.
   - Random truncation tạo query-prefix/positive-fragment pairs.
   - Fragment alignment làm retrieved fragment gần hơn với dạng query-target.
4. Retriever cần positive/negative training.
   - Identifier renaming và dead-code insertion tạo các biến thể giữ semantics.
   - In-batch negatives dùng cho InfoNCE.
5. Generator được train riêng để đọc retrieved code.
   - Retriever và generator được tối ưu stage-wise, chưa end-to-end.

## 4. Phương pháp

### 4.1. Dữ liệu và chunk

- Code corpus được chia thành các fragment có độ dài bằng nhau.
- Query là phần code chưa hoàn tất; target là continuation.
- API usage sequence được nối vào source/unfinished code để nhấn mạnh API cần dự đoán.

### 4.2. Dense retriever

- Hai encoder bidirectional tied, khởi tạo từ GraphCodeBERT.
- Embedding lấy từ [CLS], similarity là dot product.
- Objective là InfoNCE với in-batch negatives.
- Retriever continual pre-training trên khoảng 1.6M Java methods và 1.2M Python functions từ CodeSearchNet.
- Cấu hình được báo cáo: batch 256, learning rate 5e-5, 30 epochs cho mỗi ngôn ngữ.

### 4.3. Hybrid retrieval

Score cuối được mô tả là:

hybrid_score = dense_similarity + alpha * BM25, với alpha = 0.9 trong thiết lập chính.

Paper cho thấy BM25 không chỉ là fallback: ở một số benchmark nó cạnh tranh hoặc tốt hơn dense retrieval vì code completion phụ thuộc mạnh vào exact identifier/API match.

### 4.4. Data augmentation và alignment

- Identifier renaming: dùng GraphCodeBERT MLM để đề xuất top-10 thay thế, nhưng bảo toàn API/built-in lexical information.
- AST dead-code insertion: chèn code không ảnh hưởng semantics để tăng đa dạng positive.
- Query truncation: cắt ngẫu nhiên phần cuối/original code để mô phỏng unfinished query.
- Fragment alignment: biến fragment retrieved thành dạng gần với query-target continuation hơn trước khi đưa cho generator.

### 4.5. Generator

- Decoder-only CodeGPT-adapted.
- Generator được fine-tune sau khi retriever đã train.
- Input gồm in-file context, query và retrieved fragments.
- Không có online retrieval update hoặc generator–retriever joint optimization.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| Hybrid retrieval cải thiện completion | PY150: ReACC-hybrid PPL 2.311, EM 46.26, ES 74.41; Java: PPL 3.327, EM 30.70, ES 64.73 | pp. 6–7, Table 3 | So sánh với CodeGPT-adapted và các baseline | **Source fact or data** | Benchmark cũ, không phải RepoEval/CrossCodeEval hiện đại |
| Dense và hybrid tốt trên CodeNet Python | Dense EM 64.21/ES 84.57; hybrid EM 64.74/ES 84.93 | p. 8, Table 5 | Cho thấy retrieval hỗ trợ generator | **Source fact or data** | Chỉ Python và thiết lập CodeNet |
| BM25 vẫn quan trọng | BM25 có thể ngang hoặc hơn dense trên một số CodeXGLUE split | pp. 6–8 | Ủng hộ lexical branch | **Author's stated position** | Không chứng minh BM25 luôn tốt hơn |
| Augmentation có ích | Bỏ identifier renaming, dead-code insertion, API sequence hoặc query truncation đều làm giảm EM/ES trên ablation | p. 9, Table 6 | Chứng minh từng nguồn diversity/context có đóng góp | **Source fact or data** | Chưa tách riêng tác động của từng kiểu chunking |
| Stage-wise training là điểm yếu | Retriever và generator được train tách, không có gradient từ final completion quay về retriever | pp. 3–5 | Giới hạn khả năng target alignment | **Reasoned inference** | Có thể đủ tốt nếu retriever positive đúng |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Hybrid dense + lexical phù hợp với identifier-heavy code.
- Data augmentation đơn giản, rẻ và không cần human labels.
- Có explicit API sequence và fragment alignment.
- Cho thấy retriever có thể tăng cả EM và ES, không chỉ retrieval recall.

### Giới hạn

- Chunk fixed-length không bảo toàn function/class/AST boundary.
- Positive được tạo từ cùng fragment nhưng không biểu diễn dependency graph hoặc target intent sâu.
- Retriever không được tối ưu trực tiếp theo final generator loss.
- Candidate filtering và context ordering còn đơn giản.
- Kết quả cũ không đủ để suy ra hiệu quả trên AlignCoder/CrossCodeEval mới.

## 7. Hàm ý cho framework hiện tại

1. Giữ BM25/identifier branch làm recall safety net, kể cả khi dùng AST chunking.
2. Dùng AST node làm đơn vị positive/augmentation thay vì fixed line fragment.
3. Query nên gồm unfinished AST neighborhood và target-like signals, không chỉ raw prefix.
4. Nếu dùng KD, teacher cần distill ranking/utility của context cho student retriever; chỉ distill embedding similarity là chưa đủ.
5. ReACC là baseline nền, không phải lời giải cho query-target misalignment và context interaction.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** hybrid retrieval + generator fine-tuning cải thiện repository code completion.
- **Source fact or data:** BM25, dense retrieval, augmentation và API sequence đều có số liệu ablation.
- **Reasoned inference:** AST chunking hiện tại có thể kế thừa hybrid score và augmentation của ReACC nhưng thay fixed windows bằng syntax-preserving units.
- **Unverified:** joint KD từ một teacher embedding lớn có vượt AlignCoder hay không; paper này không thử.

## 9. Câu hỏi recall/transfer

1. Vì sao dense retrieval không đủ cho code completion có nhiều identifier?
2. InfoNCE của ReACC dùng positive và negative nào?
3. Nếu thay fixed fragment bằng AST subtree, phần nào của pipeline cần sửa?
4. Làm thế nào kiểm tra generator thực sự dùng retrieved context thay vì chỉ dựa vào in-file prefix?

