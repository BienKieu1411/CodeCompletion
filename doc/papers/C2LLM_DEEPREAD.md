# DeepRead report: C2LLM

## 0. Nguồn và trạng thái trích xuất

- Paper: C2LLM Technical Report: A New Frontier in Code Retrieval via Adaptive Cross-Attention Pooling.
- File nguồn: [2512.21332v1.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/2512.21332v1.pdf).
- Phạm vi: toàn bộ 10 trang.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

C2LLM là dense code retriever dựa trên Qwen2.5-Coder với PMA pooling. Thay vì lấy EOS hoặc mean pooling, một learnable query attend tới toàn bộ hidden states và tạo một embedding thích nghi. Model train trên khoảng 3M ví dụ nhiều task bằng in-batch contrastive loss với hard negatives. Kết quả MTEB-Code cao, nhưng paper không có repository-level completion experiment, không có AST chunking, query-target alignment hay knowledge distillation. Giá trị của paper với hướng hiện tại là một teacher/backbone embedding mạnh hoặc nguồn thiết kế pooling, không phải framework completion hoàn chỉnh.

## 2. Luận điểm trung tâm

**Author's stated position:** PMA pooling cho phép code embedding sử dụng toàn bộ sequence một cách thích nghi, vượt EOS/mean pooling và tạo retriever code mạnh ở nhiều kích thước.

## 3. Cây lập luận

1. Causal code LM không có EOS representation đủ tốt cho retrieval.
2. Mean pooling không biết token nào quan trọng.
3. Learnable query attention có thể chọn token phù hợp task.
4. Contrastive training với hard negatives tạo embedding phân biệt.
5. Mô hình nhỏ nhưng pooling tốt có thể cạnh tranh model lớn.

## 4. Phương pháp

### 4.1. PMA pooling

Với hidden states H và learnable query q:

- Q = qWq, K = HWk, V = HWv.
- O = softmax(QK transpose / sqrt(d))V.
- residual + layer normalization, sau đó projection/ReLU/layer normalization tạo embedding E.

PMA dùng một query học được để pool toàn bộ hidden states và có thể giảm dimension.

### 4.2. Model và training

- Base: Qwen2.5-Coder-0.5B-Instruct và 7B-Instruct.
- PMA 32 heads; LoRA rank 64, alpha 32; FlashAttention2.
- Data khoảng 3M từ CodeSearchNet, APPS, CodeFeedback, CodeEditSearch, CosQA, StackOverflowQA, SyntheticText2SQL và CodeTransOcean.
- 3 epochs, learning rate 1e-4, max sequence 1024.
- Global in-batch contrastive loss, K = 7 hard negatives, temperature 0.05.
- Batch grouped theo dataset và programming language.
- Weighted merge của 4 checkpoints.

### 4.3. Retrieval result

- C2LLM-7B MTEB-Code average 80.75, rank 1 trong bảng paper.
- C2LLM-0.5B average 75.46, rank 6 và cao nhất trong nhóm dưới 1B theo bảng.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| PMA model đạt retrieval score mạnh | C2LLM-7B 80.75 MTEB-Code; vượt các baseline được báo cáo trong bảng | pp. 5–7 | Cho thấy embedding backbone chất lượng | **Source fact or data** | MTEB-Code không phải repo completion |
| PMA giúp model nhỏ | 0.5B 75.46 và vượt các baseline dưới 1B được bảng so sánh | pp. 5–7 | Phù hợp distill/deployment | **Source fact or data** | So sánh training data/compute có thể khác |
| Training không phải KD | Objective là in-batch contrastive với hard negatives | pp. 3–4 | Phân biệt với hướng hiện tại | **Source fact or data** | Không có teacher distribution |
| Không có bằng chứng AlignCoder gain | Paper không báo cáo RepoEval/CrossCodeEval hoặc generator integration | toàn paper | Giới hạn phạm vi kết luận | **Source fact or data** | Có thể cần tự triển khai |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Pooling phù hợp với causal code LM.
- Hard negatives và domain-balanced batching thực dụng.
- Có model 0.5B tiềm năng làm student/teacher rẻ.

### Giới hạn

- Benchmark không đo repository-level target utility.
- Embedding semantic không đảm bảo exact identifier/AST dependency recall.
- Data mixture có thể chứa overlap với downstream code.
- Không có analysis của chunk size, parent context hoặc retrieval latency trên repo lớn.

## 7. Hàm ý cho KD framework

1. C2LLM-7B có thể làm teacher representation cho AST chunks; UniXcoder hoặc C2LLM-0.5B làm student.
2. Không distill chỉ cosine teacher–student; thêm target utility và lexical identifier objective.
3. Student pooling nên có PMA hoặc AST-type-aware pooling.
4. Đánh giá teacher ranking trên CrossCodeEval trước khi cam kết dùng teacher.
5. Cần so sánh trực tiếp C2LLM teacher với Jina embedding và ReVELA trên cùng corpus.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** PMA pooling tạo code retriever mạnh.
- **Source fact or data:** MTEB-Code scores được báo cáo rõ.
- **Reasoned inference:** C2LLM là ứng viên teacher/backbone, không phải code generator cho framework.
- **Unverified:** C2LLM teacher có target alignment tốt hơn Jina trên AST repository chunks hay không.

## 9. Câu hỏi recall/transfer

1. PMA khác EOS/mean pooling ở đâu?
2. Vì sao MTEB-Code score không đủ để claim vượt AlignCoder?
3. Nếu distill C2LLM vào UniXcoder, label nào cần thêm để giữ exact identifier?
4. Hard negatives của C2LLM có thể lấy từ AST sibling/dependency nào?

