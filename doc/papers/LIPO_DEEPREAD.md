# DeepRead report: LiPO

## 0. Nguồn và trạng thái trích xuất

- Paper: LiPO: Listwise Preference Optimization through Learning-to-Rank.
- File nguồn: [LiPO.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/LiPO.pdf).
- Phạm vi: toàn bộ 17 trang; phân tích chính ở trang 1–10 và appendix 12–17.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

LiPO đặt preference optimization dưới góc nhìn learning-to-rank. Mỗi prompt có một list K response và mỗi response có label/score; policy score là log-ratio giữa policy và reference. DPO, SLiC, DPOPL/PRO trở thành các ranking loss khác nhau. LiPO-lambda dùng LambdaLoss: mỗi preference pair được weighted bởi gain difference và rank-discount difference, trong đó permutation động lấy từ model score. Paper cho thấy LiPO-lambda tốt hơn DPO variants trong summarization/dialogue, nhưng toàn bộ bài vẫn thuộc preference optimization và công thức lõi vẫn phân rã thành weighted pair contributions. Vì người dùng đã loại LiPO/pairwise, paper này nên dùng như nguồn phân tích loss chứ không làm hướng chính.

## 2. Luận điểm trung tâm

**Author's stated position:** Listwise preference data nên được tối ưu bằng learning-to-rank objective phù hợp; LambdaLoss sử dụng label values và listwise permutation tốt hơn các pairwise/list-MLE objective.

## 3. Cây lập luận

1. Preference thường có nhiều response cho cùng prompt.
2. DPO/SLiC xử lý list bằng các pair, làm mất permutation/list context và score magnitude.
3. LiPO định nghĩa policy scores từ log-ratio với reference.
4. Các ranking loss khác nhau tạo các LiPO variants.
5. Lambda weighting gần mục tiêu DCG và có dynamic permutation.
6. LiPO-lambda tận dụng listwise data tốt hơn.

## 4. Phương pháp và loss

### 4.1. Listwise formulation

Dataset gồm prompt x, responses y_1…y_K và labels psi_1…psi_K. Policy ranking score:

s_i = beta * log( pi_theta(y_i | x) / pi_ref(y_i | x) ).

LiPO tối ưu một ranking loss trên vector s và labels psi.

### 4.2. LiPO-lambda

Loss:

sum trên các cặp psi_i > psi_j của Delta_i,j * log(1 + exp( -(s_i - s_j) )).

Lambda weight:

Delta_i,j = |G_i - G_j| * |1/D(rank_i) - 1/D(rank_j)|.

Thiết lập paper:

- G_i = 2 psi_i - 1.
- D(rank) = log(1 + rank).
- rank_i lấy từ permutation động do model scores.

Về mặt diễn giải, đây là weighted version của DPOBT trên tất cả cặp trong list. Vì vậy nó vẫn không phù hợp với ràng buộc “không pairwise” của framework hiện tại.

### 4.3. Training

- T5-large 770M SFT policy.
- T5-XXL 11B pairwise reward-ranking model.
- Mỗi prompt sample K = 8 response từ SFT policy.
- Reward model so sánh mọi cặp để tạo winning probability matrix rồi aggregate label psi.
- Batch 32, learning rate 2e-5, Adafactor, beta = 0.05.
- Calibration khoảng một ngày trên 32 TPU-v3.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| LiPO-lambda hơn DPO variants | Reddit TL;DR proxy reward 90.60 vs DPOBT 88.52; AutoSxS 68.26 vs 67.09; AnthropicHH 92.60/47.90 vs DPOBT 91.11/44.80 | pp. 6–8, Table 1 | Support cho LambdaLoss | **Source fact or data** | Tasks là summarization/dialogue, không code |
| Label values quan trọng | Constant gain/discount ablations kém DCG weight | p. 6–7, Figure 3 | Gain không chỉ từ rank order | **Source fact or data** | Chỉ trong preference setting |
| List size lớn hơn có lợi | LiPO-lambda hưởng lợi ổn định khi list size tăng | pp. 6–7, Figure 3 | Listwise information hữu ích | **Author's stated position** | Label list tạo qua reward model tốn quadratic comparisons |
| Framework offline | Paper explicitly nghiên cứu offline và nêu online learning là future work | p. 9 | Phân biệt với online KD | **Source fact or data** | Offline không phải online distillation |
| LiPO vẫn dùng pair contributions | LambdaLoss formula sum trên cặp, dù weight listwise-aware | pp. 4–5 | Không phù hợp yêu cầu bỏ pairwise | **Source fact or data** | Không phủ nhận giá trị lý thuyết LTR |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Phân tích rõ pointwise/pairwise/listwise dưới một framework.
- Cho thấy score magnitude và rank position có thể cùng quan trọng.
- LambdaLoss có connection với DCG.

### Giới hạn với repository code completion

- Cần list responses và reward ranking model; pipeline labeling quadratic.
- Experiments không phải code/retrieval.
- Không xử lý AST chunk, dependency hay retrieval latency.
- Loss vẫn pair-decomposable và không giải quyết candidate interaction thật.
- Chỉ offline; chưa chứng minh online stability.

## 7. Hàm ý cho hướng không pairwise

1. Không dùng LiPO-lambda làm loss chính.
2. Có thể lấy ý tưởng “label magnitude quan trọng” để tạo soft scalar target từ teacher.
3. Thay weighted pair loss bằng pointwise regression trên teacher utility:
   - student score của AST chunk;
   - target là normalized teacher utility hoặc teacher probability.
4. Nếu cần phân phối top-k, dùng list-level KL/softmax cross-entropy giữa teacher distribution và student distribution; không materialize pairwise tuples.
5. Đánh giá calibration và NDCG nhưng không dùng pairwise training.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** LambdaLoss là ranking objective tốt cho listwise preference.
- **Source fact or data:** LiPO-lambda thắng baselines trong hai preference tasks.
- **Reasoned inference:** giá trị có thể tái sử dụng là soft label magnitude và metric-aware weighting, không phải pairwise implementation.
- **Unverified:** pointwise teacher-utility regression có đạt cùng gain trong retrieval không.

## 9. Câu hỏi recall/transfer

1. LiPO score s_i được tạo từ policy/reference như thế nào?
2. Lambda weight encode những yếu tố nào?
3. Vì sao LiPO-lambda không đáp ứng yêu cầu bỏ pairwise?
4. Chuyển label magnitude thành scalar KD target ra sao?

