# ICAR CCEval — Các hướng cần sửa

## 1. Ưu tiên cao: sửa cách cắt context

### Retriever query

Query của retriever phải luôn giữ được phần code gần cursor. Đồng thời cần giữ một phần intent hints ở đầu query, thay vì chỉ đổi toàn bộ tokenizer sang left truncation rồi có nguy cơ mất hints.

### Generator input

Thống nhất cách đóng gói context giữa teacher forcing và generation. Hai đường này phải nhìn thấy cùng vùng code gần cursor để utility dùng khi train phản ánh đúng input lúc sinh prediction.

### Các đường inference

Áp dụng cùng một quy tắc context cho scalar generation, batch generation, draft generation, nhánh có retrieval và nhánh không retrieval.

## 2. Ưu tiên cao: đồng bộ train và eval

Train và eval cần dùng cùng context budget. Không nên train với 4096 token nhưng mặc định eval ở 3072 token nếu kết quả được so trực tiếp với baseline.

Các tham số quan trọng như context length, output length, truncation policy, tokenizer và postprocessing cần được lưu cùng checkpoint và thể hiện rõ trong kết quả eval.

## 3. Ưu tiên cao: bổ sung logging để xác định nguyên nhân

Trong kết quả từng mẫu cần ghi đủ thông tin để biết:

- query hoặc context có bị truncate không;
- model thực tế nhận bao nhiêu token;
- gate chọn retrieve hay skip;
- retriever chọn những chunk nào và score tương ứng;
- context được pack cuối cùng có còn vùng gần cursor không.

Metric cần được chia theo context dài/ngắn, retrieval/no-retrieval và truncated/not-truncated. Việc này giúp tách lỗi context khỏi chất lượng retriever hoặc generator.

## 4. Đồng bộ cost-aware query với gate

Gate label, gate features và retrieval context phải được tạo từ cùng một query.

Hiện tại cost-aware mode có thể dùng enhanced query để tạo utility nhưng gate lại nhìn static query. Cần chọn rõ một trong hai hướng:

- gate quyết định trước, rồi mới enhancement;
- hoặc enhancement trước và gate dùng chính enhanced query cùng các retrieval signals tương ứng.

Không nên trộn label từ enhanced retrieval với features từ static retrieval.

## 5. Tận dụng và hiệu chỉnh tín hiệu retriever cho gate

Gate nên sử dụng các tín hiệu như top-1 score, top-1 margin, entropy, score variance, identifier overlap và context cost để ước lượng retrieval có đáng dùng hay không.

Các tín hiệu này cần được đối chiếu với utility thực tế, tức mức cải thiện NLL khi có retrieval so với không retrieval. Threshold retrieve/skip nên được calibration trên validation set thay vì cố định mặc định.

Cần so sánh learned gate với hai baseline `always_retrieve` và `always_skip` để xác định gate đang giúp hay làm giảm chất lượng.

## 6. Cải thiện tín hiệu học của LiPO

LiPO hiện chấm cả top-k context bằng mean similarity, nên khó xác định chunk nào hữu ích và chunk nào gây nhiễu.

Hướng xử lý:

- đo contribution của từng chunk bằng leave-one-out utility;
- thử `top_k=1` làm baseline;
- cân nhắc weighted aggregation thay vì mean đều;
- bổ sung tín hiệu hoặc loss ở cấp chunk;
- dùng dynamic top-k để tránh đưa thêm chunk nhiễu.

Chỉ nên thay đổi objective LiPO sau khi đã sửa query truncation.

## 7. Kiểm tra retriever sau fine-tuning

So sánh retriever sau LiPO với retriever frozen và BM25 trên cùng validation set.

Không chỉ nhìn metric generation cuối cùng. Cần kiểm tra Recall@k, MRR, retrieval score/margin, utility NLL và tỷ lệ retrieved context thực sự có ích.

Nếu retriever score tăng nhưng Recall@k hoặc utility giảm, objective đang tối ưu sai proxy. Nếu Recall@k tăng nhưng generation không tăng, lỗi nhiều khả năng nằm ở gate, context packing hoặc generator.

## 8. Kiểm tra độ ổn định của BF16

Chạy ablation giữa retriever BF16 hiện tại và cấu hình FP32 hoặc mixed precision có FP32 optimizer state.

Nếu BF16 làm loss không ổn định, gradient/update quá nhỏ hoặc Recall@k giảm, nên chuyển cách train precision. Nếu không có khác biệt đáng kể thì không xem BF16 là nguyên nhân chính.

## 9. Đánh giá đúng full ICAR

Setting hiện tại dùng `ADAPTER_TYPE=none`, nên không phải full ICAR có soft prompt. Sau khi sửa các lỗi pipeline, cần chạy riêng cấu hình full:

```text
EXPERIMENT_MODE=intent_main
ADAPTER_TYPE=soft_prompt
RESAMPLE_TRAIN_EACH_EPOCH=1
```

Cần tách đóng góp của từng thành phần bằng ablation:

- frozen retriever so với LiPO retriever;
- always-retrieve so với learned gate;
- raw/static query so với intent query;
- soft prompt off so với on;
- intent mode so với cost-aware mode.

## 10. Thứ tự thực hiện đề xuất

1. Sửa truncation ở retriever query và generator input.
2. Đồng bộ context budget train/eval.
3. Eval lại checkpoint hiện tại và phân tích theo nhóm mẫu.
4. Đồng bộ query, label và features của gate.
5. Calibration gate bằng tín hiệu retriever.
6. So sánh frozen retriever với retriever sau LiPO.
7. Cải thiện objective LiPO nếu retriever vẫn học lệch.
8. Kiểm tra BF16 bằng ablation.
9. Chạy lại full ICAR có soft prompt.

## Kết luận

Hai hướng cần ưu tiên nhất là sửa truncation và đồng bộ context train/eval. Sau đó mới dùng tín hiệu retriever để kiểm tra gate và LiPO. Chỉ nên đánh giá novelty sau khi pipeline context, retrieval và gate đã nhất quán và full soft-prompt setting thực sự được chạy.
