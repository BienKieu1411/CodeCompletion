# DeepRead report: RLCoder

## 0. Nguồn và trạng thái trích xuất

- Paper: RLCoder: Reinforcement Learning for Repository-Level Code Completion.
- File nguồn: [RLCoder.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/RLCoder.pdf).
- Phạm vi: toàn bộ 13 trang; phân tích chính ở trang 1–10.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

RLCoder train retriever bằng feedback của code generator/evaluator thay vì query–candidate labels. Repository được tạo thành natural candidates bằng Split-Aggregate dựa trên blank lines. Với mỗi candidate, DeepSeekCoder-1B tính weighted perplexity của target completion khi có candidate; candidate tốt nhất nhận reward 1, còn lại 0. Retriever UniXcoder được cập nhật bằng objective kiểu policy gradient. Stop signal là empty candidate để model có thể quyết định không retrieve. Phương pháp chứng minh retriever có thể học target utility, nhưng reward winner-take-all, evaluator coupling và natural line chunk là các điểm yếu lớn.

## 2. Luận điểm trung tâm

**Author's stated position:** Có thể học repository retriever không cần labeled query–candidate pairs bằng cách dùng weighted target perplexity từ một code LM làm reward.

## 3. Cây lập luận

1. Label retrieval thủ công khó và tốn chi phí.
2. Generator perplexity có thể đánh giá candidate trực tiếp theo target.
3. Natural chunk giữ code continuity tốt hơn fixed window.
4. Retriever nhận reward từ candidate tốt nhất.
5. Empty candidate cho phép học “không cần retrieval”.
6. Retriever tốt hơn BM25/UniXcoder và tăng completion metrics.

## 4. Phương pháp và objective

### 4.1. Candidate construction

- Dữ liệu gồm 10,000 Python/Java GitHub repositories tạo trước tháng 3/2023.
- Dependency graph dựa trên import; target file được mask khỏi candidate pool.
- Split-Aggregate: split file theo blank lines thành mini-block; ghép adjacent blocks đến threshold T; block quá dài được split.
- Candidate cuối không phải AST node nên vẫn có thể cắt qua syntax/semantic unit.

### 4.2. Weighted perplexity reward

Với unfinished query x, candidate c và target y:

PPL(y | x,c) = exp( - 1/N * sum_i log P(y_i | x,c,y_<i) ).

Weighted version:

PPLw = exp( - 1/sum_i w_i * sum_i w_i log P(y_i | x,c,y_<i) ).

Weights:

- w_first cho k token đầu.
- w_api cho API/identifier token.
- 1 cho token còn lại.

Candidate c_i nhận reward 1 nếu PPLw(c_i) nhỏ nhất trong candidate set; ngược lại 0. Objective được paper viết:

L = sum_i reward(c_i | x,C) * log p(c_i | x,C).

Đây là policy-gradient-like update cho retriever, không phải supervised pairwise ranking.

### 4.3. Stop signal và inference

- Empty candidate được thêm vào codebase.
- Nếu empty candidate được chọn, không lấy repository context.
- Generator nhận các candidate được retriever chọn sau khi cắt tại stop signal.
- Generator không được train trong RLCoder; chỉ retriever được train.

### 4.4. Training

- Retriever khởi tạo UniXcoder.
- Evaluator: DeepSeekCoder-1B.
- 20 epochs, 2000 samples/epoch, batch 16, learning rate 5e-5, 2 A100.
- Early stopping.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| RLCoder tăng completion trên CrossCodeEval | DeepSeekCoder-7B Python EM/ES 30.28/74.42 so với RepoCoder 26.98/72.96 | pp. 7–8 | End-to-end gain | **Source fact or data** | Chỉ báo cáo trên benchmark cụ thể |
| RL retriever hơn retriever không train | Python EM: no retrieval 9.46, BM25 18.31, UniXcoder 23.30, UniXcoder-SFT 27.28, RL 30.28 | p. 8, Table III | Cho thấy target-PPL feedback có giá trị | **Source fact or data** | Reward/evaluator là DeepSeekCoder-1B |
| Stop signal có ích | Trên GitHubEval, bỏ stop signal làm giảm EM trung bình khoảng 7.69% | pp. 8–9, Table V | Retrieval không phải lúc nào cũng cần | **Source fact or data** | Dataset khác với CrossCodeEval |
| Natural candidate có đóng góp | Bỏ NC: Python EM 29.31 vs full 30.28 | p. 9, Table IV | Chunk continuity có ích | **Source fact or data** | Split-Aggregate chưa phải AST |
| Reward winner-take-all là giới hạn | Chỉ candidate có PPL tốt nhất nhận reward 1 | pp. 4–5 | Gradient ít thông tin và dễ bất ổn | **Reasoned inference** | Paper không so sánh reward mềm |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Không cần query–candidate annotation.
- Reward trực tiếp gắn với target completion.
- Stop action giúp tránh context không cần thiết.
- Tương thích với nhiều generator vì generator không bị cập nhật.

### Giới hạn

- PPL reward phụ thuộc evaluator; model bias có thể truyền sang retriever.
- Winner-take-all mất chênh lệch giữa candidate tốt và gần tốt.
- Dù gọi là annotation-free, target code vẫn được dùng offline để tính reward.
- Split theo blank lines chưa bảo toàn AST.
- Objective có coupling với candidate set và sampling.
- Không distill knowledge của generator thành student retriever; inference vẫn dùng retriever đã RL-train.

## 7. Hàm ý cho hướng mới

1. Thay Split-Aggregate bằng AST chunking của người dùng để loại bỏ boundary loss.
2. Dùng reward liên tục từ teacher log-likelihood hoặc teacher embedding utility, thay vì chỉ argmin PPL.
3. Distill teacher utility vào retriever student bằng pointwise regression/listwise soft target, không cần tạo pairwise.
4. Giữ stop/no-retrieval head vì RLCoder chứng minh retrieval luôn-on có thể gây hại.
5. Đánh giá theo evaluator độc lập và unit test/compile nếu có, tránh một code LM tự chấm chính nó.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** RL từ weighted PPL là cách học retriever không cần labeled pairs.
- **Source fact or data:** RL retriever vượt BM25, UniXcoder và UniXcoder-SFT trong bảng so sánh của paper.
- **Reasoned inference:** KD soft utility trên AST candidates có thể giữ lợi ích target-aware mà ổn định hơn winner-take-all RL.
- **Unverified:** student UniXcoder được distill từ teacher embedding lớn có vượt RLCoder/AlignCoder trên cùng benchmark hay không.

## 9. Câu hỏi recall/transfer

1. Tại sao PPL của target có thể làm reward cho retriever?
2. Stop signal được biểu diễn như thế nào?
3. Điều gì mất đi khi biến reward thành 0/1 winner-take-all?
4. Thiết kế soft distillation target nào thay thế reward này mà không tạo pairwise?

