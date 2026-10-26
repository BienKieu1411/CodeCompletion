# Báo cáo Modal: loss PG‑MSLD so với retriever kiểu RLCoder

**Mục tiêu.** Ghi lại cách huấn luyện loss custom và kết quả eval hiện có trên cùng tập valid 400 mẫu với các checkpoint RLCoder-style, để có thể chia sẻ và kiểm tra lại.

## Kết luận nhanh

Trong lần eval Modal này, retriever huấn luyện bằng loss custom **chưa vượt** các checkpoint RLCoder-style. Trung bình macro trên bốn bộ, PG‑MSLD đạt **22.54% EM / 56.85% ES**; checkpoint RLCoder-style fixed-chunk đạt **29.31% / 61.17%**, còn AST-chunk đạt **27.34% / 59.01%**.

Đây là kết quả quan sát được, nhưng **chưa phải ablation công bằng để quy khác biệt cho loss**: PG‑MSLD được eval với ngân sách đầu vào 1,024 token (512 cross-file), còn các checkpoint RLCoder-style dùng 3,072 token (2,344 cross-file). PG‑MSLD cũng dừng sớm hơn nhiều ở mức trung bình chỉ 3.26 snippet và STOP rate 77%. Vì vậy chưa thể kết luận loss custom tự thân kém hơn; hiện tại kết quả cho thấy cấu hình tổng thể custom chưa cạnh tranh.

## 1. Loss được thử

Tên schema của run: `rlcoder_natural_random2k_slate15_weighted_positive_gain_listwise_kd_v7`. Có thể gọi ngắn là **PG‑MSLD** (*Positive-Gain Masked Stop-Anchored Listwise Distillation*).

### Teacher chấm utility của snippet

Với mỗi task, frozen generator DeepSeek-Coder 1.3B được chấm theo teacher-forcing trên completion thật. Mỗi ngữ cảnh gồm left context cộng một snippet; lớp STOP là prompt không có snippet, dùng marker “No cross-file context is needed”. Điểm generator là weighted, length-normalized token NLL:

\[
L_i = \frac{\sum_t w_{it}\,[-\log p_{\text{DeepSeek}}(y_t\mid x_i,y_{<t})]}{\sum_t w_{it}}.
\]

Token đầu tiên của target có trọng số 5; token dạng identifier có trọng số 3; token còn lại có trọng số 1. Cách này ưu tiên phần mở đầu và identifier của completion nhưng vẫn chuẩn hóa theo tổng trọng số để target dài không tự động bị phạt.

Utility của snippet so với STOP là:

\[
g_i = L_{\text{STOP}} - L_i, \qquad u_i = \operatorname{clip}(g_i,-0.5,0.5).
\]

Do NLL càng thấp càng tốt, utility dương nghĩa là snippet giúp generator hơn so với không retrieval. Chỉ snippet hợp lệ có `u_i >= 0.02` mới được teacher xem là positive. Teacher distribution tạo từ các positive và STOP:

\[
q_i = \operatorname{softmax}(u_i / 0.2), \quad i\in\text{positive};
\qquad q_{\text{STOP}} \text{ dùng logit } 0.
\]

Snippet không positive nhận teacher probability bằng 0; padding bị loại khỏi cả teacher lẫn student normalizer. Nếu không có positive nào, STOP là toàn bộ target distribution.

### Student loss và backprop

UniXcoder tạo logit retrieval cho từng snippet và STOP; logit được scale với hệ số 20. Gọi `p` là softmax của student trên các lớp hợp lệ. Loss listwise là cross-entropy từ teacher distribution sang student:

\[
\mathcal{L}_{KD}=-\frac{1}{B}\sum_{b=1}^{B}\sum_j q_{bj}\log p_{bj}.
\]

Thêm hard anchor để giữ tín hiệu winner/STOP rõ ràng. `j*_b = argmax_j q_bj` là lớp teacher tốt nhất:

\[
\mathcal{L}_{hard}=\frac{1}{B}\sum_{b=1}^{B}-\log p_{b,j^*_b},
\qquad
\boxed{\mathcal{L}=0.75\mathcal{L}_{KD}+0.25\mathcal{L}_{hard}}.
\]

Loss này backprop trực tiếp qua student UniXcoder. Nó **không phải PPO/RRPO, không phải policy gradient và không phải pairwise loss**. Khác với nhãn winner-only, phần listwise giữ mức utility tương đối của nhiều snippet hữu ích; snippet không đạt ngưỡng bị đẩy xuống qua student softmax normalizer; STOP cho phép học khi không cần thêm context.

### Cấu hình train của run

- Student: `microsoft/unixcoder-base`; teacher: `deepseek-ai/deepseek-coder-1.3b-base`, revision `c919139c…` qua vLLM prompt log-probabilities.
- Dataset: `BienKieu/CodeCompletion`, revision `ff5af625…`; artifact `ast_repo_pool_v3`.
- Candidate pool: BM25 top-100; chunk dùng kiểu natural RLCoder blank-line mini-block, tối đa 15 dòng không rỗng.
- Slate lúc train: 15 snippet thật + 1 lớp STOP; không có exploration slot.
- Mỗi epoch chọn 2,000 repo; batch 8, microbatch 2; learning rate `5e-5`.
- Teacher temperature `0.2`; positive utility margin `0.02`; utility clip `±0.5`; hard-anchor weight `0.25`.
- Modal GPU: A100 80 GB. Run dừng do hard time limit sau `8,186 s` (~2 giờ 16 phút), checkpoint ghi `next_epoch=9`, `2,480` updates. Không xem đây là 20 epoch hoàn tất.

## 2. Kết quả eval Modal

Cùng valid Parquet gồm 400 mẫu: 100 CCEval Java, 100 CCEval Python, 100 RepoEval API, 100 RepoEval Line. Cả ba eval dùng cùng generator và revision. EM/ES dưới đây là giá trị ghi trong artifact eval; macro là trung bình số học không trọng số của bốn bộ.

| Bộ eval | PG‑MSLD custom EM / ES | RLCoder-style fixed EM / ES | RLCoder-style AST EM / ES |
|---|---:|---:|---:|
| CCEval Java | 14.00% / 60.89% | **18.00% / 61.74%** | **23.00% / 62.20%** |
| CCEval Python | 11.00% / 47.42% | **24.00% / 57.73%** | 21.00% / 55.21% |
| RepoEval API | 28.54% / 59.75% | **30.42% / 60.44%** | 27.39% / 58.83% |
| RepoEval Line | 36.61% / 59.33% | **44.83% / 64.76%** | 37.98% / 59.81% |
| Macro mean (4 bộ) | 22.54% / 56.85% | **29.31% / 61.17%** | 27.34% / 59.01% |

**Chênh lệch macro của custom:** so với fixed, `−6.78` điểm phần trăm EM và `−4.32` điểm ES; so với AST, `−4.81` điểm EM và `−2.16` điểm ES.

PG‑MSLD chọn trung bình **3.255 snippet**, token cost trung bình **223.865**, STOP rate **0.77**. So sánh, fixed chọn 8.92 snippet (token cost 1,819.305), AST chọn 7.025 snippet (token cost 2,178.173). Các số này phù hợp với khả năng PG‑MSLD đang dùng ít cross-file context hơn, nhưng chưa phân biệt được nguyên nhân là loss/STOP calibration, ngân sách prompt, hay khác biệt protocol.

## 3. Giới hạn cần nêu khi chia sẻ kết quả

1. **Prompt budget không khớp.** Eval PG‑MSLD dùng `prompt_tokens=1024`, `cross_file_tokens=512`; RLCoder-style dùng `3072` và `2344`. Đây là nhiễu lớn cho phép so sánh retrieval.
2. **K và hành vi chọn context khác.** PG‑MSLD eval top-10 nhưng có STOP và chỉ chọn trung bình 3.255 snippet; baseline được eval với 9 snippet thật và lấy trung bình gần 7–9 snippet.
3. **Đây là 400 mẫu valid, không phải test set độc lập lớn.** RepoEval mỗi nhánh có 100 mẫu nhưng chỉ 8 repo nguồn; macro không kèm confidence interval hay kiểm định paired significance.
4. **Tên baseline.** “RLCoder-style fixed/AST” ở đây là checkpoint của các run RLCoder-style lưu trong Modal, tương ứng chunking fixed và AST; không khẳng định đây là checkpoint upstream chính thức do tác giả RLCoder phát hành.
5. **Checkpoint provenance cần đối chiếu.** Training `session_result` ghi SHA256 `2d40337f…`; manifest eval ghi `30a6807a…`. Cả hai metadata đều khai báo schema v7, `next_epoch=9`, `updates=2480`, nhưng SHA byte khác nhau. Có thể artifact đã được đóng gói/lưu lại theo cách khác; chưa có xác minh tensor-level để khẳng định hai file giống hệt trọng số. Nên giải quyết điểm này trước khi dùng kết quả làm claim cuối cùng.

## 4. Diễn giải và bước kiểm chứng tiếp theo

Kết luận an toàn hiện tại: **run PG‑MSLD này cho điểm thấp hơn các baseline RLCoder-style trong lần eval đã lưu, nhưng thí nghiệm chưa cô lập được tác động của loss.** STOP rate cao cùng số snippet thấp là chẩn đoán đáng chú ý, không tự nó chứng minh loss sai.

Để trả lời riêng “loss custom có tốt hơn RLCoder không”, cần chạy lại matched evaluation: cùng checkpoint protocol, valid rows/hash, generator revision, prompt template, 3,072/2,344 token budget, candidate pool top-100, cùng số snippet + cùng STOP semantics. Đồng thời xác minh SHA/checkpoint weights. Chỉ sau đó mới quy chênh lệch cho objective; nếu vẫn thấp, ablate lần lượt soft utility KD, hard anchor, margin/temperature và STOP target.

## Artifact tham chiếu

- Modal eval metrics: `codecompletion-cur-online:/runs/pg-msld-slate15-20261004-valid-top10/metrics.json`.
- Modal eval details: `codecompletion-cur-online:/runs/pg-msld-slate15-20261004-valid-top10/top10.details.jsonl`.
- Modal training result: `codecompletion-positive-gain-listwise-kd:/runs/pg-msld-slate15-20261004/session_result.json`.
- Valid Parquet SHA256: `7658522ea93f31ce86ee50b90162cbeae9e18bb32bec5256c001e977364a8582`.
- RLCoder-style comparison metrics: `codecompletion-cur-online:/runs/trained_rlcoder_ast_vs_fixed_400/metrics.json`.
- Paper notes: [RLCoder deep-read](papers/RLCODER_DEEPREAD.md).
