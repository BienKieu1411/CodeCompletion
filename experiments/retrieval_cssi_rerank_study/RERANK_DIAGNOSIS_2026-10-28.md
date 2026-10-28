# Kiểm tra rerank top-100, 28/10/2026

## Dữ liệu và giới hạn

- Nhánh CSSI: `colab_results/bm25_top100_rerank/bm25_top100_rerank_predictions.jsonl.gz`, 8.004 prediction đã lưu từ run trước.
- Baseline tham chiếu: `experiments/rlcoder_official_repro_20261026/results_fixed/<benchmark>/prediction.jsonl`, cùng task ID, nhưng chạy với prompt, retrieval và cấu hình khác.
- CADRE v2.9: `results/cadre_v29_official_8004/cadre_v29_8004_predictions.jsonl.gz`.
- Chấm lại cả ba với `evaluate_records_official` trong `run_cadre_v29_adaptive_slack.py`. Đây là cùng một hàm chấm nội bộ; việc khớp hoàn toàn scorer của từng paper cần kiểm tra riêng.
- File run cũ không lưu top-100, điểm dense/bonus, top-10 đã chọn hay prompt. Vì vậy không thể gán từng lỗi cho một candidate cụ thể, hoặc tách tác động rerank khỏi prompt/thiết lập khác.

## Kết quả chấm lại

| Benchmark | Baseline EM / ES | Rerank cũ EM / ES | CADRE v2.9 EM / ES | Rerank thắng / thua baseline theo task |
|---|---:|---:|---:|---:|
| CCEval Python | 24.73 / 70.47 | 21.54 / 68.96 | 26.60 / 71.86 | 148 / 233 |
| CCEval Java | 22.72 / 63.62 | 24.08 / 66.93 | 27.40 / 67.89 | 142 / 113 |
| RepoEval API | 36.12 / 63.43 | 38.25 / 65.09 | 43.88 / 70.05 | 143 / 109 |
| RepoEval line | 43.81 / 66.79 | 48.25 / 69.33 | 51.12 / 71.24 | 141 / 70 |

Thắng/thua là số task có EM khác nhau, không phải hiệu EM theo repo-macro. Các cấu hình chưa được kiểm soát đồng nhất; bảng chỉ giúp xác định mẫu cần kiểm tra, không phải tác động nhân quả của rerank.

Ví dụ Python: `project_cc_python/210` có target `agg_diff_fe_calc(self.detect_data, self.anomaly_duration)`, baseline sinh đúng, run cũ sinh `rolling_agg(self.detect_data, RATE)`. Mẫu `project_cc_python/207` có target `diff_feature_calc(detect_data, self.default_point)`, baseline sinh đúng, run cũ dùng tham số `1`. Chưa có prompt và candidate của run cũ để kết luận nguyên nhân trực tiếp.

## Vấn đề trong code run trước

1. Bonus cấu trúc có các mức 0.3–3.5 cộng trực tiếp vào `2 * cosine`, nên có thể lấn át khác biệt dense. Nó còn thưởng cho việc trùng tên biến, không xác nhận quan hệ dependency.
2. STOP bị đặt điểm -999, làm mất cơ chế dừng của retriever.
3. Chunker cũ không gộp mini-block liền nhau như RLCoder; định dạng candidate encoder cũng không comment từng dòng giống `CodeBlock.__str__`.
4. Run cũ dùng prompt và tổng budget khác baseline tham chiếu. Con số baseline được ghi cứng trong script thay vì sinh/chấm cùng cấu hình.
5. Identifier EM/F1 cũ dùng toàn bộ prediction nhiều dòng, còn EM/ES dùng đoạn đã cắt; hai phép đo không cùng phạm vi.
6. Script retrieval proxy trước đây chỉ ghi Hit@K khi duyệt đúng rank K; nếu danh sách ngắn hoặc rank K là STOP, một hit ở rank trước có thể bị đếm thành miss. Đã sửa và tính lại vào thư mục tạm: macro Hit@10 dense 87.50 → 88.01, CSSI 87.30 → 88.01; Hit@1 và MRR không đổi. `study_results.json` và bảng cũ vẫn giữ nguyên. Metric này chỉ đo trùng ít nhất một identifier của ground truth, không xác nhận đúng file/định nghĩa.
7. `run_gen_and_eval_em_es.py` trước đây mặc định có nhánh giả lập lấy thẳng `groundtruth` làm prediction. Đã loại nhánh đó và bắt buộc dùng generator; bảng 40.60 EM từ script này không được coi là kết quả generator thật khi chưa có provenance/run log xác nhận.

Một probe offline trên **top-10 trace RLCoder đã lưu**, không phải top-100 của run mới, cho thấy bonus cũ lớn hơn toàn bộ khoảng `2 * cosine` trong 86.0% task Python, 37.6% Java, 69.6% RepoEval API và 70.2% RepoEval line. Trên cùng top-10 này, bonus cũ đổi top-1 tương ứng 49.6%, 19.1%, 32.9%, 31.8%; heuristic bounded mới đổi top-1 1.7%, 0.5%, 0.6%, 0.2%. Đây là probe mô tả nguy cơ ghi đè dense, không chứng minh thay đổi mới tăng EM/ES.

## Sửa đổi và bước xác nhận

Script dùng bonus tối đa 0.02 cosine. Hai chế độ: `definition` chỉ thưởng khai báo class/interface/method; `reference` cũng thưởng mức nhỏ hơn cho tên kiểu hoặc hàm xuất hiện trong code. Khi không có tín hiệu thì giữ dense. Đây là heuristic lexical, chưa phải phân giải dependency đầy đủ. STOP được xếp theo dense score; checkpoint/revision được pin; chunking và cách encode block được chỉnh theo RLCoder. Mỗi task lưu toàn bộ candidates, dense score, structural match, điểm sau rerank và hai top-10; `rlcoder_top10` trong trace là top-10 mới có thể đưa cho script v2.9 qua `--trace-path`.

Pilot `definition`: 50 task đầu mỗi bộ, tổng 200, DeepSeek 1.3B, A100. Macro EM/ES 40.50/73.39; run cũ trên đúng cùng 200 task được chấm lại 39.50/70.86. Hai run khác chunking, checkpoint/tokenization, STOP và prompt nên không thể quy mức tăng cho bonus. Trong chính trace pilot, xếp hạng top-10 đổi thứ tự ở 7/200 task và đổi tập top-10 ở 3/200: bằng chứng can thiệp `definition` quá thưa để khẳng định đây là cải thiện rerank. Bản `reference` được thêm để kiểm tra giả thuyết tăng độ phủ mà vẫn giới hạn biên độ, không có bảo đảm tăng chất lượng.

Pilot chọn 50 dòng đầu từng benchmark, không lấy mẫu ngẫu nhiên. Cả hai nhóm RepoEval đều chỉ chứa các task có prefix repository `huggingface_diffusers`; vì vậy kết quả pilot không đại diện cho toàn benchmark.

### Pilot `definition` so với `reference`

Cả hai chạy generator thật trên cùng 200 task với cùng checkpoint/model, budget, scorer và A100; hai run có cùng thứ tự top-100 dense ở cả 200 task.

| Benchmark | Definition EM / ES | Reference EM / ES | Task đổi thứ tự top-10 | Trong đó prompt thực sự thay đổi |
|---|---:|---:|---:|---:|
| CCEval Python | 22.00 / 73.38 | 22.00 / 73.82 | 19 | 12 |
| CCEval Java | 16.00 / 65.90 | 16.00 / 65.92 | 2 | 2 |
| RepoEval API | 60.00 / 77.65 | 60.00 / 77.65 | 7 | 5 |
| RepoEval line | 64.00 / 76.63 | 64.00 / 76.63 | 7 | 4 |
| Macro | 40.50 / 73.39 | 40.50 / 73.51 | 35 | 23 |

Không có task nào đổi kết quả EM giữa hai run. Trong 35 task đổi thứ tự top-10, 12 task có prompt vẫn giống hệt sau coalescing và token budget. Tám task có top-10 không đổi nhưng raw generation khác giữa hai phiên A100, nên không quy chênh lệch ES nhỏ cho heuristic. Trên 193/200 task, top-10 của `definition` trùng hệt dense-only nên có thể tái sử dụng prediction của nó làm đối chứng với `reference`; trên 193 task này cũng không có thắng/thua EM. Đây chưa phải đánh giá paired dense-only trên toàn bộ 200 task, và chưa cho thấy hai biến thể vượt baseline.

Mặc định giữ `definition`; `--cssi-signal-mode reference` chỉ là nhánh thử nghiệm. Muốn chọn phương pháp cần validation độc lập có nhiều repository, lưu prompt token IDs, và so với dense-only được tạo cùng builder/generator (`--paired-baseline`).

Trong pilot `definition`, task Python `project_cc_python/207` sinh đúng `diff_feature_calc(detect_data, self.default_point)`; task `project_cc_python/210` vẫn sinh sai `diff_percentile_func(...)`. Ở `/210`, heuristic prefix chỉ nhận diện receiver/type `Utils`; ở `/207`, nó không xác định được receiver, callee hay type, nên hai mẫu này chưa cho thấy tác động trực tiếp của bonus.

Để đo tác động thuần của rerank, chạy pilot với `--paired-baseline`: cả hai nhánh dùng chung pool, encoder, prompt, scorer và generator. Mặc định chỉ sinh nhánh rerank theo yêu cầu trước đó của người dùng. Chọn hệ số trên validation; sau đó mới đánh giá trên test. Không so sánh ID metrics cũ với metric mới vì phạm vi prediction đã thay đổi.
