# Handoff: Repo-level code completion (dòng AlignCoder)

_Cập nhật 2026-10-18. Mọi số liệu dưới đây đã được đối chiếu với `kaggle_*_out/*/*.json`._

## Mục tiêu
Cải thiện code completion cấp repo theo khung "giữ nguyên generator, hướng dẫn retriever" (dòng AlignCoder/RLCoder).
Cô hướng dẫn yêu cầu chỉ ra pain point bằng số liệu, gồm hai điểm:
- Retriever chỉ dùng left context làm query, trong khi thứ tự hàm trong file không quan trọng.
- Retriever bỏ qua quan hệ phụ thuộc (import, call, type).

## Thiết lập
- Thư mục: `NewCodeCompletion/experiments/dep_oracle/`
- Không có GPU local.
  - Từ probe likelihood trở đi chạy trên **Modal L4**: CLI nằm trong venv `C:\Users\ADMIN\WorkPlace\Vinuni\MoCap\.venv-modal`, profile `phongnhv`, Volume `dep-oracle-probe`. Dùng `modal run --detach` để job sống tiếp khi lệnh local bị dừng.
  - Các thí nghiệm trước chạy trên Kaggle bằng kaggle CLI.
  - Tài khoản `deqthnguyen`, dataset `deqthnguyen/dep-oracle`, mount tại `/kaggle/input/datasets/deqthnguyen/dep-oracle/` (tìm file bằng glob).
  - Dùng T4, bật Internet.
- Máy local không có pyarrow, nên phần đọc parquet chạy trên Kaggle. Tiến trình chạy nền ở local bị dừng khi lệnh trả về, nên mỗi lệnh phải xong dưới 3 phút.
  - Phân tích CPU trên các file prediction đã tải về thì chạy local được (`dep_oracle.simple_score` không cần pyarrow).
- KHÔNG chạy lại retriever AlignCoder (rất chậm). Dùng lại candidates đã lưu:
  - RepoEval API: `prediction_with_candidates.jsonl`
  - CCEval Python: `cceval_python_prediction_with_candidates.jsonl` (lấy từ log RQ1 trong `aligncoder-rq1-deepseek13b-20261013-225803.zip`)
- Prompt giống `construct_prompts` của AlignCoder:
  - Tổng 2048 token, mặc định 1536 cho cross-file (`MAX_CTX`, `MAX_CROSS` trong `dep_oracle.py`).
  - Generator `deepseek-coder-1.3b-base`, greedy, 64 token.
- Đo bằng simple EM (so k dòng đầu với k dòng của đáp án) và McNemar ghép cặp, 1000 mẫu/bộ (seed 0). Riêng thí nghiệm oracle API dùng 400 mẫu.
- Script (sha256 prefix):

  | Script | Vai trò | Hash |
  |---|---|---|
  | `dep_oracle.py` | lõi | `26ca377c` |
  | `budget_sweep.py` | quét ngân sách | `1017a575` |
  | `joint_retrieval.py` | xếp chung, lexical | `f597ff0f` |
  | `joint_dense.py` | xếp chung, dense | `16af64ea` |
  | `cceval_extract.py` | trích CCEval | `ad03d2fc` |
  | `oracle_constructions.py` | oracle theo cách dựng prompt (CPU, local) | `beb3a5ca` |
  | `likelihood_probe.py` | probe likelihood: `prepare` / `score` / `analyze` | `74ea5e90` |
  | `modal_probe.py` | chạy `score` (Pha 1) và `pilot` (Pha 2) trên Modal L4 | `a64766ba` |
  | `chunk_pilot.py` | Pha 2: nhãn ΔLL theo chunk + sinh với prompt xếp theo ΔLL | `83910e49` |

  Output nằm trong `kaggle_*_out/`.

## Kết quả đã chốt
1. **Thiếu định nghĩa API KHÔNG phải nút thắt.**
   - Chỉ 14.5% (RepoEval) / 49.7% (CCEval) định nghĩa cần dùng tới được prompt.
   - Nhưng oracle chèn chữ ký API của đáp án chỉ tăng ít (n=400), không có ý nghĩa thống kê. Giá trị của chunk nằm ở code tương tự.
     - RepoEval: +1.0 EM, p=0.29.
     - CCEval: +2.0 EM, p=0.13.
   - `prefixdep` (không dùng đáp án): RepoEval −2.75, CCEval 0.
2. **Ngân sách cross-file cố định 1536 token lấn chỗ của in-file context.**
   - RepoEval: left context trung bình 3934 token, chỉ 456 token vào prompt. 90.5% mẫu bị cắt left context.
   - EM theo mức token cross-file:

     | Bộ | 0 | 256 | 512 | 1024 | 1536 |
     |---|---|---|---|---|---|
     | RepoEval | 27.3 | 38.0 | **40.5** | 40.4 | 36.8 |
     | CCEval | 5.7 | 19.1 | 22.9 | 26.7 | **28.5** |

   - STOP được chọn ở 0.06% mẫu.
   - Mức trần nếu chọn được mức token tốt nhất cho từng mẫu: RepoEval 47.3, CCEval 33.3.
3. **Chọn mức token theo từng mẫu** (`budget_sweep.py`; policy học bằng 5-fold CV chia theo repo):
   - Tín hiệu rẻ (`static_learned`, chỉ sinh 1 lần): RepoEval 40.9, CCEval 28.1. Không vượt mức cố định tốt nhất của từng bộ.
   - **Sửa lại so với bản trước:** tín hiệu tự tin của generator (`conf_firstline`) vượt được mức cố định trên RepoEval.
     - Cách làm: sinh ở cả 5 mức, chọn bản có mean log-prob dòng đầu cao nhất. Không dùng đáp án, nhưng tốn 5 lần sinh.
     - RepoEval 42.7 (> 40.5, vs 1536 p<0.001).
     - CCEval 28.7 (ngang 28.5, p=0.89).
     - `static+conf_learned`: 42.9 / 29.1.
     - Kết luận: generator có tín hiệu chọn prompt tốt trên RepoEval, nhưng không có trên CCEval.
4. **Xếp chung đoạn in-file bị cắt + top-10 cross-file trong cùng ngân sách** (không train, 1000 mẫu/bộ). EM so với align1536:

   | Điều kiện | RepoEval | CCEval |
   |---|---|---|
   | `joint_lex1536` / `joint_lex1024` | 40.2 / **41.3** (+3.4 / +4.5, p<0.001) | 26.2 / 24.3 (−2.3 / −4.2) |
   | `dense_joint1536` / `dense_joint1024` (AlignRetriever) | 40.1 / 40.4 (+3.3 / +3.6, p<0.001) | 26.1 / 24.1 (−2.4 p≈0.009 / −4.4) |
   | Đối chứng chỉ xếp lại top-10 (`cross_lex`, `dense_cross`) | 37.3 / 36.7 (ns) | 26.8 / 27.3 (ns) |
   | In-file trước (≤512 token) rồi top-10 AlignCoder, dense / lex | 39.1 / 38.9 (+2.3 / +2.1, p<0.01) | 28.6 / 28.5 (ngang) |

   - Ở CCEval, đoạn in-file "giống" nhưng không "hữu ích", và đẩy mất đoạn cross-file cần thiết. Ngân sách càng nhỏ thì giảm càng nhiều.
   - "In-file trước" là cách duy nhất không làm giảm trên bộ nào, nhưng vẫn thấp hơn mức cố định 512 trên RepoEval (40.5).
   - => Các cách không train đã chạm trần.
5. **Dư địa nếu chọn đúng cách dựng prompt cho từng mẫu** (tính local từ prediction có sẵn, 2026-10-18):

   | Tập lựa chọn | RepoEval | CCEval |
   |---|---|---|
   | Tốt nhất khi dùng một cách cho mọi mẫu | 40.5 (align512) | 28.6 (dense_infile_then_cross) |
   | Oracle trên 4 cách của probe | **46.8** (+6.3) | **32.6** (+4.0) |
   | Oracle trên 2 cách tốt nhất | 45.0 (align512 + dense_joint1536) | 31.2 (align1536 + dense_joint1536) |
   | Oracle trên cả 9 cách đã chạy | 50.2 | 35.9 |

   4 cách của probe gồm align1536, align512, dense_joint1536 và dense_infile_then_cross.

6. **Probe likelihood** (2026-10-19, Modal L4, 1000 mẫu/bộ, `modal_out/full_*/probe_summary.json`).
   - Gate đạt trên cả hai bộ:
     - Số token khớp 1000/1000 ở cả 4 cách.
     - Argmax agreement 0.994–0.998, so với đối chứng đảo thứ tự 0.93–0.94.
     - Đồng thuận EM khi sinh lại 100%.
   - Anchor tái lập đúng.

   | Chính sách | RepoEval | CCEval |
   |---|---|---|
   | P0 cách cố định tốt nhất | 40.5 (align512) | 28.6 (dense_infile_then_cross) |
   | **P1 argmax likelihood của GT** (dùng GT) | **44.1** (+3.6, p<0.001, lấy lại 57% dư địa) | **31.8** (+3.2, p<0.001, lấy lại 80%) |
   | P2 tự tin của generator (không dùng GT) | 42.8 (+2.3, p=0.004, 37%) | 28.7 (+0.1, p=1.0, 3%) |
   | P3 oracle | 46.8 | 32.6 |
   | Độ chính xác theo cặp P1 / P2 | 0.80 / 0.77 | 0.95 / 0.75 |

   - **H1: GO trên cả hai bộ.**
   - **H2:** P1 hơn P2 hơn 1 EM ở cả hai bộ. Tự tin của generator không giúp được trên CCEval, đúng như dự đoán từ budget sweep.
   - Kết quả bền với cách chuẩn hoá: mean, sum và dòng đầu cho cùng kết quả.
   - **Lưu ý diễn giải: H1 là điều kiện cần, chưa phải bằng chứng mạnh.**
     - EM=1 nghĩa là greedy trùng GT, mà khi đó GT gần như chắc chắn có likelihood cao. Vì vậy P1 thắng một phần là do cấu trúc của phép đo. Điều này thấy rõ ở CCEval: GT ngắn, độ chính xác theo cặp lên 0.95.
     - Câu hỏi thật là tín hiệu này có **học được từ input không cần GT** và có **cộng dồn được ở mức chunk** không. Pha 2 trả lời câu này.
   - Chi phí khoảng $1.5 (2 container L4 × ~35 phút, cộng smoke test).

7. **Pha 2: pilot nhãn ΔLL theo chunk** (2026-10-19, Modal L4, 200 mẫu/bộ, `modal_out/pilot_*/pilot_summary.json`).
   - Mốc là `regen_P0`, tức P0 sinh lại trên L4.
     - Đồng thuận EM giữa bản sinh lại và bản T4 đã lưu: 98.5–100%.
     - Dư địa trên 200 mẫu rất nhỏ: RepoEval 5.0 điểm (≈10 mẫu), CCEval 3.5 điểm (≈7 mẫu).

   | Điều kiện | RepoEval | CCEval |
   |---|---|---|
   | regen_P0 (align512 / dense_infile_then_cross) | 38.0 | 22.5 |
   | **dll_pos_1536 (chỉ số chính)** | **41.0** (+3.0, lấy lại 60%, CI [−2, 8], p=0.29) | **26.0** (+3.5, lấy lại 100%, CI [0.5, 7], p=0.07) |
   | dll_pos_512 / dll_pos_1024 | 45.0 / 44.5 (+7.0 / +6.5, p≤0.01) | 26.5 / 25.0 |
   | rand_joint1536 (cùng pool, thứ tự ngẫu nhiên) | 27.0 | 21.5 |
   | regen dense_joint1536 (cùng pool, xếp theo dense) | 34.5 | 21.0 |
   | dll_bestB (chọn ngân sách theo likelihood GT) | 44.0 | 29.0 |
   | Oracle 4 cách của Pha 1 trên tập con (T4) | 43.0 | 27.5 |

   - **Theo ngưỡng chốt trước: ĐẠT trên cả hai bộ.**
     - RepoEval chỉ đạt sát ngưỡng (60%, CI chứa 0).
     - CCEval vẫn đạt nếu so với P0 T4 đã lưu (24.0 → lấy lại 57%).
   - **Phát hiện mô tả:**
     1. **Thứ tự quyết định, không phải pool.** Cùng pool, cùng ngân sách 1536: xếp theo ΔLL cho 41.0 / 24.5–26.0, ngẫu nhiên cho 27.0 / 21.5, xếp theo dense cho 34.5 / 21.0.
     2. **"Giống" ≠ "hữu ích".** Trong mỗi mẫu, Spearman giữa điểm dense của AlignRetriever và ΔLL chỉ 0.03–0.09 (CCEval) và 0.08–0.14 (RepoEval). Điều này có cả trong top-10 cross-file mà chính AlignCoder đã chọn.
     3. **Mức hữu ích của in-file phụ thuộc bộ.** Tỷ lệ mẫu có chunk ΔLL cao nhất là in-file: RepoEval 32–45%, CCEval 2.5–8.5%. Điều này giải thích vì sao xếp chung giúp RepoEval mà hại CCEval.
     4. **ΔLL đơn lẻ không cộng dồn tốt ở ngân sách lớn.**
        - 80–90% chunk có ΔLL > 0, nên `dll_pos_1536` gần như nhồi đầy ngân sách.
        - Ở RepoEval, 512/1024 (45.0) tốt hơn hẳn 1536 (41.0), vì cửa sổ in-file vẫn bị lấn chỗ.
        - Reranker cần học cả **ngưỡng dừng / ngân sách**, không chỉ thứ tự. `dll_bestB` cho thấy dư địa ở đây.
   - **Lưu ý:**
     - Nhãn ΔLL dùng GT, nên mọi dòng `dll_*` là cận trên của một reranker bắt chước ΔLL hoàn hảo, không phải phương pháp.
     - n=200 nên chỉ dùng để quyết định hướng đi, không dùng để báo cáo hiệu quả.
   - Chi phí khoảng $1.3 (2 container L4 × ~50–80 phút).

8. **E0: thứ tự khi tập chunk đã cố định** (2026-10-24, Modal L4, 200 mẫu pilot/bộ, `e0_order.py` `830f19fd`, `modal_e0.py` `94cb2914`, `modal_out/e0_*/e0_summary.json`).
   - Lý do: trong Pha 2, "xếp theo ΔLL" so với "ngẫu nhiên" khác nhau chủ yếu ở việc *chunk nào được chọn* (pool được xếp rồi nhồi tới khi hết ngân sách). E0 cố định tập S_B (các chunk ΔLL>0 vừa trọn trong B) và chỉ hoán vị.
   - Mốc: `repro_dll_pos_1536` khớp pilot 100% pred (RepoEval 41.0, CCEval 26.0).
   - EM trên các mẫu có |S| ≥ 2, B=1536 (ngưỡng chốt trước: chênh ≥ 3 EM, p<0.05, vượt nhiễu rand-rand2 ≥ 2):

     | Thứ tự | RepoEval | CCEval |
     |---|---|---|
     | desc (chunk tốt nhất ở xa con trỏ, như pilot) | 43.5 | 26.3 |
     | asc (chunk tốt nhất sát con trỏ, kiểu L2H) | 44.0 | **29.9** (+3.6, 8 vs 1, p=0.039) |
     | rand / rand2 | 44.0 / 45.0 | 28.4 / 27.3 |
     | in-file theo thứ tự gốc | 44.0 | 26.3 |
     | Phán quyết theo ngưỡng | ablation | "order matters" (sát ngưỡng) |

   - **Đọc kết quả:**
     1. **Sự sụp đổ của `rand_joint1536` ở Pha 2 (27.0 / 21.5) hoàn toàn do việc chọn, không phải vị trí.** Cùng tập, thứ tự ngẫu nhiên vẫn đạt 44.0 / 28.4.
     2. RepoEval: thứ tự không ảnh hưởng (mọi cặp trong ±1.5 EM, ΔGT-LL ≈ 0).
     3. CCEval: asc > desc nhất quán ở cả 3 ngân sách (+1.5 / +2.6 / +3.6), nhưng chỉ 9 mẫu khác nhau, CI của ΔGT-LL chứa 0, và có 30 phép so sánh → coi là **hiệu ứng nhỏ, cùng chiều với RepoBench L2H**, không đủ làm claim chính.
   - **Quyết định:** không claim "position-aware". Đặt mặc định đóng gói kiểu asc (chunk tốt nhất sát con trỏ, miễn phí) và để thứ tự thành một ablation. Trọng tâm là **chọn tập + ngân sách**.
   - Phụ: `set_*_1536` (chỉ chunk trọn vẹn) ~44 > `dll_pos_1536` 41.0 trên RepoEval. Khác biệt là chunk cuối bị cắt dở và in-file window rộng hơn chút, cần kiểm tra thêm khi thiết kế packing.
   - Chi phí khoảng $1.

9. **E1: scorer học ΔLL từ text** (2026-10-24, Modal L4, tài khoản `thanhnguyenvinh169`; `e1_label.py` `8fdd8ea8`, `e1_train.py` `4fbfefe7`, `e1_eval.py` `7ec648ef`, `e1_cv.py` `51112c8d`, `modal_e1.py` `6b0be485`, `modal_e1cv.py` `cf70cc14`; output `modal_out/e1eval_*`, `e1heval_*`, `e1cv/`).
   - Khung train kiểu RLCoder: dữ liệu Data4RLCoder (repo 0–2000 train, 2000–2200 dev), khởi tạo AlignRetriever, nhãn ΔLL ở B=1024, ứng viên NULL (ΔLL=0) làm stop signal. Ba biến thể: `bi_soft` (listwise KL), `bi_onehot` (đúng mục tiêu RLCoder), `cross_soft`. Ngưỡng chốt trước: G1 Spearman@1024 ≥ 0.30; G2 EM ≥ regen_P0 +3 (RepoEval) / +2 (CCEval), n=200 pilot.
   - **E1 (cắt ngẫu nhiên, 3871 mẫu):** FAIL. Nhãn gần như không có tín hiệu: ΔLL top 1.7 (eval 7–10), log p(GT) không context −8.9 (eval −16 / −32).
   - **CV trong miền trên nhãn pilot** (chẩn đoán): RepoEval held-out 0.16 (dense 0.14). Hạn chế: RepoEval chỉ 8 repo, CCEval không có tên repo nên cả bộ rơi vào một fold.
   - **E1b (cắt kiểu CCEval: GT dùng tên định nghĩa ở file khác, file ≥ 250 dòng; 4000 mẫu + 968 mẫu E1 có ΔLL top ≥ 1):** nhãn đã gần eval (top 6.05, base −16.7, 7.6 chunk in-file/mẫu). Dev Spearman vẫn chỉ ~0.06.

     | Pilot n=200 | RepoEval | CCEval |
     |---|---|---|
     | Spearman@1024 dense / bi_soft / bi_onehot / cross_soft | 0.14 / 0.17 / 0.16 / **0.20** | 0.07 / 0.12 / 0.13 / **0.14** |
     | EM regen_P0 | 38.0 | 22.5 |
     | EM dense, cùng cách đóng gói, B tốt nhất | 39.0 (1024) | 20.5 (1536) |
     | EM scorer tốt nhất | 41.5 (cross_soft 1024, 12 vs 5, p=0.14) | 21.0 (bi_onehot 1536) |
     | Cận trên ΔLL, cùng cách đóng gói | 45.0 (512) | 29.0 (1536) |
     | Phán quyết | G1 FAIL, G2 PASS (sát, n.s.) | G1 FAIL, G2 FAIL |

   - **Kết luận E1:** ba lần thử (E1, CV, E1b) đều cho Spearman ≤ 0.20. Một encoder 125M không đoán được ΔLL theo chunk từ text. Phần dư địa lớn của ΔLL (45 / 29) nằm ở tín hiệu chỉ generator mới có. Nhãn mềm và one-hot (RLCoder) không khác nhau. Stop head (NULL) gần như không bao giờ cắt (stop ≈ fill).
   - Chi phí E1 + E1b + CV: khoảng $11–13 trên tài khoản mới (~13 giờ L4).

10. **Proxy AMI không cần GT** (2026-10-25, Modal L4, `e1_ami.py` `60983240`, `modal_ami.py` `c12c99db`, `modal_out/e1ami_*/ami_summary.json`). AMI_k(c) = log p(k dòng cuối của left context | c) − log p(… | không c), cùng pool, cùng cửa sổ in-file W(1024), k ∈ {1, 3, 6}. Pilot 200 mẫu/bộ, khoảng $1.

    | | RepoEval | CCEval |
    |---|---|---|
    | Spearman@1024 dense / ami1 / ami3 / ami6 | 0.14 / 0.09 / 0.14 / 0.14 | 0.07 / 0.12 / 0.17 / **0.18** |
    | EM dense_fill_1024 (E1b, cùng đóng gói) | 39.0 | 17.0 |
    | EM AMI tốt nhất @1024 | 41.5 (ami6, 9 vs 4, p=0.27) | **22.0** (ami3_fill, 15 vs 5, p=0.04) |
    | EM regen_P0 / cận trên ΔLL @1024 | 38.0 / 44.5 | 22.5 (B=1536) / 28.0 |
    | Phán quyết | G1 FAIL, G2 FAIL | G1 FAIL, G2 PASS |

    - Đọc kết quả: tín hiệu của generator không cần GT **chỉ nhỉnh hơn dense** ở mức chunk (Spearman ≤ 0.18), ngang scorer học ở E1b. Trên CCEval nó chọn chunk tốt hơn dense ở cùng ngân sách (+5 EM) và bằng P0 với ít hơn 1/3 token cross-file, nhưng không vượt P0. Trên RepoEval không có gain. G2 CCEval là max trên 6 điều kiện, nên có chọn lọc nhẹ.
    - AMI tốn khoảng 30 lượt forward của generator mỗi mẫu, đắt hơn nhiều so với một lần sinh.
    - **Kết luận chung E1 + AMI:** không có tín hiệu nào không dùng GT (text scorer, generator proxy) vượt Spearman ~0.2 với ΔLL. Dư địa 44–45 / 28–29 phụ thuộc vào việc biết dòng cần sinh.

11. **E2: chính sách chọn ngân sách theo từng mẫu — ĐẠT, xác nhận trên toàn bộ tập test** (2026-10-25).
    - Offline trên sweep dev 1000 mẫu/bộ (`e2_offline.py` `8108b089`, CPU): chọn chính sách chính là `learned_256_1024_1536` (logistic theo từng ngân sách trên đặc trưng tĩnh + độ tự tin dòng đầu ở 3 ngân sách, một chính sách chung cho cả hai bộ).
    - Hold-out = mọi mẫu test ngoài tập dev (RepoEval 600, CCEval 1665), quét trên Modal L4 (`budget_sweep.py --holdout` `0640a452`, `modal_e2e3.py` `e7f71470`). Chính sách fit trên dev, áp nguyên lên hold-out (`e2_confirm.py` `15a787f5`): **PASS** (RepoEval 44.0 vs 42.8, CCEval 29.9 vs 29.1; gộp 57 vs 36, p=0.038).
    - **Toàn bộ tập test** (RepoEval API 1600, CCEval Python 2665), cross-fit dev↔hold-out, `modal_out/e2_fulltest_summary.json`:

      | Chính sách | Số lần sinh | RepoEval | CCEval |
      |---|---|---|---|
      | AlignCoder (1536 cố định) | 1 | 36.81 | 28.86 |
      | Ngân sách cố định tốt nhất của từng bộ (512 / 1536) | 1 | 41.38 | 28.86 |
      | Cố định 1024 (tốt nhất nếu dùng một con số chung) | 1 | 41.00 | 27.58 |
      | **learned_256_1024_1536** | 3 | **43.12** (+1.74, 65 vs 37, p=0.007) | **29.83** (+0.97, 54 vs 28, p=0.005) |
      | learned_512_1536 | 2 | 42.56 (p=0.023) | 29.64 (p=0.026) |
      | conf_256_1024_1536 (không học, chỉ chọn độ tự tin cao nhất) | 3 | 43.06 (p=0.011) | 29.72 (p=0.043) |
      | Oracle 4 ngân sách | — | 46.88 | 33.55 |

      p so với ngân sách cố định tốt nhất của từng bộ. So với AlignCoder: +6.3 / +1.0 EM.
    - Đọc kết quả: **một chính sách duy nhất thắng ngân sách cố định tốt nhất của từng bộ trên cả hai bộ**, điều mà mọi chiến lược cố định trước đó (F3) không làm được. Phần lớn tác dụng đến từ độ tự tin của generator. Phần học chỉ giúp rõ khi có 2 ngân sách (CCEval 29.64 vs 29.19).
    - Chi phí: độ tự tin chỉ dùng dòng đầu, nên lúc chạy thật có thể chỉ sinh nháp dòng đầu (~10 token) ở mỗi ngân sách rồi sinh đầy đủ một lần. Với greedy, dòng đầu trùng hệt nên kết quả không đổi; còn phải đo độ trễ.
    - Lưu ý: sweep dev chạy trên Kaggle T4, hold-out trên Modal L4 (đã thấy các bản sinh lại khớp nhau 98.5–100%). Chỉ 1 generator 1.3b, chỉ EM, chỉ Python.
12. **E3: nén in-file không cần train — KHÔNG giúp** (`e3_compact.py` `d65c64b5`, 1000 mẫu dev/bộ, B = 512 / 1536). Bỏ comment và dòng trống (`strip`) không đổi; thay phần in-file bị cắt bằng khung import + chữ ký def/class (`skel`) hơi hại; kết hợp cả hai hại có ý nghĩa trên RepoEval (−1.8 EM p=0.010; −2.0 p=0.001). CCEval không đổi. Khung chiếm chỗ của các dòng ngay trước con trỏ, mà những dòng đó có giá trị hơn. Không chạy lại trên toàn bộ test.
13. **Revela-code-500M zero-shot làm scorer** (`e1_revela.py`, `modal_revela.py`, pilot 200 mẫu/bộ): Spearman với ΔLL ngang dense (0.12 / 0.07) → G1 FAIL. EM ngược chiều: CCEval +2.5 đến +4.5 so với dense (1024: p=0.035; 1536: 23.5 > P0 22.5), RepoEval −2.5 đến −4.5. Không phải lời giải chung; chỉ có thể là ứng viên retriever cho các mẫu "cần cross-file".
- Chi phí: E2 hold-out + E3 + Revela khoảng $5; tổng trên tài khoản `thanhnguyenvinh169` khoảng $18–19 (ước tính).

14. **E4: đoán ngân sách từ trạng thái ẩn sau MỘT lần prefill — FAIL** (2026-10-25, `e4_probe.py`, `modal_e4.py`, `modal_out/e4_summary.json`). Prefill prompt B=1024 một lần cho toàn bộ test (4265 mẫu, vài phút L4), lấy trạng thái ẩn token cuối ở lớp 8/16/24, cộng entropy và xác suất top-1 của token kế tiếp, cộng 9 đặc trưng tĩnh. PCA + logistic theo từng ngân sách; siêu tham số chọn bằng CV lồng trên phía train; cross-fit dev↔hold-out như E2.

    | Chính sách (toàn bộ test) | Chi phí | RepoEval | CCEval |
    |---|---|---|---|
    | Ngân sách cố định tốt nhất của từng bộ | 1 lần sinh | 41.38 | 28.86 |
    | learned_static | 1 lần sinh | 41.94 (p=0.37) | 28.78 |
    | static + entropy token kế | 1 prefill + 1 lần sinh | 41.94 | 28.67 |
    | **Probe trạng thái ẩn** | 1 prefill + 1 lần sinh | **42.50** (45 vs 27, p=0.044) | 28.63 (p=0.53) |
    | E2 (độ tự tin ở 3 ngân sách) | 3 lần sinh | 43.12 | 29.83 |

    - Probe giữ được 64% mức tăng của E2 trên RepoEval, 0% trên CCEval; gộp p=0.34 → FAIL. CV chọn PCA 16 chiều và L2 mạnh: trạng thái ẩn chỉ thêm ít thông tin.
    - Đọc kết quả: thông tin chọn ngân sách nằm chủ yếu ở việc **thấy generator thực sự sinh gì ở mỗi ngân sách**, không đọc được từ một lần prefill. Cách giảm chi phí thực tế cho E2 là sinh nháp dòng đầu ở 3 ngân sách theo batch song song.

15. **E6: cascade theo độ tự tin để giảm chi phí E2 — FAIL** (2026-10-25, offline CPU, `e6_cascade.py` `b803f4cf`, `modal_out/e6_cascade_summary.json`). Sinh ở B=1024 trước; nếu cổng (độ tự tin dòng đầu, hoặc logistic static + độ tự tin) báo không chắc thì sinh thêm 256 + 1536 (song song) hoặc 1536 → 256 (tuần tự), chọn bằng độ tự tin hoặc selector E2. Ngưỡng chọn trên fold train, cross-fit dev↔hold-out như E2.

    | Chính sách (toàn bộ test) | Số lần sinh TB | RepoEval | CCEval |
    |---|---|---|---|
    | Ngân sách cố định tốt nhất của từng bộ | 1 | 41.38 | 28.86 |
    | Cố định 1024 | 1 | 41.00 | 27.58 |
    | Cascade tốt nhất, q chọn trên train cho chi phí ≤ 1.6 | 1.55 | 42.00 (p=0.38) | 28.59 (p=0.62) |
    | Cascade q=0.5 (cổng learned) | 2.0 | 43.0 | 29.5 |
    | E2 | 3 | 43.12 | 29.83 |

    - Ở chi phí ≤ 1.6, cascade giữ 36% mức tăng của E2 trên RepoEval và −28% trên CCEval → FAIL (ngưỡng là ≥ 80% trên cả hai). Ở chi phí 2.0 thì giữ khoảng 90% / 65%.
    - Đọc kết quả: EM tăng gần tuyến tính theo tỷ lệ mẫu được sinh thêm, nên cổng độ tự tin ở 1024 không dồn được các mẫu cần sinh thêm vào nhóm nhỏ. Trên CCEval, xuất phát ở 1024 thua 1536 cố định 1.3 EM, nên cascade rẻ phải bù lỗ trước khi có lời.
    - Hệ quả: cách giảm chi phí thực tế cho E2 vẫn là sinh nháp dòng đầu theo batch song song (chi phí token ~3× prefill + ~10 token mỗi ngân sách), không phải cascade.

16. **E7: phân rã lỗi trên toàn bộ test** (2026-10-25, CPU, `e7_errors.py` `3437d975`, `modal_out/e7_errors_summary.json`, `e7_rows_*.jsonl`). Dựng lại prompt ở 4 ngân sách y như `budget_sweep.py`. Identifier của GT "thấy được" ở ngân sách B nếu xuất hiện như một từ trong prompt đó. Dữ liệu gốc tải về `aligncoder_data/` (~1 GB, RepoEval api_level copy từ `tmp/inspection_pool_inputs`).

    | Mẫu sai ở cả 4 ngân sách | RepoEval (850 = 53.1%) | CCEval (1771 = 66.5%) |
    |---|---|---|
    | usage: mọi identifier của GT đều có trong một prompt, generator vẫn sai | 58.9% (31.3% toàn bộ) | 57.6% (38.3% toàn bộ) |
    | budget_cut: identifier thiếu có trong left context / candidates nhưng mọi ngân sách đều cắt mất | 10.7% (5.7%) | 6.1% (4.1%) |
    | retrieval: chỉ có trong file khác của repo, candidates không lấy được | 19.6% (10.4%) | 26.4% (17.5%) |
    | novel: không có ở đâu trước con trỏ | 10.7% (5.7%), hầu hết chỉ có ở right context | 9.9% (6.6%), chủ yếu là tên mới |

    - Đọc kết quả: phần lớn lỗi **không phải do thiếu thông tin trong prompt** mà do generator dùng sai thông tin đã có. Đây là cận trên lỏng (chỉ kiểm tra tên xuất hiện, không kiểm tra định nghĩa/chữ ký đúng có mặt). Retrieval miss là nguồn lỗi lớn thứ hai, nhất là CCEval. Cắt do ngân sách chỉ 4–6% toàn bộ mẫu, khớp với việc E2 chỉ tăng +1–2 EM.
    - **Kiểm tra tĩnh theo symbol của repo** (dòng đầu dùng identifier không có trong left context + repo + candidates): gặp ở 9.2% / 16.7% dự đoán sai, 1.7% / 3.8% dự đoán đúng, 6.6% / 7.9% dòng GT (dương tính giả, chủ yếu thuộc tính thư viện ngoài). Dùng làm bộ lọc trên 3 bản sinh greedy của E2: RepoEval 43.06 → 42.94, CCEval 29.72 → 29.83; trên ngân sách cố định +0.4 / +0.2. **Không đáng làm** với pool 3 bản; chỉ đáng thử lại trên pool lớn hơn (8 mẫu của E5).

17. **E7b: phân rã sâu nhóm usage** (2026-10-25, CPU, `e7b_usage.py` `dbfe527d`, `modal_out/e7b_usage_summary.json`, `e7b_rows_*.jsonl`, `e7b_usage.log`). Lấy các mẫu usage của E7 (sai ở mọi ngân sách dù mọi identifier của GT có trong một prompt): RepoEval 501 mẫu (31.3% toàn bộ), CCEval 1020 (38.3%). Phân tích dự đoán ở ngân sách cố định tốt nhất nếu ở đó thấy đủ identifier, nếu không thì ở ngân sách đầu tiên thấy đủ. So dòng đầu của dự đoán với dòng đầu của GT.

    | % nhóm usage | RepoEval | CCEval |
    |---|---|---|
    | Sai ý định: câu lệnh khác hẳn | 34.9 | 33.3 |
    | Sai tham số / thành viên: thiếu, thừa identifier, cắt ngang, cùng đầu rồi rẽ | 38.2 | 41.7 |
    | Sai literal / định dạng (cùng chuỗi identifier) | 9.6 | 24.8 (60% literal của GT không có trong prompt) |
    | Dòng đầu đúng, các dòng sau sai (GT nhiều dòng) | 17.4 | 0.2 |
    | Phần cross-file cần thiết (có identifier chỉ nằm ở phần cross-file) | 41.7 | 83.6 |
    | Định nghĩa của mọi symbol GT có trong prompt | 16.4 | 40.5 |
    | Dòng GT gần như có sẵn trong prompt (độ giống ≥ 90) | 40.3 | 1.1 |
    | Dự đoán chép nguyên một dòng trong prompt | 37.5 | 0.0 |
    | Dự đoán dùng identifier không có trong repo | 4.8 | 5.1 |

    - Đọc mẫu thủ công (~60 mẫu, in trong lần chạy đầu): RepoEval hay **chép dòng lân cận gần giống nhưng sai** (`field_output_density` thay vì `field_head_density`) và đoán sai câu lệnh tiếp theo trong code test. CCEval thường bắt đầu giữa dòng sau dấu chấm: **chọn sai thành viên hoặc tham số** (`SlabType.NODE_SLAB` thay vì `MAIN_SLAB`, `caption(img_path)` thiếu `caption_prompt`) dù định nghĩa có trong prompt.
    - Hệ quả:
      - Khoảng 1/3 nhóm usage là sai ý định: prefix không đủ để biết dòng tiếp theo làm gì. Đây là lý lẽ thứ hai để xin dùng right context.
      - Sai tham số / thành viên khi định nghĩa có trong prompt (CCEval ~40% nhóm usage, ~15% toàn bộ) là mục tiêu hợp lý nhất cho decoding tương phản kiểu CAD.
      - RepoEval chép nhầm dòng lân cận: CAD có thể làm tệ hơn vì khuếch đại việc dựa vào context. Cần theo dõi riêng nhóm này khi chạy pilot.
      - RepoEval nhiều dòng (5.4% toàn bộ): có thể chọn ngân sách lại theo từng dòng thay vì chỉ theo dòng đầu.
      - Kiểm tra symbol tĩnh chỉ chạm ~5%, xác nhận E7.

18. **E5: ngân sách như một trục bổ sung cho cách chọn kiểu Caret** (2026-10-25, `e5_budget_tts.py`, `modal_e5.py`, `modal_out/e5_summary.json`, item-level ở `modal_out/e5_*/items_*.jsonl`; 1000 mẫu dev/bộ). Mô phỏng cách chọn của Caret (arXiv 2610.04837): sampling T=0.8 / top-p 0.95, bỏ phiếu theo dòng đầu, RCV = likelihood trung bình của 128 token code sau con trỏ khi chèn ứng viên (không có cross-file, nên so được giữa các ngân sách), trộn 0.75·RCV + 0.25·vote. Giữ retriever của AlignCoder. Hai pool cùng 8 mẫu: `fixed1536` (8 mẫu ở 1536) và `mixed` (3 ở 512 + 3 ở 1024 + 2 ở 1536). Có thêm 3 bản greedy ở 512 / 1024 / 1536 kèm độ tự tin.

    | | RepoEval | CCEval |
    |---|---|---|
    | greedy 1536 / 512 | 37.0 / 40.6 | 28.2 / 22.9 |
    | fixed1536 + chọn kiểu Caret | 37.7 | 27.2 |
    | **mixed + chọn kiểu Caret** | **41.5** (73 vs 35, p=0.0003) | 27.1 (46 vs 47) |
    | Oracle của pool fixed / mixed | 44.6 / 50.2 | 38.3 / 38.3 |
    | 3 greedy + độ tự tin (E2, chỉ prefix) | 41.1 | **28.9** |
    | 3 greedy + chọn kiểu Caret (có code sau con trỏ) | **42.2** | 28.0 |
    | Chỉ prefix: mixed/vote vs fixed/vote | +2.5 (61 vs 36, p=0.014) | −0.8 (n.s.) |

    - Phán quyết theo ngưỡng chốt trước: **không rõ ràng.** Mixed thắng mạnh trên RepoEval, ngang trên CCEval. Gộp hai bộ p=0.011, nhưng ngưỡng yêu cầu thắng trên cả hai bộ.
    - Đọc kết quả: trục ngân sách tạo ứng viên đa dạng hơn sampling ở nơi có lấn chỗ (oracle RepoEval 50.2 vs 44.6), và vô ích khi không có lấn chỗ (oracle CCEval bằng nhau).
    - Bản mô phỏng Caret **không** tái lập mức tăng của Caret: với 8 mẫu, sampling + RCV/vote còn kém greedy trên CCEval. Thiếu retriever thứ hai và bước cascade của họ, nên không dùng kết quả này để so trực tiếp với Caret. Trên CCEval, E2 vẫn tốt nhất trong các cách đã thử.
    - Pool 8 mẫu ở đây có thể dùng để thử lại kiểm tra symbol tĩnh của E7.
    - **Phân tích cửa sổ 8k** (CPU, từ đặc trưng có sẵn): code trong file có trung vị RepoEval 2524 / CCEval 642 token. Ở 8k với 3800 token cross-file, RepoEval vẫn 31.7% mẫu bị cắt, CCEval chỉ 3.5%. Kể cả ở 2k, với các mẫu không bị cắt, dư địa oracle chỉ còn 4.5 / 3.3 EM (so với 6.5 / 5.8 ở các mẫu bị cắt). → Phân bổ ngân sách mạnh khi cửa sổ nhỏ hoặc file dài, yếu khi cửa sổ lớn. Báo cáo MB-SPT + SPC của partner (CCEval, 8k vs AlignCoder 2k) bị nhiễu bởi cỡ cửa sổ: cần thêm nhánh AlignCoder-8k và RepoEval.

19. **E8: decoding tương phản với cross-file (CAD) — FAIL** (2026-10-25, Modal L4, `e8_cad.py` `6dc4b975`, `modal_e8.py` `8aa202e0`, `modal_out/e8_cad_summary.json`). Mỗi token: (1+α)·logits(có cross-file) − α·logits(cùng prompt bỏ phần cross-file, giữ nguyên in-file). Greedy, pilot 200 mẫu/bộ (`chunk_pilot.pilot_ids`), B ∈ {512, 1536}, α ∈ {0, 0.25, 0.5, 1, ada (AdaCAD: α_t = JSD)}. Mốc α=0 khớp sweep (EM lệch ≤ 0.5, văn bản trùng 75–89% do batch/fp16 khác).

    | EM pilot | α=0 | 0.25 | 0.5 (chính) | 1 | ada |
    |---|---|---|---|---|---|
    | RepoEval B=512 | 38.0 | 36.0 | 35.5 (4 vs 9) | 28.5 | 37.5 |
    | RepoEval B=1536 | 31.5 | 32.0 | 28.5 | 24.5 | 32.0 |
    | CCEval B=1536 | 22.5 | 20.5 | 19.5 (4 vs 10) | 16.0 | 20.5 |
    | CCEval B=512 | 19.0 | 19.0 | 18.0 | 16.0 | 19.5 |

    - Chính (α=0.5 ở ngân sách tốt nhất từng bộ) gộp 8 thắng / 19 thua so với α=0 (p=0.052, theo chiều xấu). EM giảm đều khi α tăng trên cả hai bộ; AdaCAD chỉ gần hoà.
    - Theo nhóm lỗi E7/E7b: ở CCEval α=0.5 sửa 3 mẫu usage, làm hỏng 10 mẫu vốn đúng. Khuếch đại cross-file không sửa được lỗi tham số / thành viên.
    - Đọc kết quả: lỗi "dùng sai" không phải do generator bỏ qua cross-file. Ép nghe context nhiều hơn chỉ đẩy nó chép phần không liên quan. Bỏ hướng CAD và trộn logits theo kiểu tương phản.
20. **B1a: kiểm tra symbol tĩnh trên pool 8 mẫu của E5 — FAIL** (2026-10-25, CPU, `b1a_symbol_filter.py` `1e39ad9b`, `modal_out/b1a_symbol_filter_summary.json`; 1000 mẫu dev/bộ, không sinh mới). Bỏ ứng viên có dòng đầu dùng identifier không có trong left context + file repo + candidates (cờ `pred_unknown_id` của E7b); nếu cả pool bị gắn cờ thì giữ nguyên. Ngưỡng chốt trước: mixed/vote + lọc ≥ +1 EM so với mixed/vote trên cả hai bộ.

    | EM (không lọc → lọc) [thắng, thua, p] | RepoEval | CCEval |
    |---|---|---|
    | **mixed/vote (quyết định)** | 39.9 → 39.9 [4, 4] | 27.1 → 27.4 [7, 4, p=0.55] |
    | mixed/caret | 41.5 → 41.5 | 27.1 → 27.0 |
    | fixed1536/vote | 37.4 → 37.1 | 27.9 → 27.9 |
    | all (16 mẫu + 3 greedy)/vote | 40.3 → 40.0 | 29.0 → 29.0 |
    | oracle mixed | 50.2 → 49.3 (0 vs 9, p=0.004) | 38.3 → 37.3 (0 vs 10, p=0.002) |

    - Bị gắn cờ: 8.9% / 20.3% ứng viên trong pool mixed; 1.1% / 5.8% mẫu có cả pool bị gắn cờ.
    - Đọc kết quả: vote vốn đã ít chọn ứng viên "lạ", nên lọc gần như không đổi lựa chọn. Bộ lọc còn bỏ cả đáp án đúng (oracle giảm 1 điểm có ý nghĩa): identifier "lạ" thường là thuộc tính thư viện ngoài, đúng như dương tính giả 6.6–7.9% của E7. Kiểm tra symbol theo kiểu "tên có xuất hiện" không đủ mạnh; muốn dùng phải có thông tin kiểu / thư viện. Đóng hướng này.
21. **B1b: EM theo độ phủ (abstention) với độ tự tin E2 — mô tả** (2026-10-25, CPU, `b1b_abstention.py` `8808294e`, `modal_out/b1b_abstention_summary.json`; toàn bộ test RepoEval 1600 / CCEval 2665, chính sách học cross-fit dev↔hold-out như E2, không sinh mới). Bỏ qua các mẫu có điểm thấp nhất; điểm = độ tự tin dòng đầu của bản được chọn (hoặc xác suất logistic). Ở độ phủ 100%, learned3 = 43.1 / 29.8, khớp E2.

    | EM ở độ phủ | 100% | 80% | 60% | 50% | 30% | 10% | AUC | độ phủ lớn nhất có EM ≥ 60% |
    |---|---|---|---|---|---|---|---|---|
    | RepoEval: ngân sách cố định 512 + conf | 41.4 | 50.8 | 63.3 | 68.1 | 80.4 | 87.5 | 68.2 | 65% |
    | RepoEval: **conf3 (E2 không học)** | 43.1 | 53.1 | 64.9 | 70.5 | 80.4 | 88.1 | 69.1 | 68% |
    | RepoEval: learned3 + xác suất | 43.1 | 52.6 | 64.3 | 69.4 | 78.5 | 85.6 | 68.0 | 67% |
    | CCEval: ngân sách cố định 1536 + conf | 28.9 | 35.1 | 43.0 | 48.0 | 59.4 | 71.0 | 49.8 | 29% |
    | CCEval: **conf3** | 29.7 | 36.3 | 43.9 | 48.9 | 60.2 | 70.3 | 50.1 | 30% |
    | đối chứng: learned3, xếp ngẫu nhiên | 43.1 / 29.8 | | | | | 48.1 / 26.7 | 44.2 / 29.9 | |

    - Đọc kết quả: độ tự tin dòng đầu là tín hiệu abstention mạnh. Bỏ 50% mẫu ít chắc nhất nâng EM từ 43 lên 70 (RepoEval) và từ 30 lên 49 (CCEval); xếp ngẫu nhiên thì đường phẳng. Lợi thế của chọn ngân sách so với ngân sách cố định là khoảng +1–2 EM ở độ phủ cao, gần như biến mất khi độ phủ ≤ 30% (mẫu dễ thì ngân sách nào cũng đúng). Xác suất logistic không xếp tốt hơn độ tự tin thô.
    - Dùng trong bài: "ứng dụng thực tế": cùng một tín hiệu vừa chọn ngân sách vừa quyết định có hiện gợi ý hay không. Không phải đóng góp mới (abstention theo log-prob đã phổ biến); chỉ là phần phân tích.
22. **B1c: chọn ngân sách theo từng dòng cho GT nhiều dòng — FAIL** (2026-10-25, CPU, `b1c_perline.py` `87a74582`, `modal_out/b1c_perline_summary.json`; toàn bộ test, mô phỏng offline bằng cách ghép dòng thứ k từ bản greedy 256 / 1024 / 1536 có độ tự tin dòng đó cao nhất; không sinh mới). Ngưỡng chốt trước: hybrid (dòng 1 theo E2, các dòng sau theo độ tự tin từng dòng) > E2 trên RepoEval với p < 0.05, và không hại CCEval.

    | EM [thắng, thua so với E2] | RepoEval toàn bộ | RepoEval GT nhiều dòng (24.6%) | CCEval |
    |---|---|---|---|
    | E2 learned3 (đối chứng) | 43.12 | 23.92 | 29.83 |
    | perline_conf | 41.69 [12, 35, p=0.001] | 20.87 | 29.12 |
    | **hybrid** | 42.06 [2, 19, p=0.0002] | 20.10 | 29.83 (bằng) |
    | oracle theo mẫu (3 ngân sách) | 46.19 | 27.23 | 32.91 |
    | oracle theo dòng | 46.00 | 27.48 | 32.68 |

    - CCEval: 0% GT nhiều dòng, nên bước này chỉ có nghĩa với RepoEval. Bản ghép tái tạo đúng EM đã lưu (0 lệch) khi lấy cả bản của một ngân sách.
    - Đọc kết quả: oracle theo dòng ≈ oracle theo mẫu, nên **không có dư địa** từ việc đổi ngân sách giữa các dòng. Khi một ngân sách viết sai dòng 1, các dòng sau của nó cũng thường sai. Ghép dòng từ các ngữ cảnh khác nhau còn phá tính liền mạch (hybrid thua 19 / thắng 2). Sinh lại có điều kiện theo từng dòng cũng khó cứu, vì cận trên đã không cao hơn. Đóng hướng #3b.
23. **Tái lập mốc trên Colab L4 — ĐẠT** (2026-10-25, `budget_sweep.py` không đổi, `colab_runner.ipynb`, Drive `runs/anchor_repoeval`). Python 3.13, torch 2.5.1+cu124 (pin; gỡ torchvision/torchaudio của Colab), transformers 4.46.3, tokenizers 0.20.3. align1536 trên 1000 mẫu dev RepoEval: **EM 37.0** so với Kaggle T4 36.8. Đồng thuận EM 99.4% (Colab thắng 4 / thua 2, p=0.69), văn bản trùng 85.6%. Prompt giống hệt (cross/in-file token khớp 100%); khác biệt chỉ là đảo token gần ngang nhau do kernel fp16 / batch. Torch 2.11 mặc định của Colab cũng ổn (đồng thuận EM 98.8% trên 256 mẫu, `runs/anchor_repoeval_torch211`). **Lưu ý:** so khớp bằng đồng thuận EM, không bằng văn bản trùng (văn bản chỉ trùng 75–89% giữa các GPU / batch, như E8).
    - Guardrail CU: cell watchdog trong `colab_runner.ipynb` (xem `HANDOFF_COLAB.md` mục 3.3).
24. **D2: likelihood chéo giữa các context để chọn đáp án — FAIL** (2026-10-25, `d2_crossll.py` `a03e1ad6`, GPU local RTX 4060 trong `.venv-gpu` ở gốc project (torch 2.5.1+cu124, transformers 4.46.3), ~3.1 giờ; `modal_out/d2_{repoeval,cceval}/scores.jsonl`, `modal_out/d2_crossll_summary.json`). Hướng H2 trong `research/directions_beyond_budget_2026-10-25.md`. Không sinh mới: dòng đầu của bản greedy ở mỗi ngân sách (sweep dev + hold-out) được chấm teacher-forced dưới cả 4 prompt ngân sách (tái dùng KV cache của prompt). m(a, b) = log-prob trung bình dòng đầu của bản a dưới prompt b, cùng định nghĩa `first_line_conf`. Sanity: m(a, a) khớp độ tự tin đã lưu (độ lệch trung vị 0.0006, 99.9% < 0.05). Ngưỡng chốt trước: learned3x (đặc trưng E2 + xmean / xmin / agree theo từng ngân sách, cross-fit dev↔hold-out) ≥ E2 learned3 + 1 EM trên cả hai bộ và McNemar gộp p < 0.05.

    | EM toàn bộ test [thắng, thua so với E2] | RepoEval 1600 | CCEval 2665 |
    |---|---|---|
    | E2 learned3 (đối chứng, tái lập đúng) | 43.12 | 29.83 |
    | conf3 (độ tự tin dưới chính prompt) | 43.06 [14, 15] | 29.72 [21, 24] |
    | xmean3 | 41.44 [15, 42] | 28.11 [39, 85] |
    | xmin3 | 39.69 [17, 72] | 25.85 [39, 145] |
    | agree3 (số ngân sách cho cùng dòng đầu) | 42.12 [21, 37] | 29.01 [19, 41] |
    | **PRIMARY learned3x** | 42.31 [9, 22, p=0.03] | 29.31 [21, 35] |
    | learned4x | 43.00 [18, 20] | 29.34 [23, 36] |
    | oracle 3 / 4 ngân sách | 46.19 / 46.88 | 32.91 / 33.55 |

    - Phân biệt trong cùng mẫu (cặp ứng viên đúng/sai, AUC): độ tự tin dưới chính prompt 0.80 / 0.82; trung bình dưới 2 prompt còn lại 0.67 / 0.70; xmin 0.62 / 0.59. Prompt đã sinh ra ứng viên là "giám khảo" tốt nhất; prompt khác chấm kém hơn hẳn. xmin dồn lựa chọn về 1536 (1216/1600, 1746/2665) vì ứng viên của 1536 thường được mọi prompt chấp nhận, không phải vì đúng.
    - Đồng thuận giữa các context (agree) cũng không thắng: các ngân sách cho cùng dòng đầu không làm dòng đó đúng hơn.
    - Đọc kết quả: tín hiệu chọn nằm ở độ tự tin dưới chính context, và đã được E2 khai thác. Likelihood chéo không thêm thông tin, thêm vào logistic còn làm giảm (nhiễu). **Đóng H2**; khoảng cách oracle (3–4 EM) không lấy lại được bằng tín hiệu likelihood chỉ dùng prefix.
25. **D1: Phân giải Call-Site & chèn ví dụ gọi hàm thực tế — ĐỘT PHÁ trên CCEval, lấn chỗ ở B=512 trên RepoEval** (2026-10-25, Colab L4, `d1_callsite_cpu.py`, `d1_callsite_gen.py`, `modal_out/d1_callsite_cpu_summary.json`, `modal_out/d1_repoeval_b512_summary.json`, `modal_out/d1_cceval_b1536_summary.json`; n=400 mẫu dev/bộ).
    - **Kiểm chứng CPU (Gate 0 trên toàn bộ 1600 RepoEval + 2665 CCEval):** Trên các mẫu `base` đoán sai, **65.0% (RepoEval)** và **84.1% (CCEval)** của `line_pfx + pred_line1` đã khớp chính xác trigger cú pháp (`callee(` hoặc `receiver.`) của GT. Truy xuất Call-Site (`Oracle-U`) thu hồi identifier còn thiếu của GT ở **87.4% / 62.8%** mẫu sai (vượt xa định nghĩa `Oracle-D`: 78.4% / 52.0%). Truy xuất tự thân không cần GT (`Self-UD`) thu hồi identifier ở **51.4% (358 mẫu RepoEval)** và **38.8% (554 mẫu CCEval)**.
    - **Kết quả sinh GPU trên Colab L4 (n=400/bộ, đồng thuận mốc `base` 99.0% / 98.5%):**

      | Điều kiện | RepoEval (B=512, n=400) | CCEval (B=1536, n=400) |
      |---|---|---|
      | `base` (mốc AlignCoder tại B) | **40.50** | 28.25 |
      | `oracle_u` (Call-sites dùng GT callee) | 34.25 (−6.25, 12 vs 37, p=0.0005) | **33.00 (+4.75, 30 vs 11, p=0.0043)** |
      | `self_u` (Call-sites từ `pred_line1`, không GT) | 33.75 (−6.75, 9 vs 36, p=0.0001) | **30.00 (+1.75, 18 vs 11, p=0.265)** |
      | `self_ud` (Call-sites + Def từ `pred_line1`, không GT) | 33.00 (−7.50, 9 vs 39, p=0.0000) | **30.50 (+2.25, 21 vs 12, p=0.163)** |
      | **Oracle gộp (`base` ∪ `self_u` ∪ `self_ud`, không GT)** | **43.25 (+2.75)** | **34.00 (+5.75)** |

    - **Giải phẫu nguyên nhân đối xứng ngược (CCEval thắng lớn vs RepoEval bị lấn chỗ ở B=512):**
      1. **Trên CCEval (`B=1536`):** `oracle_u` đạt **+4.75 EM (p=0.0043)** — vượt ngưỡng chốt trước (+4.0 EM, p<0.05) và gấp hơn 2 lần `Oracle-D` của F1 (+2.0 EM, p=0.13). Bản không dùng GT `self_ud` tăng **+2.25 EM** (thu hồi **47.4%** trần `oracle_u`, vượt ngưỡng ≥40%). Đây là **phương pháp đầu tiên trong toàn bộ dự án tạo mức tăng lớn trên CCEval** (nơi E2 chỉ đạt +0.97 EM và F3/E4/E5/E8 đều ≤ 0).
      2. **Tại sao RepoEval (`B=512`) bị giảm khi thay thế vô điều kiện nhưng Oracle gộp vẫn tăng +2.75 EM (43.25%)?** Trung bình khối Call-Site chiếm **~421 token**. Ở CCEval (`B=1536`), sau khi chèn 421 token vẫn còn **~1115 token** cho `saved_blocks` của AlignCoder. Ngược lại ở RepoEval (`B=512`), chèn 421 token Call-Site làm `saved_blocks` chỉ còn **~91 token** (mất 82% chunk top-1 của AlignCoder — vốn chứa dòng giống hệt GT tới 40.3% trên RepoEval theo E7b!). Dù vậy, `self_u`/`self_ud` vẫn cứu được **11 mẫu mới (+2.75 EM)** mà `base` bó tay (`oracle_base_plus_self = 43.25%`).
      3. **Hướng khắc phục trực tiếp (D1.2):** (a) Không trừ ngân sách của `saved_blocks` ở RepoEval hoặc nén Call-Site xuống ≤ 128 token (chỉ giữ 2 dòng call-site sát nhất); và (b) Coi `self_ud` là một **View bổ sung cho E2** (chọn có điều kiện giữa `base` / các mức ngân sách và `self_ud` bằng độ tự tin dòng đầu hoặc chỉ kích hoạt khi `base` không chắc chắn ở phần tham số sau `(` / `.`).
26. **D1.2: Call-Site Không Lấn Chỗ (`compact_ud`) + Chọn ứng viên theo Độ tự tin dòng đầu (`maxconf` / `gated_cv`) — ĐẠT trên cả hai bộ** (2026-10-26, Colab L4, `d12_compact_gated.py`, `modal_out/d12_repoeval_b512_summary.json`, `modal_out/d12_cceval_b1536_summary.json`; n=400 mẫu dev/bộ).
    - **Thiết kế:**
      1. Chấm teacher-forced `first_line_conf` cho các ứng viên `base`, `self_u`, `self_ud` của D1.
      2. Sinh thêm `compact_u` và `compact_ud` (nén Call-Site xuống trung bình ~73–76 token, **giữ nguyên 100% 512 token `saved_blocks` trên RepoEval** để không mất chunk top-1 của AlignCoder).
      3. Chọn giữa `base` và ứng viên Call-Site bằng chính quy tắc `first_line_conf` không tham số (`_maxconf`, giống hệt `conf3` của E2) hoặc bộ lọc chéo 2-fold (`_gated_cv`).
    - **Kết quả thực nghiệm (n=400/bộ):**

      | Chính sách / Điều kiện | RepoEval (B=512, n=400) | CCEval (B=1536, n=400) | Ý nghĩa |
      |---|---|---|---|
      | `base` (mốc cố định tốt nhất từng bộ) | 40.50 | 28.25 | Mốc đối chứng mạnh nhất (`B=512` trên RepoEval đã cao hơn `align1536` +3.7 EM). |
      | `self_ud` thô (D1, ~421 tok) | 33.00 (−7.50) | 30.50 (+2.25, 21W/12L) | Lấn chỗ ở B=512, nhưng giàu ngữ cảnh ở B=1536. |
      | `compact_ud` thô (D1.2, ~75 tok, 0 lấn chỗ) | **39.75 (−0.75, 7W/10L)** | 29.00 (+0.75, 10W/7L) | Xoá bỏ hoàn toàn sự sụp đổ trên RepoEval (+6.75 EM so với `self_ud` thô). |
      | **`self_ud_maxconf` (`base` vs `self_ud` theo `conf`)** | 39.00 (−1.50) | **31.25 (+3.00, 16W/4L, p=0.0118)** | 🌟 **Đột phá trên CCEval:** Chọn bằng `first_line_conf` loại bỏ 8/12 mẫu thua, giữ 16 mẫu thắng (**+3.00 EM, p=0.0118**). |
      | **`compact_ud_maxconf` (`base` vs `compact_ud` theo `conf`)** | **41.00 (+0.50, 5W/3L)** | **30.00 (+1.75, 9W/2L, p=0.065)** | ✅ **Dương trên CẢ HAI BỘ** so với ngân sách cố định tốt nhất từng bộ (gộp 2 bộ: **14 thắng vs 5 thua**; trên CCEval `compact_ud_gated_cv` đạt **30.25, +2.00 EM, 8 thắng vs 0 thua, p=0.0078**). |
      | **Oracle gộp không dùng GT (`oracle_all_zero_gt`)** | **44.50 (+4.00 so với B=512; +7.70 so với align1536)** | **34.25 (+6.00 so với B=1536)** | Trần không nhìn GT vượt xa mọi phương pháp đơn lẻ. |

27. **B2: luận điểm ngân sách với DeepSeek-Coder-6.7B — ĐẠT** (2026-10-25, Colab L4, torch 2.5.1, `budget_sweep.py` không đổi với `--generator deepseek-ai/deepseek-coder-6.7b-base`, `b2_analyze.py` `0616a155`; Drive `runs/sweep67b_{repoeval,cceval}/b{256,512,1024,1536}`, `runs/b2_summary.json`; cùng 1000 mẫu dev/bộ, cùng candidates / prompt). Batch size: RepoEval b256 384 mẫu đầu + CCEval b256 840 mẫu đầu ở batch 4, còn lại batch 2 (batch 4 tràn bộ nhớ L4 ở prompt ~2048 token); khác batch chỉ đảo token gần ngang nhau (xem mục 23). Ngưỡng chốt trước: trên cả hai bộ, oracle4 − ngân sách cố định tốt nhất ≥ 4 EM VÀ learned3 > ngân sách cố định tốt nhất. Chạy lại cùng phân tích trên sweep 1.3b tái lập đúng mốc (40.5 / 36.8, 28.5).

    | EM dev [thắng, thua, p so với ngân sách cố định tốt nhất] | 256 | 512 | 1024 | 1536 | conf3 | learned3 (E2) | oracle4 | oracle − tốt nhất |
    |---|---|---|---|---|---|---|---|---|
    | RepoEval 1.3b | 38.0 | **40.5** | 40.4 | 36.8 | 41.8 | 42.4 [42, 23, p=0.025] | 46.1 | 5.6 |
    | RepoEval **6.7b** | 42.8 | **45.5** | 44.5 | 41.9 | 46.8 [34, 21, p=0.10] | 47.0 [34, 19, p=0.053] | 51.1 | **5.6** |
    | CCEval 1.3b | 19.1 | 22.9 | 26.7 | **28.5** | 29.2 | 30.0 [24, 9, p=0.014] | 32.6 | 4.1 |
    | CCEval **6.7b** | 24.1 | 27.2 | 32.4 | **33.0** | 34.1 [26, 15, p=0.12] | 33.7 [24, 17, p=0.35] | 37.4 | **4.4** |

    - Gộp hai bộ (6.7b): learned3 58 thắng / 36 thua (p=0.030); conf3 60 / 36 (p=0.018). Từng bộ riêng chưa có ý nghĩa ở n=1000.
    - Đọc kết quả: **hình dạng giữ nguyên khi tăng 5× tham số.** RepoEval vẫn tốt nhất ở 512, và 1536 (cấu hình AlignCoder) vẫn kém 3.6 EM: model lớn không bù được phần in-file bị lấn chỗ. CCEval vẫn tăng đều tới 1536. Dư địa oracle không co lại (5.6 / 4.4 so với 5.6 / 4.1). Mức tăng của bộ chọn nhỏ hơn với 6.7b (+1.5 / +0.7 so với +1.9 / +1.5), nhất là CCEval; conf3 không học còn nhỉnh hơn learned3 trên CCEval. Cần xác nhận trên hold-out nếu đưa 6.7b vào bảng chính (~6 giờ L4 cho 4 ngân sách, hoặc 3 ngân sách 256/1024/1536 ~4.5 giờ).
    - Vận hành: hai lần OOM ở batch 4 (đã ghi ở `HANDOFF_COLAB.md` 3.3); hai lần MCP gắn nhầm vào runtime CPU khác và Drive ở đó hiện trạng thái cũ, khiến tưởng mất VM. VM L4 thực tế chạy liên tục ~8.8 giờ.

## Bước tiếp theo
**Chuyển sang Colab (2026-10-25):** Modal gần hết credit. Session mới đọc `HANDOFF_COLAB.md` trước: cách chạy trên Colab (`colab_runner.ipynb`, `colab_bundle.zip`, tạo lại bằng `make_colab_bundle.py`) và danh sách việc tiếp theo theo thứ tự ưu tiên.

**Cập nhật 2026-10-25 (tối):** E6 cascade FAIL, E7 cho thấy lỗi chủ yếu là generator dùng sai thông tin đã có trong prompt (31–38% mẫu), ngân sách chỉ chạm 4–6%. Kế hoạch ưu tiên mới (decoding tương phản kiểu CAD, trộn phân phối theo ngân sách, phân rã sâu nhóm usage, kiểm tra tĩnh trên pool E5) ở `PROGRESS_2026-10-25.md`.

**Hướng chính đã có kết quả dương: chính sách ngân sách theo từng mẫu (E2).** E4 (đoán từ 1 lần prefill) không đạt. Việc còn lại để thành phương pháp:
- Train chính sách trên repo train (cắt kiểu hard), không cross-fit trên test.
- Đo chi phí thật với nháp dòng đầu.
- Kiểm tra trên generator 6.7b, thêm ES và CCEval Java.
- Baseline: Repoformer (quyết định truy xuất hay không), CODEFILTER.

(Ghi chú cũ:) Pha 2 đạt, E0 xong (thứ tự là yếu tố phụ). E1 FAIL (scorer text không học được ΔLL). AMI FAIL G1 (chỉ giúp CCEval ở mức ngang P0). Hướng đề xuất tiếp (chưa chạy, cần người dùng chọn):
- **Proxy ΔLL không cần GT, tính bằng chính generator** (kiểu AMI của LongCodeZip: chunk c làm k dòng cuối của left context dễ đoán hơn bao nhiêu). Kiểm tra rẻ trên pilot: Spearman với ΔLL thật, rồi đóng gói theo proxy. Nếu proxy tốt thì dùng làm thầy để distill, thay cho nhãn GT trên dữ liệu train lệch phân phối.
- Hoặc bỏ chấm theo chunk, chỉ học chính sách ngân sách / chế độ (in-file vs cross-file) theo từng mẫu (E2 rút gọn).

(Ghi chú cũ, trước E1:) Chuyển sang E1: train scorer utility. Theo hướng của người dùng (2026-10-24): train theo khung RLCoder (AlignCoder train giống RLCoder, chỉ khác vòng lặp lúc infer, mà train AlignCoder rất lâu), tận dụng `experiments/rlcoder_controlled_kaggle` + `Data4RLCoder`; đổi nhãn sang ΔLL mềm và thêm chunk in-file vào pool. Hướng đề xuất:
- Reranker cho cả in-file và cross-file (AlignRetriever + nhãn mềm ΔLL), kèm đầu ra quyết định dừng / ngân sách.
- Dữ liệu train lấy từ tập train của AlignCoder, không lấy repo của tập đánh giá.
- Đánh giá trên 1000 mẫu/bộ, mốc so sánh là P0 và P2.

Đang chờ ý kiến cô:
- (a) Đã có bài nào truy xuất chung in-file + cross-file chưa?
- (b) Có dùng phần code sau con trỏ không?
- (c) Khi nào xác nhận lại với generator 6.7b?

Báo cáo cho cô (kết quả 1–7, 2026-10-19): https://claude.ai/artifact/A4GkM2hapbik8xJyc13fth. Nguồn là `report_supervisor.html`.
- Bản cũ https://claude.ai/code/artifact/75b317db-f706-449f-99da-8ff63a4bb782 không mở được từ tài khoản hiện tại.

## Quy tắc làm việc
- Không quay lại vòng audit evaluator/manifest, không đổi hướng nghiên cứu khi chưa có số liệu.
- Mỗi bước là một thí nghiệm có giả thuyết, có đối chứng, có ngưỡng quyết định.
- Mỗi lần chạy phải tái lập được mốc align1536 (RepoEval 36.8, CCEval 28.5). Lệch là có lỗi.
- Khi sửa script: sửa tối thiểu, ghi hash, không đổi logic đo. Script mới viết thành file riêng và import `dep_oracle`/`joint_dense`.
