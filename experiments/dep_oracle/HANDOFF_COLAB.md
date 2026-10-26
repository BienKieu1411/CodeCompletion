# Bàn giao: chạy tiếp dòng thí nghiệm dep_oracle trên Google Colab

_2026-10-25. Viết cho một session Claude Code mới, chạy GPU bằng Colab (compute units) thay cho Modal (tài khoản Modal `thanhnguyenvinh169` gần hết $30)._

Đọc theo thứ tự:
0. `NEXT_SESSION_BRIEF.md`: mục tiêu, luận điểm, cây quyết định cho các bước tiếp theo (ngắn, đọc trước).
1. File này: trạng thái, cách chạy trên Colab, việc tiếp theo.
2. `HANDOFF.md`: số liệu chi tiết và hash script của mọi thí nghiệm (mục 1–19). Quan trọng nhất là mục 11 (E2), 15–19 (E6, E7, E7b, E5, E8).
3. `PROGRESS_2026-10-25.md`: phân rã lỗi E7/E7b và bảng hướng ưu tiên.
4. `../../research/novelty_review_2026-10-25.md`: Caret, Ciniselli và các bài gần, lý do E2 một mình chưa đủ cho bài long.

## 1. Bài toán và trạng thái

**Setting:** repo-level code completion, giữ nguyên generator (`deepseek-coder-1.3b-base`, greedy, 64 token), dùng lại top-10 chunk AlignCoder đã truy xuất và lưu sẵn. **Không chạy lại retriever AlignCoder** (rất chậm). Prompt 2048 token = cross-file (ngân sách B) + `# file path` + code trong file. Benchmark: RepoEval API (1600 mẫu) và CCEval Python (2665 mẫu). Dev = 1000 mẫu/bộ (`D.pick_subset(..., 1000, seed=0)`, phân tầng theo repo); hold-out = phần còn lại. Đo bằng simple EM (`D.simple_score`) và McNemar ghép cặp.

**Mốc bắt buộc tái lập:** align1536 trên dev = **RepoEval 36.8 / CCEval 28.5**. Lệch là có lỗi môi trường (xem mục 3.4).

**Đã chứng minh:**
- Ngân sách cross-file cố định lấn chỗ code trong file. RepoEval: 512 → 40.5, 1536 → 36.8. CCEval tăng đều tới 1536. Chọn đúng B cho từng mẫu (oracle): 47.3 / 33.3.
- "Giống" không phải "hữu ích" (Spearman 0.03–0.14). Thứ tự chunk chỉ là yếu tố phụ (E0).
- **E2 (kết quả dương duy nhất):** sinh greedy ở B = 256 / 1024 / 1536, chọn bằng độ tự tin dòng đầu, có hoặc không có logistic học thêm. Trên toàn bộ test: **43.12 / 29.83**, so với ngân sách cố định tốt nhất của từng bộ 41.38 / 28.86 (p = 0.007 / 0.005) và AlignCoder 36.81 / 28.86.

**Đã loại (đừng làm lại):** chấm điểm từng chunk không cần GT (E1, E1b, AMI, Revela zero-shot: Spearman ≤ 0.2), nén code trong file (E3), probe trạng thái ẩn từ một lần prefill (E4), cascade giảm chi phí (E6), decoding tương phản với cross-file (E8, CAD làm EM giảm).

**Vì sao chưa đủ cho bài long:**
- Caret (arXiv 2610.04837) đã làm "sinh nhiều biến thể context rồi chọn", với mức tăng lớn hơn.
- E7 cho thấy cắt do ngân sách chỉ chạm 4–6% mẫu. Lỗi lớn nhất (31–38%) là generator dùng sai thông tin đã có trong prompt.
- E5: trục ngân sách chỉ giúp nơi có lấn chỗ (RepoEval), không giúp CCEval.
- Ở cửa sổ 8k, CCEval chỉ 3.5% mẫu bị cắt.

**Đang chờ cô trả lời:** có được dùng right context (code sau con trỏ) không; có chấp nhận đổi cách decode với generator đóng băng là cùng setting không; khi nào xác nhận với 6.7b.

## 2. Bố cục file

| Nhóm | File |
|---|---|
| Lõi (import bởi mọi script) | `dep_oracle.py` (dữ liệu, dựng prompt, EM, McNemar), `joint_retrieval.py`, `joint_dense.py`, `likelihood_probe.py`, `chunk_pilot.py`, `budget_sweep.py` |
| Candidates đã lưu | `prediction_with_candidates.jsonl` (RepoEval API), `cceval_python_prediction_with_candidates.jsonl` |
| GT của dev | `probe_inputs/{repoeval,cceval}/gt.jsonl` |
| Sweep ngân sách | dev: `kaggle_sweep_{ds}_out/sweep/b{0,256,512,1024,1536}/prediction.jsonl` (Kaggle T4); hold-out: `modal_out/e2hold_{ds}/b{256,512,1024,1536}` |
| Thí nghiệm | `e0_order.py` … `e8_cad.py`; mỗi file có docstring ghi giả thuyết, cách chạy và ngưỡng chốt trước |
| Wrapper Modal | `modal_*.py`. **Không dùng trên Colab.** Mỗi wrapper chỉ gọi một lệnh CLI; lệnh tương ứng ở mục 4 |
| Kết quả | `modal_out/*` (summary JSON + jsonl theo mẫu) |

## 3. Chạy trên Colab

### 3.1 Chuẩn bị (máy local, một lần)
- Upload `colab_bundle.zip` (cùng thư mục với file này) lên Google Drive, ví dụ `MyDrive/dep_oracle/`.
- Gói zip có: toàn bộ script, candidates, `probe_inputs`, sweep dev và hold-out, summary nhỏ trong `modal_out`.
- Gói zip **không** có: `aligncoder_data/` (1 GB, tự tải lại từ HuggingFace), `modal_out/e1*_labels` và `e4_*` (lớn, chỉ cần khi chạy lại E1/E4).
- Tạo lại gói: `python make_colab_bundle.py`.

### 3.2 Notebook
Mở `colab_runner.ipynb` trên Colab (File → Upload notebook). Chọn **Runtime → Change runtime type → L4** (hoặc A100 nếu cần nhanh; T4 chạy được nhưng chậm khoảng 2–3 lần). Các cell làm theo thứ tự:
1. Mount Drive, giải nén bundle vào `/content/dep_oracle`.
2. `pip install transformers==4.46.3 peft==0.13.2 pyarrow` (giữ torch sẵn có của Colab).
3. Đặt `HF_HOME` và thư mục dữ liệu trên Drive để không tải lại model/dữ liệu mỗi phiên:
   - `HF_HOME=/content/drive/MyDrive/dep_oracle/cache/hf`
   - `--data_dir /content/drive/MyDrive/dep_oracle/cache/aligncoder_data`
4. **Kiểm tra mốc:** sinh align1536 trên 1000 mẫu dev RepoEval (khoảng 10–15 phút trên L4) rồi so EM với 36.8.
5. Cell mẫu chạy một thí nghiệm ở nền (`nohup … &`) và xem log.

### 3.3 Quy ước để không mất kết quả khi Colab ngắt
- Luôn đặt `--out` trên Drive (`/content/drive/MyDrive/dep_oracle/runs/<tên>`).
- Mọi script sinh/chấm đều **ghi nối jsonl và bỏ qua mẫu đã xong**: chạy lại đúng lệnh là chạy tiếp.
- Chạy nền bằng `!nohup python … > /content/drive/MyDrive/dep_oracle/runs/<tên>.log 2>&1 &`, xem bằng `!tail -5 <log>`. Giữ tab Colab mở; phiên miễn phí hoặc Pro vẫn có thể bị ngắt sau vài giờ.
- **Guardrail CU (cell watchdog trong `colab_runner.ipynb`, chạy ngay sau cell 3):** thread nền tự `runtime.unassign()` khi không còn job `python <script>.py` / tiến trình GPU trong 20 phút, hoặc 6 giờ kể từ lần `keepalive()` cuối (phòng job treo); flush Drive trước khi trả. Session Claude gọi `keepalive()` mỗi lần kiểm tra job; job > 6 giờ thì tăng `WD['max_hours']`. Log ở `$D/runs/watchdog.log`. Kernel restart thì chạy lại cell watchdog.
- **Khôi phục khi mất VM:** trên Drive có đủ bundle, dữ liệu (`cache/aligncoder_data`), model 1.3b + 6.7b (`cache/hf`, 16 GB), kết quả (`runs/`). Chỉ cần: mount Drive → giải nén bundle → cài torch 2.5.1 / gỡ torchvision / transformers 4.46.3 (~3 phút) → cell watchdog → `!nohup bash $D/runs/b2_resume.sh &` (lệnh B2 đã lưu; script bỏ qua mẫu đã xong). Lưu ý `rm -rf /content/dep_oracle` khi đang đứng trong thư mục đó làm hỏng các lệnh `pip` sau: `%cd /content` trước. Batch 4 với 6.7b tràn bộ nhớ L4 ở prompt ~2048 token: RepoEval dùng `--batch_size 2` (CCEval cũng OOM ở batch 4 với các prompt dài nhất: dùng 2).
- **Colab MCP có thể gắn vào runtime khác** (một notebook trống mới mỗi lần kết nối lại, đôi khi là runtime CPU riêng). Trước khi kết luận "mất VM": kiểm tra `uptime`, `ps … budget_sweep`, `nvidia-smi` và `'_WD_THREAD' in globals()`; nếu không khớp, nhờ người dùng đưa đúng tab lên trước và xem Runtime → Manage sessions (tắt runtime thừa để khỏi tốn CU).
- Phân tích (`analyze`) chạy trên CPU, ở local hay Colab đều được. Nếu chạy local: tải thư mục `runs/<tên>` về `modal_out/<tên>` (các lệnh analyze đọc từ `modal_out/`).

### 3.4 Lỗi đã gặp
- **Phiên bản:** dùng `transformers==4.46.3`. Mốc 36.8 / 28.5 được tái lập với bản này (torch 2.5.1). Bản khác có thể đổi hành vi `generate` và padding; luôn chạy cell kiểm tra mốc trước.
- **OOM khi sampling:** 24 chuỗi × prefill 2048 token tràn bộ nhớ L4 (22 GB). `e5_budget_tts.py` đã giới hạn 8 chuỗi mỗi lần gọi và bật `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Cross-encoder trong `e1_train.py` backward theo từng mẫu.
- **Dữ liệu train Data4RLCoder:** cắt ngẫu nhiên cho nhãn gần như rỗng. Muốn có tín hiệu thì dùng `--hard` trong `e1_label.py` (dòng gọi tên định nghĩa ở file khác, file ≥ 250 dòng).
- **Git Bash trên máy local:** `/vol/...` bị đổi thành đường dẫn Windows. Không ảnh hưởng Colab.
- Kết quả dev chạy trên Kaggle T4, hold-out chạy trên Modal L4. Các bản sinh lại khớp nhau 98.5–100%, nên ghép được, nhưng cần ghi rõ khi báo cáo.

## 4. Lệnh CLI (thay cho wrapper Modal)

Đặt `D=/content/drive/MyDrive/dep_oracle`, `DATA=$D/cache/aligncoder_data`. Chạy từ `/content/dep_oracle`. Với RepoEval dùng `--dataset repoeval --cands prediction_with_candidates.jsonl`; với CCEval dùng `--dataset cceval --cands cceval_python_prediction_with_candidates.jsonl`.

| Mục đích | Lệnh | Ước tính trên L4 |
|---|---|---|
| Quét ngân sách (dev) | `python budget_sweep.py sweep --dataset repoeval --cands prediction_with_candidates.jsonl --data_dir $DATA --budgets 256,512,1024,1536 --out $D/runs/sweep_repoeval` | ~10–12 phút mỗi ngân sách / 1000 mẫu |
| Quét ngân sách (hold-out) | thêm `--holdout` | RepoEval 600, CCEval 1665 mẫu |
| E2 offline / xác nhận | `python e2_offline.py`; `python e2_confirm.py` | CPU, vài phút |
| E5 (pool sampling + RCV) | `python e5_budget_tts.py run --dataset … --data_dir $DATA --out $D/runs/e5_<ds> --shard 0 --nshards 1` | ~1.5–2 giờ / 1000 mẫu |
| E8 (CAD) | xem docstring `e8_cad.py` | |
| Probe trạng thái ẩn | `python e4_probe.py extract --dataset … --out $D/runs/e4_<ds>` | vài phút cho toàn bộ test |
| Nhãn / train scorer (E1) | `e1_label.py --hard`, `e1_train.py` (docstring) | nhãn ~30 phút / 1000 mẫu |

Generator khác (ví dụ 6.7b): thêm `--generator deepseek-ai/deepseek-coder-6.7b-base` (có ở mọi script sinh). 6.7b fp16 cần khoảng 14 GB: L4 chạy được với batch nhỏ hơn (`--batch_size 4`, `--gen_batch_size 4`); A100 thoải mái hơn.

## 5. Việc tiếp theo, theo thứ tự ưu tiên

Đã thống nhất với người dùng: mục tiêu là **bài long**. E2 cần thêm một trục đóng góp mạnh hơn.

**Trạng thái 2026-10-26:** #1 FAIL, #2 xong, #3 FAIL, #5 ĐẠT (HANDOFF mục 20–22, 27). #6 (RepoEval 8k = B3) **hoãn, người dùng sẽ xem lại**. Tổng hợp và bước tiếp đề xuất: `REPORT_2026-10-26.md`.

| # | Việc | Loại | Ước tính | Ghi chú |
|---|---|---|---|---|
| 1 | **Kiểm tra tĩnh theo symbol trên pool 8 mẫu của E5** (`modal_out/e5_*/items_*.jsonl`) | CPU | 0 CU | Hướng #4 trong PROGRESS. Logic có sẵn trong `e7_errors.py` / `e7b_usage.py` |
| 2 | **Abstention / selective prediction:** đường EM theo độ phủ dựa trên độ tự tin E2 | CPU | 0 CU | Hướng #5. Dùng sweep có sẵn |
| 3 | **Chọn ngân sách theo từng dòng** cho GT nhiều dòng (RepoEval) | CPU | 0 CU | Hướng #3b. Cần độ tự tin từng dòng: có `lp`/`pieces` trong sweep |
| 4 | **Trộn phân phối token của 3 ngân sách ở từng bước** (theo entropy) | GPU, pilot 200 mẫu/bộ | ~1 giờ L4 | Hướng #2. Viết script mới theo khuôn `e8_cad.py` (đã có vòng decode thủ công) |
| 5 | **Tổng quát hoá E2:** sweep 4 ngân sách với 6.7b trên dev | GPU | ~3–4 giờ L4 | Reviewer chắc chắn đòi |
| 6 | **RepoEval ở cửa sổ 8k:** sweep vài ngân sách | GPU | ~2–3 giờ L4 | Trả lời câu "sao không dùng cửa sổ lớn". CCEval ở 8k không đáng chạy (3.5% bị cắt) |
| 7 | **Train chính sách E2 trên repo train** (Data4RLCoder, `--hard`), thay cross-fit trên test | GPU | ~2 giờ L4 | Cần cho con số "chuẩn" trong paper |
| 8 | Nếu cô cho dùng right context: chạy lại E5 với RCV, và thử tái lập Caret | GPU | | |

Mỗi bước: ghi giả thuyết, đối chứng và ngưỡng vào docstring **trước khi chạy**; chạy smoke (`--limit 4`) trước; ghi kết quả và hash script vào `HANDOFF.md`.

## 6. Bối cảnh khác

- Báo cáo mới nhất cho cô: https://claude.ai/artifact/1nmrRJLZn2AhSPVezYrEJV (nguồn `report_budget_policy.html`). Chưa có E5–E8; nên cập nhật trước buổi gặp.
- Partner (em của người dùng) có hai hướng riêng:
  - **PG-MSLD** (listwise KD + STOP): ngang RLCoder khi khớp ngân sách.
  - **MB-SPT + SPC** (CCEval, cửa sổ 8k): +2.7 / +4.8 EM so với AlignCoder 2k, nhưng bị nhiễu bởi cỡ cửa sổ. Đã khuyên thêm nhánh AlignCoder-8k và RepoEval.
- Memory của Claude Code trong project này có tóm tắt từng mốc (`dep-oracle-research-state`).
- Quy tắc làm việc của người dùng (cuối `HANDOFF.md`): không đổi hướng khi chưa có số liệu; mỗi bước có giả thuyết, đối chứng, ngưỡng; luôn tái lập mốc; sửa script tối thiểu và ghi hash. Khi chờ job chạy nền, không polling dày: ước thời gian rồi kiểm tra thưa.
