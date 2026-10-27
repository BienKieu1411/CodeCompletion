# Brief cho session tiếp theo (đọc file này trước tiên)

_2026-10-25. Ngắn có chủ đích. Chi tiết chạy ở `HANDOFF_COLAB.md`; số liệu và hash ở `HANDOFF.md` (dùng grep theo số mục, đừng đọc cả file)._

## 1. Mục tiêu
Một **bài long** (ICSE / ASE / ACL main) về repo-level code completion với **generator đóng băng** (DeepSeek-Coder, dòng AlignCoder / RLCoder). Cô hướng dẫn yêu cầu mọi claim phải có số liệu: mỗi bước có giả thuyết, đối chứng và ngưỡng chốt trước.

## 2. Luận điểm đang theo
> **Chia cửa sổ prompt giữa code trong file và code từ file khác là một đòn bẩy bị bỏ qua. Ngân sách tốt nhất thay đổi theo từng mẫu, và có thể chọn bằng tín hiệu từ đầu ra của generator.**

Bằng chứng đã có (toàn bộ test, DeepSeek-Coder-1.3B, cửa sổ 2k, candidates AlignCoder):

| | RepoEval API | CCEval Python |
|---|---|---|
| AlignCoder (cross-file 1536 cố định) | 36.81 | 28.86 |
| Ngân sách cố định tốt nhất của từng bộ | 41.38 | 28.86 |
| **E2: sinh ở 256 / 1024 / 1536, chọn theo độ tự tin dòng đầu (+ logistic)** | **43.12** (p=0.007) | **29.83** (p=0.005) |
| Oracle chọn ngân sách | 46.88 | 33.55 |

## 3. Đã chốt, không mở lại
- **Chấm điểm từng chunk không cần GT là không khả thi** (E1, E1b, AMI, Revela: Spearman ≤ 0.2). Probe từ một lần prefill (E4), cascade (E6), CAD (E8) và nén in-file (E3) đều FAIL.
- **Tín hiệu chọn ngân sách nằm ở việc thật sự sinh**, không đọc được trước khi sinh. Đổi ngân sách giữa lúc sinh tại `.` / `(` (H1, HANDOFF mục 30) FAIL: kém conf3 trên cả hai bộ. CAD: phân tích nguyên nhân ở mục 29.
- **D2 (likelihood chéo giữa các prompt) FAIL** (HANDOFF mục 24): prompt khác chấm ứng viên kém hơn chính prompt đã sinh ra nó; E2 đã là bộ chọn tốt nhất trong họ tín hiệu likelihood chỉ dùng prefix. Hướng mới còn lại: H1 / D1 trong `research/directions_beyond_budget_2026-10-25.md`.
- **E9 (HANDOFF mục 35): E2 bên trong CADRE v2.8 FAIL theo ngưỡng chốt trước.** RepoEval-API 40.6 → 42.9 (p<0.001); CCEval-py hòa. Lưu ý: CADRE v2.8 của partner thực chạy ở cửa sổ 3072, không phải 2560.
- **E9c (HANDOFF mục 36): dải budget chỉnh theo cửa sổ {384, 1536, 2304} → PASS.** CADRE 40.6 / 25.0 → CADRE + E2 44.3 / 26.9 (scorer CADRE, p<0.001 cả hai bộ). Ở CCEval phần lớn là nhờ cross-file được thêm chỗ; conf3 chỉ hơn b2304 cố định +0.3. E9d (conf6) đang chạy trên Modal.
- **Rủi ro novelty:** Caret (arXiv 2610.04837) và Ciniselli et al. 2025 đã có cơ chế "sinh nhiều biến thể context rồi chọn". Vì vậy đóng góp phải là **trục ngân sách và phân tích**, không phải cơ chế chọn.
- **Giới hạn đã biết:**
  - Trục ngân sách chỉ mạnh khi có lấn chỗ: RepoEval mạnh, CCEval yếu (E5). Ở cửa sổ 8k, CCEval chỉ 3.5% mẫu bị cắt, RepoEval còn 31.7%.
  - Lỗi lớn nhất (31–38% mẫu) là generator dùng sai thông tin đã có trong prompt (E7).

## 4. Cây quyết định (làm theo thứ tự; dừng nhánh khi FAIL)

```
B1  CPU, 0 CU: ba phân tích trên dữ liệu có sẵn — XONG 2026-10-25 (HANDOFF mục 20–22)
    a. Kiểm tra symbol tĩnh trên pool 8 mẫu của E5  → FAIL (+0.0 / +0.3; lọc còn bỏ đáp án đúng). Đóng.
    b. Đường EM theo độ phủ (abstention) dùng độ tự tin E2 → xong: độ phủ 50% cho EM 70.5 / 48.9 (từ 43.1 / 29.8)
    c. Chọn ngân sách theo từng dòng cho GT nhiều dòng (RepoEval) → FAIL (oracle theo dòng ≈ theo mẫu; ghép dòng hại). Đóng.
B2  GPU ~3–4 giờ L4: sweep 4 ngân sách với deepseek-coder-6.7b trên dev 1000/bộ — ĐẠT 2026-10-25 (HANDOFF mục 27)
    → oracle gap 5.6 / 4.4; learned3 47.0 vs 45.5 / 33.7 vs 33.0 (gộp p=0.030); RepoEval vẫn tốt nhất ở 512. Thực tế ~8.8 giờ L4 (batch 2).
    ĐẠT nếu: oracle − ngân sách cố định tốt nhất ≥ 4 EM VÀ E2 (offline) thắng ngân sách cố định tốt nhất trên cả hai bộ
    ├─ ĐẠT  → (B3 hoãn, xem "Chạy sau") → chờ người dùng chọn bước tiếp (REPORT_2026-10-26.md mục 7)
    └─ FAIL → luận điểm chỉ đúng với generator nhỏ: dừng GPU, báo người dùng (chuyển hướng bài phân tích / short)

CHẠY SAU (người dùng sẽ xem lại, 2026-10-26; đừng tự chạy):
B3  GPU ~2–3 giờ L4: RepoEval ở cửa sổ 8k (ngân sách cross-file 1024 / 2048 / 3800)
    ĐẠT nếu: oracle − ngân sách cố định tốt nhất ≥ 3 EM
    ├─ ĐẠT  → câu trả lời cho "sao không dùng cửa sổ lớn": vẫn còn tác dụng với file dài
    └─ FAIL → giới hạn phạm vi bài ở cửa sổ nhỏ / độ trễ thấp, ghi rõ
B4  GPU ~2 giờ L4: train chính sách E2 trên repo train (Data4RLCoder, e1_label.py --hard; nhãn = EM hoặc log p(GT) ở từng ngân sách)
    ĐẠT nếu: trên toàn bộ test thắng ngân sách cố định tốt nhất trên cả hai bộ, McNemar gộp p < 0.05
B5  Chỉ khi cô cho dùng right context: E5 với RCV + tái lập Caret làm baseline (cần code Caret)
B6  Phương pháp mạnh hơn (nếu còn CU): trộn phân phối token của 3 ngân sách ở từng bước (pilot 200 mẫu/bộ,
    ~1 giờ L4; viết theo khuôn e8_cad.py); đạt nếu vượt E2 trên cả hai bộ
```

So với branch BienKieu của partner (repo `../CodeCompletion`, chấm cùng scorer, 2026-10-26): CADRE 8k đạt CCEval-py 30.58 so với AlignCoder-2k 28.74 (p = 0.0008), RepoEval-API 42.19, còn E2 ở 2k đạt 43.12. (Baseline RepoEval 41.25 mà branch BienKieu dùng là một lần chạy không chuẩn ngày 6/10; baseline tái tạo được là align1536 = 36.81. Xem HANDOFF mục 28.) Phần thắng của CADRE chủ yếu là giảm lấn chỗ; nó chỉ sửa khoảng 4% lỗi "dùng sai". Đề xuất cho CADRE (chưa chạy): dòng call site trong Channel B; chọn theo độ tự tin giữa CADRE 2k / 4k / 8k; thêm nhánh AlignCoder-8k và thống nhất scorer. Code E/D đã push lên branch `Phong` (`experiments/dep_oracle/`, commit `0cb7e9d`).

Báo cáo tổng hợp mới nhất (gộp các lần chạy + báo cáo Antigravity D1/D1.2/D2, kèm hiệu chỉnh): `REPORT_2026-10-26.md`. Ứng viên bước tiếp (chưa chốt): hợp nhất E2 + D1.2 với một chính sách chốt trước, xác nhận D1.2 trên hold-out, hold-out 6.7B.

Ngân sách CU: B2 + B3 + B4 khoảng 8–9 giờ L4. Làm B1 trước vì miễn phí. Ưu tiên B2: nếu luận điểm không giữ được với 6.7b thì các bước sau vô nghĩa.

## 5. Cách làm việc
- Mỗi bước: viết giả thuyết, đối chứng và ngưỡng vào docstring của script mới → smoke `--limit 4` → chạy thật → ghi một mục mới (số thứ tự tiếp theo) vào `HANDOFF.md` kèm hash script → trả lời người dùng 5–10 dòng.
- Script mới là file riêng, import `dep_oracle` / `budget_sweep`; không sửa logic đo của script cũ.
- Luôn tái lập mốc trước khi tin số mới: align1536 trên dev = 36.8 / 28.5 (cell kiểm tra trong `colab_runner.ipynb`).
- **Giữ context nhỏ:**
  - Đọc file bằng grep, head, tail; không mở jsonl lớn.
  - Tóm tắt output bằng script.
  - Chờ job thì ước thời gian rồi kiểm tra thưa.
  - FAIL thì ghi lại và dừng, không tự mở hướng mới.
- Cập nhật mục 2–4 của file này khi một bước làm thay đổi cây quyết định, để session sau đọc đúng trạng thái.

## 6. Còn chờ người dùng / cô
Có được dùng right context không? Có chấp nhận đổi cách decode với generator đóng băng không? Báo cáo mới nhất: https://claude.ai/artifact/1nmrRJLZn2AhSPVezYrEJV (chưa có E5–E8).
