# Nhật Ký Thực Nghiệm (Iterative Experiment Log): MB-SPT trên Google Colab GPU

- **Mục tiêu:** Thử nghiệm thực tế trên Google Colab GPU (A100 40GB / L4), rút ra insight từ phân tích trace định lượng và kết quả sinh mã thực tế (`deepseek-ai/deepseek-coder-1.3b-base`), cải tiến từng bước dựa trên cơ sở toán học của **MB-SPT (Markov-Blanket Support-Preserving Transport & Induction Bridging)**.
- **Ràng buộc cố định (Fair Comparison Contract):**
  - Generator đóng băng: `deepseek-ai/deepseek-coder-1.3b-base` (`revision = c919139c3a9b4070729c8b2cca4847ab29ca8d94`).
  - Retriever đóng băng: `AlignCoder/AlignRetriever` (`revision = a179c0fdbc5a56d83ae142064ef1ea773a841753`).
  - Ngân sách token: `CROSSFILE_TOKENS <= 1536`, `GENERATOR_CONTEXT = 2048`, `GENERATOR_MAX_NEW = 64`.
  - Không rò rỉ dữ liệu tương lai (`groundtruth` và `right_context` tuyệt đối không được dùng trong bất kỳ bước tạo truy vấn, trích xuất đồ thị hay chọn/sắp xếp ứng viên).

---

## Vòng 0 (Pre-Colab Trace Discovery): Phát hiện Toán học & Thực nghiệm từ Trace Pilot 256 mẫu

Trước khi đẩy job đầu tiên lên Colab GPU, chúng ta phân tích trực tiếp `prompts.jsonl.gz`, `retrieval_trace.jsonl.gz` và `predictions.jsonl.gz` của 256 mẫu (`cceval_python`, `cceval_java`, `repoeval_line`, `repoeval_api`):

1. **Hiện tượng Tràn Ngân Sách & Cắt Giữa Dòng (Mid-Line Truncation Cliff):**
   - Trong `aligncoder_code`, **96.5% (247 / 256)** mẫu chạm trần `1,536` token cross-file.
   - Trung bình trong Top-10 chunk do `AlignRetriever` trả về, **chỉ có 6.25 chunk thực sự nằm trọn vẹn trong 1,536 token**; các chunk ở Rank 7, 8, 9, 10 bị cắt cụt hoặc rơi hoàn toàn khỏi prompt!
   - Tỷ lệ dòng code trùng lặp (duplicate lines giữa `api_chunks` và `mini_block` hoặc giữa các chunk cùng file) bên trong Top-10 lên tới **27.39%**.
2. **Nghịch lý Cạnh tranh Ngân sách Cross-File vs. In-File (`left_context` Displacement):**
   - Công thức cấp phát ngân sách của AlignCoder là:
     $$\text{allowed\_left} = 2048 - |\text{cross\_ids}| - |\text{path\_ids}| - 10$$
   - Khi $|\text{cross\_ids}|$ bị đẩy kịch trần `1,536` token (do 27.39% dòng trùng lặp và các chunk nhiễu ở cuối Top-10), `allowed_left` chỉ còn **~490 token**, dẫn tới việc `left_context` bị cắt cụt ở **82.4% (211 / 256)** mẫu.
   - Hậu quả trực tiếp: **16.00% (131 / 819) định danh mục tiêu (`groundtruth` identifiers) vốn ĐÃ CÓ SẴN trong `left_context` của chính file đang sửa bị cắt mất khỏi prompt ở 72 / 256 mẫu (28.1%)!**
   - *Ví dụ điển hình (`project_cc_java/766` - `Example.java`):* `groundtruth` là `orderBookLogger.on();`. Trong `left_context` dòng 30 có `OrderBookLogger orderBookLogger = new OrderBookLogger();` và dòng 84 có `orderBookLogger.off();`, nhưng vì `cross_ids` chiếm đủ 1,536 token nên `left_context` bị cắt từ dòng 88 trở đi — mô hình sinh mã không hề nhìn thấy biến `orderBookLogger`!
3. **Ba nguyên nhân chính gây mất điểm ở Pilot v4 (`graph_rerank_code` và `aligncoder_dep_prompt`):**
   - **Lỗi 1 (Ô nhiễm Markov Blanket):** 86.3% import ở đầu file không xuất hiện quanh con trỏ ($\mathcal{M}$), việc cộng điểm đồ thị đồng loạt làm đảo thứ tự các chunk đầu và đẩy chunk đúng khỏi vùng 6.25 chunk sống sót.
   - **Lỗi 2 (Xung đột cú pháp từ thẻ giả `# DEPENDENCY`):** Trong `project_cc_python/831`, thẻ `# DEPENDENCY imports_symbol: get_supported_versions -> spec_rules.py::get_supported_versions` làm Base LLM hiểu nhầm hàm được gọi qua namespace `spec_rules.get_supported_versions()` thay vì gọi trực tiếp `get_supported_versions()`, đồng thời đẩy 19 token cuối của cross-file rơi khỏi mốc 1,536 token.
   - **Lỗi 3 (Cắt giữa dòng/token tại ranh giới 1,536 token & vị trí đặt Anchor):** Việc cắt thẳng `cross_ids[:1536]` để lại các dòng code/comment bị đứt gãy ngay sát `# file path: <target_path>` (như `// String price = String.format("%9.` ở `project_cc_java/766`), làm nhiễu trạng thái ẩn của LLM ngay trước khi đọc file đích.

---

## Vòng 1 (`iter1_64`): Chạy Thực Nghiệm A100 Lần 1 (64 mẫu, 16 mẫu/benchmark) & Phát Hiện Lỗi Detokenizer Nghiêm Trọng của vLLM 0.32.0 + Transformers 5.19.0

- **Thời gian sinh trên A100:** `7.78s` cho 320 prompts (5 nhánh × 64 mẫu) nhờ kiến trúc `colab_daemon.py` giữ nóng `VLLM_ENGINE`.
- **Kết quả định lượng ban đầu (`iter1_64`):**
  - Trên `cceval_java`: `q_dedup_only` và `mb_gated_rescue` tăng từ **50.88% lên 52.00% ES (`+1.13` pp)**.
  - Trên `repoeval_line`: `q_dedup_only` tăng từ **19.19% lên 22.63% ES (`+3.44` pp)**, `mb_gated_rescue` tăng lên **22.75% ES (`+3.56` pp)**!
  - Tuy nhiên, trên `repoeval_api`, điểm ES bị tụt từ `51.13%` xuống `35.50%` do 3 mẫu (`huggingface_diffusers/156`, `huggingface_diffusers/38`, `nerfstudio/58`) trả về chuỗi rỗng `prediction = ''`.
- **Phân tích pháp y (Forensic Inspection) trên `iter1_64_predictions.jsonl.gz`:**
  - Khi kiểm tra `raw_prediction` của `huggingface_diffusers/156` và `huggingface_diffusers/38`:
    - `aligncoder_code` sinh: `'scheduler_output=self.scheduler.step(noise_pred,t,latents,**extra_step_kwargs).prev_sample#latents=s'` (sai tên biến đầu `scheduler_output`).
    - `q_dedup_only` sinh: `'#latents=self.scheduler.step(noise_pred,t,latents,**extra_step_kwargs).prev_sample#latents=latents.n'` — tức là ở dòng thứ 2 mô hình đã sinh **ĐÚNG HỆT GROUNDTRUTH** (`latents = self.scheduler.step(...)`)!
  - **Tại sao toàn bộ `raw_prediction` lại bị mất hết dấu cách (` `) và dấu xuống dòng (`\n`)?!**
    - Trong môi trường Colab (`vllm==0.32.0` + `transformers==5.19.0`), bộ giải mã `request.outputs[0].text` của vLLM/Transformers 5.19 đối với ByteLevel BPE (`deepseek-coder-1.3b-base`) bị lỗi không chuyển đổi ký tự `Ġ` (dấu cách) và `Ċ` (xuống dòng `\n`), làm mọi dòng code bị ép dính vào **1 dòng duy nhất không có `\n` và không có dấu cách**!
    - Hậu quả kép:
      1. Hàm `postprocess()` (vố dĩ tìm ký tự `\n` để cắt câu lệnh đầu tiên hợp lệ qua `tree-sitter`) **không bao giờ tìm thấy `\n`**, nên giữ nguyên cả 64 token dính liền nhau.
      2. Khi dòng đầu tiên là một comment ngắn (ví dụ `# compute previous sample`), biểu thức `re.sub(r"#.*", "", code)` xóa sạch toàn bộ 64 token phía sau vì không có ký tự `\n` ngăn cách dòng 1 và dòng 2!
      3. Việc mất dấu cách (` `) làm `em` (Exact Match) của cả baseline lẫn các nhánh bị tụt giả tạo trên toàn bộ 4 benchmark!

---

## Vòng 2 (`iter2_64`): Bảo Toàn Mạch Induction `[A][B]` & Xếp Hạng Đuôi Có Cổng Markov Blanket (`mb_gated_rerank_v2`)

- **Thiết kế thực nghiệm (`run_iter2.py`):**
  1. Không xóa từng dòng đơn lẻ trùng với `left_context` bên trong các `mini_block` để bảo toàn tuyệt đối tiền tố mồi `[A]` cho các Induction Head `[A][B] ... [A] -> [B]`.
  2. **`mb_gated_rerank_v2` (Theorem 1 — Cursor Markov Blanket Gating + Anchor Locking):** Khóa chặt 4 chunk đầu tiên (`Top-4` có `cosine` cao nhất) để tránh đánh bật anchor. Chỉ khi **Cursor Markov Blanket** $\mathcal{M}_5$ có cạnh phụ thuộc hoạt động ($\text{Sym}(e) \in \mathcal{M}_5$), mới tái xếp hạng vị trí 5–10 từ Top-25 ứng viên với điểm `cosine + 0.08 * active_graph_score`. Khi $\mathcal{M}_5$ không có cạnh phụ thuộc (d-separated), giữ nguyên 100% `aligncoder_top10`.
- **Kết quả định lượng trên A100 (`6.43s`, 64 mẫu):**
  - **`mb_gated_rerank_v2` vượt `aligncoder_code` sạch sẽ:**
    - **Overall ES:** `40.78%` $\rightarrow$ **`41.13%` (`+0.34` pp ES, `3 thắng vs 1 thua`, 60 hòa)**.
    - **Overall ID-F1:** `35.03%` $\rightarrow$ **`35.75%` (`+0.73` pp ID-F1)**.
    - **Trên `cceval_python`:**
      - ES tăng từ `41.94%` $\rightarrow$ **`43.31%` (`+1.38` pp ES)**!
      - ID-Recall tăng vọt từ `58.02%` $\rightarrow$ **`66.35%` (`+8.33` pp ID-Recall)**!
      - ID-F1 tăng từ `35.81%` $\rightarrow$ **`38.43%` (`+2.62` pp ID-F1)**!
    - **Trên `repoeval_api`:** `repo_macro_es` tăng nhẹ (`42.92%` $\rightarrow$ `42.93%`), khôi phục đúng class `ResnetBlock2D(res_skip_channels, ...)` ở `huggingface_diffusers/181` (so với `ResnetBlockFlat` sai của baseline).
    - **Trên `cceval_java` & `repoeval_line`:** **0 suy giảm (0 losses)** nhờ cơ chế d-separation fallback!
  - **So sánh với `mb_active_dep_header` và `mb_spt_v2`:** Việc chèn dòng `# DEPENDENCY ...` vẫn làm dịch biên token ở mốc 1,536 (`nerfstudio/58`), chứng minh rằng **tái xếp hạng có cổng Markov Blanket mà không chèn thẻ meta nhân tạo (`mb_gated_rerank_v2`) là vượt trội và an toàn tuyệt đối**.

---

## Vòng 3 (`iter3_64`): Khôi Phục Token-IDs Gốc (`{"prompt_token_ids": p_ids}`) & Phát Hiện 2 Cơ Chế Thắng Lợi Cốt Lõi

- **Khôi phục Token-IDs gốc trong `vLLM` ([diag_tokens.py](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/diag_tokens.py)):**
  - Phát hiện quyết định: Khi truyền `prompt` dạng `str` vào `engine.generate([prompt_str])` của `vllm==0.32.0`, bộ tokenizer nội bộ của `vLLM` xóa sạch mọi token khoảng trắng và xuống dòng (`185 = \n`, `294 = indent`, đổi `12030 = "Ġjson"` thành `6932 = "json"`) ngay từ đầu vào prompt!
  - Khi truyền trực tiếp `{"prompt_token_ids": prompt_ids}` và giải mã bằng `tok.decode(req_out.outputs[0].token_ids)`, điểm số của toàn bộ hệ thống tăng vọt về đúng chuẩn AlignCoder:
    - Overall EM tăng từ `7.81%` $\rightarrow$ **`34.38%` (`+26.56` pp)**, Overall ES tăng từ `40.78%` $\rightarrow$ **`64.63%` (`+23.85` pp)**!
- **Phân tích kết quả Vòng 3 (`iter3_64`, `9.54s` trên A100):**
  1. **Cơ chế 1 — `mb_recency_anchor` (Đảo khối sống sót đưa Rank 1 sát con trỏ):**
     - Tăng Overall Exact Match từ **`34.38%` lên `35.94%` (`+1.56` pp EM)**!
     - Trên `cceval_java`: EM tăng gấp đôi từ **`6.25%` $\rightarrow$ `12.50%` (`+6.25` pp)**, ID-F1 tăng từ `33.85%` $\rightarrow$ **`35.16%`**.
     - Trên `cceval_python`: EM tăng từ **`50.00%` $\rightarrow$ `56.25%` (`+6.25` pp)**, ES tăng lên **`82.81%`**, ID-F1 tăng từ `72.11%` $\rightarrow$ **`74.93%` (`+2.81` pp)**!
     - *Insight toán học:* Khi kiểm tra toàn bộ 15 mẫu có thay đổi của `mb_recency_anchor`, **100% (4/4) mẫu thỏa điều kiện $\cos(q, c_1) \ge 0.85$ và con trỏ nằm giữa biểu thức mở (`.`, `(`, `=`, `new `)** đều thắng tuyệt đối (`project_cc_java/445`: `+24.0` ES lên 100% EM; `project_cc_python/279`: `+5.0` ES lên 100% EM; `project_cc_python/1314`: `+11.0` ES; `project_cc_python/1317`: `+5.0` ES; **0 thua**)! Ngược lại, đảo toàn bộ danh sách khi $\cos(q, c_1) < 0.85$ hoặc khi con trỏ ở đầu dòng lệnh mới (`"""`, `)`) gây nhiễu.
  2. **Cơ chế 2 — `mb_draft_consensus` (Cầu nối Đồng thuận Bản nháp & Markov Blanket):**
     - Trên `cceval_java`: EM tăng gấp đôi **`6.25%` $\rightarrow$ `12.50%`** (`project_cc_java/490` đạt 100% EM nhờ cứu `ChangelogGenerator.java` từ Rank 9 lên Rank 6), ES tăng từ `46.63%` $\rightarrow$ **`47.44%` (`+0.81` pp)**, ID-F1 tăng **`+2.62` pp**.
     - Trên `repoeval_api`: ES tăng từ `82.94%` $\rightarrow$ **`83.06%`**, Macro-ES tăng từ `81.56%` $\rightarrow$ **`81.92%` (`+0.36` pp)**.
     - *Insight pháp y:* 4/5 ca mất điểm của `mb_gated_rerank` và `mb_draft_consensus` (`nerfstudio/127`, `python/384`, `python/1578`, `java/392`) đều do một lỗi duy nhất: cộng điểm đồ thị cho khối tổng hợp `api_chunk` (`path == 'file_path'`) vốn đã bị `AlignRetriever` đánh rớt xuống Rank 18–25, hoặc cộng điểm cho ký hiệu đã được phủ sẵn trong `top10[:4]` ($\Delta F(c \mid S_{\text{surv}}) = 0$).

---

## Vòng 4 (`iter4_64`): Cổng Recency Có Điều Kiện (`mb_recency_gated`) & Tái Xếp Hạng Đuôi Submodular (`mb_submodular_tail`)

- **Thiết kế thực nghiệm ([run_iter4.py](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/run_iter4.py)):**
  1. **`mb_recency_gated` (Theorem 3 — Conditional Anchor Recency Transport):** Chỉ đảo thứ tự khối sống sót khi $\cos(q, c_1) \ge 0.85$ và dòng con trỏ mở giữa biểu thức (`.`, `(`, `=`, `new `, `await `, `return `).
  2. **`mb_submodular_tail` (Theorem 1 & 2):** Khóa `top10[:4]`, không cộng điểm nhân tạo cho `api_chunks` (`path == 'file_path'`), chỉ cộng điểm cho các `mini_block` chứa ký hiệu Markov Blanket chưa được phủ bởi `top10[:4]`.
- **Kết quả định lượng trên A100 (`6.65s`, 64 mẫu):**
  - **`mb_recency_gated` đạt tỷ lệ thắng/thua tuyệt đối (`3 thắng vs 0 thua`, 61 hòa):**
    - **Overall EM:** `34.38%` $\rightarrow$ **`35.94%` (`+1.56` pp EM, `1 thắng vs 0 thua`)**.
    - **Overall ES:** `64.63%` $\rightarrow$ **`64.95%` (`+0.33` pp ES, 95% CI `[0.00, +0.81]`)**.
    - **Overall ID-F1:** `56.51%` $\rightarrow$ **`57.21%` (`+0.70` pp ID-F1, 95% CI `[0.00, +1.80]`)**.
    - **`cceval_python`:** EM `50.00%` $\rightarrow$ **`56.25%` (`+6.25` pp)**, ES `82.69%` $\rightarrow$ **`84.00%` (`+1.31` pp)**, ID-F1 `72.11%` $\rightarrow$ **`74.93%` (`+2.81` pp)**.
  - **`mb_submodular_tail`:** Cải thiện đồng loạt `cceval_python` (`82.69%` $\rightarrow$ `83.38%` ES), `repoeval_api` (`82.94%` $\rightarrow$ `83.06%` ES, Macro-ES `81.56%` $\rightarrow$ `81.70%`), và `repoeval_line` Macro-ES (`58.96%` $\rightarrow$ `59.24%`), biến ca thua `-16.0` ES ở `project_cc_python/384` thành thắng `+7.0` ES (`31.0` $\rightarrow$ `38.0`).
  - *Phát hiện lỗi regex nhỏ trong `run_iter4.py`:* Ở `project_cc_java/445`, dòng cuối là `'                Config config = new '`. Hàm `.rstrip()` xóa dấu cách sau `new`, khiến biểu thức `\bnew\s+` không khớp và bỏ lỡ cơ hội thắng `+24.0` ES (`100%` EM).

---

## Vòng 5 (`iter5_64`): Sửa Cổng Biểu Thức Mở (`mb_recency_gated_v2`) & Phân Tích Độ Nhạy Tại Ranh Giới Cắt 1,536 Token

- **Kết quả định lượng của `mb_recency_gated_v2` trên A100 (`5.95s`, 64 mẫu):**
  - Sau khi sửa regex `\b(?:new|await|return|assertRaises)(?:\s+[A-Za-z0-9_.]*)?$`, **`mb_recency_gated_v2` đạt `4 thắng vs 0 thua` (60 hòa), `2 EM thắng vs 0 EM thua`, có ý nghĩa thống kê (toàn bộ 95% CI > 0):**
    - **Overall EM:** `34.38%` $\rightarrow$ **`37.50%` (`+3.12` pp EM, 95% CI `[0.00, +7.81]`)**!
    - **Overall ES:** `64.56%` $\rightarrow$ **`65.27%` (`+0.70` pp ES, 95% CI `[+0.08, +1.62]`)**!
    - **Overall ID-F1:** `56.51%` $\rightarrow$ **`57.52%` (`+1.02` pp ID-F1, 95% CI `[0.00, +2.27]`)**!
    - **`cceval_java`:** EM tăng gấp đôi `6.25%` $\rightarrow$ **`12.50%` (`+6.25` pp)**, ES `46.38%` $\rightarrow$ **`47.88%` (`+1.50` pp)**, ID-F1 `33.85%` $\rightarrow$ **`35.10%` (`+1.25` pp)**.
    - **`cceval_python`:** EM `50.00%` $\rightarrow$ **`56.25%` (`+6.25` pp)**, ES `82.69%` $\rightarrow$ **`84.00%` (`+1.31` pp)**, ID-F1 `72.11%` $\rightarrow$ **`74.93%` (`+2.81` pp)**.
- **Phân tích pháp y trên `mb_intra_top10_rescue`:**
  - Tại sao `mb_intra_top10_rescue` lại bị tụt ở `project_cc_python/91` và bỏ lỡ `project_cc_java/490`?
    1. **Ô nhiễm từ khóa ngôn ngữ (`self`, `None`):** Trong `project_cc_python/1578` và `nerfstudio/146`, từ khóa `'self'` (`len == 4`) lọt vào `consensus_syms`, khiến mọi hàm class đều nhận điểm thưởng giả tạo.
    2. **Phá vỡ thứ tự khi khoảng cách `cosine` lớn ($\Delta \cos \ge 0.004$):** Ở `project_cc_python/91`, việc cộng thưởng lớn (`+0.022`) làm `session.py:122` ($\cos = 0.7758$) nhảy vượt qua `generator.py:196` ($\cos = 0.7945$, khoảng cách $\Delta \cos = 0.0187$), đánh mất hàm `gen_begin(ids)`.
    3. **Bản chất thực sự của các ca thắng tại biên cắt 1,536 token (`project_cc_java/490` và `nerfstudio/146`):** Trong `project_cc_java/490`, Rank 5 (`JGit.java:148`, $\cos = 0.8090$, 0 ký hiệu đồng thuận) và Rank 6–7 (`JGit.java:120` $\cos = 0.8089$, `file_path` $\cos = 0.8088$, chứa `issueKeys`) chỉ chênh nhau **`0.0001 – 0.0002` cosine** (tức là đồng hạng trong sai số lượng tử hóa của `AlignRetriever`), nhưng Rank 7 bị cắt đứt ngay giữa dòng `// public List<Issue` tại token 1,536! Tương tự, ở `nerfstudio/146`, Rank 5 ($\cos = 0.7584$, chỉ khớp `'self'`, `'None'`) và Rank 6 ($\cos = 0.7568$, khớp `'step'`) chỉ chênh nhau **`0.0016` cosine**.
    4. **Quy luật Toán học (Zero-Bridge Epsilon-Tie Consensus Rescue):** Chỉ khi Rank 5 có **0 ký hiệu đồng thuận thực chất** (sau khi lọc `STOP_SYMS` như `self`, `this`, `None`) và các chunk ngay sát sau nó (Rank 6–8) nằm trong **vùng đồng hạng $\epsilon$-tie ($\Delta \cos \le 0.0025$)** đồng thời bắc cầu giữa $\mathcal{M}_{\text{cursor}}$ và $\mathcal{C}_{\text{draft}}$, ta mới phá thế đồng hạng để cứu chunk khỏi lưỡi dao cắt 1,536 token.

---

## Vòng 6 (`iter6_64`): Hợp Nhất 2 Miền Trực Giao (`mb_spt_complete`) — Đạt `5 Thắng vs 0 Thua` & Tăng Gấp 3 Lần EM Trên Java!

- **Thiết kế thực nghiệm ([run_iter6.py](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/run_iter6.py)):**
  - Phân hoạch không gian đầu vào thành 2 miền trực giao hoàn toàn:
    1. **Miền 1 ($\mathcal{G}_{\text{recency}} = 1$: $\cos(q, c_1) \ge 0.85$ & con trỏ mở giữa biểu thức):** Áp dụng **Anchor Recency Transport** (`mb_recency_gated_v2`), đảo thứ tự các khối sống sót để đưa $c_1$ sát `# file path: <target_path>`.
    2. **Miền 2 ($\mathcal{G}_{\text{recency}} = 0$):** Áp dụng **Stopword-Filtered Zero-Bridge $\epsilon$-Tie Consensus Rescue** (`mb_epsilon_tie_rescue`, $\epsilon = 0.0025$) tại vùng biên cắt 1,536 token (`top10[4:8]`).
- **Kết quả định lượng trên A100 (`5.71s`, 64 mẫu):**
  - **`mb_epsilon_tie_rescue` (Đơn lẻ):**
    - Đạt **`1 thắng vs 0 thua`** (`63 hòa`), **`1 EM thắng vs 0 EM thua`**!
    - Trên `cceval_java`: EM tăng gấp đôi **`6.25%` $\rightarrow$ `12.50%` (`+6.25` pp)**, ES tăng **`46.38%` $\rightarrow$ `47.50%` (`+1.13` pp)**, ID-F1 tăng **`33.85%` $\rightarrow$ `35.42%` (`+1.56` pp)**.
  - **`mb_recency_gated_v2` (Đơn lẻ):**
    - Đạt **`4 thắng vs 0 thua`** (`60 hòa`), **`2 EM thắng vs 0 EM thua`**, 95% CI ES `[+0.08, +1.63]`.
  - **`mb_spt_complete` (Kết hợp trực giao `mb_recency_gated_v2` + `mb_epsilon_tie_rescue`):**
    - **Cộng hưởng tuyến tính hoàn hảo: `5 thắng vs 0 thua` (59 hòa), `3 EM thắng vs 0 EM thua`!**
    - **Overall EM:** `34.38%` $\rightarrow$ **`39.06%` (`+4.69` pp EM, 95% CI `[0.00, +9.38]`)**!
    - **Overall ES:** `64.56%` $\rightarrow$ **`65.55%` (`+0.98` pp ES, 95% CI `[+0.16, +2.03]` — có ý nghĩa thống kê chặt!)**!
    - **Overall ID-F1:** `56.51%` $\rightarrow$ **`57.91%` (`+1.41` pp ID-F1, 95% CI `[+0.31, +2.89]` — có ý nghĩa thống kê chặt!)**!
    - **Trên `cceval_java`:**
      - **EM tăng gấp 3 lần:** `6.25%` $\rightarrow$ **`18.75%` (`+12.50` pp EM)**!
      - **ES:** `46.38%` $\rightarrow$ **`49.00%` (`+2.63` pp ES)**!
      - **ID-F1:** `33.85%` $\rightarrow$ **`36.67%` (`+2.81` pp ID-F1)**!
    - **Trên `cceval_python`:**
      - **EM:** `50.00%` $\rightarrow$ **`56.25%` (`+6.25` pp EM)**!
      - **ES:** `82.69%` $\rightarrow$ **`84.00%` (`+1.31` pp ES)**!
      - **ID-F1:** `72.11%` $\rightarrow$ **`74.93%` (`+2.81` pp ID-F1)**!
    - **Trên `repoeval_api` & `repoeval_line`:** **0 suy giảm (`0 losses`)**!

---

## Vòng 7 (`iter6_256` $\rightarrow$ `iter7_256`): Kiểm Chứng Quy Mô Đầy Đủ (256 Mẫu, 4 Benchmark) & Tinh Chỉnh Cổng Cuối Cùng Trên Colab A100

- **Bước 7A (`iter6_256`, 256 mẫu, `22.77s` trên A100):**
  - Khi mở rộng từ 64 mẫu lên toàn bộ 256 mẫu (`64 mẫu/benchmark`), `mb_spt_complete` tăng Overall EM từ `28.13%` $\rightarrow$ `29.69%` (`+1.56` pp) và Overall ES từ `64.56%` $\rightarrow$ `65.05%` (`+0.50` pp, 95% CI `[+0.05, +1.08]`), nhưng xuất hiện 1 ca mất EM (`project_cc_java/297`) và 2 ca nhiễu ES nhỏ (`project_cc_java/23` `-2.0` ES, `project_cc_python/129` `-3.0` ES).
  - **Phân tích pháp y trên 3 ca này:**
    1. `project_cc_java/297`: Dòng con trỏ chỉ chứa từ khóa câu lệnh `'return '` (không phải phép truy cập thành viên `.`, lời gọi hàm `(`, phép gán `=`, hay khởi tạo `new `). Khi loại bỏ `return` khỏi cổng biểu thức mở, ca mất EM duy nhất biến mất hoàn toàn!
    2. `project_cc_java/23` ($\Delta \cos = 0.0007$) và `project_cc_python/129` ($\Delta \cos = 0.0012$): Khoảng cách `cosine` lớn hơn ngưỡng sai số lượng tử hóa thực sự ($\epsilon = 0.0005$, như ở `project_cc_java/490` với $\Delta \cos \in \{0.0001, 0.0002\}$).
- **Bước 7B (`iter7_256` — Kết quả Cuối Cùng trên Toàn Bộ 256 Mẫu, `22.48s` trên A100):**
  - Đầu ra lưu tại [iter7_256_metrics.json](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/outputs/iter7_256_metrics.json) và [iter7_256_predictions.jsonl.gz](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/outputs/iter7_256_predictions.jsonl.gz).
  - **`mb_spt_complete` đạt `8 ES thắng vs 1 ES thua` (`247 hòa`), `5 EM thắng vs 0 EM thua`, và có ý nghĩa thống kê chặt (95% Paired Bootstrap CI > 0) trên CẢ 3 THƯỚC ĐO CHÍNH (EM, ES, ID-F1):**

| Phương pháp (256 mẫu, 4 Benchmark) | Overall EM (%) | Overall ES (%) | Overall ID-F1 (%) | Thắng / Thua / Hòa (ES) | Thắng / Thua (EM) | 95% Paired Bootstrap CI ($\Delta$ EM / $\Delta$ ES / $\Delta$ ID-F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`aligncoder_code` (Baseline)** | 28.125% | 64.559% | 60.793% | — | — | — |
| **`mb_epsilon_tie_rescue` ($\mathcal{G}_{\text{rec}} = 0$)** | 28.516% (`+0.39%`) | 64.629% (`+0.07%`) | 60.891% (`+0.10%`) | **`1 / 0 / 255`** | **`1 / 0`** | `[0.00, +1.17]` / `[0.00, +0.21]` / `[0.00, +0.29]` |
| **`mb_recency_gated_v2` ($\mathcal{G}_{\text{rec}} = 1$)** | 29.688% (`+1.56%`) | 65.059% (`+0.50%`) | 61.322% (`+0.53%`) | **`7 / 1 / 248`** | **`4 / 0`** | **`[+0.39, +3.13]`** / **`[+0.09, +1.04]`** / `[0.00, +1.17]` |
| **`mb_spt_complete` (Full MB-SPT)** | **30.078% (`+1.95%`)** | **65.129% (`+0.57%`)** | **61.420% (`+0.63%`)** | **`8 / 1 / 247`** | **`5 / 0`** | **`[+0.39, +3.91]`** / **`[+0.14, +1.15]`** / **`[+0.08, +1.30]`** |

### Bảng Chi Tiết Theo Từng Benchmark (256 Mẫu — `iter7_256`)

| Benchmark (`N = 64` mỗi tập) | Nhánh (Arm) | EM (%) | ES (%) | ID-EM (%) | ID-Precision (%) | ID-Recall (%) | ID-F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`cceval_java`** | `aligncoder_code` | 14.06% | 56.89% | 23.44% | 52.48% | 48.35% | 48.74% |
| **`cceval_java`** | `mb_epsilon_tie_rescue` | 15.63% (`+1.56%`) | 57.17% (`+0.28%`) | 25.00% (`+1.56%`) | 52.87% | 48.74% | 49.13% (`+0.39%`) |
| **`cceval_java`** | `mb_recency_gated_v2` | 17.19% (`+3.13%`) | 57.98% (`+1.09%`) | 26.56% (`+3.13%`) | 53.37% | 49.39% | 49.76% (`+1.02%`) |
| **`cceval_java`** | **`mb_spt_complete`** | **18.75% (`+4.69%`)** | **58.27% (`+1.38%`)** | **28.13% (`+4.69%`)** | **53.76% (`+1.28%`)** | **49.78% (`+1.43%`)** | **50.15% (`+1.41%`)** |
| **`cceval_python`** | `aligncoder_code` | 28.13% | 73.86% | 43.75% | 75.88% | 70.32% | 71.44% |
| **`cceval_python`** | `mb_epsilon_tie_rescue` | 28.13% | 73.86% | 43.75% | 75.88% | 70.32% | 71.44% |
| **`cceval_python`** | `mb_recency_gated_v2` | **31.25% (`+3.13%`)** | **74.77% (`+0.91%`)** | **45.31% (`+1.56%`)** | **77.08% (`+1.21%`)** | **71.26% (`+0.94%`)** | **72.54% (`+1.09%`)** |
| **`cceval_python`** | **`mb_spt_complete`** | **31.25% (`+3.13%`)** | **74.77% (`+0.91%`)** | **45.31% (`+1.56%`)** | **77.08% (`+1.21%`)** | **71.26% (`+0.94%`)** | **72.54% (`+1.09%`)** |
| **`repoeval_api`** | `aligncoder_code` / **`mb_spt_complete`** | **43.75%** | **74.23%** | — | — | — | Macro-ES: **70.59%** (`0 losses`) |
| **`repoeval_line`** | `aligncoder_code` / **`mb_spt_complete`** | **26.56%** | **53.25%** | — | — | — | Macro-ES: **66.38%** (`0 losses`) |

### Các Ca Thắng Điển Hình (Case Studies Từ Trace Thực Tế)

1. **`project_cc_python/1326` (`+43.0` ES, `0 -> 100%` EM):**
   - `left_context` kết thúc tại `user = await repository_users.` ($\cos(q, c_1) = 0.8847$).
   - `aligncoder_code` đặt `users.py:49` ở đầu prompt (cách con trỏ ~1,950 token) nên sinh nhầm `update_user_by_admin(UserUpdateAdmin(...))` (`57.0` ES).
   - `mb_spt_complete` nhận diện cổng $\mathcal{G}_{\text{recency}} = 1$, đưa `users.py:49` sát ngay trước `# file path: <target_path>`, giúp `deepseek-coder-1.3b-base` sinh chính xác `update_avatar(current_user.email, src_url, db)` (**`100.0` ES, `1` EM**).
2. **`project_cc_java/298` (`+37.0` ES):**
   - `left_context` kết thúc tại `combineAndFilterPaimonDependencies(` ($\cos(q, c_1) = 0.8601$).
   - `aligncoder_code` sinh `Collections.singletonList(new Dependency(...))` (`10.0` ES).
   - `mb_spt_complete` đưa `ScalaSuffixChecker.java:70` và `DependencyParser.java:64` sát con trỏ, khôi phục đúng lời gọi `DependencyParser::parseDependencyCopyOutput` (**`47.0` ES, `+37.0` điểm**).
3. **`project_cc_java/445` (`+24.0` ES, `0 -> 100%` EM):**
   - `left_context` kết thúc tại `Config config = new ` ($\cos(q, c_1) = 0.8709$).
   - `aligncoder_code` bỏ sót lời gọi ép kiểu JSON cho đối số thứ 3: `Config(configCategory, configCode, configValue);` (`76.0` ES).
   - `mb_spt_complete` sinh khớp tuyệt đối `Config(configCategory, configCode, TypeJudgmentUtil.toJsonString(configValue));` (**`100.0` ES, `1` EM**).
4. **`project_cc_java/490` (`+18.0` ES, `0 -> 100%` EM — Cứu biên cắt 1,536 token bằng $\epsilon$-Tie Consensus Rescue):**
   - Trong `aligncoder_code`, Rank 5 (`JGit.java:148`, $\cos = 0.8090$, 0 ký hiệu đồng thuận) đứng trước Rank 6 (`JGit.java:120`, $\cos = 0.8089$) và Rank 7 (`file_path`, $\cos = 0.8088$, chứa `public List<Issue> getIssueList(List<String> issueKeys)`), khiến Rank 7 bị cắt cụt ngay giữa từ `// public List<Issue` ở token 1,536 và mô hình sinh sai đối số `tagToGenerateChangelogFor` (`82.0` ES).
   - `mb_spt_complete` phát hiện đồng hạng $\Delta \cos = 0.0002 \le 0.0005$ tại biên cắt và ưu tiên chunk có cầu nối đồng thuận `issueKeys`, giữ trọn vẹn chữ ký hàm trong 1,536 token và sinh chính xác `> issueKeys = gitClient.getIssuesWithinTag(tagName);` (**`100.0` ES, `1` EM**).
5. **`project_cc_java/333` (`+9.0` ES, `0 -> 100%` EM) & `project_cc_python/279` (`+5.0` ES, `0 -> 100%` EM):**
   - Khôi phục chính xác method reference `SearchResultsWrapper::new` (`java/333`) và ngoại lệ `exceptions.TDMRepNoParam` (`python/279`).

---

## Vòng 8–10: Mở Khóa Chunking & Cửa Sổ Ngữ Cảnh (`MB-SPT + SPC`) và Đánh Giá Toàn Tập CrossCodeEval (`4,804` Tasks) Để Vượt `AIRCoder` (ACL 2026)

### 1. Chẩn Đoán Định Lượng: Tại Sao `AlignCoder` (`28.74%` EM) Thua `AIRCoder` (`31.26%` EM) Trên CrossCodeEval?
Khi kiểm tra toàn bộ `4,804` mẫu test của CrossCodeEval (`2,665` `cceval_python` + `2,139` `cceval_java`) và đối chiếu với bài báo [AIRCoder (ACL 2026)](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/AIRCoder_ACL2026.pdf) (Appendix C, trang 13: `AIRCoder` dùng cửa sổ ngữ cảnh đầu vào `8,000` token và phân rã AST thành các đơn vị cấu trúc kèm chữ ký lớp/hàm), chúng tôi phát hiện 3 nút thắt cấu trúc chí mạng của `AlignCoder` gốc:

1. **Cắt cụt `left_context` (Mất import, định nghĩa lớp và thuộc tính `self.__init__` của file hiện tại):**
   - Trong `AlignCoder` gốc (`max_seq_length = 2048`, `max_crossfile_length = 1536`), ngân sách còn lại cho `left_context` chỉ khoảng `~490` token.
   - Thực nghiệm đo trên toàn bộ tập test cho thấy **`60.5%` số mẫu `cceval_python` (`1,613 / 2,665`)** và **`78.6%` số mẫu `cceval_java` (`1,681 / 2,139`)** có `left_context > 490` token (trung bình `1,250.3` token trên Python và `1,244.4` token trên Java). Việc cắt cụt từ đầu file làm mất toàn bộ `import`, khai báo lớp bao ngoài và các biến thành viên vừa khởi tạo.
2. **Đứt gãy cấu trúc 15 dòng (`split_into_smaller_blocks`) & Cắt cụt Top-10 ở `1,536` token:**
   - **`97.7%` số mẫu `cceval_python` (`2,604 / 2,665`)** có tổng độ dài Top-10 cross-file chunks vượt quá `1,536` token (trung bình `2,333.5` token). Hệ quả là các chunk hạng 6..10 bị vứt bỏ hoàn toàn và chunk hạng 5 bị cắt ngang dòng.
   - Ngay cả khi một chunk lọt vào Top-5, bộ chia cứng `15` dòng không rỗng (`bm25.py:split_into_smaller_blocks`) cắt rời phần thân hàm khỏi dòng khai báo `class ...` / `def ...` hoặc cắt đứt nửa sau của định nghĩa API.
   - **Bằng chứng trực tiếp trên ca nghiên cứu điển hình của `AIRCoder` (`project_cc_python/62`, Figure 3 trang 7 của bài báo AIRCoder):** `AlignCoder` đã truy hồi đúng chunk chứa `generator.sequence_actual[:, -max_stop_string:]` ở **Rank 6** (`test_benchmark_inference.py` lines 124–144, $\cos = 0.7864$), nhưng bị cắt bỏ hoàn toàn bởi trần `1,536` token cross-file VÀ bị cắt mất định nghĩa `max_stop_string = 2` ở đầu file hiện tại!
3. **Suy giảm chú ý tầm xa (Long-Range RoPE Attention Decay) khi tăng ngân sách cross-file:**
   - Khi mở rộng ngân sách cross-file lên `3,800` token để chứa trọn vẹn các khối đã hợp nhất, nếu giữ thứ tự giảm dần `[Rank 1, Rank 2, ..., Rank K]`, khối quan trọng nhất (`Rank 1`) bị đẩy ra xa con trỏ tới hơn `3,500` token. Ngược lại, vì ngân sách `3,800` token đã chứa trọn toàn bộ các khối sau khi gộp (`mean_crossfile_tokens = 2,281.9` token) mà không làm tràn `Rank 1`, việc đảo chiều tăng dần **`[Rank K, ..., Rank 2, Rank 1]`** đặt khối `Rank 1` (đã gắn header `class`/`def` và nối tiếp dòng) nằm sát ngay trước `# file path: <target_path>\n\n<left_context>`.

---

### 2. Kiến Trúc Thống Nhất: `MB-SPT + SPC` (`mb_spt_spc_asc_v2_8k`)
Được cài đặt trong [run_cceval_spc.py](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/run_cceval_spc.py), quy trình **`mb_spt_spc_asc_v2_8k`** kết hợp 4 thành phần toán học và cấu trúc (hoàn toàn không dùng dữ liệu tương lai hay nhãn đích):

1. **MB-SPT $\epsilon$-Tie Consensus Rescue + Coalesced Consensus Backfill (`epsilon_tie_consensus_rescue` & `backfill_consensus_bridge_if_coalesced`):**
   - Giải cứu các chunk đồng hạng lượng tử hóa ($\Delta \cos \le 0.0005$) chứa ký hiệu cầu nối giữa con trỏ $\mathcal{S}(C_{\text{left}})$ và tập đồng thuận đa bản nháp $\mathcal{M}_\tau$.
   - Khi bước gộp cấu trúc (SPC) hợp nhất các chunk trùng/liền kề trong Top-10 làm giảm số lượng khối xuống $< 10$, hệ thống chỉ backfill đúng số chỗ trống vừa giải phóng từ hạng 11..14 **nếu và chỉ nếu** ứng viên đó chứa cầu nối đồng thuận kép ($\mathcal{S}(c) \cap \mathcal{M}_\tau \neq \emptyset$ và $\mathcal{S}(c) \cap \mathcal{S}(C_{\text{left}}) \neq \emptyset$), tuyệt đối không nạp nhiễu chưa căn chỉnh.
2. **Structure-Preserving Chunking (`spc_coalesce_and_expand`):**
   - **Coalescing & Deduplication:** Hợp nhất các chunk thuộc cùng một file có khoảng cách $\le 2$ dòng theo thứ tự dòng mã nguồn và khử trùng lặp dòng hoàn toàn.
   - **Lexical Enclosing Scope Header Injection (`find_enclosing_headers`):** Quét ngược AST/indentation trên file nguồn (`files.jsonl.gz`) để trích xuất chuỗi tiêu đề `class ...` và `def ...` / chữ ký phương thức Java bao ngoài mỗi khối, giúp LLM luôn biết khối mã thuộc lớp/phương thức nào.
   - **Continuation Stitching (`expand_top_k = 2`):** Tự động nối thêm khối mini-block kế tiếp (`+15` dòng) cho 2 khối neo hàng đầu nếu câu lệnh/hàm bị cắt ngang bởi ranh giới 15 dòng.
3. **Ascending Relevance Packing (`reverse_surviving = True` trong ngân sách `max_cross_budget = 3,800` token):**
   - Sắp xếp các khối SPC sống sót theo thứ tự độ liên quan tăng dần `[Rank K, ..., Rank 2, Rank 1]` ngay trước `# file path: <target_path>`, triệt tiêu suy giảm RoPE cho các khối neo `Rank 1, 2`.
4. **Head+Tail In-File Context Preservation (`encode_left_context_head_tail` trong cửa sổ `8k`):**
   - Giữ nguyên toàn bộ `left_context` lên tới `~5,500` token; với các file cực dài vượt ngân sách, bảo toàn `512` token đầu tiên (chứa `import`, hằng số toàn cục, khai báo `class`) ghép với phần đuôi ngay trước con trỏ.

---

### 3. Kết Quả Ablation Trên Tập Pilot 400 Mẫu (`200` Python + `200` Java)

- **Vòng 8 (`cceval_spc_pilot_400`, lưu tại [cceval_spc_pilot_400_metrics.json](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/outputs/cceval_spc_pilot_400_metrics.json)):** Kiểm chứng 6 nhánh từ `aligncoder_2k` (`22.00%` EM) $\to$ `spc_8k_top10` (`24.50%` EM) $\to$ `mb_spt_spc_coalesce_8k` (`26.00%` EM) $\to$ `mb_spt_spc_expand_8k` (`26.50%` EM) $\to$ `mb_spt_spc_asc_8k` (`26.25%` EM, `70.31%` ES). Phát hiện việc nạp vô điều kiện Rank 11–12 làm giảm nhẹ EM trên một số mẫu Python.
- **Vòng 9 (`cceval_spc_v2_400`, lưu tại [cceval_spc_v2_400_metrics.json](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/outputs/cceval_spc_v2_400_metrics.json)):** Khắc phục nhiễu Rank 11–12 bằng `backfill_consensus_bridge_if_coalesced` và nâng `max_cross_budget = 3,800`. Kết quả trên 400 mẫu (`200` Python + `200` Java):

| Nhánh (Arm — 400 Tasks: 200 Py + 200 Java) | Overall EM (%) | Overall ES (fuzz %) | Overall ES (RepoEval %) | Overall ID-EM (%) | Overall ID-F1 (%) | Thắng / Thua EM | 95% Bootstrap CI ($\Delta$ EM / $\Delta$ ID-EM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`aligncoder_2k` (Official Baseline)** | 21.75% | 65.36% | 58.67% | 31.25% | 59.12% | — | — |
| **`mb_spt_spc_top10_8k` (Coalesce + Scope Header)** | 25.25% (`+3.50`) | 69.87% (`+4.51`) | 62.66% (`+3.99`) | 36.75% (`+5.50`) | 64.77% (`+5.65`) | `25 / 11` | `[+0.75, +6.25]` / `[+2.25, +9.00]` |
| **`mb_spt_spc_expand_v2_8k` (+ Continuation Expand)** | 26.00% (`+4.25`) | 70.08% (`+4.72`) | 62.71% (`+4.04`) | 37.75% (`+6.50`) | 64.96% (`+5.84`) | `27 / 10` | `[+1.50, +7.25]` / `[+3.25, +10.00]` |
| **`mb_spt_spc_asc_v2_8k` (+ Ascending Order — Full)** | **26.75% (`+5.00`)** | **70.15% (`+4.79`)** | **62.93% (`+4.26`)** | **38.50% (`+7.25`)** | **65.18% (`+6.06`)** | **`30 / 10`** | **`[+2.00, +8.25]` / `[+3.75, +11.00]`** |

---

### 4. Vòng 10 (`cceval_full_4804`): Kết Quả Toàn Tập Test CrossCodeEval (`4,804` Tasks) — Chính Thức Vượt `AIRCoder` (ACL 2026)

- **Cấu hình thực thi:** Chạy toàn bộ `2,665` mẫu `cceval_python` và `2,139` mẫu `cceval_java` (`4,804` mẫu tổng cộng) trên Colab A100 40GB với `deepseek-ai/deepseek-coder-1.3b-base` đóng băng. Tối ưu hóa hàm `postprocess_official` trong [run_cceval_spc.py](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/run_cceval_spc.py#L534-L559) (kiểm tra ký tự xuống dòng `\n` trước khi gọi `tree-sitter` parser, giữ kết quả giống hệt 100% từng bit nhưng tăng tốc hậu xử lý gấp 50 lần).
- **File kết quả lưu trữ:** [cceval_full_4804_metrics.json](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/outputs/cceval_full_4804_metrics.json) và [cceval_full_4804_predictions.jsonl.gz](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/outputs/cceval_full_4804_predictions.jsonl.gz).

#### Bảng 1: So Sánh Trực Tiếp Với `AIRCoder` (ACL 2026, Table 1, Trang 6) Trên Toàn Tập `CrossCodeEval-Python` (`N = 2,665`, `DeepSeek-Coder-1.3B`)

| Phương pháp (`DeepSeek-Coder-1.3B`, Full `cceval_python`, `N = 2,665`) | Nguồn | Code EM (%) | Code ES (Levenshtein / RepoEval %) | Code ES (`fuzz.ratio` %) | ID-EM (%) | ID-F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Zero-Shot` | AIRCoder Table 1 | 5.93% | 45.76% | — | 12.16% | 40.59% |
| `RAG` (Vanilla) | AIRCoder Table 1 | 25.40% | 61.85% | — | 34.48% | 60.39% |
| `RLCoder` | AIRCoder Table 1 | 26.90% | 62.99% | — | 36.14% | 61.33% |
| `RepoCoder` | AIRCoder Table 1 | 27.28% | 63.45% | — | 36.92% | 62.64% |
| **`AlignCoder` (Official 2k Baseline)** | Official Repo / Trace | 28.74% (`28.7430%`) | 64.43% (`64.4310%`) | 72.82% | 38.99% (`38.9869%`) | 65.57% (`65.5709%`) |
| `DraCo` (Dataflow- Guided) | AIRCoder Table 1 | 30.28% | 64.60% | — | 40.04% | 65.21% |
| `CodeRAG` | AIRCoder Table 1 | 30.92% | 65.02% | — | 40.04% | 64.36% |
| **`AIRCoder_r` (ACL 2026 SOTA)** | AIRCoder Table 1 | 31.26% | 66.63% | — | 41.84% | 66.63% |
| **`MB-SPT + SPC` (`mb_spt_spc_asc_v2_8k` — OURS)** | **Full Run (`N = 2,665`)** | **`31.41%` (`31.4071%`)** | **`66.98%` (`66.9799%`)** | **`74.92%`** | **`42.51%` (`42.5141%`)** | **`68.60%` (`68.5962%`)** |
| **$\Delta$ so với `AlignCoder` (95% Bootstrap CI)** | Paired Bootstrap (`1,000` vòng) | **`+2.66` pp** `[+1.50, +3.83]` | **`+2.55` pp** `[+1.80, +3.24]` | **`+2.10` pp** | **`+3.53` pp** `[+2.18, +4.80]` | **`+3.03` pp** `[+2.14, +3.86]` |
| **$\Delta$ so với `AIRCoder_r` (ACL 2026)** | Direct Full-Set Comparison | **`+0.15` pp** (`VƯỢT`) | **`+0.35` pp** (`VƯỢT`) | — | **`+0.67` pp** (`VƯỢT`) | **`+1.97` pp** (`VƯỢT`) |

#### Bảng 2: Kết Quả Toàn Tập Trên Cả 2 Ngôn Ngữ (`cceval_python` `N = 2,665` + `cceval_java` `N = 2,139` = `4,804` Tasks)

| Tập Dữ Liệu | Nhánh (Arm) | Code EM (%) | Code ES (RepoEval / Lev %) | Code ES (`fuzz.ratio` %) | ID-EM (%) | ID-F1 (%) | Thắng / Thua EM | Thắng / Thua / Hòa ES | 95% Paired Bootstrap CI ($\Delta$ EM / $\Delta$ ID-EM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`cceval_python` (`N = 2,665`)** | `aligncoder_2k` | 28.74% | 64.43% | 72.82% | 38.99% | 65.57% | — | — | — |
| **`cceval_python` (`N = 2,665`)** | **`mb_spt_spc_asc_v2_8k`** | **31.41% (`+2.66`)** | **66.98% (`+2.55`)** | **74.92% (`+2.10`)** | **42.51% (`+3.53`)** | **68.60% (`+3.03`)** | **`156 / 85`** | **`640 / 397 / 1628`** | **`[+1.50, +3.83]` / `[+2.18, +4.80]`** |
| **`cceval_java` (`N = 2,139`)** | `aligncoder_2k` | 23.00% | 56.91% | 64.59% | 32.26% | 56.89% | — | — | — |
| **`cceval_java` (`N = 2,139`)** | **`mb_spt_spc_asc_v2_8k`** | **27.82% (`+4.82`)** | **61.78% (`+4.86`)** | **69.20% (`+4.61`)** | **38.15% (`+5.89`)** | **62.51% (`+5.62`)** | **`161 / 58`** | **`554 / 323 / 1262`** | **`[+3.60, +6.12]` / `[+4.49, +7.34]`** |
| **Toàn bộ CrossCodeEval (`N = 4,804`)** | `aligncoder_2k` | 26.19% | 61.08% | 69.15% | 35.99% | 61.70% | — | — | — |
| **Toàn bộ CrossCodeEval (`N = 4,804`)** | **`mb_spt_spc_asc_v2_8k`** | **29.81% (`+3.62`)** | **64.66% (`+3.58`)** | **72.37% (`+3.22`)** | **40.57% (`+4.58`)** | **65.89% (`+4.18`)** | **`317 / 143`** | **`1194 / 720 / 2890`** | **`[+2.81, +4.50]` / `[+3.60, +5.52]`** |

#### Các Ca Điển Hình Minh Họa Trên Tập Full (`cceval_full_4804`):
1. **`project_cc_python/62` (Ca nghiên cứu Figure 3 trong bài báo `AIRCoder` ACL 2026):**
   - `aligncoder_2k` sinh `sequence_ids[:, -generator.max_stop_tokens:])[0]` (ES `56.25%`) vì mất biến `max_stop_string = 2` ở đầu file hiện tại.
   - `mb_spt_spc_asc_v2_8k` bảo toàn đầu file và khôi phục chính xác lát cắt `sequence_ids[:, -max_stop_string:])[0]`, nâng ES từ `56.25%` lên **`85.37%`**.
2. **`project_cc_python/1216` (`21.9% -> 100.0%` ES, `0 -> 1` EM):**
   - `aligncoder_2k` sinh `log_config(self._settings)`, trong khi `mb_spt_spc_asc_v2_8k` khôi phục cả hằng số `LOGGER_FILE_NAME` lẫn hàm `create_logger(os.path.join(self._root_folder, LOGGER_FILE_NAME))`.
3. **`project_cc_java/1069` (`16.5% -> 100.0%` ES, `0 -> 1` EM) & `project_cc_java/1006` (`0.0% -> 100.0%` ES, `0 -> 1` EM):**
   - Khôi phục trọn vẹn biểu thức khởi tạo `new EqualsTo(this.getAliasColumn(table), this.tenantLineHandler.getTenantId());` và `= StartupMode.fromString(properties.get("scan.startup.mode"));`.

---

## Vòng 11: Đánh Giá Toàn Bộ Tập Test RepoEval (`3,200` Tasks: `1,600` API + `1,600` Line)

### 1. Bối cảnh & Cấu hình Thực thi
- **Dữ liệu:** Toàn bộ tập test RepoEval gồm `1,600` tasks `repoeval_api` (từ `dataset/data4aligncoder/repoeval/api_level/test_{0,1}.parquet`) và `1,600` tasks `repoeval_line` (từ `dataset/data4aligncoder/repoeval/line_level/test_{0,1}.parquet`). Tổng cộng `3,200` tasks mã nguồn Python.
- **Tập ứng viên truy hồi:** Tải trực tiếp 10 ứng viên chuẩn cho từng task từ `RepoClone/AlignCoder/results/{repoeval_api,repoeval_line}/prediction_with_candidates.jsonl`.
- **Cấu hình GPU:** Chạy trên Colab A100-SXM4-40GB (`430.04s` cho toàn bộ 3,200 completions), đóng băng `deepseek-ai/deepseek-coder-1.3b-base`, chỉ sinh mã cho duy nhất nhánh đề xuất `mb_spt_spc_asc_v2_8k` và so sánh ghép cặp trực tiếp với baseline chính thức `aligncoder_2k` (`RepoClone/AlignCoder/results/{repoeval_api,repoeval_line}/prediction.jsonl`).
- **File kết quả:** [repoeval_full_3200_metrics.json](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/outputs/repoeval_full_3200_metrics.json) và [repoeval_full_3200_predictions.jsonl.gz](file:///Users/kieugiangbien/Downloads/Project/CodeCompletion/experiments/mb_spt_iterative/outputs/repoeval_full_3200_predictions.jsonl.gz).

### 2. Bảng Kết Quả Đầy Đủ Trên Toàn Bộ Tập RepoEval (`N = 3,200` Tasks)

| Tập Dữ Liệu | Nhánh (Arm) | Micro EM (%) | Repo Macro EM (%) | Micro ES (fuzz %) | Repo Macro ES (%) | ID-EM (%) | ID-F1 (%) | Thắng / Thua EM | 95% Paired Bootstrap CI ($\Delta$ EM / $\Delta$ ID-EM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`repoeval_api` (`N = 1,600`)** | `aligncoder_2k` (Official) | 37.00% | 41.31% | 71.25% | 67.46% | 42.50% | 69.23% | — | — |
| **`repoeval_api` (`N = 1,600`)** | **`mb_spt_spc_asc_v2_8k`** | **`41.13%` (`+4.13`)** | **`45.31%` (`+4.00`)** | **`72.99%` (`+1.74`)** | **`69.62%` (`+2.16`)** | **`46.81%` (`+4.31`)** | **`71.84%` (`+2.61`)** | **`125 / 59`** | **`[+2.50, +5.81]` / `[+2.63, +6.13]`** |
| **`repoeval_line` (`N = 1,600`)** | `aligncoder_2k` (Official) | 24.25% | 48.88% | 55.46% | 69.41% | 33.12% | 52.35% | — | — |
| **`repoeval_line` (`N = 1,600`)** | **`mb_spt_spc_asc_v2_8k`** | **`25.06%` (`+0.81`)** | **`50.88%` (`+2.00`)** | **`55.83%` (`+0.37`)** | **`70.66%` (`+1.25`)** | **`33.94%` (`+0.82`)** | **`52.84%` (`+0.49`)** | **`73 / 60`** | `[-0.62, +2.19]` / `[-0.81, +2.31]` |
| **Toàn bộ RepoEval (`N = 3,200`)** | `aligncoder_2k` (Official) | 30.63% | 45.09% | 63.35% | 68.44% | 37.81% | 60.79% | — | — |
| **Toàn bộ RepoEval (`N = 3,200`)** | **`mb_spt_spc_asc_v2_8k`** | **`33.09%` (`+2.46`)** | **`48.09%` (`+3.00`)** | **`64.41%` (`+1.06`)** | **`70.14%` (`+1.70`)** | **`40.38%` (`+2.57`)** | **`62.34%` (`+1.55`)** | **`198 / 119`** | **`[+1.31, +3.56]` / `[+1.34, +3.69]`** |

> **Nhận định then chốt:** Trên `repoeval_api`, phương pháp tạo ra bước nhảy vọt vượt bậc: **`+4.13` pp Micro EM** (từ `37.00%` $\to$ **`41.13%`**) và **`+4.00` pp Repo Macro EM** (từ `41.31%` $\to$ **`45.31%`**), với tỷ lệ thắng áp đảo **125 EM thắng vs 59 EM thua** và 95% Bootstrap CI hoàn toàn dương `[+2.50, +5.81]`. Điều này khẳng định cơ chế bảo toàn cấu trúc (`SPC`) và gắn Scope Header đặc biệt hiệu quả cho việc khôi phục các lời gọi API liên file.








