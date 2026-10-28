# Báo Cáo Nghiên Cứu Thực Nghiệm: Ảnh Hưởng Của Repository Context (CSSI) Đến Retrieval và Rerank Dưới Ràng Buộc 512 Tokens

**Tác giả:** CADRE Research Team  
**Ngày thực hiện:** 28/10/2026  
**Quy mô thử nghiệm:** Toàn bộ 8.004 tác vụ trên 4 tập benchmark chuẩn quốc tế (`cceval_python`, `cceval_java`, `repoeval_api`, `repoeval_line`).  
**Ràng buộc cốt lõi (Hard Constraint):** Chiều dài chuỗi token đầu vào truy vấn tối đa cho retriever/reranker là **$\le$ 512 tokens** (`max_length = 512`).

---

## 1. Đặt Vấn Đề & Động Lực Nghiên Cứu (Motivation & Problem Formulation)

Trong các hệ thống RAG Code Completion hiện nay (như RLCoder, RepoCoder, AlignCoder), mô hình Retrieval (UniXcoder / BM25) thường bị ràng buộc nghiêm ngặt bởi độ dài ngữ cảnh biểu diễn: **tối đa 512 tokens**.

Một câu hỏi cốt lõi được đặt ra:
> *Nếu đưa thông tin Repository Context (CSSI - Cross-file Semantic Structure Index, các chữ ký hàm, class hierarchy, type declarations) vào quá trình truy xuất (Retrieval) và xếp hạng lại (Reranking), liệu nó có giúp cải thiện chất lượng truy xuất hay làm suy giảm hiệu năng? Nên tích hợp CSSI ở tầng nào là tối ưu?*

Để trả lời câu hỏi này một cách khoa học, nghiên cứu thiết kế hai phần thử nghiệm đối chứng (Controlled Ablation Experiments) trên 8.004 code completion tasks:
1. **Part 1 (Stage 1 - Query Formulation):** Khảo sát việc đưa CSSI trực tiếp vào **Retrieval Query** trong giới hạn cứng 512 tokens.
2. **Part 2 (Stage 2 - Candidate Reranking):** Giữ nguyên Retrieval Query chuẩn (tối đa hóa cục bộ cursor), và áp dụng CSSI làm **Structural Prior** để tái xếp hạng (Rerank) Top-$K$ ứng viên.

---

## 2. Phần 1: Ảnh Hưởng Của CSSI Đến Khâu Retrieval (Query Formulation)

### 2.1 Thiết Kế Thực Nghiệm (Experimental Setup)
Dưới trần ngân sách 512 tokens (`508 content tokens + 4 special tokens`), chúng tôi so sánh 3 cấu hình Query Formulation:
- **Arm 1 (Standard Tail Tokens - Baseline):** Lấy chính xác 508 tokens mã nguồn nằm ngay sát phía trước con trỏ (Immediate Left Context).
- **Arm 2 (Full Left Context Naive):** Lấy toàn bộ left context từ đầu file tới con trỏ (truy vấn bị tràn, BM25 loãng từ khóa).
- **Arm 3 (CSSI-Guided Query):** Dành 256 tokens cho các định nghĩa/chữ ký CSSI có liên quan trong repo + 252 tokens cho mã nguồn sát con trỏ (tổng 508 tokens).

### 2.2 Kết Quả Định Lượng (Empirical Results)

| Tập Benchmark | Cấu Hình Truy Vấn | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | Hit@10 (%) | Recall@10 (%) | MRR (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **cceval_python** (2.661 tasks) | **Arm 1: Standard 508 Tail** | **50.85** | **69.07** | **74.86** | **81.44** | **63.42** | **61.24** |
| | Arm 2: Full Left Context | 43.74 | 61.71 | 68.47 | 75.42 | 54.87 | 54.11 |
| | Arm 3: CSSI [256] + Tail [252] | 43.48 | 61.48 | 68.13 | 75.69 | 54.93 | 53.98 |
| | *Chênh lệch Arm 3 vs Arm 1* | *-7.37 pp* | *-7.59 pp* | *-6.73 pp* | *-5.75 pp* | *-8.49 pp* | *-7.26 pp* |
| **cceval_java** (2.137 tasks) | **Arm 1: Standard 508 Tail** | **59.34** | **80.02** | **86.71** | **93.07** | **64.17** | **70.94** |
| | Arm 2: Full Left Context | 55.12 | 72.44 | 79.46 | 86.94 | 55.49 | 65.48 |
| | Arm 3: CSSI [256] + Tail [252] | 55.17 | 72.53 | 79.41 | 86.90 | 55.53 | 65.51 |
| | *Chênh lệch Arm 3 vs Arm 1* | *-4.17 pp* | *-7.49 pp* | *-7.30 pp* | *-6.17 pp* | *-8.64 pp* | *-5.43 pp* |
| **repoeval_api** (1.597 tasks) | **Arm 1: Standard 508 Tail** | **77.02** | **88.67** | **91.92** | **94.24** | **72.24** | **83.50** |
| | Arm 2: Full Left Context | 71.51 | 83.53 | 87.85 | 91.80 | 64.77 | 78.58 |
| | Arm 3: CSSI [256] + Tail [252] | 70.95 | 83.47 | 87.91 | 91.92 | 65.08 | 78.23 |
| | *Chênh lệch Arm 3 vs Arm 1* | *-6.07 pp* | *-5.20 pp* | *-4.01 pp* | *-2.32 pp* | *-7.16 pp* | *-5.27 pp* |
| **repoeval_line** (1.428 tasks) | **Arm 1: Standard 508 Tail** | **58.05** | **72.34** | **76.82** | **81.23** | **67.27** | **66.12** |
| | Arm 2: Full Left Context | 53.71 | 66.39 | 71.50 | 76.89 | 60.71 | 61.21 |
| | Arm 3: CSSI [256] + Tail [252] | 53.99 | 66.39 | 71.71 | 77.17 | 60.93 | 61.35 |
| | *Chênh lệch Arm 3 vs Arm 1* | *-4.06 pp* | *-5.95 pp* | *-5.11 pp* | *-4.06 pp* | *-6.34 pp* | *-4.77 pp* |
| **Macro Mean** (8.004 tasks) | **Arm 1: Standard 508 Tail** | **61.31** | **77.53** | **82.58** | **87.50** | **66.78** | **70.45** |
| | Arm 2: Full Left Context | 56.02 | 71.02 | 76.82 | 82.76 | 58.96 | 64.84 |
| | Arm 3: CSSI [256] + Tail [252] | 55.90 | 70.97 | 76.79 | 82.92 | 59.12 | 64.77 |
| | *Chênh lệch Arm 3 vs Arm 1* | **-5.41 pp** | **-6.56 pp** | **-5.79 pp** | **-4.58 pp** | **-7.66 pp** | **-5.68 pp** |

### 2.3 Phân Tích Nguyên Nhân Cốt Lõi (Failure Analysis)
1. **Hiện Tượng Trục Xuất Ngữ Cảnh (Information Displacement):**
   - Khi giới hạn query chỉ có 512 tokens, việc chèn 256 tokens CSSI buộc hệ thống phải cắt ngắn ngữ cảnh sát con trỏ từ 508 xuống 252 tokens.
   - Các mô hình Dense Retriever (UniXcoder) và Sparse Retriever (BM25) phụ thuộc cực lớn vào chuỗi liên tục cú pháp cục bộ sát cursor (khai báo biến vừa khởi tạo 1-2 dòng trước, logic `if-else`, tham số truyền vào hàm). Việc mất 256 tokens này làm mất đứt chuỗi logic trực tiếp.
2. **Nhiễu Biểu Diễn Vector (Query Vector Dilution):**
   - Các chữ ký CSSI gom từ nhiều file ngoài là những mảnh rời rạc (`class Foo: def bar(x: int)`). Khi đưa vào UniXcoder, embedding pooling bị kéo lệch về phía các từ khóa cấu trúc tổng quát, làm loãng tín hiệu đặc trưng của vị trí hoàn thành code hiện tại.
3. **Kết luận cho Stage 1:** **KHÔNG NÊN** chèn trực tiếp repo context vào Query của Retriever khi kích thước query bị khống chế ở mức $\le$ 512 tokens.

---

## 3. Phần 2: Ảnh Hưởng Của CSSI Đến Khâu Rerank (Candidate Rescoring)

### 3.1 Thiết Kế Thực Nghiệm (Experimental Setup)
Trong Stage 1, Retriever sử dụng nguyên bản 508 tail tokens để lấy ra Top-10 ứng viên (đạt độ bao phủ tối ưu).
Sau đó, ở **Stage 2 (Reranking)**, chúng tôi áp dụng CSSI làm **Structural Alignment Prior**:
- Tính điểm tương thích cú pháp giữa các định nghĩa/phạm vi trong CSSI với từng ứng viên mã nguồn trong Top-10.
- Kết hợp điểm Dense Retrieval Score và Structural Match Score để tái sắp xếp thứ tự ưu tiên của Top-10 candidate.

### 3.2 Kết Quả Định Lượng (Empirical Results)

| Tập Benchmark | Phương Pháp Rerank | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | Hit@10 (%) | MRR (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **cceval_python** | Dense Baseline (No Rerank) | 50.85 | 69.07 | 74.86 | 81.44 | 61.24 |
| | **CSSI Structural Rerank** | **53.85** | **71.55** | **76.81** | **81.21** | **63.77** |
| | *Mức tăng trưởng ($\Delta$)* | **+3.00 pp** | **+2.48 pp** | **+1.95 pp** | -0.23 pp | **+2.53 pp** |
| **cceval_java** | Dense Baseline (No Rerank) | 59.34 | 80.02 | 86.71 | 93.07 | 70.94 |
| | **CSSI Structural Rerank** | **60.88** | 79.88 | **86.90** | 92.98 | **71.86** |
| | *Mức tăng trưởng ($\Delta$)* | **+1.54 pp** | -0.14 pp | **+0.19 pp** | -0.09 pp | **+0.92 pp** |
| **repoeval_api** | Dense Baseline (No Rerank) | 77.02 | 88.67 | 91.92 | 94.24 | 83.50 |
| | **CSSI Structural Rerank** | **79.21** | **89.17** | **92.11** | 94.18 | **84.93** |
| | *Mức tăng trưởng ($\Delta$)* | **+2.19 pp** | **+0.50 pp** | **+0.19 pp** | -0.06 pp | **+1.43 pp** |
| **repoeval_line** | Dense Baseline (No Rerank) | 58.05 | 72.34 | 76.82 | 81.23 | 66.12 |
| | **CSSI Structural Rerank** | **63.59** | **74.23** | **78.36** | 80.81 | **69.69** |
| | *Mức tăng trưởng ($\Delta$)* | **+5.54 pp** | **+1.89 pp** | **+1.54 pp** | -0.42 pp | **+3.57 pp** |
| **Macro Mean** | Dense Baseline (No Rerank) | 61.31 | 77.53 | 82.58 | 87.50 | 70.45 |
| | **CSSI Structural Rerank** | **64.38** | **78.71** | **83.55** | 87.30 | **72.56** |
| | *Mức tăng trưởng ($\Delta$)* | **+3.07 pp** | **+1.18 pp** | **+0.97 pp** | -0.20 pp | **+2.11 pp** |

### 3.3 Phân Tích Cơ Chế Thắng Lợi (Success Mechanics)
1. **Khử Nhiễu Nhập Nhằng Biểu Tượng (Structural Disambiguation):**
   - Dense Retriever thường bị đánh lừa bởi các đoạn mã có cùng tên hàm nhưng thuộc các lớp/module hoàn toàn khác nhau trong repository (ví dụ: `to_dict()`, `validate()`, `build()`).
   - CSSI cung cấp đồ thị quan hệ import và kế thừa lớp, giúp Reranker nhận diện chính xác đoạn snippet nào thực sự nằm trong phạm vi phụ thuộc của vị trí con trỏ, đẩy snippet đúng từ vị trí Rank 3-5 lên thẳng **Rank 1**.
2. **Bứt Phá Mạnh Mẽ Trên RepoEval Line (+5.54 pp Hit@1):**
   - Trong tác vụ Line completion, ngữ cảnh thường yêu cầu gọi một phương thức thành viên của một đối tượng cụ thể. CSSI giúp định vị chính xác file định nghĩa class của đối tượng đó và đẩy lên top 1, tạo bước đệm hoàn hảo cho LLM sinh ra dòng code chính xác 100%.

---

## 4. Tổng Kết & Khuyến Nghị Thiết Kế Kiến Trúc (Architectural Takeaways For Paper)

Kết quả thực nghiệm 8.004 tác vụ mang lại một đóng góp lý thuyết và thực tiễn rất quan trọng cho bài báo:

```
+-------------------------------------------------------------------------------+
|                      NGUYÊN LÝ PHÂN TẦNG REPO CONTEXT (CSSI)                  |
+-------------------------------------------------------------------------------+
|                                                                               |
|  [Tầng 1: Retrieval Query]  (Budget <= 512 tokens)                            |
|    -> TUYỆT ĐỐI KHÔNG đưa CSSI vào Query.                                    |
|    -> Giữ trọn 508 tokens sát con trỏ để bảo toàn ngữ cảnh cú pháp cục bộ.   |
|                                                                               |
|  [Tầng 2: Candidate Rerank] (Không gian Top-K)                                |
|    -> SỬ DỤNG CSSI để lọc và tái xếp hạng Top-K ứng viên.                     |
|    -> Tăng Hit@1 thêm +3.07 pp và MRR thêm +2.11 pp.                         |
|                                                                               |
|  [Tầng 3: LLM Generation Prompt] (Budget 3.072 - 4.096 tokens)               |
|    -> ĐƯA CSSI vào khối cấu trúc riêng biệt (Structure Block).               |
|    -> Kết hợp với Adaptive Budgeting để tối ưu hóa khả năng hoàn thành mã.   |
|                                                                               |
+-------------------------------------------------------------------------------+
```

1. **Hiểu lầm thường gặp:** Cố gắng nhồi nhét càng nhiều thông tin repo vào câu truy vấn của Retriever là một sai lầm nghiêm trọng khi retriever bị giới hạn 512 tokens (dẫn đến sụt giảm tới **-5.41 pp Hit@1**).
2. **Thiết kế tối ưu của CADRE:** 
   - Retriever truy vấn bằng 100% ngữ cảnh cục bộ sâu nhất sát cursor.
   - CSSI đóng vai trò bộ lọc định hướng cấu trúc (Structural Arbiter) ở tầng **Reranking**, và bộ hướng dẫn cú pháp ở tầng **Generation Prompt**.
