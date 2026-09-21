# DeepRead report: RepoCoder

## 0. Nguồn và trạng thái trích xuất

- Paper: RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation.
- File nguồn: [RepoCoder.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/RepoCoder.pdf).
- Phạm vi: toàn bộ 14 trang; phân tích chính ở trang 1–9 và appendix.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

RepoCoder không train retriever hay generator mới. Đóng góp là biến completion sinh ra ở vòng trước thành một phần query cho vòng retrieval tiếp theo. Vòng đầu dùng unfinished code; các vòng sau nối một đoạn cuối của unfinished code với một đoạn đầu của completion trước. Retrieval và generation lặp lại trong inference, nhờ đó query dần gần target hơn. Đây là baseline quan trọng vì nó cho thấy target-alignment có thể đạt được bằng feedback loop, nhưng cái giá là latency, error propagation và số vòng lặp khó ổn định.

## 2. Luận điểm trung tâm

**Author's stated position:** Iterative retrieval–generation giúp thu hẹp semantic gap giữa unfinished code và target mà không cần retrain retriever hoặc generator.

## 3. Cây lập luận

1. Query ban đầu thường thiếu thông tin về phần code sẽ được sinh.
2. Completion ở vòng trước chứa các identifier/API tiềm năng của target.
3. Dùng prefix của completion trước để mở rộng query.
4. Vòng retrieval mới lấy context sát target hơn.
5. Generator đọc context mới và sinh completion tốt hơn.
6. Lặp vài vòng có lợi, nhưng nhiều vòng có thể đưa lỗi/noise vào query.

## 4. Phương pháp

### 4.1. Chunk và query

- Repository được chia bằng sliding window.
- Với line/API completion: window 20 lines, stride 10.
- Với function body: window 50 lines, stride 10.
- Query vòng đầu là 20 dòng cuối của unfinished code.
- Query vòng sau gồm phần cuối của unfinished code và phần đầu của completion trước, với phần overlap theo stride.

### 4.2. Retrieval và generation

- Main experiments dùng sparse Jaccard retrieval; appendix có UniXcoder dense retrieval.
- Generator có thể là GPT-3.5-Turbo hoặc CODEGEN 350M/2B/6B.
- Prompt chứa retrieved snippets/path và unfinished code.
- Không cập nhật parameters trong inference.
- K tối đa 10 snippets; context tối đa 4096 token cho GPT-3.5 và 2048 cho CODEGEN.

### 4.3. RepoEval

- RepoEval gồm 14 GitHub repositories từ năm 2022, phần lớn Python, có unit tests.
- Line-level và API-level có 1600 mẫu mỗi loại; function-body có 373 mẫu.
- Tác giả cố gắng tránh leakage bằng repository-disjoint split và dữ liệu sau 2022.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| Iteration cải thiện line/API completion | GPT-3.5 line EM 40.56 in-file → 55.31 vòng 1 → 57.00 vòng 3; API 34.06 → 47.69 → 49.44 | pp. 5–6, Table 2 | Cho thấy generated query có ích | **Source fact or data** | Oracle vẫn cao hơn một ít |
| Mô hình nhỏ cũng hưởng lợi | CODEGEN-350M line EM 29.56 → 43.06 sau vòng 2; API 22.19 → 33.88 | p. 6, Table 2 | Tác động không chỉ do model lớn | **Source fact or data** | Vẫn thấp hơn GPT-3.5 tuyệt đối |
| Function-body unit-test pass rate tăng | GPT-3.5 in-file 23.32 → 42.63 vòng 2 | p. 7, Table 3 | Retrieval lặp hỗ trợ semantic completion | **Source fact or data** | Vòng 3/4 có dao động |
| Recall tăng theo vòng | GT-Code EM 53.63 → 55.07 và recall 86.04 → 90.34 từ vòng 1 sang vòng 2 | p. 8, Table 4 | Retrieval query tốt hơn | **Source fact or data** | Golden context có tính oracle |
| Nhiều vòng không luôn tốt | Một số mẫu bị hỏng do misleading snippets/noisy generated prefix | pp. 8–9 | Chỉ ra error propagation | **Author's stated position** | Chưa có stopping policy học được |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Không cần nhãn query–document và không cần train thêm model.
- Query enhancement được hình thành trực tiếp từ hành vi generator.
- Dễ cài trên retriever/generator sẵn có.
- Có benchmark RepoEval thực tế hơn các split cũ.

### Giới hạn

- Chậm do nhiều pass retrieval + generation.
- Completion sai ở vòng sớm sẽ làm query vòng sau sai.
- Sliding-window line chunk làm mất function/AST continuity.
- Số vòng tối ưu phụ thuộc task; oracle retrieval chỉ là upper bound.
- Không có target-aware loss khi train retriever.

## 7. Hàm ý cho framework AST/KD

1. Có thể dùng một vòng query refinement rẻ bằng AST candidates thay vì full generated completion.
2. Sinh query proposal từ decoder nhỏ hoặc teacher offline, nhưng phải giới hạn token và lọc identifier.
3. AST chunking giúp tránh query cắt giữa node và giảm error propagation do context rời rạc.
4. Có thể distill một bước iterative teacher thành một student single-pass retriever để giữ chất lượng nhưng giảm latency.
5. Đánh giá phải báo riêng gain của retrieval và gain do thêm generation rounds.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** iterative retrieval–generation là cách hiệu quả để đưa target signal vào retrieval.
- **Source fact or data:** 1–3 vòng thường cải thiện EM/ES và unit-test pass rate.
- **Reasoned inference:** một student retriever được train trên các query đã được teacher-refine có thể thay thế inference loop.
- **Unverified:** query refinement bằng AST plus KD có thể vượt AlignCoder hay không; chưa có thử nghiệm trực tiếp.

## 9. Câu hỏi recall/transfer

1. Query ở vòng hai của RepoCoder gồm những đoạn nào?
2. Vì sao completion trước vừa là tín hiệu hữu ích vừa là nguồn error propagation?
3. Làm sao distill iterative teacher thành single-pass retriever?
4. Nếu candidate là AST node, nên giữ prefix/suffix nào để mô phỏng RepoCoder?

