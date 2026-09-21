# DeepRead report: AlignCoder

## 0. Nguồn và trạng thái trích xuất

- Paper: AlignCoder: Aligning Retrieval with Target Intent for Repository-Level Code Completion.
- File nguồn: [AlignCoder.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/AlignCoder.pdf).
- Phạm vi: toàn bộ 14 trang; phân tích chính ở trang 1–11, gồm Table I–III, sampling study, ablation và case study.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

AlignCoder giải quyết query–target semantic misalignment bằng hai thành phần: query enhancement và AlignRetriever. Trước tiên BM25 lấy coarse chunks; generator lấy unfinished code cùng chunks này và sampling nhiều candidate completions. Các candidate được nối vào unfinished query để tạo enhanced query. Sau đó AlignRetriever lấy context fine-grained. Retriever được train offline bằng reward target-PPL: trong mỗi candidate set, chỉ candidate có PPL target thấp nhất nhận tín hiệu. Framework thêm dependency chunks từ import/function/class signatures. Kết quả vượt RLCoder trên nhiều benchmark, nhưng vẫn có ba nút thắt: sampling nhiều candidate tạo latency/noise, reward winner-take-all vẫn là policy-gradient-style objective, và chunking theo blank-line khiến mất cấu trúc AST.

## 2. Luận điểm trung tâm

**Author's stated position:** Nếu query được bổ sung bởi nhiều candidate completion do generator suy ra, rồi retriever được train để tận dụng các token dự đoán đó, repository retrieval sẽ gần target intent hơn và completion tốt hơn.

## 3. Cây lập luận

1. Unfinished query thiếu các token target quan trọng.
2. Generator có thể dự đoán một số identifier/API của target.
3. Nhiều candidate tăng xác suất chứa token hữu ích.
4. Enhanced query giúp fine retriever tìm dependency/base chunks phù hợp hơn.
5. Target-PPL reward dạy retriever candidate nào thực sự giúp generator.
6. Dependency chunks bổ sung context không giống query nhưng có quan hệ chương trình.
7. Ba thành phần cùng nhau cải thiện EM/ES.

## 4. Phương pháp

### 4.1. Chunking

- Base chunks: Split-Aggregate theo blank lines, ghép mini-block đến giới hạn L lines.
- Dependency chunks:
  - parse imports bằng tree-sitter;
  - giữ intra-repository imports;
  - function/method: signature;
  - class: class signature, methods và nested class signatures/methods.
- Đây là chunking có dependency awareness nhưng chưa phải AST subtree chunking hoàn chỉnh.

### 4.2. Query enhancement

Pipeline inference:

1. BM25 coarse retrieval từ unfinished code.
2. Generator lấy coarse chunks + unfinished code.
3. Sampling k candidate completions.
4. Nối candidate vào query gốc.
5. AlignRetriever fine retrieval.
6. Generator cuối đọc retrieved code + unfinished code.

Thí nghiệm k = 1…6; k = 4 là cấu hình tốt nhất trung bình. k > 4 có thể thêm noise.

### 4.3. AlignRetriever reward

Với candidate c_i, query q và target t:

PPL(t | q,c_i) = exp( - 1/L * sum_j log P(t_j | c_i,q,t_<j) ).

Candidate có target PPL nhỏ nhất được indicator I(c_i)=1, còn lại bằng 0. Reward/objective:

Reward = sum_i I(c_i) * log( exp(s_i,q) / sum_j exp(s_j,q) ).

Trong đó s_i,q là cosine similarity giữa embedding candidate và enhanced query. Đây là objective policy-gradient-like với hard winner, dù paper không dùng pairwise loss.

### 4.4. Training và model

- Retriever khởi tạo UniXcoder.
- Evaluator: DeepSeekCoder-1B.
- Data: 10,000 Python/Java repositories, tạo trước tháng 3/2023; dùng dependency clusters và loại CrossCodeEval/RepoEval khỏi training.
- 20 epochs, 3000 samples/epoch, learning rate 5e-5, 2 A100.
- Generator inference/evaluation: CodeLlama-7B, StarCoder-7B, StarCoder2-7B, DeepSeekCoder-1B/7B.
- Generator được dùng để sample và cuối cùng sinh completion; paper không train generator trong AlignCoder.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| AlignCoder vượt RLCoder nhiều setting | DeepSeekCoder-7B CrossCodeEval Python 33.92/76.97 vs RLCoder 30.09/74.43; RepoEval API 41.88/67.75 vs 39.75/66.01 | p. 8, Table I | Chứng minh gain end-to-end | **Source fact or data** | Không phải mọi metric đều gain lớn |
| Gain lớn nhất trên CrossCodeEval Python | DeepSeekCoder-1B EM 28.37, cải thiện 18.1% so với RLCoder | p. 8 | Claim chính của paper | **Source fact or data** | Phần trăm là relative improvement |
| Multiple sampling tốt hơn single | k=4: Python EM 28.37 vs k=1 27.20; API EM 37.25 vs 36.69 | p. 9, Table II | Candidate đa dạng giúp query | **Source fact or data** | Gain nhỏ ở RepoEval line |
| Sampling quá nhiều gây noise | k=5/6 giảm một số Java/RepoEval metrics so với k=1 | pp. 8–9 | Có trade-off quality–latency | **Author's stated position** | Chưa có confidence filtering |
| Dependency context có ích | Bỏ DC làm Python EM 33.92 → 31.44; Java 28.28 → 27.44 | p. 9, Table III | Structural context bổ sung lexical query | **Source fact or data** | Dependency chunks vẫn heuristic |
| Query enhancement và RL đều quan trọng | Bỏ QH: Python 31.33; bỏ RL: 28.14 so với full 33.92 | p. 9, Table III | Retriever phải học token mới | **Source fact or data** | RL ablation đồng thời loại target-aware training |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Đánh trúng failure mode query–target misalignment.
- Kết hợp lexical coarse retrieval, inferred query và dependency context.
- Không cần pairwise preference labels.
- Có ablation chứng minh QH, DC và retriever training đều đóng góp.

### Giới hạn cần khắc phục

- Base chunking theo blank lines làm mất block/function/AST boundary.
- k candidate sampling chạy generator nhiều lần, tăng latency.
- Candidate lỗi/noise được nối thẳng vào query, không có uncertainty filtering.
- Reward chỉ giữ candidate tốt nhất; bỏ qua mức độ tốt/xấu và nhiều candidate cùng hữu ích.
- Target PPL offline gắn chặt với evaluator DeepSeekCoder-1B; chưa có calibration với execution/compile.
- Fine retriever được train bằng reward target nhưng generator cuối không được joint optimize.
- Kết quả chưa chứng minh vượt một teacher embedding lớn hoặc KD student.

## 7. Hàm ý cho framework mới

1. Giữ insight query enhancement nhưng thay candidate sampling bằng teacher signals/AST-aware probes rẻ hơn.
2. Dùng AST chunks cho cả base và dependency context; lưu node type, parent, file, span.
3. Distill soft utility từ teacher: target PPL/log-prob, teacher embedding similarity, và optional code structure score.
4. Tránh pairwise construction: student học regression trên utility hoặc KL giữa teacher/student distributions; nếu cần chọn top-k, dùng differentiable listwise soft target hoặc weighted BCE trên independent candidates.
5. Thêm no-retrieval head và context budget loss.
6. Dùng held-out generator hoặc unit test để kiểm định teacher bias.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** AlignCoder tăng target alignment nhờ query enhancement + RL retriever.
- **Source fact or data:** paper báo cáo gain tối đa 18.1% EM trên CrossCodeEval Python; k=4 và ba ablation đều có số liệu.
- **Reasoned inference:** AST chunking + soft KD có thể giữ lợi ích của AlignCoder trong khi loại boundary loss và winner-take-all instability.
- **Unverified:** student UniXcoder distill từ Jina/C2LLM/Revela có vượt AlignCoder; cần experiment cùng generator/dataset.

## 9. Câu hỏi recall/transfer

1. Enhanced query được tạo ở bước nào và chứa tín hiệu gì?
2. Vì sao k=4 tốt hơn k=1 nhưng k>4 có thể giảm?
3. Reward của AlignCoder khác supervised pairwise ranking ở điểm nào?
4. Nếu thay hard indicator bằng soft teacher utility, loss chính nên là gì?
5. AST chunking khắc phục cụ thể failure mode nào của paper?

