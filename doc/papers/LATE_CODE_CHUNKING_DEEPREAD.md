# DeepRead report: Late Code Chunking

## 0. Nguồn và trạng thái trích xuất

- Paper: Late Code Chunking: A Code Chunking Strategy for Repository-Level Code Completion.
- File nguồn: [2026.acl-short.64.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/2026.acl-short.64.pdf).
- Phạm vi: toàn bộ 7 trang.
- Trạng thái: PDF có text layer; không cần OCR.
- Đọc lại 2026-10-01: xác nhận đủ 7 trang; đối chiếu [ACL record](https://aclanthology.org/2026.acl-short.64/) và [author implementation](https://github.com/bonohubby/late-code-chunking). LC² khác **Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models**: đây là mở rộng code sau retrieval, không pooling embedding sau whole-document encoding.

## 1. Tổng hợp

Late Code Chunking (LC2) tách retrieval context và comprehension context. Retrieval dùng chunk token cố định để index/rank ổn định; sau khi chọn chunk, hệ thống mở rộng về phía trước để lấy semantic antecedents và augment bằng signatures/docstrings của các function được gọi. Query dùng 64 token cuối của unfinished code, trong khi retrieved chunks mặc định 512 token. Kết quả cho thấy việc trì hoãn mở rộng context đến sau retrieval tốt hơn việc index các chunk quá dài ngay từ đầu.

## 2. Luận điểm trung tâm

**Author's stated position:** Retrieval nên dùng đơn vị ngắn và bất đối xứng để match chính xác; comprehension mới nên mở rộng/augment retrieved chunks để khôi phục semantics.

## 3. Cây lập luận

1. Chunk lớn giúp hiểu code nhưng làm retrieval nhiễu.
2. Chunk nhỏ match query tốt nhưng thiếu antecedent/definition.
3. Tách retrieval và comprehension giải quyết trade-off.
4. Query gần cursor chứa identifier/API quan trọng.
5. Context expansion và function augmentation khôi phục missing semantics.
6. Generator nhận context giàu hơn với cùng budget.

## 4. Phương pháp

### 4.1. Retrieval context

- Token-based chunks, default 512 tokens.
- Query là 64 token cuối của unfinished code.
- Kết quả top-5, tổng retrieved budget 4096; generator input tối đa 8192.

### 4.2. Context expansion

- Sau khi retrieval, lấy thêm preceding content tối đa 512 tokens.
- Mục tiêu là nối initialization/semantic antecedent bị cắt khỏi chunk.

### 4.3. Context augmentation

- Phát hiện function calls trong retrieved chunk.
- Retrieve corresponding definition.
- Chỉ append function signature + docstring, tối đa 3 functions.
- Đây là augmentation nhẹ hơn append full implementation.

### 4.4. Setup

- Retrievers: BM25, UniXcoder và CodeRank variants.
- Generator: DeepSeekCoder-1.3B trong bảng chính.
- Baselines: in-file, fixed-window 10 lines, Split-Aggregate, function-level, fixed-token 512.
- Decoding nucleus p = 0.95, max 50 tokens.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| LC2 vượt các chunk baseline | RepoEval line LC2 EM/ES 45.12/72.87 vs fixed-token 44.88/72.36; API 45.00/74.48 vs 44.62/74.39 | PDF p.4 / printed p.783, Table 2 | Tách retrieval/comprehension có lợi | **Source fact or data** | Gain nhỏ trên RepoEval |
| CCEval gain rõ hơn | Python LC2 26.35/71.92 vs fixed-token 22.01/69.05; Java 24.22/66.79 vs 23.38/66.10; C# 21.72/68.19 vs 19.98/66.95 | PDF p.4, Table 2 | Generalization đa ngôn ngữ | **Source fact or data** | Không phải controlled comparison với AST pipeline của mình |
| Retrieval model vẫn ảnh hưởng | CodeRankLLM-7B Python 29.47/72.95 cao hơn BM25 26.20/72.29 và UniXcoder 26.35/71.92 | PDF p.4, Table 3 | Có thể thay retriever | **Source fact or data** | Thay retriever, không teacher distillation trong bảng này |
| Expansion/augmentation bổ sung gain | Ablation cho thấy retrieval context + asymmetric sizing đã tốt, expansion/augmentation thêm gain; augmentation có thể hại C# | PDF p.5, §4.4 và Limitations | Context không phải cứ thêm là tốt | **Author's stated position** | Selective augmentation còn future work |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Giải trade-off retrieval precision và semantic completeness.
- Chi phí augmentation bị giới hạn bằng signature/docstring.
- Tương thích với BM25/dense retriever.
- Có connection trực tiếp với AST parent/child expansion.

### Giới hạn

- Function-call detection và definition retrieval vẫn heuristic.
- Fixed token retrieval chunk chưa chắc bảo toàn AST.
- Augment không chọn theo target utility, có thể thêm noise.
- Chưa có learned stopping/selection.

## 7. Hàm ý cho AST/KD

Phần dưới là suy luận lịch sử theo hướng KD, **không còn là quyết định hiện hành**. Xem §10 cho bản AST+late-context không KD đã triển khai.

1. Dùng AST node nhỏ làm retrieval unit, rồi late-expand parent/sibling/definition.
2. Teacher KD có thể chấm cả retrieval unit và expansion edges.
3. Không index context mở rộng; chỉ mở rộng sau khi student chọn node.
4. Tạo auxiliary loss để student dự đoán có nên expand/augment hay không.
5. Kết hợp với stop head của RLCoder để tránh expansion khi candidate không có ích.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** late expansion/augmentation giải quyết retrieval–comprehension trade-off.
- **Source fact or data:** LC2 vượt fixed-token và line/function baselines trên các benchmark được báo cáo.
- **Reasoned inference:** đây là cách rẻ để kết hợp AST chunking với context completeness, phù hợp hơn append full graph.
- **Unverified:** KD teacher có thể học expansion policy tốt hơn heuristic hay không.

## 9. Câu hỏi recall/transfer

1. Vì sao retrieval chunk và comprehension chunk nên khác nhau?
2. LC2 augment phần nào của function definition?
3. AST parent expansion khác late token expansion ra sao?
4. Loss nào có thể dạy student quyết định expand?

## 10. Đọc lại và triển khai AST + late context, 2026-10-01

### Kết luận quyết định

**Reasoned inference:** nên kiểm tra late enrichment trên cùng AST retrieval cores trước, chưa bỏ AST để chuyển fixed-token toàn bộ. AST bảo toàn một số ranh giới cú pháp; LC² giải quyết việc generator cần context rộng hơn retriever. Hai cơ chế không loại trừ nhau.

**Unverified:** chunking/target generation là nguyên nhân khiến PPO/RRPO fail; LC² tốt hơn AST hiện tại; dataset mới sẽ giúp CUR thắng AlignCoder. Chưa có các paired runs để xác nhận.

Paper so Function-Level, Split-Aggregate, line/token windows, không so đúng cây AST recursive/statement fallback đang dùng ở project. Chênh lệch CCEval Python +4.34 EM percentage points tương ứng khoảng +19.7% relative, không +19.7 điểm. Còn setup paper có retrieved cap4096 và input8192, lớn hơn cap2344/input3072 của plan hiện tại; không bê kết quả đó sang cấu hình mình.

### Paper–code discrepancy cần giữ trong provenance

Paper §2.2 mô tả lấy preceding context. [Author CCEval UniXcoder script](https://github.com/bonohubby/late-code-chunking/blob/main/retriever/run_cceval_unixcoder_lc2.py) kiểm tra ngày2026-10-01 đặt `CHUNK_SIZE=500`, embed đoạn `[idx:idx+CHUNK_SIZE]`, nhưng comprehension lấy `[idx:idx+2*CHUNK_SIZE]` — **forward extension**, không preceding. Script còn bỏ dòng trắng; bản local không làm vậy. Chưa tái lập author pipeline hay kiểm tra toàn bộ scripts khác; không gọi hybrid local là exact LC² reproduction.

### Các hạn chế xác nhận được của pipeline hiện tại

- `src/ast_ppo_unixcoder_kaggle.py:ast_chunks`: node vừa cap thì emit nguyên node và dừng descent; node lớn tách header rồi xuống children. Điều này có thể cho ra statements không kèm enclosing signature/initialization. Đây là cơ chế có thật, chưa phải bằng chứng gây benchmark fail.
- `build_row` loại candidate không fit UniXcoder length rồi mới BM25 top64. Chunk tốt có thể bị loại trước learner. Cần thống kê dropped candidates chứ không chỉ tăng K.
- `src/build_rrpo_train_parquet.py:example_for_repo`: target lấy từ AST spans, kiểm có candidate hợp budget; quality filtering không xác nhận target bắt buộc cần cross-file context. Có thể sinh nhiều task giải được từ prefix. Chưa đo tỷ lệ này trên train thật.
- Serialized train payload thiếu source offsets/full related sources; không thể mở rộng chắc chắn chỉ từ vài snippets. Bản mới yêu cầu source gốc với SHA khớp; snippet xuất hiện nhiều lần mà thiếu offset thì fail, không chọn lần xuất hiện đầu tùy tiện.

### Code đã tạo

- [late_code_context.py](../../src/late_code_context.py): AST core giữ nguyên, preceding expansion, bounded signature/docstring augmentation, byte provenance, canonical rendering và overlap dedup.
- [prepare_late_code_context.py](../../src/prepare_late_code_context.py): CLI bổ sung `late_context_payload` vào file mới. So sánh mọi original field, kể cả compressed train payload, trước khi finalize. Giữ gold target, prefix, query IDs cũ, split và task IDs; không tái tạo nhãn.
- [test_late_code_context.py](../../src/test_late_code_context.py): tests cho Python/Java, UTF-8/CRLF, exclusion current file, ambiguous calls/offsets, budget, overlap, source hashes và Parquet invariance.

Trial config: giữ core đã có; thêm tối đa128 preceding tokens; tối đa3 functions, mỗi signature/docstring package≤128 generator tokens; tổng một candidate rendered≤768. Không cố ép3 functions nếu budget không đủ. Chỉ augment unique simple-name definitions, bỏ qualified calls cần receiver resolution và overloaded/ambiguous names. Đây vẫn là name heuristic, không binding/type resolver; có thể bỏ lỡ hoặc nhận sai quan hệ khi shadowing. Không copy full function body để augment.

Renderer `mode='ast'` chỉ core; `mode='ast_late'` dùng core+extensions. Cả hai dùng cùng pool/canonical renderer để làm controlled comparison; baseline này khác formatter cũ, cần giữ thêm original-runtime baseline để tách tác động formatting. Mọi candidate cố định trước khi so set utility. Slate không fit budget thì infeasible; không trim lại core/left context. Tổng cost phải tính sau merge shared spans, không cộng candidate costs máy móc.

**Integration boundary:** notebook PPO/RRPO chưa đổi và không tự đọc cột mới. Artifact schema mới chủ động khác schema train cũ. CUR chưa được implement: khi triển khai oracle/trainer/eval, phải dùng `render_selection` chung; không reuse ES labels/checkpoint manifest từ context cũ. Retriever input lấy `retrieval_text` mới và tokenize lại (IDs cũ được giữ chỉ để bảo toàn dữ liệu/baseline). Phải tính prompt wrappers+left context+output reserve riêng; renderer chỉ enforce cross-file budget.

### Cách chạy prepare

Chạy từ project root, ở subprocess/terminal để parser native không làm chết notebook kernel. Tokenizer SHA dưới đây đã có trong cache máy này:

```bash
/Users/kieugiangbien/bienkieu_env/bin/python -m src.prepare_late_code_context \
  --input /absolute/path/cceval_python.parquet \
  --output /absolute/path/late/cceval_python.parquet \
  --language python \
  --revision c919139c3a9b4070729c8b2cca4847ab29ca8d94 \
  --local-files-only
```

Java đổi `--language java`. Train dùng `--input .../train.parquet` và thêm `--source-root .../data4aligncoder_prepared` (root có `data/github_repos/python/train.parquet` và Java tương ứng). Loader kiểm combined SHA của filtered sources bằng metadata train trước khi dùng repo_id. Không nhận raw unfiltered repo numbering thay thế. Output luôn file mới; lỗi để lại `.incomplete` để kiểm tra, không finalize hoặc ghi đè. Chưa có automatic per-row resume trong CLI này.

### Verification và gate áp dụng

2026-10-01: **13 CPU tests passed** (10 mới +3 existing preparation tests). Smoke test với tokenizer DeepSeek thật cached và AST chunker thật: fixture Python tạo34 cores; selected core context24 tokens, late context196 tokens, có thêm helper signature/docstring và không đổi core. Đây là correctness fixture, không benchmark performance.

Không tìm thấy full train/test Parquet trong workspace; chưa rebuild full artifacts, chưa decode generator, chưa kiểm tra benchmark improvement.

Experiment-design gate: frozen UniXcoder+frozen DeepSeek, cùng validation IDs/pool/generation seed/prefix/cross-file budget, so original AST formatter, core-only new formatter, AST+preceding, AST+preceding+signature. Chưa đổi query64 cùng lúc; asymmetric query là ablation riêng. Primary paired ES/EM; secondary token use, truncation/dropped cores, unresolved signatures, no-retrieval gap. Chỉ chuyển default khi paired validation ES tăng có CI>0 và EM không regression material theo margin định trước; nếu không, giữ AST core và sửa nguyên nhân qua error analysis. Test labels không tham gia quyết định này.
