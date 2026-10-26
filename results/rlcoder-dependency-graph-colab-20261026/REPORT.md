# RLCoder RLRetriever: dependency graph pilot

## Kết quả

Thử nghiệm paired trên 256 mẫu có dependency rõ ràng: 64 mẫu cho mỗi benchmark. Cả hai nhánh dùng RLCoder chunking, BM25 top-100, RLRetriever top-10 và DeepSeek-Coder-1.3B-Base greedy. Nhánh graph thay query bằng dependency graph và cũng thay toàn bộ left context của generator bằng graph.

| Benchmark | Raw ES | Graph ES | Δ Graph−Raw (95% CI) | Raw EM | Graph EM |
|---|---:|---:|---:|---:|---:|
| CCEval Java | 58.75 | 18.52 | −40.23 [−47.42, −33.08] | 12.50% | 0.00% |
| CCEval Python | 73.94 | 16.03 | −57.91 [−64.55, −51.11] | 32.81% | 0.00% |
| RepoEval API | 72.50 | 21.91 | −50.59 [−58.11, −42.59] | 43.75% | 0.00% |
| RepoEval line | 53.30 | 17.08 | −36.22 [−44.92, −27.25] | 26.56% | 0.00% |

ES là thang 0–100, dùng logic postprocess/scoring của RLCoder; bản rescoring offline dùng Tree-sitter grammar Python/Java cài trong môi trường local. Các CI là paired bootstrap theo task. RepoEval chỉ có hai repository trong tập mẫu nên không đại diện cho độ bất định giữa các repository.

## Retrieval

| Benchmark | Edge recall@10 raw | Edge recall@10 graph | Δ Graph−Raw (95% CI) |
|---|---:|---:|---:|
| CCEval Java | 0.311 | 0.501 | +0.190 [+0.099, +0.280] |
| CCEval Python | 0.304 | 0.288 | −0.016 [−0.120, +0.090] |
| RepoEval API | 0.049 | 0.000 | −0.049 [−0.098, −0.011] |
| RepoEval line | 0.099 | 0.000 | −0.099 [−0.167, −0.041] |

Gộp bốn nhóm bằng nhau: edge recall@10 là 0.191 raw và 0.197 graph; edge recall trong các snippet thực sự đưa vào generator là 0.191 raw và 0.140 graph. Số snippet trung bình được dùng lần lượt là 9.68 và 7.14.

## Diễn giải và giới hạn

Kết quả retrieval không nhất quán: graph query cải thiện rõ trên CCEval Java, gần như không đổi trên CCEval Python, và giảm trên hai nhóm RepoEval. Chưa có bằng chứng graph query cải thiện retrieval xuyên benchmark.

ES/EM của graph arm cho thấy **cách thay left context thô bằng chuỗi graph trong prompt completion hiện tại thất bại**. Prompt kết thúc bằng các dòng như `DEPENDS_ON=... SYMBOLS=...`; DeepSeek-Coder-Base tiếp tục sinh các dòng `DEPENDS_ON` thay vì code. Cấu hình này không giữ code prefix/cursor và không có chỉ dẫn hay boundary báo cho generator chuyển từ graph sang code. Vì vậy không nên diễn giải đây là bằng chứng rằng mọi cách đưa dependency graph vào generator đều kém; nó kiểm tra trực tiếp việc dùng graph text làm literal next-token prefix.

Ngoài ra, vLLM trả token surface theo byte-level BPE (`Ġ`, `Ċ`, `ĉ`) thay vì whitespace đã decode. File prediction gốc và CSV đầu tiên giữ nguyên output thô đó; các điểm trong bảng này được tính lại sau khi giải mã các byte marker. File `predictions_decoded.jsonl.gz` chứa output đã giải mã và score tính lại. Không dùng `outputs/metrics_by_benchmark.csv` thay cho bảng đã rescored này.

## Artifact

- `outputs/predictions_decoded.jsonl.gz`: prediction đã giải mã và metrics theo mẫu.
- `outputs/metrics_by_benchmark_rescored.csv`: bảng ES/EM đã rescored sau khi decode output vLLM.
- `outputs/retrieval_trace.jsonl.gz`: dependency graph, BM25 top-100, RLRetriever top-10 và snippet đưa vào generator.
- `outputs/prompts.jsonl.gz`: prompt/token IDs thực tế của cả hai nhánh.
- `outputs/retrieval_metrics.jsonl.gz`: retrieval metrics theo mẫu.
- `outputs/summary.json`: cấu hình và thống kê retrieval gốc.

`retrieval_trace.jsonl.gz` lưu toàn bộ BM25 top-100 và RLRetriever top-10; run này **không lưu điểm RLRetriever cho từng candidate hạng 11–100**. Chọn mẫu theo dependency và RepoEval chỉ có hai repo, nên đây là pilot định hướng, không phải kết quả full benchmark.
