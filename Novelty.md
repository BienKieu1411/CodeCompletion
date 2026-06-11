# Co-Retrieval: Co-Training RL Retriever and Soft Prompt for Repository-Level Code Completion

## 1. Problem Setting

Repository-level code completion yêu cầu hoàn thiện code tại vị trí con trỏ dựa trên:

- left context (code phía trước con trỏ trong file hiện tại);
- current scope: imports, class/function signature, local variables, decorators;
- cross-file definitions: hàm, class, method được định nghĩa ở file khác trong cùng repository;
- usage patterns: cách các hàm/class đó được gọi ở các vị trí khác trong repo.

Thách thức chính: code tại con trỏ thường đang viết dở, không hoàn chỉnh. Left context chứa identifier cụt, thiếu thông tin về hàm/class cần gọi. Retriever phải tìm đúng cross-file evidence dù query (left context) có semantic gap lớn so với target completion.

## 2. Prior Work

### 2.1 RLCoder

RLCoder đóng góp việc dùng Reinforcement Learning để train retriever cho code completion mà không cần labeled retrieval data. Retriever được tối ưu bằng RL với reward dựa trên chất lượng completion (weighted perplexity). RLCoder cũng đề xuất stop signal để quyết định khi nào retrieval không cần thiết.

Hạn chế:

- Generator (Code LLM) hoàn toàn đóng băng trong quá trình train retriever. Retriever phải tự mò xem generator cần gì, nhưng generator không được điều chỉnh để tận dụng context mà retriever chọn.
- Dùng PPO để tối ưu retriever. PPO yêu cầu thiết kế hàm reward, có rủi ro reward hacking và cần tuning nhiều hyperparameters.

### 2.2 AlignCoder

AlignCoder nhận ra semantic gap giữa left context (incomplete code) và target completion. Giải pháp: dùng Code LLM sinh ra sampled completions từ left context, rồi dùng các sampled tokens này để enhance query cho retriever. Pipeline gồm coarse-to-fine retrieval và RL-trained AlignRetriever.

Hạn chế:

- Sampled completions có thể chứa tokens sai (nhiễu), dẫn đến query enhancement sai hướng.
- Generator vẫn hoàn toàn đóng băng. Retriever được tối ưu cho một generator cố định, nhưng generator không được dạy cách tận dụng retrieved context hiệu quả hơn.
- Luôn thực hiện retrieval cho mọi sample, kể cả khi completion không cần cross-file context (ví dụ: `for i in range(`). Retrieval không cần thiết có thể đưa noise vào prompt và làm giảm chất lượng completion.
- Dùng PPO, gặp các vấn đề tương tự RLCoder về reward design.

## 3. Research Gap

RLCoder và AlignCoder đều train retriever bằng RL để cải thiện code completion. Tuy nhiên, cả hai đều chia pipeline thành hai phần tách biệt:

```text
[Retriever] → chọn context → [Generator (đóng băng hoàn toàn)] → sinh code
```

Có ba vấn đề chưa được giải quyết:

### 3.1 Generator không được adapt để đọc retrieved context

Code LLM được pretrained trên code, nhưng không được train để đọc và tận dụng retrieved cross-file snippets một cách tối ưu. Retrieved context có format, ordering, và mức độ chi tiết khác với code bình thường trong pretraining data. Một generator được adapt nhẹ (ví dụ qua soft prompt) có thể tận dụng retrieved context tốt hơn generator gốc.

Nếu generator được adapt, thì retriever cũng cần được tối ưu cho generator đã adapt đó (không phải generator gốc). Điều này tạo ra nhu cầu co-training.

### 3.2 Retriever và generator lệch pha khi train tuần tự

Nếu train tuần tự (retriever trước, prompt tuning sau), retriever được tối ưu cho generator gốc. Nhưng sau prompt tuning, generator thay đổi cách đọc context. Retriever cũ không còn tối ưu cho generator mới. Đây là distribution mismatch.

### 3.3 Thiếu adaptive retrieval decision

Không phải mọi completion đều cần cross-file retrieval. Nhiều completions đơn giản (ví dụ: syntax patterns, common API calls) không cần thêm context và việc thêm retrieved context có thể gây nhiễu. Cần một cơ chế học khi nào retrieval có ích, khi nào nên skip.

## 4. Proposed Method

### 4.1 Tên

**Co-Retrieval: Co-Training RL Retriever and Soft Prompt with Adaptive Retrieval Gate for Repository-Level Code Completion**

### 4.2 Tổng quan pipeline

```text
1. Offline: AST chunking toàn bộ repository → candidate chunks
2. Mỗi training sample:
   a. Trích xuất left context làm query
   b. Adaptive Retrieval Gate quyết định: retrieve hay skip?
   c. Nếu retrieve: Retriever chọn context từ candidate chunks
   d. Soft Prompt + Frozen LLM sinh completion
   e. So sánh kết quả giữa các bộ context (và giữa retrieve vs skip)
   f. DPO loss cập nhật Retriever + Gate
   g. Generation loss cập nhật Soft Prompt
   h. Retriever và Soft Prompt co-adapt trong cùng training loop
```

### 4.3 Core claim

> Co-Retrieval là framework đầu tiên đồng huấn luyện (co-train) RL retriever và soft prompt cho code completion. Retriever học chọn context mà generator (với soft prompt) tận dụng được tốt nhất. Generator (qua soft prompt) học đọc loại context mà retriever có xu hướng chọn. Adaptive gate học khi nào retrieval có ích, khi nào nên skip.

### 4.4 Ba đóng góp chính

| # | Contribution | Giải quyết vấn đề gì | Khác prior work ở đâu |
|---|---|---|---|
| 1 | **Co-training Retriever + Soft Prompt** | Distribution mismatch giữa retriever và generator | RLCoder/AlignCoder chỉ train retriever, generator hoàn toàn đóng băng |
| 2 | **DPO thay PPO cho retriever** | Reward hacking, hyperparameter tuning phức tạp | RLCoder/AlignCoder dùng PPO với reward function thiết kế thủ công |
| 3 | **Adaptive Retrieval Gate** | Retrieval không cần thiết gây nhiễu | AlignCoder luôn retrieve; RLCoder có stop signal nhưng không được co-train với generator |

## 5. AST Chunking

### 5.1 Mục đích

AST chunking được dùng để chuẩn bị data, không phải novelty chính. Mục đích là tạo ra candidate chunks có boundary trùng với code entities (function, method, class) thay vì cắt theo số dòng cố định.

### 5.2 Chunk types

| Chunk type | Mô tả |
|---|---|
| `global` | imports, constants, module-level statements |
| `function` | top-level function hoặc async function |
| `class_header` | class signature, base classes, docstring |
| `method` | method body kèm class header |
| `class_body` | class không có method (enum, config, data holder) |
| `fallback` | line-based block khi AST parse thất bại |

### 5.3 Metadata mỗi chunk

Mỗi chunk lưu: `file_path`, `start_line`, `end_line`, `chunk_type`, `text`, `defined_symbols`, `used_symbols`, `call_names`, `parent_class`, `class_bases`, `method_names`.

Metadata được dùng để xây dựng index cho retriever (BM25/dense search).

### 5.4 Xử lý file không parse được

- Ưu tiên tree-sitter (tolerant parsing, chịu lỗi tốt).
- Fallback: Python `ast` module.
- Nếu cả hai thất bại: cắt thành small line-based blocks, đánh dấu `fallback`.
- Current file tại cursor thường incomplete. AST chunking áp dụng cho repository files đã commit/ổn định, không yêu cầu file đang edit phải parse hoàn chỉnh.

## 6. Co-Training Framework

### 6.1 Kiến trúc tổng quan

Hệ thống có 3 thành phần trainable:

```text
┌─────────────────────────────────────────────────────────┐
│                    Trainable Components                  │
│                                                         │
│  1. Retriever R_θ:  scoring function cho candidate      │
│                     chunks, output: s_θ(chunk | query)  │
│                                                         │
│  2. Retrieval Gate G_φ:  scalar gate quyết định         │
│                          retrieve hay skip              │
│                          output: g_φ(query) ∈ [0, 1]   │
│                                                         │
│  3. Soft Prompt P_ψ:  learnable prompt tokens được      │
│                       prepend vào input của LLM         │
│                       (LLM parameters đóng băng)        │
└─────────────────────────────────────────────────────────┘
```

Generator (Code LLM) hoàn toàn đóng băng. Chỉ soft prompt tokens được cập nhật gradient.

### 6.2 Forward pass cho một sample

```text
Input: left_context, candidate_chunks, ground_truth_target

Step 1: Gate decision
   g = G_φ(left_context)
   if g < threshold → skip retrieval → go to Step 3 with empty context

Step 2: Retrieval
   scores = R_θ(chunk_i | left_context)  for all chunk_i in candidates
   selected_context = top-k chunks by score

Step 3: Generation
   prompt = [P_ψ] + [selected_context] + [left_context]
   completion = LLM(prompt)   (LLM frozen)

Step 4: Evaluation
   quality = metric(completion, ground_truth_target)
```

### 6.3 Co-training loop

Trong mỗi training iteration, cả 3 thành phần được cập nhật:

```text
For each batch of training samples:

  1. Với mỗi sample, chạy forward pass nhiều lần với các retrieval strategies khác nhau:
     - Strategy A: top-k chunks (temperature sampling trên R_θ)
     - Strategy B: top-k chunks (khác temperature hoặc khác k)
     - Strategy C: không retrieve (skip)

  2. Đánh giá quality của mỗi strategy qua completion output:
     - quality_A = metric(completion_A, target)
     - quality_B = metric(completion_B, target)
     - quality_C = metric(completion_no_retrieval, target)

  3. Tạo preference pairs:
     - Nếu quality_A > quality_B: pair (chosen=A, rejected=B) → DPO loss cho R_θ
     - Nếu quality_A > quality_C: retrieve tốt hơn skip → gradient cho G_φ (tăng gate)
     - Nếu quality_C > quality_A: skip tốt hơn retrieve → gradient cho G_φ (giảm gate)

  4. Cập nhật Soft Prompt P_ψ:
     - Dùng strategy có quality cao nhất (winner)
     - Chạy generation loss (cross-entropy) với prompt = [P_ψ] + [winner_context] + [left_context]
     - Backprop qua frozen LLM → chỉ cập nhật P_ψ

  5. Cập nhật R_θ và G_φ bằng DPO loss từ preference pairs ở bước 3
```

### 6.4 Tại sao co-training, không phải sequential training

| Aspect | Sequential (train R_θ xong rồi train P_ψ) | Co-training (train đồng thời) |
|---|---|---|
| R_θ tối ưu cho | Generator gốc (chưa có soft prompt) | Generator + Soft Prompt (đúng đối tượng) |
| P_ψ tối ưu cho | Context cố định từ R_θ đã đóng băng | Context đang được cải thiện liên tục |
| Distribution mismatch | Có. R_θ chọn context cho generator cũ, nhưng P_ψ thay đổi generator | Không. Cả hai co-adapt |
| Kết quả kỳ vọng | R_θ suboptimal cho P_ψ, hoặc P_ψ phải bù lỗi cho R_θ | R_θ và P_ψ bổ trợ lẫn nhau |

Claim cần kiểm chứng thực nghiệm:

> Co-training phải thắng sequential training trên cùng model, cùng data, cùng budget. Nếu không, co-training không có giá trị.

## 7. DPO cho Retriever

### 7.1 Tại sao DPO thay PPO

PPO yêu cầu:

- Thiết kế hàm reward: `R = Q + λ₁X - λ₂Y - λ₃Z ...`
- Tuning nhiều hyperparameters (các λ).
- Reward model hoặc reward function phải ổn định. Nếu reward function sai hoặc bị hack, policy học sai.
- Credit assignment khó khi retriever chọn nhiều chunks.

DPO chỉ yêu cầu:

- Preference pairs: bộ context A tốt hơn bộ context B (đánh giá qua completion output).
- Không cần thiết kế reward function thủ công.
- Không cần reward model riêng.

### 7.2 DPO loss cho retriever

Cho query q, retriever R_θ gán score cho mỗi chunk:

```math
s_\theta(c_i \mid q) \quad \text{for each candidate chunk } c_i
```

Xác suất retriever chọn một bộ context R = {c₁, ..., c_k}:

```math
\pi_\theta(R \mid q) = \frac{\exp\left(\sum_{c_i \in R} s_\theta(c_i \mid q)\right)}{\sum_{R' \in \mathcal{R}} \exp\left(\sum_{c_j \in R'} s_\theta(c_j \mid q)\right)}
```

Trong đó R là tập tất cả các bộ context khả dĩ (trong thực tế, chỉ sample một số bộ).

DPO loss:

```math
\mathcal{L}_{DPO} = -\log \sigma \left( \beta \left[ \log \frac{\pi_\theta(R^+ \mid q)}{\pi_{ref}(R^+ \mid q)} - \log \frac{\pi_\theta(R^- \mid q)}{\pi_{ref}(R^- \mid q)} \right] \right)
```

Trong đó:
- R⁺ là bộ context dẫn đến completion tốt hơn (chosen);
- R⁻ là bộ context dẫn đến completion kém hơn (rejected);
- π_ref là retriever tham chiếu (frozen copy tại đầu mỗi epoch hoặc đầu training);
- β là temperature parameter;
- σ là sigmoid function.

### 7.3 Cách tạo preference pairs

Cho mỗi training sample (left_context, target):

1. Sample N bộ context khác nhau từ retriever hiện tại (temperature sampling, khác k, random dropout trên scores).
2. Với mỗi bộ context, chạy Soft Prompt + Frozen LLM → sinh completion → đánh giá bằng metric (EM, edit similarity, identifier F1).
3. Xếp hạng N bộ context theo quality.
4. Lấy pairs: (top-ranked, lower-ranked) làm (chosen, rejected).

Số lượng samples N nên nhỏ (2-4) để kiểm soát chi phí compute.

### 7.4 So với PPO

| Aspect | PPO (RLCoder/AlignCoder) | DPO (Co-Retrieval) |
|---|---|---|
| Cần reward function | Có, phải thiết kế thủ công | Không |
| Hyperparameters reward | Nhiều (các λ) | Chỉ có β (temperature) |
| Reward hacking | Rủi ro cao | Thấp hơn (chỉ học preference) |
| Credit assignment | Khó (thưởng cho tổng, không biết chunk nào đóng góp) | Vẫn khó ở mức chunk đơn lẻ, nhưng ở mức bộ context thì rõ ràng hơn |
| Computational cost | On-policy rollout + value function | Cần chạy LLM N lần/sample để tạo pairs |

## 8. Adaptive Retrieval Gate

### 8.1 Động lực

Không phải mọi completion đều cần cross-file retrieval:

| Ví dụ | Cần retrieve? |
|---|---|
| `for i in range(` | Không. Syntax pattern, LLM tự biết. |
| `user = await db.fetch_` | Có. Phụ thuộc cross-file definition. |
| `x = x + 1` | Không. Local arithmetic, retrieve gây nhiễu. |
| `result = UserService.` | Có. Cần biết methods của UserService. |

AlignCoder luôn retrieve cho mọi sample. RLCoder có stop signal nhưng nó được train riêng biệt, không co-train với generator.

### 8.2 Thiết kế Gate

Gate G_φ là một network nhỏ nhận left context embedding và output scalar:

```math
g = \sigma(G_\phi(\text{encode}(\text{left\_context})))
```

Quyết định:

```text
if g ≥ threshold:
    retrieve context bằng R_θ → [P_ψ] + [context] + [left_context] → LLM
else:
    skip retrieval → [left_context] → LLM (không dùng soft prompt, không thêm context)
```

### 8.3 Training Gate cùng DPO

Gate được train bằng cách so sánh chất lượng completion khi retrieve vs khi skip:

```text
quality_retrieve = metric(LLM(P_ψ + context + left_context), target)
quality_skip     = metric(LLM(left_context), target)
```

- Nếu quality_retrieve > quality_skip: label = retrieve (tăng g).
- Nếu quality_skip ≥ quality_retrieve: label = skip (giảm g).

Gate loss có thể là binary cross-entropy với label trên, hoặc được tích hợp vào DPO bằng cách coi "skip" là một retrieval strategy đặc biệt (bộ context rỗng).

### 8.4 Lợi ích

- Giảm latency trung bình: skip retrieval cho các completion đơn giản.
- Giảm noise: không đưa context không cần thiết vào prompt.
- Co-train với soft prompt: gate biết rằng với soft prompt hiện tại, retrieve có ích hay không. Nếu soft prompt mạnh lên, gate có thể học rằng một số trường hợp trước đây cần retrieve giờ không cần nữa (hoặc ngược lại).

## 9. Soft Prompt

### 9.1 Tại sao Soft Prompt, không phải LoRA hay Full Fine-tuning

| Phương pháp | Số parameters trainable | Ảnh hưởng đến LLM | Chi phí |
|---|---|---|---|
| Full fine-tuning | Toàn bộ LLM | Thay đổi hoàn toàn | Rất cao |
| LoRA | Adapter layers | Thay đổi internal representations | Trung bình |
| Soft Prompt | Chỉ prompt tokens (rất ít) | Không thay đổi LLM weights | Thấp nhất |

Lý do chọn soft prompt:

- Co-training yêu cầu backprop qua generator nhiều lần (mỗi iteration, mỗi sample). Soft prompt có ít parameters nhất → nhanh nhất.
- Không thay đổi LLM weights → LLM vẫn giữ nguyên khả năng gốc cho code completion.
- Soft prompt học "cách đọc" retrieved context, không phải học "code knowledge" mới.
- Dễ ablation: bỏ soft prompt = quay về LLM gốc.

### 9.2 Training Soft Prompt

Trong co-training loop, soft prompt được cập nhật bằng generation loss:

```math
\mathcal{L}_{prompt} = -\sum_t \log P_{LLM}(y_t \mid P_\psi, \text{context}^*, \text{left\_context}, y_{<t})
```

Trong đó:
- P_ψ là soft prompt tokens;
- context* là bộ context tốt nhất (winner) từ bước DPO;
- y là ground truth target;
- LLM parameters đóng băng, gradient chỉ chảy về P_ψ.

## 10. Training Strategy

### Phase 1: Khởi tạo Retriever

Khởi tạo R_θ từ pretrained code encoder (ví dụ: UniXcoder, CodeBERT, hoặc encoder tương đương). Chạy retrieval cơ bản (BM25 + dense scoring) để retriever có khả năng ranking ban đầu trước khi vào co-training.

### Phase 2: Co-training

Chạy co-training loop (mô tả ở Section 6.3). Trong mỗi epoch:

1. Sample preference pairs cho DPO (chạy retriever nhiều lần + chạy LLM để đánh giá).
2. Cập nhật R_θ và G_φ bằng DPO loss.
3. Cập nhật P_ψ bằng generation loss với winner context.
4. Cập nhật π_ref (frozen retriever copy) theo schedule (mỗi epoch hoặc mỗi N steps).

### Phase 3: Evaluation

Sau co-training, evaluate trên test set:

- Retriever R_θ chọn context.
- Gate G_φ quyết định retrieve hay skip.
- Soft Prompt P_ψ + Frozen LLM sinh completion.
- So sánh với baselines.

## 11. Research Questions

### RQ1: Co-training có hơn sequential training không?

So sánh:

- **Sequential**: train R_θ (DPO, generator gốc) → đóng băng R_θ → train P_ψ.
- **Co-training**: train R_θ + P_ψ đồng thời.

Nếu co-training không thắng sequential, contribution chính sụp đổ. Đây là RQ quan trọng nhất.

### RQ2: DPO có hơn PPO cho retriever training không?

So sánh:

- PPO với reward function (weighted PPL hoặc EM-based).
- DPO với preference pairs.

Trên cùng retriever architecture, cùng data, cùng generator.

### RQ3: Adaptive gate có cải thiện không?

So sánh:

- Luôn retrieve (không có gate).
- Luôn skip (không retrieve, chỉ LLM gốc).
- Gate cố định (rule-based: retrieve nếu left context chứa cross-file identifier).
- Learned gate (co-trained).

### RQ4: Soft prompt có cần thiết không?

So sánh:

- R_θ (DPO) + LLM gốc (không soft prompt).
- R_θ (DPO) + Soft Prompt P_ψ (sequential).
- R_θ (DPO) + Soft Prompt P_ψ (co-trained).

### RQ5: Pipeline có hơn flat retrieval baselines không?

So sánh:

- BM25 top-k.
- Dense retriever top-k.
- RLCoder (nếu reproduce được).
- AlignCoder (nếu reproduce được).
- Co-Retrieval.

## 12. Ablation Design

| Ablation | Mục đích |
|---|---|
| `sequential training` | Kiểm tra giá trị của co-training |
| `PPO thay DPO` | Kiểm tra giá trị của DPO |
| `w/o gate` (luôn retrieve) | Kiểm tra giá trị của adaptive gate |
| `w/o soft prompt` (LLM gốc) | Kiểm tra giá trị của soft prompt |
| `w/o co-training soft prompt` (soft prompt train riêng sau) | Kiểm tra co-adapt vs sequential adapt |
| `fixed-window chunks` | Kiểm tra AST chunking vs line-based |
| `BM25 only` | Baseline không có RL |
| `dense only` | Baseline không có RL |

## 13. Evaluation Metrics

### Completion quality

- Exact Match (EM);
- Edit Similarity;
- Identifier EM;
- Identifier F1.

### Retrieval behavior

- Retrieval rate (% samples được retrieve, không bị gate skip);
- Average retrieved tokens;
- Gate accuracy (retrieve khi cần, skip khi không cần).

### Efficiency

- Inference latency per sample;
- Training cost (GPU hours);
- So sánh training cost với PPO-based methods.

## 14. Risks and Mitigations

### Risk 1: Co-training không hội tụ

Co-training hai thành phần đồng thời có thể gây oscillation (retriever thay đổi → soft prompt phải adapt → retriever lại thay đổi → lặp lại).

Mitigation:

- Cập nhật π_ref (reference retriever) theo schedule chậm (mỗi epoch) để stabilize DPO.
- Learning rate của P_ψ nhỏ hơn R_θ để soft prompt adapt từ từ.
- Warm-start R_θ ở Phase 1 trước khi co-training, giúp retriever đã có khả năng ranking cơ bản.
- Monitor validation metrics mỗi epoch, early stop nếu diverge.

### Risk 2: Chi phí tạo DPO pairs cao

Mỗi sample cần chạy LLM N lần (N bộ context + 1 lần skip) để tạo preference pairs.

Mitigation:

- Giữ N nhỏ (2-3 bộ context + 1 skip = 3-4 forward passes).
- Dùng LLM nhỏ (1B-3B) trong training. Evaluate trên LLM lớn hơn.
- Cache LLM outputs cho các context giống nhau.
- Có thể dùng proxy metric (ví dụ: token overlap thay EM) để giảm chi phí tạo pairs trong giai đoạn đầu.

### Risk 3: Soft prompt quá yếu để tạo sự khác biệt

Soft prompt chỉ có vài chục đến vài trăm tokens. Nếu nó không đủ capacity để học "cách đọc" retrieved context, contribution về co-training mất giá trị.

Mitigation:

- Ablation: so sánh soft prompt vs LoRA vs no adaptation. Nếu soft prompt không đủ, có thể thay bằng LoRA (tăng capacity nhưng vẫn giữ LLM frozen).
- Thử nhiều prompt lengths (10, 50, 100, 200 tokens).
- Report ablation trung thực. Nếu soft prompt không giúp, acknowledge trong paper.

### Risk 4: DPO pairs không đủ đa dạng

Nếu retriever đã khá tốt, tất cả N bộ context đều tương tự nhau → preference pairs không informative.

Mitigation:

- Dùng temperature sampling đủ cao để tạo diversity.
- Thêm random dropout trên retriever scores.
- So sánh với strategy "no retrieval" luôn tạo ra pair có contrast rõ.

### Risk 5: Kết quả không vượt AlignCoder

AlignCoder có query enhancement mạnh (sampled completions). Co-Retrieval không có query enhancement, chỉ dùng left context thô.

Mitigation:

- Co-training + soft prompt có thể bù đắp semantic gap bằng cách khác: soft prompt học "đọc" context mà retriever chọn, thay vì enhance query.
- Nếu kết quả không vượt AlignCoder, có thể kết hợp query enhancement vào Co-Retrieval (không mâu thuẫn, là orthogonal contribution).
- Report trung thực kết quả so sánh.

## 15. Final Novelty Statement

Co-Retrieval đề xuất framework đồng huấn luyện (co-training) giữa RL retriever và soft prompt cho repository-level code completion.

Ba đóng góp chính:

1. **Co-training Retriever và Soft Prompt.** Retriever và generator (qua soft prompt) được tối ưu đồng thời trong cùng training loop. Retriever học chọn context mà generator tận dụng được tốt nhất. Generator (soft prompt) học đọc loại context mà retriever chọn. Điều này giải quyết distribution mismatch khi train tuần tự, mà các phương pháp trước (RLCoder, AlignCoder) chưa giải quyết.

2. **DPO thay PPO cho retriever training.** Thay vì thiết kế hàm reward thủ công với nhiều hyperparameters, retriever được tối ưu bằng preference pairs: bộ context nào dẫn đến completion tốt hơn thì được prefer. Loại bỏ nhu cầu reward shaping.

3. **Adaptive Retrieval Gate.** Gate được co-train cùng retriever và soft prompt, học khi nào retrieval có ích và khi nào nên skip. Giảm noise từ retrieval không cần thiết, giảm latency trung bình.

Claim cần kiểm chứng thực nghiệm:

> Co-Retrieval chỉ mạnh nếu (a) co-training thắng sequential training, (b) DPO không kém PPO, (c) adaptive gate cải thiện accuracy hoặc giảm latency so với always-retrieve, và (d) toàn bộ pipeline cạnh tranh được với RLCoder, AlignCoder, và flat retrieval baselines trên cùng benchmark.
