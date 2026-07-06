# Nhận xét Approach: Intent-Conditioned Adaptive Co-Retrieval

## Tổng quan nhanh

Repo đề xuất framework **Intent-Conditioned Adaptive Co-Retrieval** cho bài toán repo-level code completion, phát triển từ hướng của AlignCoder. Pipeline gồm 4 thành phần chính:

| Thành phần | File chính | Vai trò |
|---|---|---|
| Intent Sketch | [intent.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/intent.py) | Tạo query có cấu trúc từ code chưa hoàn chỉnh |
| Context Utility | [context_utility.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/context_utility.py) | Đo lợi ích thực sự của retrieved context |
| Adaptive Gate | [neural_gate.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/neural_gate.py) | Quyết định retrieve hay skip |
| DPO Retriever + Soft Prompt | [dense_retriever.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/dense_retriever.py), [soft_prompt.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/soft_prompt.py) | Retriever được tối ưu bằng preference + generator adaptation |

---

## 1. Đánh giá các điểm mạnh

### 1.1 Utility-anchored training signal — Điểm sáng nhất

Việc đo context bằng `U(C) = NLL(stop) - NLL(C)` là **đóng góp có giá trị thực sự**. So với AlignCoder chọn positive chunk đơn thuần bằng PPL thấp nhất, cách tiếp cận này:

- **Có baseline rõ ràng** (no-retrieval), nên biết được context thực sự *giúp* hay chỉ *không quá hại*
- **Phát hiện được harmful retrieval** (`U(C) < 0`), thông tin mà absolute PPL ranking không cho
- **Tạo nhãn tự nhiên cho gate**: retrieve khi `max U(C) > δ`, skip khi mọi context đều không vượt stop

> [!TIP]
> Đây nên là **core contribution #1** của paper, không phải intent sketch hay DPO loss.

### 1.2 Adaptive gate tách biệt khỏi retriever training

Thiết kế tách gate supervision (BCE từ utility labels) khỏi retriever training (DPO từ preference pairs) là đúng về mặt lý thuyết. Gate trả lời câu hỏi *"có nên retrieve không?"*, retriever trả lời *"retrieve cái gì?"* — hai câu hỏi khác nhau cần supervision khác nhau.

### 1.3 Infrastructure tốt cho ablation

Code đã có:
- 9 experiment modes rõ ràng
- Inference-safe strategy whitelist (chặn oracle leak)
- Metrics toàn diện: EM, Edit Similarity, Identifier F1, gate calibration
- Sequential baselines cho co-training comparison

Đây là điểm mà nhiều repo nghiên cứu thiếu.

### 1.4 Co-training loop có cấu trúc

7-phase pipeline với alternating update giữa adapter, retriever, gate và index refresh là hợp lý về mặt design. Ý tưởng retriever học lấy context mà generator dùng được, và generator học đọc context mà retriever đang trả về, tạo ra vòng feedback có ý nghĩa.

---

## 2. Các vấn đề nghiêm trọng cần sửa

### 2.1 🔴 Static Intent Sketch quá yếu so với claim

> [!CAUTION]
> Đây là **rủi ro lớn nhất** của toàn bộ approach.

[IntentSketcher](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/intent.py) chỉ dùng regex-based extraction:

```python
# Chỉ có regex pattern matching
_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
_MEMBER_ACCESS_RE = re.compile(r"...")
_ASSIGN_CALL_RE = re.compile(r"...")
```

**Vấn đề cốt lõi**: AlignCoder dùng LLM sinh sampled completions → query chứa *token chưa xuất hiện* trong left context (ví dụ `refund_payment`). Static sketch **không bao giờ** tạo được token mới — nó chỉ reorganize token đã có.

**Ví dụ cụ thể:**

```python
# Left context:
client = PaymentClient(...)
client.ref

# Intent sketch chỉ có: member_owner=client, member_prefix=ref
# KHÔNG có token "refund_payment" hay "RefundRequest"

# AlignCoder sample: "client.refund_payment(request)"  
# → Query CHỨA "refund_payment" → retrieve đúng definition
```

**Hệ quả**: Nếu chạy head-to-head, `intent_main` có khả năng cao thua AlignCoder trên những case mà intent cần suy luận. Khi đó title paper phải hạ theo bảng fallback trong Novelty.md — và đó là kịch bản rất likely.

### 2.2 🔴 Gate label bị oracle leak

Đã được chỉ ra trong [APPROACH_EXPLAINED.md](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/APPROACH_EXPLAINED.md) (section 6.1) nhưng **chưa được sửa trong code**. Nếu strategy pool cho gate label chứa `oracle`, gate học *"trong repo có context tốt"* thay vì *"retriever hiện tại lấy được context tốt"*.

Đây là **train-test mismatch** cổ điển và sẽ inflate kết quả gate metrics.

### 2.3 🔴 Data split theo sample, không theo repository

`runner._split_train_eval` shuffle rồi chia → **data leakage** nghiêm trọng. Samples từ cùng repo xuất hiện ở cả train và eval → model ghi nhớ API names, class hierarchy, codebase patterns.

> [!WARNING]
> Bất kỳ kết quả nào trên internal split hiện tại đều **không đáng tin** cho paper claim. Phải split theo `repository_id` hoặc chuyển sang benchmark chuẩn (CrossCodeEval, RepoEval).

### 2.4 🟡 DPO trên retrieval scoring chưa được justify

Code gọi objective là "DPO-style loss" nhưng bản chất là:

```
L = -log σ(β × [(S_θ(q,C⁺) - S_θ(q,C⁻)) - (S_ref(q,C⁺) - S_ref(q,C⁻))])
```

Trong khi AlignCoder đã dùng cross-entropy ranking (gần contrastive learning). Câu hỏi quan trọng: **DPO-style có thực sự tốt hơn cross-entropy đơn giản?**

Nếu không chứng minh được advantage, việc gắn nhãn "DPO" chỉ thêm complexity mà không thêm contribution. Reviewer sẽ hỏi: *"tại sao không dùng InfoNCE/cross-entropy như AlignCoder?"*

### 2.5 🟡 Soft prompt có thể quá yếu cho generator adaptation

Soft prompt (50 virtual tokens) là cách cheapest để adapt generator, nhưng expressiveness rất hạn chế so với LoRA. Nếu adapter contribution không rõ trong ablation (`intent_main` vs `retriever_only` không khác biệt đáng kể), thì co-training claim bị yếu đi nhiều.

---

## 3. Đánh giá tính khả thi

### 3.1 Khả thi về engineering ✅

Code scaffold tương đối hoàn chỉnh. Các module được tách rõ ràng, có unit tests, có CLI, có proxy mode để debug. Nếu có GPU đủ mạnh (≥ 24GB VRAM cho Jina 1.5B + Qwen 7B), pipeline có thể chạy.

### 3.2 Khả thi về research contribution ⚠️

| Claim | Khả thi? | Ghi chú |
|---|---|---|
| Utility-anchored training signal | ✅ Cao | So sánh trực tiếp được với absolute PPL ranking |
| Adaptive gate giảm harmful retrieval | ✅ Trung bình-Cao | Cần gate metrics thuyết phục |
| Intent sketch vượt AlignCoder query | ❌ Thấp | Static regex vs LLM sampling — khó thắng |
| DPO tốt hơn CE cho retriever | ⚠️ Chưa rõ | Cần ablation rõ ràng |
| Co-training tốt hơn sequential | ⚠️ Chưa rõ | Compute budget phải match |
| Soft prompt thêm giá trị đáng kể | ⚠️ Chưa rõ | Soft prompt thường weak |

### 3.3 Khả thi về reproducibility

> [!IMPORTANT]
> - Chưa có official CrossCodeEval/RepoEval evaluation command
> - Chưa có statistical significance testing
> - Chưa report latency/memory
> - `test.jsonl` format có thể không match DatasetLoader schema

---

## 4. Hướng cải thiện đề xuất

### 4.1 🎯 Hybrid Adaptive Query Enhancement (Ưu tiên cao nhất)

Thay vì coi static intent sketch là replacement cho AlignCoder, biến nó thành **first stage** trong adaptive pipeline:

```
left context
  → static intent sketch (cheap, luôn chạy)
  → query confidence estimator
      → confidence cao → retrieve bằng static query (70-80% cases)
      → confidence thấp → gọi lightweight LLM sinh 2-4 draft completions
                         → merge vào enhanced query
  → retriever
```

**Novelty thực sự ở đây**: không phải query method mới, mà là **adaptive compute allocation** — chỉ trả chi phí LLM sampling khi cần. Đây mới là điểm có thể vượt AlignCoder (luôn sample 4 completions cho mọi query).

**Implementation gợi ý:**

```python
class AdaptiveQueryEnhancer:
    def __init__(self, sketcher: IntentSketcher, sampler: LightweightSampler):
        self.sketcher = sketcher
        self.sampler = sampler
        self.confidence_gate = QueryConfidenceGate()
    
    def enhance(self, left_context: str) -> str:
        sketch = self.sketcher.build(left_context)
        confidence = self.confidence_gate.score(sketch)
        
        if confidence > self.threshold:
            return sketch.query  # cheap path
        
        # expensive path — only when needed
        drafts = self.sampler.generate(left_context, n=2)
        return self._merge(sketch.query, drafts)
```

> [!TIP]
> **Confidence signal có thể đến từ**: entropy của retriever scores trên top-k results. Nếu retriever confident (low entropy), static query đủ. Nếu uncertain (high entropy, flat score distribution), cần richer query.

### 4.2 Utility-aware Listwise Retrieval Loss

Thay vì ép utility thành binary chosen/rejected pairs, dùng toàn bộ utility spectrum:

```
p*(c_i | q) = softmax(U(c_i) / τ)        # target distribution từ utility
p_θ(c_i | q) = softmax(s_θ(q, c_i))      # retriever distribution

L = KL(p* || p_θ)   hoặc   CE(p*, p_θ)
```

**Ưu điểm**:
- Tận dụng được continuous utility values, không mất thông tin khi discretize
- So sánh trực tiếp và công bằng với AlignCoder cross-entropy objective
- Không cần chọn `preference_margin` hyperparameter

**DPO vẫn giữ làm ablation**, nhưng listwise utility alignment nên là default.

### 4.3 Sửa Gate Label — Counterfactual Supervision

```python
# HIỆN TẠI (sai): gate label từ toàn bộ strategy pool bao gồm oracle
gate_label = max_C_in_all_strategies U(C) > utility_margin

# CẦN SỬA: gate label chỉ từ deployed retriever  
current_top_k = retriever.retrieve(query, chunks, top_k=k)
gate_label = U(concat(current_top_k)) > utility_margin
```

Gate phải trả lời: *"retriever HIỆN TẠI có lấy được context hữu ích không?"*, không phải *"có tồn tại context hữu ích nào đó trong repo không?"*.

### 4.4 Repository-disjoint Splitting

```python
# HIỆN TẠI (sai):
samples = shuffle(all_samples)
train, eval = split(samples, ratio=0.8)

# CẦN SỬA:
repos = get_unique_repos(all_samples)
train_repos, eval_repos = split(repos, ratio=0.8)
train_samples = [s for s in all_samples if s.repo_id in train_repos]
eval_samples = [s for s in all_samples if s.repo_id in eval_repos]
```

### 4.5 Factorial Adapter Ablation

Hiện tại khi skip, soft prompt bị tắt → không biết improvement đến từ đâu.

| # | Context | Soft Prompt | Mục đích |
|---|---|---|---|
| 1 | Không | Tắt | Pure baseline |
| 2 | Không | Bật | Soft prompt standalone effect |
| 3 | Có (retrieved) | Tắt | Retriever standalone effect |
| 4 | Có (retrieved) | Bật | Full pipeline |

Comparison chính cho gate: row 1 vs row 4 (có gate switching giữa hai).
Adapter contribution: row 3 vs row 4.
Gate contribution: so sánh `always_retrieve` (luôn row 4) vs learned gate (mix row 1 và row 4).

### 4.6 LoRA Adapter Option

Nếu soft prompt quá yếu (ablation row 3 ≈ row 4), cần LoRA:

```python
# Gợi ý: dùng peft library
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=16, lora_alpha=32, 
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
generator = get_peft_model(base_model, lora_config)
```

Thêm `adapter_type = "lora"` vào config, song song với `"soft_prompt"` và `"none"`.

### 4.7 Retrieval Loss Ablation Bắt Buộc

Phải chạy ít nhất 4 objectives trên cùng data:

| Objective | Mô tả |
|---|---|
| Cross-Entropy (AlignCoder style) | `CE(argmin_PPL_label, retriever_scores)` |
| InfoNCE | Contrastive loss chuẩn |
| Pairwise Logistic | `log(1 + exp(s_neg - s_pos))` |
| DPO-style (hiện tại) | Reference-regularized pairwise |

Nếu DPO không thắng rõ, **bỏ DPO khỏi core claim** và focus vào utility + gate.

---

## 5. Đề xuất Positioning cho Paper

### Nếu Hybrid Query + Utility Gate thắng AlignCoder:

> **Utility-Calibrated Adaptive Retrieval with Cost-Aware Query Enhancement for Repository-Level Code Completion**
>
> We propose an adaptive retrieval framework that makes two learned decisions per completion: (1) whether to invest in LLM-based query enrichment based on retriever uncertainty, and (2) whether retrieved context will benefit the generator based on utility measured as NLL improvement over no-retrieval. This dual gating eliminates unnecessary computation while maintaining retrieval quality.

### Nếu chỉ Utility Gate thắng:

> **Utility-Calibrated Adaptive Retrieval for Repository-Level Code Completion**

### Core contributions nên theo thứ tự:

1. **Generator-utility supervision**: retriever optimized by actual downstream benefit, not surface similarity
2. **Adaptive retrieve/skip gate**: learned decision to avoid harmful retrieval
3. **Cost-aware query enhancement** (nếu hybrid query được implement)
4. Co-training / soft prompt as secondary engineering contributions

---

## 6. Checklist trước khi chạy experiments

- [ ] **Sửa gate label**: tách oracle khỏi gate supervision pool
- [ ] **Sửa data split**: split theo repository_id
- [ ] **Implement hybrid query**: thêm lightweight sampler cho low-confidence cases
- [ ] **Thêm listwise utility loss**: làm default, giữ DPO làm ablation
- [ ] **Factorial ablation**: 4 combinations context × adapter
- [ ] **Retrieval loss comparison**: CE, InfoNCE, pairwise, DPO
- [ ] **CrossCodeEval evaluation**: chạy trên benchmark chuẩn
- [ ] **Repository-disjoint split**: cho internal evaluation
- [ ] **Statistical significance**: bootstrap CI hoặc paired test, ≥ 3 seeds
- [ ] **Latency/memory report**: quan trọng cho adaptive efficiency claim
- [ ] **LoRA option**: backup nếu soft prompt không đủ

---

## 7. Tóm tắt nhận xét

### Điểm mạnh
Approach có **3 ý tưởng thực sự có giá trị**: utility-anchored training, adaptive gate, và co-training loop. Code scaffold tốt, có infrastructure cho ablation. Các tài liệu MD rất chi tiết và tự phản biện tốt (đặc biệt [APPROACH_EXPLAINED.md](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/APPROACH_EXPLAINED.md) và [Novelty.md](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/Novelty.md) đã nhận ra phần lớn vấn đề).

### Điểm yếu
Static intent sketch là **weakest link** — nó khó thắng AlignCoder's sampled query enhancement. Gate label bị oracle leak và data split bị leakage là hai bugs cần fix trước khi chạy bất kỳ experiment nào. DPO claim chưa có backing evidence.

### Bottom line

> [!IMPORTANT]
> Approach khả thi ở mức **workshop paper hoặc short paper** với setup hiện tại. Để lên **top venue** (ICSE, FSE, ASE, EMNLP), cần: (1) hybrid adaptive query thay vì static sketch, (2) sửa 3 bugs nghiêm trọng (gate label, data split, adapter confound), (3) CrossCodeEval benchmark, và (4) ablation chứng minh từng component.
>
> **Contribution mạnh nhất không phải retriever mới hay DPO**, mà là **adaptive computation and retrieval calibrated by downstream utility** — đây nên là thesis chính của paper.
