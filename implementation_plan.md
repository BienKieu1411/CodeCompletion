# Implementation Plan: Co-Retrieval Critical Fixes

## Tóm tắt

5 thay đổi cần implement, sắp theo mức ưu tiên:

---

## 1. Fix Gate Oracle Leak — `neural_training.py`

### Vấn đề
Trong `phase2_build_preference_data` (line 743-847), gate label được tạo từ **toàn bộ strategy pool** bao gồm oracle. Gate học "có context tốt tồn tại trong repo" thay vì "retriever hiện tại lấy được context tốt".

### Thay đổi

#### [MODIFY] [neural_training.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/neural_training.py)

Tách logic tạo gate label: gate label chỉ dùng **current retriever** strategy, không dùng oracle/hard_neg.

```python
# Hiện tại (line 774-792): gate label từ TOÀN BỘ scored (bao gồm oracle)
best_retrieve = max(
    (score for score in scored if not score.is_stop),
    key=lambda score: score.utility,
    default=None,
)

# Sửa: gate label chỉ từ INFERENCE-SAFE strategies
GATE_SAFE_STRATEGIES = {"current", "bm25", "dense_frozen", "learned_retriever"}
gate_scored = [s for s in scored if s.name in GATE_SAFE_STRATEGIES]
best_gate_retrieve = max(
    (score for score in gate_scored if not score.is_stop),
    key=lambda score: score.utility,
    default=None,
)
# Gate label từ deployed retriever, không từ oracle
retrieve_is_better = (
    best_gate_retrieve is not None
    and best_gate_retrieve.utility > self.config.utility_margin
)
```

Preference pairs vẫn dùng toàn bộ pool (bao gồm oracle) — chỉ gate label bị giới hạn.

---

## 2. Fix Data Split — `runner.py`

### Vấn đề
`_split_train_eval` (line 72-86) shuffle rồi chia → samples từ cùng repo ở cả train và eval = data leakage.

### Thay đổi

#### [MODIFY] [runner.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/runner.py)

Thêm `_split_by_repository` function mới, split theo `file_path` prefix (lấy repository identifier).

```python
def _split_by_repository(
    samples: Sequence[TrainingSample],
    eval_ratio: float,
    max_eval_samples: int,
    random_seed: int,
) -> Tuple[List[TrainingSample], List[TrainingSample]]:
    """Split by repository to prevent data leakage."""
    # Group samples by repository (extract from file_path)
    repo_to_samples: Dict[str, List[TrainingSample]] = {}
    for sample in samples:
        repo_id = _extract_repo_id(sample.file_path)
        repo_to_samples.setdefault(repo_id, []).append(sample)
    
    # Split repos, not samples
    repos = list(repo_to_samples.keys())
    rng = random.Random(random_seed)
    rng.shuffle(repos)
    
    eval_repo_count = max(1, int(len(repos) * eval_ratio))
    eval_repos = set(repos[:eval_repo_count])
    
    train = [s for r, ss in repo_to_samples.items() if r not in eval_repos for s in ss]
    eval_ = [s for r, ss in repo_to_samples.items() if r in eval_repos for s in ss]
    
    # Cap eval size
    if len(eval_) > max_eval_samples:
        rng.shuffle(eval_)
        eval_ = eval_[:max_eval_samples]
    
    return train, eval_
```

Thay thế `_split_train_eval` trong `_train_neural` bằng `_split_by_repository`.

---

## 3. Thêm LiPO Loss — `dense_retriever.py`

### Thay đổi

#### [MODIFY] [dense_retriever.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/dense_retriever.py)

Thêm method `lipo_loss` vào class `DenseRetriever`:

```python
def lipo_loss(
    self,
    query_text: str,
    candidate_chunks_list: List[List[CodeChunk]],
    utilities: List[float],
    tau: float = 1.0,
) -> torch.Tensor:
    """LiPO: Listwise Preference Optimization with utility-derived soft labels."""
    q_vec = self.encode_query(query_text)
    
    # Target distribution from utility (soft labels)
    utility_tensor = torch.tensor(utilities, device=self._device, dtype=torch.float32)
    target_dist = F.softmax(utility_tensor / tau, dim=-1)
    
    # Retriever score distribution
    scores = []
    for chunks in candidate_chunks_list:
        if not chunks:
            scores.append(torch.tensor(0.0, device=self._device))
        else:
            c_vecs = self.encode_chunks(chunks)
            scores.append(self.retrieval_score(q_vec, c_vecs))
    score_tensor = torch.stack(scores)
    log_pred_dist = F.log_softmax(score_tensor, dim=-1)
    
    return F.kl_div(log_pred_dist, target_dist, reduction='batchmean')
```

#### [MODIFY] [neural_training.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/neural_training.py)

- Thêm config option: `retriever_loss: str = "lipo"` (options: `"lipo"`, `"dpo"`)
- Thêm config option: `lipo_tau: float = 1.0`
- Thêm `PreferenceData` field: `lipo_groups` cho listwise training data
- Sửa `phase3_dpo_training` → `phase3_retriever_training` để hỗ trợ cả LiPO và DPO
- Trong phase 2: ngoài tạo pairwise pairs, cũng nhóm candidates theo query thành listwise groups cho LiPO

---

## 4. Thêm CostAwareQueryEnhancer — `intent.py`

### Thay đổi

#### [MODIFY] [intent.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/intent.py)

Thêm class `CostAwareQueryEnhancer`:

```python
class CostAwareQueryEnhancer:
    """Conditionally invoke LLM sampling based on retriever score entropy.
    
    Inherits AlignCoder's query enhancement idea but only activates
    LLM sampling when retriever is uncertain (high score entropy).
    """
    def __init__(self, sketcher: IntentSketcher, entropy_threshold: float = 1.5):
        self.sketcher = sketcher
        self.entropy_threshold = entropy_threshold
    
    def enhance(
        self,
        left_context: str,
        retriever_scores: Optional[torch.Tensor] = None,
        draft_completions: Optional[List[str]] = None,
    ) -> Tuple[str, bool]:
        """Returns (enhanced_query, used_sampling)."""
        sketch = self.sketcher.build(left_context)
        
        if retriever_scores is not None:
            entropy = self._score_entropy(retriever_scores)
            if entropy < self.entropy_threshold:
                return sketch.query, False
        
        # Expensive path: merge draft completions into query
        if draft_completions:
            return self._merge(sketch, draft_completions), True
        return sketch.query, False
```

> [!NOTE]
> Phần gọi LLM sinh draft completions cần generator hoặc lightweight sampler riêng. Trong lần implement này, class `CostAwareQueryEnhancer` nhận `draft_completions` từ bên ngoài (caller truyền vào). Phần tích hợp với generator sẽ ở `NeuralCoTrainer._retrieval_query`.

---

## 5. Fix Adapter Confound trong Evaluation — `neural_training.py`

### Vấn đề
Trong `phase6_evaluate` (line 1157-1163), khi skip retrieval, `use_soft_prompt=False`. Khi retrieve, `use_soft_prompt=self.use_adapter`. Comparison đồng thời thay đổi cả context VÀ adapter → không biết improvement đến từ đâu.

### Thay đổi

#### [MODIFY] [neural_training.py](file:///c:/Users/ADMIN/WorkPlace/ISE%20-%20lab/CodeCompletetion/CodeCompletion/src/co_retrieval/neural_training.py)

Sửa skip branch để cũng dùng `use_soft_prompt=self.use_adapter` (giữ adapter treatment nhất quán):

```python
# Skip branch — giữ adapter treatment giống retrieve branch
else:
    ctx = []
    pred = self.generator.generate(
        sample.left_context,
        max_new_tokens=self.config.max_new_tokens,
        use_soft_prompt=self.use_adapter,  # SỬA: không tắt adapter khi skip
    )
```

Tương tự cho NLL comparison (line 1179, 1183).

---

## Verification Plan

### Automated Tests
```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_chunking tests.test_training tests.test_intent_utility tests.test_cli_config
```

### New Tests
- Test gate label chỉ dùng inference-safe strategies
- Test repo-disjoint split không có overlapping repos giữa train/eval
- Test LiPO loss gradient flows correctly
- Test CostAwareQueryEnhancer chọn đúng cheap/expensive path
- Test adapter treatment nhất quán giữa retrieve và skip

---

## Thứ tự Implement

1. **Fix gate oracle leak** (neural_training.py) — critical bug
2. **Fix data split** (runner.py) — critical bug  
3. **Fix adapter confound** (neural_training.py) — critical bug
4. **Add LiPO loss** (dense_retriever.py + neural_training.py) — new method
5. **Add CostAwareQueryEnhancer** (intent.py + neural_training.py) — new feature
> Implementation status (2026-07-06): all five planned changes are implemented. Repository identity is preserved for disjoint evaluation; gate labels follow the deployed strategy; adapter treatment is consistent; LiPO is the default with DPO retained as an ablation; and cost-aware intent enhancement is available as an opt-in mode. The CLI and unit tests cover the new behavior.
