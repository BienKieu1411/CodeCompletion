# Quantum Fidelity: Deviation from QIEPSM Paper

> **File**: `retriever/quantum_retriever.py` → `_quantum_fidelity()`  
> **Reference**: QIEPSM `compression experiments/utils/utils.py` → `fidelity()`

---

## 1. Công thức QIEPSM gốc (Faithful)

```python
# QIEPSM reference (utils.py)
def fidelity(b_q_qubits, b_ps_qubits):
    b_q_qubits = b_q_qubits.conj()
    inner = (b_q_qubits[:,:,:,0] * b_ps_qubits[:,:,:,0]) + \
            (b_q_qubits[:,:,:,1] * b_ps_qubits[:,:,:,1])
    # inner: complex tensor, shape (batch, n_pairs, n_qubits)
    return inner.prod(dim=2).abs().square()
    #        ^^^^ PROD trước  ^^^^ ABS sau
```

### Toán học:

```
F = |Π_k z_k|²

trong đó z_k = ⟨ψ_q^k|ψ_d^k⟩ = r_k · e^(iφ_k)   (complex number)

F = |r_1·e^(iφ_1) × r_2·e^(iφ_2) × ... × r_256·e^(iφ_256)|²
  = |( Π_k r_k ) · e^(i·Σ_k φ_k)|²
  = ( Π_k r_k )²
```

**Quan sát**: Khi viết ra đầy đủ, `|Π z_k|² = (Π |z_k|)²` — tức **nếu mỗi inner product đều có |z_k| = magnitude**, thì kết quả giống nhau bất kể phase.

**NHƯNG**: Trong QIEPSM, `inner` là phép nhân complex giữa `conj(q)` và `d`:

```
z_k = q_k^* · d_k[0] + q_k^* · d_k[1]
```

Đây là **tổng** 2 tích complex → kết quả KHÔNG nhất thiết có `|z_k|` bằng tích các magnitudes. Phases của 2 thành phần tương tác (constructive/destructive interference).

---

## 2. Bản Implementation hiện tại (Phase-blind)

```python
# quantum_retriever.py
inner = (q_conj[..., 0] * doc_q[..., 0]) + (q_conj[..., 1] * doc_q[..., 1])
log_abs_inner = torch.log(inner.abs() + 1e-12)   # ← ABS per qubit TRƯỚC
log_fidelity = 2.0 * log_abs_inner.sum(dim=1)     # ← SUM log (= log Π)
```

### Toán học:

```
log_F_approx = 2 × Σ_k log|z_k|
             = Σ_k log(|z_k|²)
             = log( Π_k |z_k|² )

F_approx = Π_k |z_k|²
```

### So sánh:

```
QIEPSM:   F_true  = |Π_k z_k|²  = (Π_k |z_k|)² · 1     (phase triệt tiêu sau prod)
Bản này:  F_approx = Π_k |z_k|²  = (Π_k |z_k|)² · 1     (phase bị bỏ trước prod)
```

**Kết quả**: `|Π z|² = (Π |z|)²` luôn đúng vì `|a·b| = |a|·|b|`.

> ⚠️ **Thực ra hai công thức cho cùng giá trị số**:
> `|z_1 · z_2 · ... · z_n|² = |z_1|² · |z_2|² · ... · |z_n|²`
>
> Đây là tính chất cơ bản: `|Π z_k|² = Π |z_k|²`

---

## 3. Vậy khác biệt thực sự là gì?

### Khác biệt KHÔNG phải ở toán học, mà ở **numerical behavior**:

| Aspect | QIEPSM (prod trước) | Bản này (log-space) |
|---|---|---|
| **Underflow** | `prod(256 numbers < 1)` → **0.0** (float32 limit ~1e-38) | `sum(log)` → **giá trị hữu hạn** |
| **Gradient** | `d/dx Π = Π/x_k` → vanishing khi product ≈ 0 | `d/dx Σ log = 1/x_k` → **ổn định** |
| **Output range** | `[0, 1]` (true fidelity) | `(-∞, 0]` (log-fidelity) |
| **Softmax behavior** | Tất cả scores ≈ 0 → uniform softmax → **không phân biệt** | Scores phân tán → softmax phân biệt tốt |

### Kết luận quan trọng:

**QIEPSM gốc hoạt động vì**:
- Dùng BERT-base (768D) → 768 qubits → **không compress**
- Hoặc compress nhưng paper scale kết quả `× 20.0` (xem `score_fn`)
- Và QIEPSM paper dùng ít qubits hơn (64, 128, 256, 384) trong compression experiments

**Bản implementation này**:
- Dùng 256 qubits sau compression
- **Không nhân × 20.0** — log-space tự nhiên phù hợp hơn cho softmax
- Log-space cho gradient ổn định hơn cho PPO training

---

## 4. Khi nào cần sửa lại?

### Nên giữ nguyên log-space (hiện tại) nếu:
- ✅ Chỉ cần ranking (top-k selection) — thứ tự giống nhau
- ✅ Dùng PPO training — cần gradient ổn định qua softmax
- ✅ Số qubits ≥ 128 — prod sẽ underflow

### Nên đổi sang faithful QIEPSM nếu:
- ❓ Cần giá trị fidelity thực (0-1) cho metric/report
- ❓ Số qubits ít (< 64) — prod không underflow
- ❓ Muốn reproduce kết quả QIEPSM paper chính xác

### Cách sửa nếu cần faithful:
```python
# Option 1: Faithful nhưng có scale factor (như QIEPSM)
fidelity = inner.prod(dim=-1).abs().square()  # có thể underflow
score = fidelity * 20.0  # scale up cho softmax

# Option 2: Chunked product (compromise)
# Chia 256 qubits thành 16 chunks × 16 qubits
# Product trong chunk → log → sum across chunks
chunks = inner.reshape(N, 16, 16)
chunk_prod = chunks.prod(dim=-1)  # 16 numbers, ít underflow hơn
log_fidelity = 2.0 * torch.log(chunk_prod.abs() + 1e-12).sum(dim=-1)
```

---

## 5. Tóm tắt

| | QIEPSM gốc | Bản hiện tại |
|---|---|---|
| **Công thức** | `\|Π z_k\|²` | `exp(2·Σ log\|z_k\|)` |
| **Giá trị số** | **Giống nhau** (toán học tương đương) | **Giống nhau** |
| **Numerical stability** | Kém (underflow 256 qubits) | **Tốt** |
| **Gradient flow** | Kém (vanishing) | **Tốt** |
| **Phù hợp PPO** | Không (scores ≈ 0) | **Có** |

**→ Giữ bản hiện tại là hợp lý cho use case retrieval + PPO training.**
