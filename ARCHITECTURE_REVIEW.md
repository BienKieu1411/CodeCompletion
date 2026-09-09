# Ghi chú kiến trúc từ các technical report LLM

> Mục đích: đây là ghi chú tra cứu lâu dài cho EviSeq/LLM2Seq. Nó tách rõ (i) cơ chế mà từng report thực sự mô tả, (ii) bằng chứng mà report có hoặc không có, và (iii) phần nào có thể chuyển thành giả thuyết kiến trúc cho EviSeq.
>
> Ngày tổng hợp: 2026-09-04. Mọi số trang dưới đây là **số trang PDF**, không phải số trang in trong tài liệu. Các kết luận về hiệu quả là kết luận của report gốc; không được diễn giải chúng như bằng chứng trực tiếp cho ROUGE PubMed của EviSeq.

## 1. Phạm vi và nguyên tắc đọc

Đã đọc các tệp trong thư mục này:

1. [DeepSeekV4.pdf](DeepSeekV4.pdf) — 58 trang.
2. [kimi-k3.pdf](kimi-k3.pdf) — 47 trang.
3. [Instella-MoE Technical Report .pdf](Instella-MoE%20Technical%20Report%20.pdf) — 33 trang.
4. [Qwen3.8-Next Architecture.pdf](Qwen3.8-Next%20Architecture.pdf) — 28 trang.
5. [Qwen3.8-Flash-Next.pdf](Qwen3.8-Flash-Next.pdf) — 28 trang.

Hai PDF Qwen là hai bản dàn trang/cập nhật của **cùng report về Qwen3.8-Flash-Next**, không phải hai kiến trúc độc lập. Bản `Qwen3.8-Next Architecture.pdf` ngày 2026-09-01 mới hơn bản `Qwen3.8-Flash-Next.pdf` ngày 2026-08-26, nên được dùng làm nguồn chính. Bản mới chỉ làm rõ lookup n-gram bằng multi-head hashing và contextual gating, cùng một số citation/dàn trang; không thay đổi GR, GDN, QSA, optimizer hay kết quả cốt lõi.

### Ràng buộc kiến trúc EviSeq dùng để đánh giá khả năng chuyển giao

- Một encoder pretrained, một bridge/interface, một decoder pretrained.
- Một full source memory duy nhất đi vào decoder; không được lén cắt source chỉ để có latency đẹp.
- Inference chất lượng dùng greedy decoding (`num_beams=1`, không sampling), để so sánh công bằng với T5Gemma greedy.
- Đích chính là tóm tắt abstractive, đặc biệt PubMed; các thước đo cuối là ROUGE-1/2/L, length/factuality/evidence recall và latency.
- Không coi MoE, MTP, speculative decoding, teacher/RL hay thay self-attention của Qwen là cải tiến bridge mặc định. Chúng là những phạm vi thí nghiệm khác.

### Quy tắc chống suy diễn quá mức

Một report mô hình trillion tham số có thể là nguồn **động cơ thiết kế**, nhưng không phải chứng minh cho EviSeq. Chỉ một ablation kiểm soát trên PubMed/WikiLingua/CNNDM của chính EviSeq mới có thể chứng minh:

- một khối bridge tăng ROUGE;
- một evidence metric cao hơn làm generation tốt hơn;
- tốc độ tốt hơn không làm mất coverage/factuality;
- một lựa chọn hyperparameter tổng quát giữa các dataset.

Vì vậy, trong paper nên dùng các từ *inspired by*, *motivated by* hoặc *adapted at the interface*, không viết rằng EviSeq “adopts” nguyên xi mHC, AttnRes, QSA, KDA, MLA hay GR của các model lớn.

## 2. Kết luận điều hành

### Những nguyên lý đáng mang sang EviSeq

| Nguyên lý | Nguồn chính | Cách chuyển giao an toàn vào EviSeq | Lý do |
|---|---|---|---|
| Giữ đường pretrained chính, thêm nhánh mới dạng residual bị chặn | DeepSeek mHC; Qwen GR; Kimi gated output | Final encoder state/identity projection là anchor; nhánh mới chỉ học delta có gate nhỏ | Giảm nguy cơ bridge mới phá interface encoder–decoder đang usable |
| Chọn representation theo chiều sâu thay vì mặc định dùng một layer/mean | Kimi Attention Residuals | Depth-selective residual readout trên một số final encoder states | Có thể chọn độ sâu phù hợp task mà không thay backbone |
| Coarse relevance trước, fine evidence sau | DeepSeek CSA; Qwen QSA | Static salience tạo coarse source context, sau đó evidence slots tạo dynamic score; vẫn giữ full source memory | Mượn tư duy indexer mà không làm mất fact vì hard top-k |
| Normalization, centering, gate dương/bị chặn | mHC, Qwen GR/GatedNorm, KDA | RMS/L2 norm cho q/k, centered dynamic evidence score, residual gate có max | Kiểm soát scale và tránh dynamic branch chiếm source attention ngay từ đầu |
| Warm-up selector trước khi selector điều khiển đường chính | DeepSeek/Qwen sparse attention | Warm-up evidence/salience route; anneal alignment sau warm-up | Một selector non-random cần tín hiệu trước khi được tin cậy |
| Đánh giá quality, efficiency và stability đồng thời | Qwen report | Report ROUGE + paired bootstrap + VRAM/latency + train/validation diagnostics | Loss/CL accuracy không đủ để kết luận quality generation |

### Những phần không nên bê nguyên

| Khối | Vì sao không phù hợp cho EviSeq hiện tại |
|---|---|
| KDA/Gated DeltaNet, MLA, NoPE, CSA/HCA/QSA trong decoder | Thay projection, cache và self-attention đã pretrain của Qwen/PPLX; các report dùng chúng cho 256K–1M context hoặc from-scratch training |
| MoE/LatentMoE/Quantile Balancing/Hash routing | Tăng scope, số tham số, routing instability và communication; không giải trực tiếp mismatch source–summary |
| mHC/GR đa nhánh đầy đủ trong backbone | Cần sửa residual stream của mọi layer, có memory/activation cost; EviSeq chỉ cần interface bridge nhỏ |
| Hard top-k evidence pruning | Các report phải train indexer + backbone rất lâu trước khi pruning. PubMed summary dễ mất fact khi bỏ một sentence evidence |
| MTP/EAGLE/speculative decoding | Chủ yếu tăng throughput; không cải thiện quality greedy output của target decoder |
| Muon, DPO, RL, MOPD như thay đổi mặc định | Là training/post-training ở quy mô khác; thay nhiều biến cùng lúc làm hỏng ablation bridge |

### Kết luận kiến trúc cho EviSeq

Phiên bản hợp lý nhất không phải “dual bridge” với hai source memory hay một mini-MoE. Nó là:

\[
\text{one full source memory}
+\text{one bounded, selective residual route}
\longrightarrow \text{one fused decoder interface}.
\]

Nói cách khác: decoder vẫn có quyền đọc toàn bộ source tokens; route evidence/depth chỉ giúp **định hình ưu tiên**, không thay thế hoặc lọc bỏ source. Đây là điểm chung có thể rút ra từ các report: bảo toàn đường chính, thêm đường chọn lọc nhỏ và có kiểm soát.

## 3. DeepSeek-V4

Nguồn: [DeepSeekV4.pdf](DeepSeekV4.pdf), đặc biệt PDF pp. 6–14, 24–26.

### 3.1. Toàn cảnh và mục tiêu thật sự

DeepSeek-V4 là decoder-only MoE ở quy mô rất lớn, với hai cấu hình Flash/Pro, context tối đa một triệu token. Các thành phần chính là DeepSeekMoE, MTP kế thừa V3, hybrid attention CSA/HCA, Manifold-Constrained Hyper-Connections (mHC), và Muon optimizer. Mục tiêu trung tâm là giảm FLOPs/KV cache của long-context inference/training, không phải encoder–decoder abstractive summarization (PDF pp. 6–7, 24–26).

Vì thế, những số liệu hiệu quả cache/FLOPs không thể dùng làm bằng chứng rằng một block sẽ tăng ROUGE trên PubMed. Giá trị của report với EviSeq là nguyên lý về routing, residual stability và curriculum selector.

### 3.2. Manifold-Constrained Hyper-Connections (mHC)

#### Cơ chế

Hyper-Connections mở residual stream từ một vector \(d\) chiều thành \(n_{hc}\) streams. Một block nhận tổ hợp của các streams và ghi output trở lại:

\[
X_{\ell+1}=B_\ell X_\ell+C_\ell F_\ell(A_\ell X_\ell).
\]

- \(A_\ell\): read/gom nhiều residual streams làm input của block.
- \(B_\ell\): residual mixing giữa streams.
- \(C_\ell\): write output block về streams.

mHC buộc \(B_\ell\) nằm trong Birkhoff polytope (ma trận doubly stochastic): phần tử không âm, tổng từng hàng và cột bằng một. Report lập luận điều này chặn spectral norm không quá một, giúp stacked residual mixing không khuếch đại tín hiệu. \(A\) và \(C\) đi qua sigmoid; phần động phụ thuộc input được khởi tạo scale nhỏ. Sinkhorn–Knopp được dùng để project \(B\) về tập hợp bị ràng buộc (PDF pp. 7–8).

#### Phần mới và trade-off

- Tính mới nằm ở ràng buộc manifold cho dynamic residual mixing, không phải ở ý tưởng residual rộng nói chung.
- Điểm mạnh: nhiều đường residual và propagation ổn định hơn ở mạng rất sâu.
- Giá phải trả: activation/memory/communication lớn hơn, fused kernel và recomputation phức tạp hơn (PDF pp. 19–21).
- Report không có ablation trực tiếp nào chứng minh riêng mHC tăng summarization quality.

#### Chuyển giao sang EviSeq

**Không port mHC nguyên vẹn.** Nó sẽ thay đổi residual state, parameter shapes và forward graph của backbone pretrained.

Điều có thể mượn là nguyên lý residual interpolation bị chặn:

\[
M=(1-g(x))H^{(L)}+g(x)\sum_j\pi_jH^{(j)},
\qquad
0\leq g(x)\leq g_{\max},\quad \sum_j\pi_j=1,\ \pi_j\geq0.
\]

Ở đây \(H^{(L)}\) là final encoder representation, \(\pi\) là convex depth mixture và \(g\) nhỏ. Công thức này có đường anchor, không có mixing âm tùy ý, nhưng **không phải mHC**: không có multiple residual streams, Sinkhorn hay guarantee spectral norm của mHC. Trong paper chỉ nên gọi nó là *bounded depth-selective residual interpolation*.

### 3.3. CSA: compressed sparse attention

CSA thực hiện ba bước (PDF pp. 9–11):

1. Nén KV của các token lân cận thành compressed entries, có overlap giữa hai chuỗi block.
2. Một lightning indexer tạo query low-rank, chấm relevance của compressed blocks.
3. Core attention chỉ dùng top-k blocks được chọn, cộng thêm sliding-window KV không nén để giữ local dependency.

Indexer có dạng khái quát:

\[
I_{t,s}=\sum_h w^I_{t,h}\,\operatorname{ReLU}
\left(q^I_{t,h}\cdot K^{I,\mathrm{Comp}}_s\right),
\]

rồi chọn top-k block cho mỗi query token. CSA dùng shared-KV MQA, grouped output projection, partial RoPE, RMSNorm trên q/KV compressed và attention sink (PDF pp. 11–13).

#### Insight hợp lệ

CSA có cấu trúc **coarse relevance → fine attention**. Tương ứng an toàn cho EviSeq là:

\[
c=\rho\,\operatorname{Mean}(U)+(1-\rho)
\sum_i\operatorname{Softmax}(a_i/\tau_c)u_i,
\]

trong đó \(U=\{u_i\}\) là sentence/unit representations, \(a_i\) là static salience và \(c\) là coarse source context. Sau đó prompt/source context tạo evidence scores ở mức sentence. Đây chỉ là **analogy ở cấp bridge**, không phải CSA.

#### Không chuyển giao trực tiếp

- Không nén/cắt decoder KV hoặc sửa Qwen native attention.
- Không hard top-k sentence trong cấu hình chính. CSA đã có local window để bù information gap sau compression; EviSeq không nên tự tạo lỗ hổng information như vậy.
- Không thêm attention sink vào source cross-attention: sink cho phép decoder rút attention mass khỏi source; điều này có thể tăng reliance vào LM prior và hallucination trong summarization.
- Partial RoPE / shared-KV / grouped projection là tối ưu long-context decoder cache, không phải giải pháp source–summary grounding.

### 3.4. HCA, MoE, MTP, Muon và training systems

**HCA** nén KV mạnh hơn CSA nhưng không top-k; vẫn là giải pháp cache 1M context (PDF pp. 11–13). Không áp dụng vào source memory 4K.

**DeepSeekMoE** dùng shared + routed experts, fine-grained expert routing, balancing không auxiliary loss và Hash routing ở vài layer đầu (PDF p. 7, pp. 24–25). Không thêm vào EviSeq nếu chưa có bằng chứng capacity bridge là bottleneck.

**MTP** là auxiliary pretraining objective kế thừa V3, không phải contribution chất lượng generation của report và không dùng ở inference. Nó không phải đường ngắn để tăng ROUGE greedy (PDF p. 7, pp. 24–26).

**Muon** được dùng cho phần lớn matrix weights nhưng AdamW vẫn dùng cho embedding, prediction head, gates/static mHC và RMSNorm. Nó cần hạ tầng distributed/bucket chuyên biệt ở scale này (PDF pp. 14, 19–20). Nếu thử ở EviSeq, chỉ là ablation riêng cho projection bridge; không đổi optimizer toàn mô hình cùng lúc với đổi kiến trúc.

**Curriculum selector** là phần có ích nhất: report train dense trước, warm-up indexer ngắn rồi mới để sparse selector điều khiển attention (PDF pp. 24–26). EviSeq có thể giữ route source an toàn trong interface warm-up, học static/evidence selector, rồi anneal loss alignment khi full fine-tuning. Không copy số step hay loss weight từ DeepSeek.

### 3.5. Phán quyết DeepSeek-V4 cho EviSeq

| Thành phần | Quyết định | Lý do |
|---|---|---|
| Bounded residual, gate nhỏ, convex mixing | Adapt | Phù hợp bridge pretrained-to-pretrained |
| Coarse index → fine relevance | Adapt | Hữu ích cho evidence routing nếu vẫn giữ full memory |
| Selector warm-up | Adapt | Tránh random route điều khiển generation quá sớm |
| q/k normalization | Keep | Hạn chế logit/scale collapse |
| mHC đầy đủ | Reject | Thay backbone residual và tốn tài nguyên |
| CSA/HCA/MLA/MQA/RoPE/sink | Reject | Long-context decoder architecture, không phải bridge |
| MoE/Hash routing/MTP | Reject | Scope khác, không chứng minh ROUGE |

## 4. Kimi K3

Nguồn: [kimi-k3.pdf](kimi-k3.pdf), đặc biệt PDF pp. 3–10, 12–14, 22–25, 43–45.

### 4.1. Toàn cảnh theo ba trục information flow

Kimi K3 tổ chức architecture quanh ba chiều (PDF pp. 3–10):

| Chiều | Khối | Bài toán gốc |
|---|---|---|
| Theo sequence length | Kimi Delta Attention (KDA) xen Gated MLA | Long-context token mixing |
| Theo network depth | Attention Residuals (AttnRes) | Selective access tới representations layer trước |
| Theo channel/width | Stable LatentMoE | Sparse specialization ở quy mô 2.8T |

Trong ba trục, chỉ **depth selection** có transfer architectural gần trực tiếp tới encoder–bridge interface của EviSeq.

### 4.2. Kimi Delta Attention (KDA) và Gated MLA

KDA là recurrent/linear attention dùng delta-rule. Với state \(S_t\), query/key/value \(q_t,k_t,v_t\), write gate \(\beta_t\) và channel-wise retention \(\alpha_t\):

\[
S_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_tk_tv_t^\top,
\qquad \tilde o_t=S_t^\top q_t.
\]

K3 dùng ShortConv + Swish + L2Norm cho q/k, và thay mapping decay âm không bị chặn bằng:

\[
g_t=g_{\min}\operatorname{Sigmoid}(e^Az_t),\quad g_{\min}=-5,
\quad \alpha_t=e^{g_t},
\]

để tránh BF16 overflow và tăng khả năng dùng Tensor Cores (PDF pp. 4–5). Sau ba KDA layers, một Gated MLA layer cung cấp global token interaction. Output gate là full-rank, input-conditioned:

\[
y_t=W_o\left[\sigma(W_gx_t)\odot\operatorname{RMSNorm}(\tilde o_t)\right].
\]

Kimi dùng NoPE cho MLA; KDA cung cấp tín hiệu position/recency (PDF p. 5).

**Không chuyển KDA/MLA/NoPE sang EviSeq.** Chúng thay QKV projections, positional prior, state/cache và kernel của backbone pretrained. Chúng tối ưu 1M-token efficiency, không giải evidence selection ở interface. Bài học duy nhất là: route mới cần normalization, gate và main path an toàn.

### 4.3. Attention Residuals (AttnRes): nguồn cảm hứng trực tiếp nhất

#### Cơ chế Kimi thật sự

Residual chuẩn truyền lịch sử layer qua một state tuần tự. AttnRes thay accumulation cố định bằng attention qua depth. Với receiving layer \(\ell\), Kimi có pseudo-query learned riêng \(q_\ell=w_\ell\). Key/value là embedding hoặc output các layer trước:

\[
k_i=v_i=
\begin{cases}
h_1,&i=0,\\
f_i(h_i),&1\leq i\leq \ell-1.
\end{cases}
\]

\[
\alpha_{i\rightarrow\ell}=
\frac{\exp(q_\ell^\top\operatorname{RMSNorm}(k_i))}
{\sum_{j<\ell}\exp(q_\ell^\top\operatorname{RMSNorm}(k_j))},
\qquad
h_\ell=\sum_{i<\ell}\alpha_{i\rightarrow\ell}v_i.
\]

Đây là attention **qua depth**, không phải self-attention qua token positions, và không làm causal encoder thành bidirectional. RMSNorm tránh một layer thắng chỉ vì norm lớn (PDF p. 6).

Block AttnRes gom outputs vào \(N\) block representations để giảm memory/communication từ \(O(Ld)\) xuống \(O(Nd)\); report nói khoảng 8 block lấy lại phần lớn lợi ích ở backbone 93 layers (PDF p. 6).

#### Cách diễn giải đúng cho EviSeq

`DepthSelectiveAttentionResidual` chỉ là một **AttnRes-inspired depth-selective residual readout**, không phải Kimi AttnRes đầy đủ.

| Kimi AttnRes | Depth readout ở bridge EviSeq |
|---|---|
| Nằm trong backbone, ở rất nhiều receiving layers | Chạy một lần sau encoder |
| Pseudo-query learned riêng cho từng receiving layer | Query có thể là pooled final document state |
| Layers sau thay đổi computation vì có access mới | Không thể thay computation của encoder layers đã chạy |
| Reads embedding và toàn bộ previous depths/blocks | Chỉ đọc một tập final hidden states |
| Selection khác nhau ở từng depth receiving | Một phân bố depth chung cho tài liệu |
| Pretrained from scratch quy mô cực lớn | Fine-tuned nhẹ ở interface pretrained models |

Công thức adaptation an toàn:

\[
\bar H=\sum_{j=1}^{m}\pi_j\operatorname{Norm}(H^{(L-m+j)}),
\qquad
M=H^{(L)}+g(x)\left(P(\bar H)-P(H^{(L)})\right).
\]

- \(H^{(L)}\): anchor final encoder state.
- \(P\): projection encoder space → decoder space; nếu tương thích chiều, khởi tạo theo identity/residual identity.
- \(\pi\): depth weights softmax, nên convex.
- \(g(x)\): bounded residual gate nhỏ, không cho readout thay final state ngay từ đầu.

Điểm cần theo dõi trong ablation:

1. `final layer only`.
2. Uniform mean của sáu final layers.
3. Learned depth mixture.
4. Learned mixture + bounded residual gate.
5. Entropy depth weights, final-layer weight, gate, delta RMS, ROUGE và latency.

Số “8 blocks” của Kimi không phải lý do để dùng tám layers cho EviSeq. Kimi có 93 layers; sáu final states là điểm khởi đầu hợp lý, còn 8 là ablation, không phải kết luận lý thuyết.

### 4.4. Stable LatentMoE

Kimi kết hợp shared full-width experts và compact routed experts trong latent space. Routed mixture được RMSNorm trước up-projection để biến thiên scale của router/expert không làm hỏng shared path. K3 thêm SiTU-GLU (soft-cap hai nhánh) để hạn chế activation outlier và Quantile Balancing để cân bằng top-k expert dispatch qua histogram global batch (PDF pp. 6–9, 43–45).

Không thêm các khối này vào EviSeq main architecture. EviSeq không có bottleneck expert routing; evidence weights không phải discrete MoE routing. Điều có thể học là:

- norm aggregate trước merge/projection;
- bảo toàn shared/main path;
- chỉ thêm specialised route khi có ablation chứng minh cần thiết.

### 4.5. Optimizer, MTP/EAGLE và long-context curriculum

- Per-head Muon là optimizer pretraining quy mô lớn; không có evidence trực tiếp cho bridge fine-tuning (PDF p. 10).
- Kimi chỉ tăng context theo curriculum và tạo dữ liệu buộc retrieval xuyên toàn document; tăng max length đơn thuần không đủ (PDF p. 12). Điều này hữu ích nếu sau này tăng length EviSeq: phải kiểm tra evidence distant có còn visible và có loss thưởng cho nó.
- MTP/EAGLE là draft/speculative decoding để tăng tốc, không tăng quality greedy target decoder (PDF p. 14).
- MOPD/RL là online post-training quy mô lớn. Không dùng như mặc định cho EviSeq bridge.

### 4.6. Phán quyết Kimi K3 cho EviSeq

| Thành phần | Quyết định | Lý do |
|---|---|---|
| Depth-selective residual readout | Adapt | Gần đúng vấn đề chọn representation depth ở interface |
| Final-state anchor + bounded gate | Keep | Bảo toàn pretrained encoder–decoder interface |
| RMSNorm trước aggregate | Keep/ablate | Calibration scale hợp lý |
| KDA/Gated MLA/NoPE | Reject | Thay backbone/caching/positional prior |
| Stable LatentMoE/QB/SiTU-GLU | Reject | Không phải bottleneck EviSeq |
| Per-head Muon | Optional riêng | Không thay cùng kiến trúc |
| MTP/EAGLE | Reject | Throughput, không phải greedy ROUGE |

## 5. Instella-MoE

Nguồn: [Instella-MoE Technical Report .pdf](Instella-MoE%20Technical%20Report%20.pdf), đặc biệt PDF pp. 4–8, 11–19, 26.

### 5.1. Toàn cảnh

Instella-MoE-16B-A3B là decoder-only MoE 27 layers (một dense, 26 MoE), 16B total/2.8B active parameters. Mỗi token đi qua top-6 trong 64 routed experts cộng hai shared experts. Hai contribution architecture/system nổi bật là Gated MLA và FarSkip-Collective; MTP tồn tại ở pretraining/mid-training (PDF pp. 4–6).

### 5.2. Gated MLA: một gate đơn giản nhưng có ablation rõ

MLA nén KV thành latent để giảm KV cache. Instella chèn element-wise sigmoid gate giữa scaled-dot-product attention output và output projection:

\[
y_t=W_o\left[\sigma(W_gx_t)\odot\hat o_t\right].
\]

Gate được condition theo normalized layer input \(x_t\), nên mỗi token có thể modulate từng channel trong concatenated head output. Trong controlled 200B-token ablation của report, Gated MLA tăng average benchmark từ 49.86 lên 50.33; thêm FarSkip vẫn giữ 50.38 (PDF pp. 5–7, 18).

**Ý nghĩa cho EviSeq:** đây là bằng chứng tốt rằng *gated residual/channel read* có thể giúp expressivity/stability. Nhưng gate đó nằm trong MLA được pretrain từ đầu, không phải bridge. Analog an toàn ở EviSeq là:

\[
\tilde U=P_{id}(U)+g_h(p,c)\odot\Delta P(U),
\]

trong đó `P_id` giữ đường projection an toàn, `ΔP` là delta trainable và \(g_h\) là feature-wise gate nhỏ. Không dùng công thức này để thay hẳn source memory; phải ablate riêng vì bridge hiện đã có nhiều gate.

### 5.3. FarSkip-Collective: không phải block chất lượng

FarSkip sửa connectivity để overlap expert-parallel communication với compute. Nó cho attention/MoE dùng partial hoặc stale activation trong lúc dispatch/combine all-to-all chạy, rồi merge full result khi sẵn sàng:

\[
h_{k+1}=h_k+f_{k+1}(\hat h_k).
\]

Đây là co-design hệ thống cho MoE; report cho throughput pretraining +12.7% và TTFT +39.2% trên serving expert-parallel (PDF pp. 6–7, 18–19, 26). Nó không phải evidence rằng stale activation sẽ tăng summary quality.

Không port FarSkip vào EviSeq. Nếu generation chậm, ưu tiên profiling, Qwen cache chuẩn, tránh CPU–GPU sync và batched greedy evaluation; không làm decoder đọc stale representations.

### 5.4. Các bài học training/data, không phải architecture

Report có một ý tưởng hữu ích cho continual training: feedback-driven data curation. Student trả lời held-out seed set; judge phân tích lỗi; reflection model biến lỗi thành weighted retrieval queries; phần còn lại được uniform sample. Ở experiment SFT cuối, selection 512K examples tăng average benchmark 1.5 points so với uniform sample cùng budget (PDF pp. 11–13).

Nếu dùng cho EviSeq continual summarization, phải tuân thủ:

- Chỉ dùng **train + validation/held-out development**, tuyệt đối không dùng test references/labels để chọn data.
- Mục tiêu retrieval phải dựa trên lỗi đo được (coverage, factuality, length, R2), không chỉ mô tả tự do.
- Đây là data curation phase riêng, không phải claim contribution core nếu chưa có ablation.

MOPD/GRPO/DPO của Instella là training agent/reasoning; không đưa vào bridge EviSeq chỉ vì report thành công (PDF pp. 13–16).

### 5.5. Phán quyết Instella cho EviSeq

| Thành phần | Quyết định | Lý do |
|---|---|---|
| Gated residual/projection branch nhỏ | Optional ablation | Đây là phần architecture đơn giản nhất có rationale |
| Identity/main path + gate | Keep | Tương thích pretrained interface |
| Feedback-driven data curation | Future continual-training experiment | Có thể đúng nếu không đụng test |
| FarSkip | Reject | MoE communication optimization |
| MoE routing/balancing/MTP | Reject | Phạm vi khác, không tăng greedy ROUGE trực tiếp |
| MOPD/RL/DPO | Reject mặc định | Cần teacher/rollouts, làm nhiễu bài toán bridge |

## 6. Qwen3.8-Flash-Next

Nguồn chính: [Qwen3.8-Next Architecture.pdf](Qwen3.8-Next%20Architecture.pdf), PDF pp. 1–22. Bản cũ: [Qwen3.8-Flash-Next.pdf](Qwen3.8-Flash-Next.pdf).

### 6.1. Thông điệp quan trọng nhất của report

Report đánh giá mỗi thay đổi theo ba trục: downstream quality, training/prefill/decode cost, và stability. Thông điệp quan trọng nhất cho EviSeq là **training loss không đồng nghĩa downstream quality**. Ví dụ n-gram tables có thể tiếp tục hạ loss trong khi benchmark bão hòa/dao động; sparse branch read có thể gần như không hại pretraining loss nhưng làm post-training quality thấp hơn (PDF pp. 1, 11–15, 22).

Với EviSeq, điều này nghĩa là không được kết luận từ `evi_cl`, `evi_acc`, `cross_res` hay CE đơn lẻ. Cần chọn kiến trúc theo ROUGE test/validation chuẩn và paired bootstrap, với length/factuality/latency.

### 6.2. Gated DeltaNet (GDN) hybrid

Qwen xen ba GDN layers và một global attention layer. GDN nén prefix vào fast-weight recurrent state; global attention giữ direct content retrieval mà finite-state memory không tái tạo chính xác:

\[
\tilde S_{t-1}=\alpha_tS_{t-1},\quad
e_t=v_t-\tilde S_{t-1}^\top k_t,
\]

\[
S_t=\tilde S_{t-1}+\beta_t k_te_t^\top,
\quad y_t=S_t^\top q_t.
\]

Q/K được L2-normalize, có short causal convolution và output gate. Ablation native 25B-A3B report GDN hybrid đạt 53.81 average, cao hơn full attention 49.87 và SWA hybrid 51.15 (PDF pp. 3–5).

Không graft GDN vào Qwen decoder pretrained. Điều rút ra là **compressor không thay được direct retrieval hoàn toàn**. Điều này ủng hộ EviSeq giữ full source memory và xem bridge evidence route là guidance, không là bộ nhớ thay thế.

### 6.3. Qwen Sparse Attention (QSA)

QSA làm coarse-to-fine sparse retrieval trong từng attention layer:

1. Average-pool keys theo micro-block \(r\) tokens.
2. MQA indexer nhẹ chấm block-level relevance:

\[
\bar k_b=\operatorname{RMSNorm}(\operatorname{AvgPool}(k_{br:br+r-1})),
\]

\[
I_{ib}=\sum_h\operatorname{ReLU}(\langle q_i^h,\bar k_b\rangle),
\qquad B_i=\operatorname{TopK}_{\lceil K/r\rceil}(I_{i,:}).
\]

3. Expand selected blocks về token indices, cộng tail tokens, rồi chạy sparse core attention (PDF pp. 5–7).

QSA train hai phase:

- Dense distillation: max-pool dense teacher attention về blocks, KL với `softmax(indexer score)`; chỉ train indexer 1,000 steps.
- Sparse adaptation: bật top-k và jointly train backbone/indexer 8,000 steps ở 256K sequence length; KL chỉ trên selected blocks, teacher distribution được renormalize (PDF pp. 6–7).

Tại 1M context, QSA report speed-up đáng kể; điều này không áp dụng trực tiếp input 4096 của EviSeq (PDF pp. 8–9).

#### Transfer đúng và sai

**Đúng:** static salience → coarse source context → evidence score là bản mềm của coarse-to-fine retrieval.

**Sai:** gọi EviSeq là QSA, hoặc dùng hard top-k sentence vì QSA. QSA cần dense-attention distillation và sparse backbone adaptation ở 200B tokens; decoder source summaries không có luxury đó. Hard top-2 branch read của Qwen gần như không làm hại pretraining loss nhưng làm post-training quality giảm, là cảnh báo trực tiếp chống pruning source evidence quá sớm (PDF p. 14).

### 6.4. Gated Residual (GR): nguồn thiết kế residual rõ nhất

GR mở residual stream thành bốn branches. Mỗi branch được RMSNorm riêng; read là feature-wise, data-dependent sigmoid gate từ toàn bộ branches; write là scalar data-dependent cho mỗi branch:

\[
\hat R_i=\operatorname{RMSNorm}(R_i;\gamma_i),
\]

\[
G=\operatorname{unvec}\left[\sigma\left(W_u\operatorname{SiLU}
\left(\frac1{n_r}W_d\operatorname{vec}(\hat R)\right)\right)\right],
\]

\[
x=\frac1{n_r}\sum_iG_i\odot\hat R_i,
\qquad
s=2\sigma\left(\frac1{n_r}W_w\operatorname{vec}(\hat R)\right),
\qquad R_i'=R_i+s_iy.
\]

Các ablation đáng nhớ (PDF pp. 10–14):

- Chỉ widening static đã tăng benchmark average 50.91 → 52.49.
- Dynamic read/write tăng lên 54.47; GR lên 54.66.
- Element-wise **read** quan trọng hơn element-wise write.
- Đọc mọi branch tốt hơn pooling/chỉ đọc branch cuối.
- Cross-branch mixing \(H_{res}\) gần như không đáng giá nên bị bỏ.
- Một branch thường giữ đường long-range, ba branch còn lại chủ yếu local.
- Sparsify read xuống top-2 branch có pretraining metrics gần giữ được nhưng **post-training quality giảm rõ**.

Transfer an toàn cho EviSeq không phải 4 full residual branches trong Qwen. Nó là một main path + small, gated projection residual ở interface. Tư duy cần giữ: new route không được thay main representation; gate là rescaling mechanism, không phải “càng lớn càng tốt”.

### 6.5. N-gram tables, Muon và stability

N-gram embedding table của Qwen scale capacity ngoài backbone, lookup qua multi-head hashing, injected bằng contextual gating. Loss giảm khi tăng vocabulary nhưng downstream không tăng đơn điệu (PDF pp. 14–15). Không thêm n-gram table cho EviSeq chỉ để hạ loss.

Qwen dùng Muon cho matrix weights phù hợp, AdamW cho router/embedding/GR low-rank projections. Bản report nhấn mạnh phải split fused matrices theo semantic sub-matrix trước Muon; không phải mọi tensor đều hợp Muon (PDF p. 16). Đây là lý do không dùng Muon như thay đổi mặc định ở bridge.

GR/GatedNorm tăng stability khi LR stress cao bằng cách giảm activation outlier và gradient spikes (PDF pp. 19–22). Nó củng cố nhu cầu bounded gate, nhưng không suy ra rằng tăng cross-gate/cross-res tới 0.2 hoặc cao hơn chắc chắn tăng ROUGE.

### 6.6. Phán quyết Qwen cho EviSeq

| Thành phần | Quyết định | Lý do |
|---|---|---|
| Main path + gated residual delta | Adapt | Gần interface mismatch encoder-space → decoder-space |
| Coarse-to-fine indexer principle | Adapt mềm | Dùng prior/bias, không hard-prune |
| Selector distillation/warm-up principle | Adapt ngắn | Static salience là weak teacher đầu training |
| GDN/QSA native decoder | Reject | Cần retrain backbone, mục tiêu 256K–1M context |
| GR đa branch toàn backbone | Reject | Scope/memory quá lớn cho EviSeq |
| Sparse top-k branch/source | Reject | Qwen thấy hậu quả post-training xấu |
| N-gram/MoE/MTP | Reject | Không giải source evidence mismatch |

## 7. Tổng hợp: thiết kế bridge nên được kiểm tra

### 7.1. Hypothesis kiến trúc chặt chẽ

Một bridge để thử nghiệm, không phải claim đã được chứng minh, có các phần:

1. **Full memory protected.** Encoder tạo \(H\in\mathbb{R}^{T\times d_e}\); mọi source token visible vẫn được đưa vào decoder cross-attention sau projection.
2. **Depth-selective readout.** Một số final hidden states tạo mixture \(\bar H\), giữ \(H^{(L)}\) làm anchor.
3. **Identity-preserving space adaptation.** Projection từ encoder sang decoder space bắt đầu gần đường identity/copy nếu dimensions cho phép; delta branch có gate nhỏ.
4. **Coarse context.** Sentence/unit salience tạo context có blend global mean và salience-weighted mean, tránh mean toàn document làm loãng evidence.
5. **Coverage-aware evidence route.** Ba evidence slots, vì labels extractive hiện có là ba câu, tạo dynamic score theo prompt + coarse context. Ba slots cần diversity nhẹ để không collapse.
6. **Centered bounded residual bias.** Dynamic score chỉ là residual lên static salience:

\[
b_i=a_i+g(x)\,[r_i-\operatorname{mean}(r)],
\]

trong đó \(a_i\) là static score, \(r_i\) dynamic score và \(g\) bounded. Centering ngăn offset của route mới làm tất cả source tokens bị tăng/giảm cùng lúc.

7. **One fused interface.** `b_i` chỉ bias/gate single source memory; không tạo two-bank cross-attention hoặc two independent decoder routes.

### 7.2. Vì sao không dùng hard top-k

Hard top-k có lợi cho 1M context inference khi indexer đã được distill và backbone adapted. EviSeq có source 4K, summary cần coverage nhiều facts, và query evidence chỉ được tạo một lần/prompt. Nếu drop một sentence evidence, decoder không có cách lấy lại fact đó. Vì vậy:

- main configuration: soft bias, full memory;
- top-k: chỉ negative-control ablation sau khi full model đã ổn;
- report evidence recall của positive labels, not only ROUGE/latency.

### 7.3. Loss và gradient: nguyên tắc, không phải thêm vô hạn loss

Một loss chỉ hợp lệ nếu nó có đường gradient đến component được nói là cải thiện:

| Loss | Đường gradient cần tồn tại | Cảnh báo |
|---|---|---|
| CE | Decoder cross-attention → fused bridge → projection/depth/evidence route nếu route được dùng trong forward | Không tính CE trên một memory khác route inference |
| Salience | Salience logits → static score/coarse context/bridge bias | Label chỉ dùng train/validation, không dùng test inference |
| Evidence contrastive | slot query/key projections và, nếu intentional, salience logits qua differentiable coupling | `evi_acc` cao không thay ROUGE; negative quá dễ làm objective vô ích |
| Diversity | Các slot queries, chỉ khi có từ hai positive visible | Không ép diversity khi truncation chỉ còn một evidence label |
| Alignment warm-up | Dynamic score → static score distribution, static side stop-gradient | Anneal về zero; giữ mãi sẽ khóa route mới vào static route |

Không cần thêm MTP, RL, ranking candidate, KD hay nhiều auxiliary loss cùng lúc để “cứu” bridge. Mỗi loss phải có ablation tắt/mở, loss weight rõ và metric chẩn đoán tương ứng.

### 7.4. Protocol ablation tối thiểu

Thứ tự giúp tìm nguyên nhân thay vì trộn thay đổi:

1. Static single-memory EviSeq baseline.
2. `+` identity-initialized projection residual.
3. `+` depth-selective residual readout.
4. `+` salience-conditioned coarse context.
5. `+` 3 evidence slots + hard negatives trong document.
6. `+` centered confidence/bounded residual bias.
7. Tắt từng thành phần: no depth, no salience coupling, no evidence CL, no residual gate.

Mỗi variant cần cùng seed/data split/generation length/greedy decode và ROUGE backend. Báo paired bootstrap khi difference nhỏ, vì chênh vài phần mười ROUGE có thể là noise.

### 7.5. Metrics cần đọc cùng nhau

| Nhóm | Metric | Cách hiểu đúng |
|---|---|---|
| Quality | R1/R2/RL, length ratio, repetition, factuality | R2 và RL đặc biệt quan trọng cho coverage/ordering; không chỉ chọn CE thấp nhất |
| Evidence | evi_acc, positive/negative similarity, gap, evidence recall | Chỉ là proxy; xem có chuyển thành ROUGE không |
| Interface | cross residual ratio, projection delta RMS, depth entropy/final weight, gate | Quá 0 = không hẳn tốt; quá lớn có thể phá main path; gần 0 có thể là branch không dùng |
| Stability | grad norm, loss spikes, activation RMS/max | Dùng để bắt lỗi numerical/routing, không dùng thay quality metric |
| Efficiency | tokens/s, eval docs/s, latency/sample, peak VRAM | Report cùng quality, không đổi decode mode để lấy speed |

## 8. Câu dùng an toàn trong paper

Có thể viết:

> Recent frontier architectures improve information flow by selectively retrieving representations across depth and by constraining newly introduced residual routes. Motivated by these principles, EviSeq preserves one full source memory and adds a bounded depth-selective residual readout at the encoder–decoder interface.

> Coarse-to-fine evidence scoring is used only as a differentiable source prior. Unlike sparse-attention systems designed for million-token inference, EviSeq does not prune the source memory during greedy summarization.

Không nên viết:

- “EviSeq implements Kimi K3 Attention Residuals.”
- “EviSeq uses DeepSeek-V4 mHC/QSA.”
- “AttnRes makes a causal encoder bidirectional.”
- “Kimi’s eight blocks proves eight bridge layers are optimal.”
- “Long-context efficiency proves better abstractive summarization.”
- “Higher contrastive accuracy proves higher ROUGE.”
- “Các technical report chứng minh EviSeq sẽ vượt T5Gemma.”

## 9. Checklist trước khi đưa một ý tưởng từ report vào code

- [ ] Có giải đúng bottleneck hiện tại của EviSeq hay chỉ giải pretraining/serving của model lớn?
- [ ] Có bảo toàn encoder/decoder pretrained path không?
- [ ] Có giữ một full source memory không?
- [ ] Có đường gradient rõ từ loss tới block mới không?
- [ ] Có gate/normalization/residual khởi tạo an toàn không?
- [ ] Có smoke test shape, BF16, gradient checkpointing và greedy cache không?
- [ ] Có baseline và ablation một-biến-thay-đổi không?
- [ ] Có đánh giá cùng một ROUGE backend/generation setting không?
- [ ] Có tránh test leakage qua labels/reference/data selection không?
- [ ] Có lý do để tin improvement sẽ chuyển thành ROUGE, thay vì chỉ làm auxiliary metric đẹp hơn không?

## 10. Kết luận cuối

Các report không chỉ ra một “khối thần kỳ” có thể lắp vào EviSeq để thắng T5Gemma. Chúng nhất quán ở một bài học hữu ích hơn: model lớn thành công khi thêm capability mới **mà không phá đường thông tin đã học** — thông qua residual bảo toàn, normalized/gated readout, selector được warm-up và đánh giá đầy đủ quality–efficiency–stability.

Do đó EviSeq nên tiếp tục là một kiến trúc encoder–bridge–decoder gọn: full source memory vẫn là nền, còn evidence/depth route là delta nhỏ, có giám sát, có residual gate và có ablation. Đây vừa là hướng kỹ thuật hợp lý, vừa là câu chuyện paper trung thực hơn việc ghép MoE/linear attention/sparse decoding từ các model trillion tham số.
