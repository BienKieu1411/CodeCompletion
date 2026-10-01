# Plan: retriever học utility của tập context cho repo-level code completion

Ngày: 2026-10-01. Trạng thái: **thiết kế để triển khai/kiểm chứng, chưa có kết quả thực nghiệm**.

Đây là plan hiện hành sau khi người dùng chấp nhận chấm nhiều context/sample. Nó thay giới hạn 10 scored contexts của [thiết kế trước](research/retriever_training_alternatives_2026_10_01/2026-10-01_architecture_loss_decision.md), không thay code, dữ liệu hay checkpoint hiện tại. Không triển khai hay thuê GPU trong bước lập plan này.

## 1. Chốt một hướng, bốn yêu cầu

**Full fine-tune UniXcoder để dự đoán utility của tập snippet; dùng chênh lệch utility để thêm, bỏ, thay snippet và dừng.** Frozen DeepSeek-Coder sinh completion để cung cấp nhãn tự động. Không tiếp tục actor–critic PPO/RRPO; không LiPO, preference-pair mining hay embedding distillation.

Tên làm việc: **Conditional Utility Retriever (CUR)**. Đây là tên mô tả, không phải tuyên bố thuật toán hoàn toàn mới.

| Model cần học | Quan sát dùng để dạy | Hành vi cần kiểm chứng |
|---|---|---|
| Snippet hữu ích | Completion tốt lên khi thêm c vào S | Ưu tiên gain dương, không chỉ semantic similarity |
| Snippet gây nhiễu | Completion tệ đi khi thêm c; tốt lên khi bỏ c | Không thêm hoặc loại bỏ c |
| Chọn snippet đúng với context hiện có | Cùng c được chấm trong nhiều S; thêm cặp, bỏ, thay | Phân biệt bổ sung thông tin với lặp lại thông tin |
| Bao nhiêu là đủ | State gần đủ, dư thừa và gần hết budget; gain của các bước tiếp theo | STOP nếu không còn thay đổi có lợi trong neighborhood được xét |

“Đủ” được định nghĩa theo chất lượng completion và chi phí context, **không phải cố học chọn đúng K=10**. K=10 là trần. Không bảo đảm tìm được tập tối ưu toàn cục.

## 2. Gap được phép nói với RLCoder và AlignCoder

Trong training paths đã đọc, RLCoder và AlignCoder chấm các candidate, lấy chỉ số thắng bằng argmin loss/argmax score, rồi dùng CE. RLCoder đã có empty candidate/STOP; AlignCoder đã có AST dependency context, query enhancement và nhánh `avg_es`. Vì vậy không claim chúng không có AST, không biết dừng hoặc không dùng ES. [RLCoder code](https://github.com/DeepSoftwareAnalytics/RLCoder/blob/main/main.py), [AlignCoder paper](https://arxiv.org/html/2601.19697v2).

**Gap nhắm tới:** hard-winner candidate supervision trong những paths đó không trực tiếp fit độ lớn và dấu của lợi ích thay đổi theo tập context đã chọn, cũng không trực tiếp dạy sửa một tập context bằng add/remove/swap. Đây là khác biệt về đối tượng học và supervision, không chứng minh baseline không thể học được hành vi tốt bằng cách khác.

RepoShapley đã có signed effects, coalition verification và context filtering trong chính task này. Deep Sets/EquiVSet đã học set functions; structural pooling có MGS3. Contribution không thể chỉ là “AST + MSE + adaptive K”. [RepoShapley](https://aclanthology.org/2026.findings-acl.505/), [Deep Sets](https://arxiv.org/abs/1703.06114), [EquiVSet](https://arxiv.org/abs/2203.01693), [MGS3](https://arxiv.org/abs/2505.24274).

**Contribution dự kiến, có điều kiện:** retriever code-specific học một potential chung từ measured context interventions, dùng chính potential đó để xử lý helpfulness/interference/complementarity và quyết định đủ context. Cần ablation chứng minh conditional supervision, kiến trúc và adaptive selection tạo giá trị riêng. Chưa đủ bằng chứng gọi là novelty mạnh hoặc vượt AlignCoder.

## 3. Contract dữ liệu và generator — khóa trước khi train

- Retriever: `microsoft/unixcoder-base`, khởi tạo từ pretrained, fine-tune toàn bộ; head mới train cùng encoder. Không dùng checkpoint PPO thất bại làm default.
- Generator: `deepseek-ai/deepseek-coder-1.3b-base`, frozen, pin revision/tokenizer; greedy decoding với cấu hình cố định, output cap và stop rules được lưu manifest.
- **Update theo yêu cầu tạo lại data 2026-10-01:** dùng pipeline [AST data v2](AST_TRAINING_DATA_V2.md), tạo lại target Python+Java và giữ repo-level split cũ khi ánh xạ được; loại conflict, không chia ngẫu nhiên lại các repo cũ. Không cắt validation về 128. Số dòng mới phải lấy từ artifact/report hoàn thành; 3.335 train rows dưới đây chỉ là số từ log cũ. `target_code` là nhãn completion; nhãn utility chưa được tạo ở bước prepare.
- Giữ AST chunks; thêm role spans/token masks vào sidecar nếu cần. Query chỉ đọc code trước con trỏ. Gold target chỉ dùng tính oracle score, không dùng chọn pool, tạo query, chọn prefix hay làm feature.
- **Update 2026-10-01 — chunking implementation:** đã có optional [AST + LC²-inspired context layer](papers/LATE_CODE_CHUNKING_DEEPREAD.md#10-đọc-lại-và-triển-khai-ast--late-context-2026-10-01). Retrieval cores giữ nguyên; generator nhận fixed preceding/signature enrichment nếu mode được chọn. Chưa đổi default production. Nếu dùng mode này, oracle utility và inference bắt buộc gọi cùng `render_selection`; cost tính sau overlap dedup, labels cũ không reuse. Parquet mới có separate `late_context_payload`, original targets/splits không đổi; legacy retrieval IDs không đại diện formatter mới và cần tokenize lại khi tích hợp CUR.
- Pool khởi đầu 64 chunks, K≤10, cross-file cap 2.344 tokens, generator prompt cap 3.072 tokens; output tokens nằm ngoài prompt cap và trong server context window. Đây là cấu hình thử phù hợp nhánh PPO terminal, không phải cấu hình tối ưu đã chứng minh.
- Giữ prefix **cố định trong mọi intervention của một query**, dùng context selector hiện tại sau kiểm thử. Phần prefix và wrappers phải fit phần budget còn lại, đo bằng tokenizer thật. Không trim lại snippet cũ khi thêm snippet mới; intervention không fit thì bỏ và log.
- Render tập theo thứ tự cố định `(path,start,end,chunk_id)`, giống train/inference. Insertion có thể dịch vị trí snippet cũ: đo tác động của **rendered context**, không claim causal effect thuần của nội dung snippet.
- Query/code được encode ở cap đã khai báo; generator có thể thấy chunk dài hơn phần retriever đọc. Log tỷ lệ truncation; không giấu vấn đề này bằng gọi metadata là “full code representation”.

### Oracle utility

Với prefix x, target y và tập context S:

\[
U_x(S)=\mathrm{ES}_{\text{versioned metric}}\big(G(\mathrm{render}(x,S)),y\big)\in[0,1].
\]

**ES của completion thực tế là nhãn chính của plan mới.** Likelihood không phải loss chính mặc định; chỉ là ablation rẻ hơn nếu cần. Log EM và identifier F1 nhưng chưa cộng các trọng số tùy ý vào U.

Trước khi score số lượng lớn, metric adapter phải đối chiếu reference RLCoder/AlignCoder và evaluator benchmark bằng fixtures. Không mặc định “mọi target chỉ lấy dòng đầu”: line/block và các benchmark có postprocessing khác nhau. Dataset targets giữ nguyên; mọi phép chuẩn hóa chỉ ở metric, có version. Nếu hai reference khác nhau, ghi riêng hai metric và chọn một contract trước khi train, không đổi theo kết quả test. ES đo tương đồng chuỗi, không bảo đảm code đúng ngữ nghĩa.

Kiểm tra lặp cùng prompt với batch size/order khác nhau để đo độ ổn định. Greedy không tự bảo đảm bitwise deterministic trên mọi backend. Cache khóa bằng model revision, full rendered prompt, decoding config, target hash và metric version; lưu raw completion để có thể chấm lại mà không decode lại.

## 4. Kiến trúc cụ thể

Một shared UniXcoder encode query và từng chunk. Từ token states lấy global pool 768 chiều và ba AST role pools: definition/signature (D), references/calls/imports (U), implementation (B). Project mỗi role về 64 chiều, có presence/parse-failure flags. Không chạy encoder ba lần cho ba role. Không coi AST role là symbol resolution chính xác.

Head nhận global similarities, 3×3 directed role similarities, query/chunk projections, token cost và role-presence flags. MLP hidden 128 tạo scalar có dấu u(x,c) và vector v(x,c)∈R64. Không softmax scalar utility; noise được phép có utility âm.

\[
F_\theta(x,S)=\sum_{c\in S}u_\theta(x,c)
+\psi_\theta(e_x,\sum_{c\in S}v_\theta(x,c),|S|,C(S)/B)
-\psi_\theta(e_x,0,0,0).
\]

ψ là MLP 128-hidden; F(empty)=0. Không ép monotonicity/submodularity: một snippet có thể gây hại, hoặc chỉ có ích khi đi cùng snippet khác. C(S) là chi phí rendered cross-file context theo tokenizer, quy ước C(empty)=0; cheap additive estimate chỉ dùng prefilter, quyết định cuối kiểm tra actual tokens. Fixed-dimensional pooling vẫn có bottleneck, chưa bảo đảm biểu diễn được dependency phức tạp.

Giá trị cần fit là `[U(S)−U(empty)]/a`, a>0 là scale từ train rồi freeze; không center theo từng query/list. Dấu gain và mốc gain=0 được giữ. Inference đổi về đơn vị ES bằng aF.

Matching head này là **selector trong shortlist**, không còn pure dot-product ANN. Stage-1 BM25/global dense mining và refresh dense index phải có provenance. CUR không cứu được snippet đã nằm ngoài pool. First ablations dùng pool cố định để không lẫn thay đổi mining với thay đổi loss.

## 5. Thu nhãn: chấm các tập, lấy mọi phép so có sẵn

Các nhãn sau được tạo tự động từ gold target đã có, không phải người dùng tạo preference pairs:

\[
\Delta(c\mid S)=U(S\cup\{c\})-U(S).
\]

Δ dương: giúp trong context S; âm: gây hại; gần 0: không thấy lợi ích theo metric. **Không tự gán nhãn âm mạnh cho mọi Δ≈0.** Một snippet có thể neutral do dư thừa hoặc do cần partner.

Member contribution `D(c|S)=U(S)−U(S\{c})`: D<0 nghĩa là bỏ c giúp hơn. Joint addition `U(S+a+b)−U(S)` và hai bước trung gian cho biết complementarity. Swap `U(S−a+b)−U(S)` cho biết thay a bằng b có tốt hơn không, đặc biệt khi budget đầy.

### Recipe khởi đầu: tối đa 123 scored contexts/query/refresh

| Phép đo | Phân bổ ban đầu | Số context tối đa trước dedup |
|---|---|---:|
| State gốc | empty, S2, S5, S8 | 4 |
| Add một snippet | 16 feasible probes cho mỗi state | 64 |
| Add hai snippet | 4 cặp từ chính probes mỗi state | 16 |
| Leave-one-out | Tất cả thành viên của S2/S5/S8 | 15 |
| Swap | 8 thay thế ở mỗi nonempty state | 24 |
| **Tổng** | | **123** |

Đây là **contexts/sequences**, không 123 HTTP requests. Gom qua nhiều query thành generator batches; dedup theo full prompt. Pair-of-snippets là intervention context, không pairwise preference loss hay DPO.

Cardinality là coverage target chứ không ép mọi state phải có đúng từng ấy chunks. Không đủ candidates/budget thì dùng state khả thi và log. Refresh tiếp xoay S8 thành S9, bổ sung actual stop states và states sát stop. Recipe S2/S5/S9 có tối đa 120 contexts vì không thể add hai vào S9 với K=10. State đã K=10 vẫn được học qua kết quả add vào S9 và qua remove/swap khi được sample làm state thực tế.

Nguồn states trộn: BM25/dense selection, selector checkpoint hiện tại, random-diverse, và deliberately redundant contexts. Probes trộn top score, random exploration, lexical hard cases, cùng dependency/role heuristics. Không dùng gold để chọn ứng viên. Các heuristic chỉ tạo coverage; generator mới quyết định helpful/harmful. Trong refresh mới, ưu tiên thêm states/actions mà inference thực sự gặp, không chỉ các state dễ/đẹp.

Recipe chỉ là starting allocation; chưa biết 16 probes hay 4 cặp đã đủ. Ghi sampling provenance và held-out coverage. Không oversample dấu âm rồi mặc định loss vẫn ước lượng cùng phân phối; nếu cân bằng labels theo outcome phải công bố weighting và ablate.

## 6. Loss chính và backprop

Mỗi query có graph gồm scored sets (nodes) và observed changes (edges). Một scored context được reuse cho nhiều derived labels, **không coi các labels chung baseline là độc lập**.

\[
d_{ST}=\mathrm{stopgrad}\left[\frac{U(T)-U(S)}a\right],\qquad
b_T=\mathrm{stopgrad}\left[\frac{U(T)-U(\varnothing)}a\right].
\]

\[
\mathcal L_{gain}
=\operatorname{mean}_{g\in\{add,member,joint,swap\}}
\operatorname{mean}_{(S,T)\in E_g}
\left[F_\theta(x,T)-F_\theta(x,S)-d_{ST}\right]^2,
\]

\[
\mathcal L_{set}=\operatorname{mean}_{T\ne\varnothing}
\left[F_\theta(x,T)-b_T\right]^2,
\qquad \boxed{\mathcal L=\mathcal L_{gain}+0.25\mathcal L_{set}.}
\]

Mean chỉ trên nonempty groups; sau đó mean theo query để query nhiều interventions không tự có trọng số lớn hơn. Group joint chứa joint edge và các second-add edges suy từ measured singleton/joint states. Canonicalize unordered endpoint pairs, giữ một hướng duy nhất: đảo chiều cho cùng squared loss, không thêm evidence. Dedup cả cross-group edges bằng ownership cố định `member > add > joint > swap`; label-derived second-add edges thuộc joint nếu không có ở add. Các groups được weighting có chủ đích, không quảng cáo mỗi derived edge là thêm một quan sát độc lập. Log node degrees để phát hiện state bị overweight; thử degree-balancing riêng nếu imbalance lớn, không đổi weighting ngầm.

0.25 là trial, không tối ưu đã chứng minh. So với β=0 và direct set regression chỉ dùng L_set. Set anchor giúp fit mức utility chung nhưng không bắt buộc cho marginal-only inference và có thể làm gain fit tệ đi. MSE là scalar utility regression, không “PPO viết lại thành CE”.

Với residual e=F(T)−F(S)−d, gradient:

\[
\nabla_\theta\ell=2e\,[\nabla_\theta F(T)-\nabla_\theta F(S)].
\]

Gradient đi vào **u, v, ψ, role projections và toàn bộ UniXcoder cho query/candidates/members**. Generator và labels stop-gradient. Không old-policy ratio, critic, GAE, entropy bonus; không backprop qua generator.

Potential bảo đảm telescoping và reverse-removal sign về mặt cấu trúc; không cần cộng thêm consistency loss trùng lặp. Điều này không bảo đảm potential accurate/calibrated. Reuse cùng encoder features trong bundle, head không dropout ở bản đầu để tránh phá identity bằng các lần random head evaluation khác nhau.

## 7. Inference: chọn gì và bao nhiêu bằng cùng một thang utility

Encode query và pool một lần ở checkpoint cố định, cache profiles. Khởi đầu S=empty, objective:

\[
J(S)=aF(x,S)-\lambda C(S)/B.
\]

Xét các thay đổi khả thi:

1. Add một candidate bất kỳ trong pool.
2. Remove một member.
3. Add hai candidates trong shortlist 8 có cả predicted-useful và complementary/diverse proposals: tối đa 28 cặp.
4. Swap một member với một candidate trong shortlist: tối đa 80 thay thế khi K=10.

Chọn thay đổi có ΔJ lớn nhất nếu ΔJ>δ; nếu không thì STOP. Giá trị khởi đầu λ=0.01, δ=0 ở thang ES[0,1]; calibrate trên validation với một grid nhỏ khai báo trước, không trên test. Cost penalty xử lý context dư thừa; **độ hại về completion phải đến từ U thực đo**, không giả định dài là sai. λ=0 là ablation bắt buộc.

Strict improvement, visited-set guard và max 30 moves tránh loop; nếu hit move cap phải log `search_censored`, không gọi đó là learned STOP. Tất cả render/token constraints dùng chung scorer; không reclaim prefix budget sau khi STOP ở bản đầu.

Ví dụ giả lập: U(empty)=0.40, U(a)=0.65, U(a+b)=0.65, U(a+n)=0.50. Model lý tưởng chọn a; b redundant; n harmful. Nhưng U(p)=U(q)=0.40 và U(p+q)=0.75 thì greedy singleton có thể bỏ lỡ: joint-add labels/search là để xử lý trường hợp này. Đây là minh họa, không kết quả dataset.

Ở deployment **không có gold target hoặc oracle scoring để chọn tập**. Tất cả moves dùng learned head; chỉ gọi generator cho completion cuối. STOP chỉ chứng nhận theo predicted local neighborhood, không global optimum; pair shortlist có thể bỏ lỡ partner tốt và tương tác từ 3 snippets trở lên vẫn khó.

## 8. Training schedule và tối ưu chi phí

Tách hai phase có thể resume:

1. Frozen generator tạo/làm mới oracle bundles, cache completions/scores.
2. UniXcoder fit các bundles nhiều supervised passes; không gọi lại generator mỗi optimizer step.

Trial full run: **2 refresh rounds × tối đa 3 supervised passes/round**, early stop theo full validation và lưu best/last. Sau round 1, checkpoint hiện tại tạo thêm on-policy-like states để giảm distribution shift; đây là supervised dataset aggregation, không policy-gradient RL. Các ablation chính phải dùng cùng oracle bundles để so loss công bằng; adaptive-refresh experiment là bước riêng.

Với 3.335 train rows và ceiling 123: **410.205 scored contexts/round**, tối đa 820.410 cho hai rounds trước cache/invalid reductions; chưa gồm validation, diagnostics, reruns. Không còn hứa chi phí gần RLCoder: mặc định RLCoder khoảng 10 scored singleton contexts/query, batched teacher-forced scoring, không 10 sequential calls. Decode dài và set context lớn có thể làm plan này đắt hơn đáng kể. [Batching audit](research/retriever_training_alternatives_2026_10_01/diffs/2026-10-01_rlcoder_batching_correction.md).

Oracle annotations được reuse qua nhiều epochs, seeds và ablations ở checkpoint-independent bundles. Model-dependent refresh vẫn sinh context mới. Chỉ gọi lại những prompt chưa có trong cache; frozen generator không đồng nghĩa cache embeddings UniXcoder train được giữ mãi.

### Memory/backprop implementation

- Batch oracle theo token lengths và server capacity, starting request batch 108 nếu backend đã probe; request size không bảo đảm không OOM. Không giữ assumption vLLM tự chặn mọi OOM.
- Một query có thể cần query+64 unique chunk encoder sequences, **không còn bound 14 của recipe cũ**. Nhiều set-head evaluations nhỏ không cần encode lại từng set.
- Full fine-tune không được dùng detached stale feature cache như thể có encoder gradients. Bản đúng trước: query bundle batch nhỏ, mixed precision, bucket lengths, gradient accumulation với normalization theo query/group chính xác.
- Nếu activations của cả bundle không fit: dùng representation-gradient caching/replay (GradCache) với weights cố định đến khi replay xong, RNG replay nhất quán; hoặc chia thành edge subgraphs có measured endpoints rồi accumulate theo objective weights. Chia encoder microbatch rồi giữ tất cả graphs vẫn không giải quyết tổng activation memory.
- GradCache thêm encoder forward/replay, không tự nhanh hơn. Gradient checkpointing là runtime switch sau memory profile; không mặc định tắt. So gradient small-batch reference trước khi bật tối ưu này. [Gradient Cache](https://arxiv.org/abs/2101.06983).
- Starting optimizer trial: AdamW encoder LR 2e-5, new heads 1e-4, weight decay 0.01, warmup 5%, max grad norm 2. Đây là config mới đề xuất, không sửa ngầm LR/runtime hiện tại. Log preclip norm/skipped steps; chưa có bằng chứng tối ưu.
- Nếu chung một GPU: score phase và train phase có thể time-share để không giữ 80% VRAM vLLM trong lúc supervised training. Lưu manifest/cache trước chuyển phase. So wall-time cold load versus concurrent service trước khi quyết định; không triển khai trong plan này.
- Checkpoint atomic gồm encoder/heads/optimizer/scheduler/scaler/RNG, refresh id, bundle manifest/hash, pass+cursor, sampler order, normalization a, metric/prompt versions. Lỗi không skip tùy tiện: lưu trạng thái hợp lệ cuối, log failed context, dừng; resume không âm thầm đổi pool/metric/model.

## 9. Experiment plan: tách nguyên nhân trước khi claim contribution

### Stage 0 — correctness gate, chưa chạy full

- Fixtures Python/Java line/block: labels/splits bất biến, không right-context leakage, actual token accounting, deterministic render, metric parity.
- Synthetic helpful/noisy/redundant/complementary/swap examples: F(empty)=0, telescoping, target signs, stop/cap semantics.
- Autograd test thực tế: encoder query/candidate/member và mọi head nhận gradient; numeric check nhỏ cho gradient accumulation/replay nếu dùng. Overfit một tiny bundle để phân biệt bug với thiếu capacity.
- End-to-end một batch nhỏ: cache hit giữ raw output/score, checkpoint resume cùng next update ở tolerance phù hợp; fail-safe không mất pass/cursor.

### Stage 1 — chứng minh supervision, chưa đòi kiến trúc mới thắng

Subset train cố định 256 query, stratified language/target-kind; dùng validation split hiện có, không lấy test để chọn threshold. Đây là thí nghiệm bác bỏ rẻ trước full run, không thay production train split.

Trên cùng pool/prompt/oracle bundles, so:

1. Hard-winner CE với global features (controlled analogue, không gọi là full AlignCoder).
2. Signed singleton regression với cùng global features.
3. Conditional set-potential với global features, cùng utility labels.
4. Role-aware conditional set-potential.

CE/singleton chỉ dùng singleton measurements phù hợp; log các oracle measurements không dùng. Report hai chế độ: **same available annotations** và **matched total scoring budget** (baseline được dùng thêm independent training queries/refresh trong pool train đã định nghĩa). Không gọi việc cho method nhiều context hơn là architecture win.

Ở Stage 1 ưu tiên generalization của signed conditional gain, error gần 0 và trên actions do chính search chọn; không chọn mô hình chỉ vì train MSE thấp. Nếu singleton/general dense retrieval đã giải thích toàn bộ improvement thì thu hẹp claim.

### Stage 2 — full data và ablations tối thiểu

- Full existing train split, full existing valid, 3 seeds cho kết luận chính.
- Baselines: no retrieval; BM25 AST budget; frozen UniXcoder; reproduced RLCoder; reproduced AlignCoder (bao gồm các thành phần của official configuration); closest-prior RepoShapley nếu artifact/protocol tái lập được. Không thay một baseline bằng bản đã bỏ query enhancement rồi claim thắng paper gốc.
- Core ablations trên cùng contexts: singleton vs conditional; global vs roles (parameter-matched head); gain+anchor vs gain-only vs set-only.
- Kiến trúc interaction: additive-only potential versus nonlinear set potential, giữ conditional labels, encoder và search giống nhau. Representation role pooling và interaction head là hai yếu tố khác nhau, không gộp thành một lời giải thích.
- Selection ablations cùng trained scorer: fixed K versus learned stop, single-add greedy versus add/remove/joint/swap; λ=0 versus calibrated token cost. Không gộp lợi ích search với lợi ích learned representation.
- Shortcut control: count/token-only utility head và fixed-K được tune cùng validation budget. Trên held-out content swaps giữ cardinality và token cost gần bằng nhau, model phải phân biệt nội dung hữu ích/gây hại tốt hơn đối chứng này mới claim học chọn snippet.
- Label coverage ablation: bỏ joint/removal/swap supervision từng nhóm, nhưng giữ nguyên available measured nodes khi có thể; nói rõ phần thông tin còn suy ra được từ set targets. Không claim xóa edge loss đồng nghĩa xóa mọi thông tin về intervention đó.
- Kiểm tra learning curve theo số unique contexts/query (ví dụ 32/64/123 từ bundles có node coverage phù hợp), held-out probes và unseen joint additions. 123 là allocation thử, không bằng chứng coverage đã đủ; query/repo validation phải tách train, không chỉ hold out edges trong cùng query rồi gọi là generalization giữa repos.

### Stage 3 — test một lần sau khi khóa thiết kế

CCEval Python+Java là primary; RepoEval line/API là secondary transfer. Dùng prepared test Parquet, chỉ đọc chunks và labels để chấm cuối; không sửa labels hay tune trên test. Report official-compatible EM/ES và từng benchmark; primary tổng hợp là macro ES của hai CCEval languages.

Gate đề xuất, **phải khóa trước full confirmatory run**:

- So strongest matched baseline: macro ES tăng ít nhất 1 percentage point, paired repo-cluster bootstrap 95% CI của delta >0.
- EM guardrail: lower bound của CI delta EM >−0.5 percentage point; nếu thiếu power thì kết luận chưa rõ, không tự gọi non-inferior.
- Report quality–token frontier, không chỉ một λ. Đo noise acceptance, redundant picks, number selected, actual tokens, local missed-benefit khi STOP, oracle GPU-hours và train/inference time.
- Independent validation audit: score all feasible single additions và pair shortlist tại các STOP states đã chọn trước; nếu >10% states còn một measured ES gain >0.01 thì chưa được claim learned sufficient context. Với token-cost objective report thêm missed positive net-gain. Đây là local coverage gate, không chứng minh global sufficiency.
- Audit pool coverage bằng expanded pool 128 trên subset validation riêng; nếu lỗi chính là first-stage miss thì ưu tiên mining thay vì thêm loss. Report unresolved candidate coverage thay vì chỉ báo 100% samples có candidates.

Thresholds trên là trial acceptance criteria, không paper facts. Các số đo actions trong cùng query/repo không độc lập; bootstrap ở repo, không ở từng intervention. Report per-language và line/block slices để macro không che regression lớn.

## 10. Thứ tự triển khai và deliverables

1. **Contract/metrics:** `context_contract.py`, `utility_metrics.py`, manifest + fixture tests. Chốt metric/prompt parity trước oracle budget lớn.
2. **Oracle builder:** `build_utility_bundles.py`, resumeable completion cache; `utility_nodes.parquet`, `utility_edges.parquet`, `bundle_manifest.json`. Đây là sidecars tự động, train.parquet gốc giữ nguyên.
3. **Retriever:** `conditional_utility_model.py`, full encoder + optional role branch + potential; reference gradient/identity tests.
4. **Trainer:** `train_conditional_utility.py`, normal supervised batches, query-balanced losses, finite checks, checkpoints/fail-stop. Stage 0 reference trước optimized replay.
5. **Selector/eval:** `select_conditional_context.py`, search trace, actual budgets, head-only moves; `eval_conditional_utility.py` chung metric với scorer.
6. **Packaging:** sau Stage 0 mới đóng gói notebook/Modal entry point. Không thêm một notebook lớn chưa kiểm được từng phần.

Các tên file này là **deliverables dự kiến**, chưa phải file đã được tạo hay APIs đã tồn tại. Không rewrite notebook PPO/RRPO trong bước lập plan.

## 11. Điều kiện đổi hướng

- ES labels phần lớn hòa hoặc quá bất ổn: kiểm tra task/metric, sau đó mới thử expected ES nhiều decodes hoặc likelihood ablation; không thêm loss phụ tùy ý.
- Potential không generalize tốt hơn independent conditional head/set regression: bỏ ràng buộc kiến trúc, giữ bài học về data coverage; coherent không đồng nghĩa expressive hơn.
- Head bị search khai thác ở OOD sets: bổ sung measured deployment states và validation calibration; chưa claim đủ context chỉ vì predicted gains âm.
- Chỉ giảm K mà EM/ES giảm: cost penalty đang ép sparse, chưa học chọn tốt.
- Pool/query/prompt fixes giải thích improvement hoặc closest prior đã có cơ chế tương đương: thu hẹp contribution, không đổi tên để che overlap.

**Đích cần chứng minh:** với cùng generator, dữ liệu và điều kiện retrieval, CUR chọn context có utility tốt hơn và bỏ context không đáng dùng, thay vì chỉ học xếp hạng một snippet thắng. Hiệu quả, novelty và thời gian vẫn cần đo; nguồn hiện có xác nhận prior/gap hẹp, không xác nhận đề xuất đã thành công.
