# DeepRead report: REPOSHAPLEY

## 0. Nguồn và trạng thái trích xuất

- Paper: REPOSHAPLEY: Shapley-Enhanced Context Filtering for Repository-Level Code Completion
- Tác giả: Yu Huo và cộng sự.
- Venue: Findings of ACL 2026, trang 10390–10412.
- File nguồn: /Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/2026.findings-acl.505.pdf
- Phạm vi đã đọc: toàn bộ 23 trang, gồm main paper và appendix.
- Chất lượng nguồn: PDF có text layer; đã kiểm tra nội dung theo từng nhóm trang, không cần OCR.
- Mục đích của report: lưu lại phương pháp, objective/loss, training và inference pipeline, bằng chứng thực nghiệm, giới hạn, và các hàm ý trực tiếp cho hướng repository-level code completion đang phát triển.

Nhãn bằng chứng trong report:

- Author's stated position: tuyên bố hoặc diễn giải trực tiếp của tác giả.
- Source fact or data: công thức, thiết lập, số liệu hoặc mô tả có trong paper.
- Reasoned inference: suy luận để phục vụ thiết kế nghiên cứu tiếp theo.
- Unverified: giả thuyết chưa được paper này kiểm chứng.

## 1. Tóm tắt điều hành

REPOSHAPLEY giải quyết một vấn đề mà các bộ lọc context độc lập thường bỏ sót: giá trị của một chunk không cố định mà phụ thuộc vào các chunk khác trong context. Hai loại interaction chính là:

1. Complementarity: một chunk định nghĩa interface và một chunk khác chứa implementation; giữ riêng lẻ thì không đủ, giữ cùng nhau thì hữu ích.
2. Conflict: các chunk chứa API cũ và API mới, hoặc hai convention mâu thuẫn; thêm cả hai có thể làm generator tệ hơn.

Thay vì chấm từng chunk độc lập rồi chọn top chunks, REPOSHAPLEY:

1. Dùng một bộ chấm offline tên ChunkShapley để ước lượng utility của các coalition.
2. Khởi tạo interaction bằng ảnh hưởng single-chunk.
3. Dùng một surrogate logistic có positive/negative votes và saturation để mô hình hóa complementarity/conflict.
4. Tính exact Shapley value trên surrogate cho pool nhỏ.
5. Chọn một tập candidate subsets từ Shapley prefixes, Delta prefixes và một số combinations nhỏ.
6. Dùng frozen generator để post-verify các candidate subset bằng decoding metrics.
7. Huấn luyện một generator/controller duy nhất để:
   - quyết định có cần retrieval không;
   - dự đoán KEEP/DROP cho từng candidate;
   - sinh completion với packed context đã lọc.

Kết quả chính của paper cho thấy post-verification là thành phần quan trọng nhất và Shapley proposal tốt hơn Delta-only proposal. Tuy nhiên, REPOSHAPLEY không phải một retriever thuần túy: nó thay đổi cả cách tạo label offline, controller, generator input format và số pass autoregressive lúc inference. Vì vậy, kết quả của nó không thể được quy trực tiếp thành bằng chứng rằng framework này sẽ vượt AlignCoder hoặc một framework AST/KD khác.

Hàm ý chính cho nghiên cứu hiện tại:

- Đây là prior art trực tiếp cho vấn đề AST chunk interaction.
- Nếu chỉ dùng KD độc lập trên từng chunk, ta vẫn có nguy cơ bỏ sót complementarity/conflict.
- Nhưng full Shapley có chi phí offline cao và không loại bỏ hoàn toàn nhu cầu verify.
- Chiến lược thực tế là giữ framework KD làm baseline chính, sau đó thêm một nhánh coalition-aware proposal và bounded verification để kiểm tra xem interaction có tạo gain thực sự trên cùng pool AST hay không.

## 2. Paper đang giải quyết bài toán nào?

### 2.1. Repository-level FIM

Một instance có dạng:

- X_in = (X_p, X_s): prefix và suffix trong file đang cần completion.
- X_out: pool các cross-file chunks lấy từ repository.
- Y: completion target.

Retriever ban đầu trả về top-K chunks:

X_cc = {cc_1, cc_2, ..., cc_K}

Một subset được chọn là X_S với S là subset của D = {1, ..., K}. Mục tiêu của context filtering là tìm subset S hữu ích nhất cho generation, thay vì luôn đưa toàn bộ top-K chunks vào generator.

### 2.2. Failure mode của scoring độc lập

Nếu đánh giá mỗi chunk bằng một score độc lập Delta_i, score đó không biểu diễn được:

- chunk i chỉ hữu ích khi chunk j cũng xuất hiện;
- chunk i gây hại chỉ khi chunk j cùng xuất hiện;
- một nhóm chunk có tổng utility bị bão hòa, nên thêm chunk mới không còn lợi;
- các chunk có thông tin trùng lặp hoặc có API conflict.

Đây là lý do tác giả cho rằng context selection là một cooperative game, trong đó mỗi chunk là một player và subset các chunk là coalition.

### 2.3. Điểm cần lưu ý về candidate pool

Paper tập trung vào filtering sau retrieval, không học một retriever từ đầu. Candidate pool mặc định gồm top-10 chunks từ Jaccard retrieval; một thí nghiệm robustness dùng dense retrieval với UniXcoder. Do đó chất lượng và bias của candidate pool vẫn quyết định giới hạn của toàn bộ phương pháp.

## 3. Utility function và Shapley labeling

### 3.1. Utility dựa trên teacher-forced log-likelihood

Paper định nghĩa normalized target log-likelihood:

ell(C) = (1 / |Y|) sum_t log p_theta(y_t | y_<t, C)

Trong đó C là context được đưa cho generator và Y là target completion.

Coalition value:

v(S | X_in, Y) = ell(X_in, X_S) - ell(X_in)

Ý nghĩa:

- v(S) > 0: thêm subset X_S làm target dễ sinh hơn theo teacher-forced likelihood.
- v(S) < 0: subset làm giảm likelihood.
- v(empty) = 0.

Đây là utility để tạo nhãn offline, không phải reward do model sampling online.

### 3.2. Single-chunk effect

Với từng chunk i, paper tính:

Delta_i = ell({i}) - ell(empty)

Sau đó lưu:

- y_i = sign(Delta_i): chunk có vote positive hoặc negative.
- omega_i = |Delta_i|: độ mạnh của vote.

Delta_i được dùng như probe ban đầu, không được xem là utility cuối cùng của chunk trong mọi coalition.

### 3.3. Shapley value

Shapley value của chunk i:

phi_i = sum over S subset D excluding i of
        |S|! (K-|S|-1)! / K!
        multiplied by [v(S union {i}) - v(S)]

Các tính chất chính:

- phi_i là marginal contribution trung bình của chunk i trên mọi ordering.
- tổng phi_i bằng v(D), gọi là efficiency.
- score này phản ánh đóng góp ở cấp coalition tốt hơn Delta_i nếu value function được ước lượng tốt.

Tuy nhiên, exact Shapley trên value function thật cần đánh giá nhiều subset. Vì vậy paper không chạy generator trên toàn bộ 2^K subsets ở quy mô lớn mà dùng surrogate rồi chỉ verify một pool có giới hạn.

## 4. ChunkShapley: surrogate và bounded verification

### 4.1. Logistic surrogate

Paper xây dựng surrogate:

g(S) = sum over i in S of omega_i multiplied by y_i

v_sur(S) = sigmoid(beta multiplied by g(S)) - sigmoid(0)

Trong đó:

- y_i biểu diễn hướng tốt/xấu của chunk.
- omega_i biểu diễn độ mạnh của single-chunk signal.
- beta điều khiển độ dốc của surrogate.
- sigmoid tạo saturation và diminishing returns.
- negative y_i cho phép mô hình hóa chunk conflict.
- exact Shapley được tính trên v_sur vì K nhỏ.

Điểm quan trọng: surrogate này không học trực tiếp mọi higher-order interaction. Nó chỉ tái cấu trúc single-chunk evidence bằng một hàm phi tuyến có saturation và negative votes. Các interaction phức tạp hơn được xử lý một phần qua candidate verification.

### 4.2. Candidate proposal

Candidate pool C bao gồm:

1. Prefixes theo thứ tự Shapley, với số lượng N_v.
2. Prefixes ngắn theo thứ tự Delta.
3. Combinations size 2 và size 3 từ top-L chunks theo Delta.

Các candidate này phục vụ hai mục tiêu:

- Shapley prefixes kiểm tra các coalition có ranking interaction-aware.
- Delta prefixes giữ một proposal đơn giản và rẻ.
- size-2/size-3 combinations tạo cơ hội bắt các complementarity nhỏ mà prefix không thể biểu diễn.

### 4.3. Post-verification

Với mỗi S trong C, frozen generator decode một completion Y_hat_S.

Paper chọn:

S* = argmax over S in C of
      (ES(Y_hat_S, Y), EM(Y_hat_S, Y))

Theo thứ tự lexicographic:

1. tối đa Exact Match hoặc metric ES được paper sử dụng trong RepoEval;
2. nếu hòa thì dùng metric còn lại.

Trong bảng verification-scope, paper báo cáo rõ rằng tăng L làm tăng chi phí đáng kể nhưng vẫn có gain nhỏ sau L=3. Cấu hình mặc định dùng L=3, N_v=10.

Post-verification là điểm rất quan trọng:

- Shapley chỉ đề xuất candidate.
- Label cuối cùng vẫn phụ thuộc vào output thực tế của frozen generator.
- Đây là bước giải quyết sự lệch giữa utility surrogate và generation quality.
- Nhưng nó làm pipeline offline đắt hơn nhiều so với scoring độc lập.

### 4.4. Trigger và token supervision

Từ subset đã verify S*, paper tính:

Delta_ES = ES(Y_hat_S*, Y) - ES(Y_hat_empty, Y)

- Nếu Delta_ES <= epsilon, target retrieval token là DONE.
- Nếu Delta_ES > epsilon, target retrieval token là NEED.

Với trường hợp NEED, mỗi chunk nhận target selection token:

- KEEP nếu chunk thuộc S*.
- DROP nếu chunk không thuộc S*.

Như vậy, supervision có ba phần:

1. retrieval decision: NEED/DONE;
2. chunk selection: KEEP/DROP;
3. final code generation: target Y.

## 5. Input format, training và inference

### 5.1. Hai format được dùng trong training

#### Format 1: selection

Cấu trúc khái quát:

PFX [X_p] SFX [X_s] NEED
C_1 [cc_1] /C_1 ... C_K [cc_K] /C_K
SELECT q_1 ... q_K DONE

Generator phải đọc candidate chunks và sinh sequence KEEP/DROP.

#### Format 2: generation

Cấu trúc khái quát:

PFX [X_p] SFX [X_s] NEED
Pack(C_S*) DONE MID [Y]

Generator nhận subset đã lọc và học sinh completion.

#### No-retrieval format

Khi subset rỗng hoặc retrieval không cải thiện:

PFX [X_p] SFX [X_s] DONE MID [Y]

### 5.2. Losses

Paper mask loss trên context tokens. Các token control, selection và target completion mới nhận loss.

Retrieval loss:

L_R = - log P_G(r* | X_in)

Selection loss:

L_S = - sum_i log P_G(q_i* | X_in, X_cc, r*, q_<i*)

Generation loss:

L_Y = - sum_t log P_G(y_t | y_<t, X_in, X_S*, r*)

Format 1:

L_F1 = lambda_R L_R + lambda_S L_S

Format 2:

L_F2 = lambda_R L_R + L_Y

Training objective cuối là expectation qua các format được sampling. Ở inference, retrieval token là model prediction, không phải oracle input.

### 5.3. Inference pipeline

1. Generator dự đoán NEED hoặc DONE.
2. Nếu DONE, generator thực hiện FIM completion không retrieval.
3. Nếu NEED, retriever lấy K candidate chunks.
4. Generator dự đoán K token KEEP/DROP.
5. Các chunk KEEP được pack vào context.
6. Generator chạy lần cuối để sinh completion.

Đây là một model nhưng có nhiều autoregressive passes. Chi phí inference vì thế phụ thuộc cả số lượng selection token và final generation.

### 5.4. Phân biệt offline và online

- ChunkShapley và post-verification là offline label construction.
- Generator/controller được train sau khi offline labels đã tồn tại.
- Khi inference, model mới tự quyết định NEED/DONE và KEEP/DROP.
- Paper không phải online RL và không dùng gradient update sau mỗi retrieval request.
- Paper cũng không phải KD giữa teacher distribution và student distribution; supervision chính là discrete control labels, selected context labels và generation target.

## 6. Dữ liệu và experimental setup

Các thiết lập được paper báo cáo:

- 290k Python repositories từ The Stack sau filtering.
- 7.5k repositories và 50k labeled instances.
- Repository-disjoint split 95/5 trước khi tạo instances.
- Candidate retrieval mặc định: top-10 chunks bằng Jaccard trên các symbol/token features của repository.
- Robustness retrieval: dense retrieval bằng UniXcoder.
- 50% context-only query và 50% oracle-assisted query trong offline candidate construction.
- Oracle-assisted query dùng target Y để hỗ trợ candidate retrieval lúc tạo dữ liệu; paper nói Y không được đưa vào model input hay query ở inference.
- K = 10.
- Chunk window = 512 tokens.
- Stride = 256 tokens.
- N_v = 10.
- L = 3.
- beta = 1.0.
- tau_es = 50.
- epsilon = 0.
- Trigger threshold t_c = 0.5.
- Backbone: StarCoderBase 1B/3B/7B và CodeLlama 7B/13B.
- Training: 2 epochs, learning rate 2e-5, 5% warmup, max length 4096, lambda_R = lambda_S = 2.0, global batch 512, 8 H100 80GB.

Cần coi thông tin oracle-assisted là một biến experimental quan trọng. Candidate pool được tạo offline với một phần query có target signal, dù target không đi vào inference. Điều này có thể làm candidate distribution thuận lợi hơn deployment thực tế.

## 7. Kết quả thực nghiệm chính

### 7.1. RepoEval trên StarCoderBase 1B

Line completion:

- Full retrieval: EM 52.27, ES 73.13.
- CODEFILTER: EM 57.19, ES 78.84.
- REPOSHAPLEY: EM 61.34, ES 82.78.

API completion:

- Full retrieval: EM 44.18, ES 69.09.
- CODEFILTER: EM 48.37, ES 75.66.
- REPOSHAPLEY: EM 53.62, ES 79.53.

### 7.2. RepoEval trên StarCoderBase 7B

Line completion:

- Full retrieval: EM 58.26, ES 77.79.
- CODEFILTER: EM 61.49, ES 81.41.
- REPOSHAPLEY: EM 65.81, ES 86.59.

API completion:

- Full retrieval: EM 50.38, ES 75.01.
- CODEFILTER: EM 53.62, ES 79.29.
- REPOSHAPLEY: EM 58.79, ES 84.11.

### 7.3. CodeLlama 13B

- CODEFILTER line: EM 64.16, ES 82.71.
- REPOSHAPLEY line: EM 68.89, ES 87.11.
- CODEFILTER API: EM 53.01, ES 78.99.
- REPOSHAPLEY API: EM 57.66, ES 83.41.

### 7.4. Left-to-right completion

Trên StarCoderBase 7B:

- Full retrieval: EM 51.22, ES 71.55.
- CODEFILTER: EM 53.95, ES 74.22.
- REPOSHAPLEY: EM 59.88, ES 80.55.

Điều này cho thấy tác giả không chỉ đánh giá FIM. Tuy nhiên, đây vẫn là cùng family benchmark và không đủ để suy ra superiority trên benchmark AlignCoder nếu protocol khác nhau.

### 7.5. CrossCodeEval đa ngôn ngữ

Trên StarCoderBase 7B, ES/F1:

- Python: 62.91 / 53.30.
- Java: 66.20 / 58.29.
- C#: 71.83 / 64.23.
- TypeScript: 60.23 / 51.63.

Trên CodeLlama 13B, ES/F1:

- Python: 59.37 / 51.39.
- Java: 67.11 / 57.17.
- C#: 70.95 / 63.20.
- TypeScript: 56.98 / 50.80.

## 8. Ablation và điều học được

### 8.1. Post-verification là thành phần cốt lõi

Với StarCoderBase 1B:

- Full REPOSHAPLEY line: EM 61.34, ES 82.78.
- Không post-verification line: EM 38.50, ES 54.44.
- Full REPOSHAPLEY API: EM 53.62, ES 79.53.
- Không post-verification API: EM 36.15, ES 55.81.

Tác động rất lớn. Kết luận thực tế là surrogate/Shapley một mình chưa đủ đáng tin; quality của label phụ thuộc vào generator-based verification.

### 8.2. Shapley proposal tốt hơn Delta-only

Line:

- Delta-only: EM 58.45, ES 77.12.
- REPOSHAPLEY: EM 61.34, ES 82.78.

API:

- Delta-only: EM 48.46, ES 75.26.
- REPOSHAPLEY: EM 53.62, ES 79.53.

Shapley có gain, nhưng paper không tách hoàn toàn gain do logistic surrogate khỏi gain do candidate combinations và post-verification. Vì vậy không nên kết luận rằng exact Shapley riêng lẻ là nguồn gain duy nhất.

### 8.3. Surrogate logistic và weights

Line:

- Linear surrogate: EM 59.92, ES 76.41.
- Uniform weights: EM 60.18, ES 80.97.
- Full logistic weighted surrogate: EM 61.34, ES 82.78.

Điều này ủng hộ hai yếu tố:

- weighted evidence từ Delta_i hữu ích;
- saturation/nonlinearity tốt hơn additive linear scoring.

### 8.4. Format selection và generation

- Format 1 only: line EM 5.56, ES 8.27; API EM 2.34, ES 5.66.
- Format 2 only: line EM 59.88, ES 79.11; API EM 52.12, ES 77.49.
- Full multi-format: line EM 61.34, ES 82.78; API EM 53.62, ES 79.53.

Format 1 only bị suy giảm nghiêm trọng trong thiết lập ablation của paper. Format 2 chứa generation signal là phần quan trọng; selection control cần được gắn với generation training chứ không nên xem như nhiệm vụ độc lập.

### 8.5. Retrieval trigger

- Có trigger: line EM 61.34, ES 82.78; API EM 53.62, ES 79.53.
- Không trigger: line EM 61.26, ES 81.33; API EM 52.15, ES 78.81.

Trigger có gain nhỏ hơn post-verification, nhưng giúp model học khi nào retrieval không đáng dùng.

### 8.6. Verification budget

Theo bảng scope L:

- L=0: EM 8.41, thời gian khoảng 33 giây, ES 70.86.
- L=1: EM 10.78, khoảng 94 giây, ES 76.27.
- L=2: EM 12.56, khoảng 187 giây, ES 78.40.
- L=3: EM 18.29, khoảng 348 giây, ES 82.78.
- L=4: EM 25.07, khoảng 671 giây, ES 83.22.
- L=7: EM 68.94, khoảng 1528 giây, ES 83.59.

ES gần bão hòa sau L=3/L=4, trong khi offline cost tăng mạnh. Đây là lý do L=3 là trade-off thực dụng, không phải optimum tuyệt đối.

### 8.7. K sensitivity

Bảng sensitivity cho thấy:

- K=7: EM 52.16, ES 70.44, latency khoảng 513 ms.
- K=10: EM 61.34, ES 82.78, latency khoảng 1053 ms.
- K=13: EM 61.88, ES 81.62, latency khoảng 3539 ms.
- K=20: EM 61.76, ES 79.92, latency khoảng 18825 ms.

K lớn hơn không luôn tốt hơn. K=13/20 làm chi phí tăng mạnh, còn ES giảm do context noise, model difficulty và selection burden. Exact surrogate có chi phí exponential theo K nên K nhỏ là yêu cầu của thiết kế.

### 8.8. Utility choice

Paper thử các utility:

- v_EM: EM 58.12, ES 77.45.
- v_ES: EM 59.03, ES 78.10.
- v_Log: EM 60.50, ES 79.07.

Log-likelihood utility tốt nhất trong so sánh này. Đây là bằng chứng ủng hộ dùng teacher-forced likelihood để tạo proposal, nhưng vẫn cần verify bằng decoding metric vì log-likelihood và EM/ES không đồng nhất.

### 8.9. Proposal quality và candidate-pool ceiling

Paper báo cáo:

- Delta + Verify: ES 74.21.
- Shapley + Verify: ES 82.78.
- Oracle best trên candidate pool:
  - Full: ES 71.52.
  - CODEFILTER: ES 85.23.
  - REPOSHAPLEY: ES 95.68.

Giá trị oracle này phải được diễn giải cẩn thận: đó là best subset trong candidate pool đã được sinh ra, không phải global optimum trên mọi subset và không phải upper bound tuyệt đối cho repository. Nó cho thấy vẫn còn headroom ở candidate proposal/verification, nhưng không chứng minh model có thể đạt mức oracle.

## 9. Failure modes và limitations

### 9.1. Higher-order interaction chưa được mô hình hóa đầy đủ

Surrogate khởi đầu từ single-chunk effects. Nó có thể bỏ sót:

- interaction bậc cao hơn;
- non-monotone coalition;
- trường hợp một chunk xấu riêng lẻ nhưng tốt trong một nhóm cụ thể;
- trường hợp utility phụ thuộc thứ tự hoặc format của context.

### 9.2. Exact Shapley vẫn exponential

Surrogate evaluation sử dụng 2^K subsets. K mặc định chỉ 10 để giữ chi phí thực tế. Nếu AST chunking tạo nhiều evidence units hơn, phải có:

- hierarchical grouping;
- sampled Shapley;
- influence approximation;
- candidate pruning;
- hoặc chỉ verify các prefix/combinations giới hạn.

### 9.3. Bounded verification có thể bỏ lỡ subset tối ưu

S* chỉ tối ưu trong candidate set C. Nếu complementarity yêu cầu một combination không thuộc prefix hoặc size-2/size-3 proposals, label vẫn sai.

### 9.4. Label phụ thuộc greedy decoding

Post-verification lấy completion do frozen generator decode và dùng EM/ES. Điều này làm label phụ thuộc:

- backbone cụ thể;
- decoding strategy;
- context packing;
- tokenizer;
- benchmark metric.

Đổi generator hoặc decoding có thể làm thứ tự subset thay đổi.

### 9.5. Chunk granularity

Main setup dùng fixed token windows 512/256 stride. Cách này vẫn có thể cắt giữa:

- function/class;
- decorator và function;
- call site và definition;
- type/interface và implementation;
- các node AST liên quan.

Do đó REPOSHAPLEY giải quyết interaction ở cấp chunk nhưng bản thân chunk boundary vẫn có thể gây mất thông tin. Đây là điểm mà AST chunking hiện tại có thể cải thiện trực tiếp.

### 9.6. Offline cost

Cấu hình L=3 mất khoảng 348 giây mỗi sample trong bảng của paper; L tăng làm cost lên hàng trăm hoặc hơn một nghìn giây. Đây là rào cản lớn cho dữ liệu nhiều triệu instances.

### 9.7. Oracle-assisted candidate construction

Một nửa query offline dùng target Y để hỗ trợ candidate retrieval. Paper khẳng định Y không vào model input/inference, nhưng candidate pool offline có thể vẫn chịu target-aware selection bias. Cần đánh giá thêm hoàn toàn context-only để biết độ bền trong deployment.

### 9.8. Controller/generator coupling

Gain không chỉ đến từ filtering. Nó đến từ cùng lúc:

- candidate pool;
- offline labels;
- NEED/DONE;
- KEEP/DROP;
- packed input format;
- generation training.

Vì vậy paper chưa cô lập hoàn toàn đóng góp của context selection khỏi đóng góp của multi-format training.

### 9.9. Failure cases theo paper

Tác giả báo cáo khoảng 8% trường hợp trên RepoEval mà REPOSHAPLEY kém hơn full retrieval. Các nguyên nhân được nêu:

- candidate pool chất lượng thấp;
- non-monotone interaction;
- implicit repository conventions không có trong các chunk được retrieve.

## 10. Evidence ledger

### Claim A: utility của chunk có thể phụ thuộc vào coalition

- Nhãn: Author's stated position.
- Bằng chứng: paper phân tích complementarity và conflict, rồi mô hình hóa context filtering như cooperative game.
- Giới hạn: đây là framing của paper; cần kiểm chứng trên benchmark cụ thể của ta.

### Claim B: utility v(S) dùng normalized teacher-forced log-likelihood

- Nhãn: Source fact or data.
- Bằng chứng: công thức ell(C) và v(S) trong phần method.
- Ý nghĩa: có thể dùng cùng signal này để ranking/soft supervision, nhưng không đồng nghĩa với generation metric.

### Claim C: full pipeline gồm surrogate Shapley và bounded post-verification

- Nhãn: Source fact or data.
- Bằng chứng: ChunkShapley tạo candidate subsets rồi frozen generator verify các candidates.
- Ý nghĩa: Shapley không thay thế verification.

### Claim D: post-verification là thành phần tạo gain lớn nhất

- Nhãn: Source fact or data.
- Bằng chứng: ablation không post-verification giảm line ES từ 82.78 xuống 54.44 và API ES từ 79.53 xuống 55.81 trên StarCoderBase 1B.
- Giới hạn: mức giảm lớn có thể phụ thuộc ablation protocol và label format cụ thể.

### Claim E: Shapley proposal tốt hơn Delta-only proposal

- Nhãn: Source fact or data.
- Bằng chứng: line ES 82.78 so với 77.12; API ES 79.53 so với 75.26.
- Giới hạn: Shapley proposal đi kèm surrogate và combination candidates; chưa cô lập exact contribution của từng phần.

### Claim F: log-likelihood utility tốt hơn EM/ES utility trong utility ablation

- Nhãn: Source fact or data.
- Bằng chứng: v_Log đạt EM/ES 60.50/79.07, cao hơn các utility khác trong bảng ablation.
- Giới hạn: kết quả chỉ trong setup của paper.

### Claim G: REPOSHAPLEY đạt gain so với full retrieval và CODEFILTER

- Nhãn: Source fact or data.
- Bằng chứng: nhiều bảng RepoEval trên StarCoderBase và CodeLlama.
- Giới hạn: không phải so sánh trực tiếp với AlignCoder nếu backbone, retrieval, split hoặc protocol khác.

### Claim H: oracle best cho thấy candidate pool còn headroom

- Nhãn: Reasoned inference.
- Bằng chứng: REPOSHAPLEY oracle ES 95.68, trong khi learned result thấp hơn nhiều.
- Giới hạn: oracle chỉ trong candidate pool; không phải global upper bound.

### Claim I: oracle-assisted query có thể tạo distribution shift khi chuyển sang deployment

- Nhãn: Reasoned inference.
- Bằng chứng: một phần offline candidate construction được hỗ trợ bởi target Y, dù Y không có trong inference.
- Cần kiểm chứng: train/eval hoàn toàn context-only và đo candidate recall.

### Claim J: framework này đã thay thế nhu cầu KD từ teacher embedding sang student encoder

- Nhãn: Unverified.
- Paper không thực hiện Jina-to-UniXcoder KD và không so sánh với AST-aware embedding distillation. Không được suy ra từ kết quả REPOSHAPLEY.

### Claim K: khoảng 8% RepoEval cases bị kém hơn full retrieval

- Nhãn: Author's stated position.
- Bằng chứng: phần failure analysis/limitations.
- Ý nghĩa: filtering không nên được giả định luôn an toàn; trigger và abstention cần được đánh giá.

## 11. Quan hệ với hướng AST chunking và KD hiện tại

### 11.1. AST chunking giải quyết một điểm yếu khác

REPOSHAPLEY dùng fixed windows, nên interaction-aware scoring vẫn có thể hoạt động trên các chunk bị cắt sai. AST chunking có thể:

- giữ nguyên function/class/interface/implementation unit;
- giữ quan hệ parent-child;
- tránh cắt giữa declaration và body;
- tạo evidence units có semantic boundary rõ hơn;
- cho phép expand từ node nhỏ sang ancestors/related nodes khi cần.

Điều này không tự động giải quyết selection interaction, nhưng làm player trong cooperative game có ý nghĩa hơn.

### 11.2. Independent KD hiện tại không bắt được coalition

Nếu student UniXcoder học một distribution:

p_T(i) proportional to exp(-ell_i / tau)

trên từng AST chunk i, student chỉ học marginal preference. Nó không trực tiếp học:

- p(i, j) khi pair i/j bổ trợ nhau;
- conflict giữa i và j;
- utility của một nhóm nodes;
- context budget allocation cho coalition.

Vì vậy, kết quả tốt của single-chunk KD không loại trừ việc coalition-aware teacher có thể tạo thêm gain.

### 11.3. Không nên chuyển ngay toàn bộ framework sang full Shapley

Lý do:

- offline verification rất đắt;
- surrogate bắt đầu từ single-chunk signal;
- exact Shapley cần 2^K;
- bounded candidate set có thể bỏ lỡ higher-order combination;
- paper chưa đánh giá AST chunks hoặc Jina-to-UniXcoder KD;
- chưa chứng minh gain trên protocol của AlignCoder.

Thiết kế kiểm chứng hợp lý trên cùng data split và cùng generator:

1. Independent AST-KD: baseline hiện tại.
2. AST-KD + deterministic top-1/top-2/top-3 packing verification.
3. AST-KD + coalition proposal kiểu surrogate Shapley + bounded verification.
4. Chỉ bật full candidate set nếu bước 2 cho thấy interaction thật sự là bottleneck.

Đây là recommendation nghiên cứu, không phải kết quả đã được paper chứng minh.

## 12. Protocol đề xuất để kiểm chứng công bằng

Giữ cố định:

- target file/span;
- repo-disjoint split;
- generator backbone;
- tokenizer;
- candidate retriever;
- AST chunking;
- maximum context budget;
- decoding;
- EM/ES evaluation.

So sánh tối thiểu:

### Control 1: independent scoring

- Mỗi AST unit được encode/scoring độc lập.
- Student nhận KD target cho từng unit.
- Pack top units trong cùng budget.
- Không có coalition verification.

### Control 2: small-coalition verification

- Dùng independent score để đề xuất top-1, top-2, top-3 units.
- Generator teacher-forced likelihood hoặc greedy decode chấm từng candidate coalition.
- Chọn coalition tốt nhất trong budget.
- Đo gain và offline cost.

### Control 3: surrogate coalition proposal

- Tính Delta_i từ generator log-likelihood.
- Tạo signed weighted logistic surrogate.
- Tính approximate/exact Shapley cho K nhỏ.
- Đề xuất Shapley prefixes và combinations nhỏ.
- Verify candidates bằng cùng generator và cùng metric.
- So với Control 2 để biết Shapley có đáng chi phí hay không.

Các metric cần log:

- RepoEval EM và ES.
- candidate recall của oracle best coalition.
- average number of packed AST units.
- context tokens và budget utilization.
- offline labeling time per instance.
- inference latency và số model passes.
- tỷ lệ filtering làm giảm score so với full retrieval.
- calibration của abstention/NEED-DONE.
- gain theo interaction type: complementarity, conflict, redundancy.

## 13. Kết luận đọc paper

### Kết luận có độ tin cậy cao

1. REPOSHAPLEY là một framework context filtering theo coalition cho repository-level code completion.
2. Nó dùng teacher-forced log-likelihood để tạo utility, logistic surrogate để proposal, Shapley để ranking, và frozen generator verification để chọn subset.
3. Post-verification là thành phần thiết yếu trong ablation.
4. Shapley-based proposal tốt hơn Delta-only trong các bảng chính.
5. Chi phí offline tăng nhanh theo verification budget và K.
6. Main fixed-window chunking vẫn để lại khoảng trống mà AST chunking có thể cải thiện.

### Kết luận cần giữ ở mức giả thuyết

1. REPOSHAPLEY có thể cải thiện framework AST/KD hiện tại nếu interaction giữa AST units là bottleneck.
2. Jina embedding distillation có thể bổ sung semantic retrieval signal, nhưng paper này không cung cấp bằng chứng cho hướng đó.
3. Có thể thay full exact Shapley bằng sampled/hierarchical coalition distillation để giảm cost.
4. Không thể tuyên bố vượt AlignCoder trước khi chạy cùng backbone, split, retrieval pool, budget và evaluation protocol.

### Một câu chốt

Paper này cho thấy vấn đề không chỉ là học chunk nào gần target nhất; vấn đề là học subset các evidence units cùng nhau tạo ra completion tốt. Tuy nhiên, cách triển khai của REPOSHAPLEY trả giá bằng offline generator verification rất đắt. Hướng đáng kiểm chứng nhất cho framework hiện tại là AST-aware coalition supervision với candidate verification nhỏ, thay vì copy nguyên vẹn full Shapley pipeline.

## 14. Câu hỏi recall và transfer

1. Vì sao Delta_i không đủ để chọn context?
2. Utility v(S) khác gì với single-chunk Delta_i?
3. Logistic surrogate mô hình hóa saturation và conflict bằng thành phần nào?
4. Tại sao paper không verify toàn bộ 2^K subsets bằng generator?
5. Candidate set C gồm những nhóm proposal nào?
6. Retrieval token NEED/DONE được tạo từ tín hiệu nào?
7. Vì sao post-verification làm kết quả giảm mạnh khi bỏ đi?
8. K và L ảnh hưởng thế nào tới accuracy và cost?
9. Oracle best trong candidate pool không phải global upper bound vì sao?
10. AST chunking có thể cải thiện phần nào của REPOSHAPLEY?
11. Independent KD khác coalition-aware supervision ở đâu?
12. Nếu muốn kiểm chứng interaction mà không dùng pairwise training, control experiment nhỏ nhất là gì?

## 15. Tài liệu nguồn

- [REPOSHAPLEY PDF](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/2026.findings-acl.505.pdf)

