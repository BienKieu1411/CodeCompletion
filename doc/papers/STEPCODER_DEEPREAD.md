# DeepRead report: StepCoder

## 0. Nguồn và trạng thái trích xuất

- Paper: StepCoder: Improve Code Generation with Reinforcement Learning from Compiler Feedback.
- File nguồn: [StepCoder.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/StepCoder.pdf).
- Phạm vi: toàn bộ 13 trang; phân tích chính ở trang 1–8 và appendix.
- Trạng thái: PDF có text layer; paper có một lỗi màu PDF khi render nhưng text extraction đầy đủ, không cần OCR.

## 1. Tổng hợp

StepCoder là framework RL cho program synthesis chứ không phải repository retrieval. Nó giải hai vấn đề: exploration khó khi code sequence dài và reward compiler chỉ phản ánh phần code được execute. CCCS tạo curriculum bắt đầu từ phần gần canonical solution rồi dần lùi về đầu bài toán; FGO mask token/snippet không được unit test execute để advantage chỉ cập nhật phần có liên quan. Paper còn làm sạch APPS thành APPS+ và báo cáo pass@1 tốt hơn PPOCoder/RLTF. Với repo-level completion, giá trị chính là AST/coverage-aware credit assignment, không phải dùng PPO trực tiếp cho retriever.

## 2. Luận điểm trung tâm

**Author's stated position:** Curriculum of Code Completion Subtasks và Fine-Grained Optimization từ execution coverage giúp RL code generation khám phá và cập nhật chính xác hơn vanilla PPO.

## 3. Cây lập luận

1. Human requirement tạo sequence code dài, unit-test reward sparse.
2. Xác suất sinh đúng toàn bộ code giảm theo số conditional/decision points.
3. Bắt đầu từ prefix canonical gần goal giúp exploration dễ.
4. Khi pass rate tăng, curriculum lùi dần về đầu prompt.
5. Unit tests chỉ execute một phần code.
6. Mask unexecuted tokens để tránh gradient nhiễu.
7. Hai component cùng cải thiện compiler pass rate.

## 4. Phương pháp

### 4.1. CCCS

- AST canonical solution để tìm conditional statements.
- Mỗi sample có curriculum starting position s* ở đầu conditional statement.
- Prompt gồm human requirement và prefix canonical đến s*.
- Policy sinh phần còn lại.
- Reward:
  - +1 nếu pass tất cả tests;
  - -0.3 nếu fail test;
  - -0.6 runtime error;
  - -1 compile error.
- PPO objective có KL penalty với reference SFT model.
- Khi moving-average pass rate vượt threshold rho_t, starting point tiến về đầu solution.
- Số curriculum cho sample là ceiling square-root của số conditional statements.

### 4.2. FGO

- Theo dõi snippets/tokens được execute trong unit tests.
- Mask action không execute khi tính RL loss.
- Chỉ token execute đóng góp vào policy update.

### 4.3. Dataset và training

- APPS+ được làm sạch: bỏ missing input/output/solution, syntax errors, irrelevant code, API misuse và missing libraries; chuẩn hóa I/O.
- 7456 instances: 2850 Introductory, 4020 Interview, 586 Competition.
- Backbone: DeepSeek-Coder-Instruct-6.7B.
- SFT 3 epochs, learning rate 2e-5, 8 A100 80G, global batch 64.
- PPO policy lr 5e-7, critic lr 1.5e-6, 16 rollouts, temperature 0.8, top-p 0.9, max 1024 tokens, KL beta 0.05.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| StepCoder thắng RL baselines trên APPS+ | Pass@1 overall: PPO 31.7, PPOCoder 32.1, RLTF 32.7, StepCoder 36.1 | pp. 6–7, Table 1 | CCCS+FGO có gain | **Source fact or data** | Benchmark program synthesis, không repository completion |
| Hai component đều cần | w/o CCCS 34.6, w/o FGO 35.5, full 36.1 | p. 6 | Ablation trực tiếp | **Source fact or data** | So với vanilla PPO 31.7 |
| Generalization cải thiện | HumanEval 78.7 và MBPP 67.0, cao hơn baselines trong bảng | p. 7, Table 2 | Compiler feedback có transfer | **Source fact or data** | Có thể phụ thuộc APPS+ overlap |
| FGO giải credit assignment | Paper minh họa chỉ 75% code được execute trong một test | pp. 2–3, Figure 1 | Lý do mask | **Author's stated position** | Coverage không hoàn toàn đo semantic relevance |
| Curriculum dùng canonical solution | CCCS prompt có prefix gold trong training | pp. 3–5 | Khó chuyển trực tiếp sang repo inference | **Source fact or data** | Đây là oracle-like training signal |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Dùng AST để xác định decision points.
- Dùng execution coverage cho fine-grained credit assignment.
- Có dataset cleaning và unit-test verification.
- Cho thấy RL có thể vượt SFT next-token trong program synthesis.

### Giới hạn

- CCCS phụ thuộc canonical solution prefix, không có trong production completion.
- PPO đắt, cần rollout/critic/compiler.
- Execution coverage có thể bỏ sót code cần thiết cho input khác.
- Runtime/compile sandbox và test quality là bottleneck.
- Không xử lý retrieval candidate/chunk relevance.

## 7. Hàm ý cho AST/KD framework

1. Mượn AST decision-point idea để tạo auxiliary target mask/importance cho completion tokens.
2. Nếu có unit tests, dùng coverage làm quality check offline cho teacher KD.
3. Không dùng PPO làm loss chính cho retriever vì chi phí và credit assignment phức tạp.
4. Có thể weight teacher target utility cao hơn ở API/identifier/branch tokens.
5. AST node boundaries cho phép transfer FGO từ token mask sang context-unit mask.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** CCCS + FGO cải thiện RL code generation.
- **Source fact or data:** StepCoder có số liệu pass@1 và ablation rõ.
- **Reasoned inference:** execution-aware weighting là auxiliary signal tốt cho KD, nhất là khi benchmark có tests.
- **Unverified:** coverage-weighted context distillation có giúp vượt AlignCoder trong repo completion hay không.

## 9. Câu hỏi recall/transfer

1. CCCS di chuyển curriculum như thế nào?
2. FGO mask theo tín hiệu nào?
3. Vì sao canonical prefix làm CCCS khó chuyển thẳng sang inference?
4. Làm sao biến execution coverage thành teacher weight cho AST chunk?

