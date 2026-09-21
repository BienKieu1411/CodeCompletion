# DeepRead report: REVELA

## 0. Nguồn và trạng thái trích xuất

- Paper: REVELA: Dense Retriever Learning via Language Modeling.
- File nguồn: [ICLR-2026-revela-dense-retriever-learning-via-language-modeling-Paper-Conference.pdf](/Users/kieugiangbien/Downloads/Project/CodeCompletion/Paper/ICLR-2026-revela-dense-retriever-learning-via-language-modeling-Paper-Conference.pdf).
- Phạm vi: toàn bộ 24 trang; phân tích chính ở trang 1–18 và appendix 19–24.
- Trạng thái: PDF có text layer; không cần OCR.

## 1. Tổng hợp

REVELA là một hướng non-pairwise đáng chú ý: train dense retriever bằng next-token prediction. Trong mỗi batch document, embedding similarity giữa các document tạo trọng số attention cross-document; LM dự đoán token của document hiện tại sau khi nhìn prefix của nó và thông tin từ các document khác. NTP loss truyền gradient đồng thời vào LM và retriever, không cần query-document pair labels. Đây không phải KD và không phải repository completion-specific, nhưng gợi ý một loss chính thuần language modeling có thể học retrieval representation. Nếu thích nghi cho code completion, cần thiết kế batch AST chunks/target masks để similarity không học quan hệ topic chung mà học context giúp dự đoán target.

## 2. Luận điểm trung tâm

**Author's stated position:** Dense retriever có thể được học self-supervised từ language modeling khi similarity của các document quyết định cross-document attention, không cần cặp query-document gán nhãn.

## 3. Cây lập luận

1. Query-document labels đắt và domain-specific.
2. Các document liên quan giúp LM dự đoán token dễ hơn.
3. Retriever similarity có thể làm soft attention weights.
4. NTP loss cung cấp tín hiệu trực tiếp cho cả LM và retriever.
5. Batch lớn hơn tạo nhiều context/negative relationships.
6. Retriever học được representation cạnh tranh supervised retriever.

## 4. Phương pháp

### 4.1. In-batch document attention

Với các document D_i trong batch:

- Retriever E_Theta tạo normalized cosine similarity S_ij.
- Softmax với temperature tạo Sim(D_i,D_j), j khác i.
- Self-attention của D_i xử lý prefix riêng.
- Cross-document outputs b_ij lấy từ cached K/V của document j.
- b_i là weighted sum theo Sim.
- Hidden state dùng cho NTP là self-attention output cộng b_i.
- Attention mask xử lý duplicate/causal constraints.

NTP loss:

P(x_i,l | x_i,<l, các D_j khác i).

Gradient đi qua cả LM Phi và retriever Theta.

### 4.2. Dữ liệu và training

- Code corpus CoIR được chunk theo raw text tối đa 120 words, complete sentences.
- Batch 16; chunks cùng document được interleave.
- Wiki cho BEIR/BRIGHT; code cho CoIR.
- LM LLaMA3.2-1B.
- Retriever backbones SmolLM2-135M, Qwen2.5-0.5B, Llama3.2-1B/3B.
- LoRA rank 256, temperature 1e-4, learning rate 1e-4, warmup 100, bf16, 4 A100 80G.
- Khoảng một epoch: Wiki 10k steps/44h, code 11k steps/48h trong setup báo cáo.

### 4.3. Không phải distillation

REVELA không có teacher–student KL. Đây là joint/self-supervised training, trong đó NTP là loss chính và retriever similarity chỉ là một phần của forward computation.

## 5. Bằng chứng chính

| Claim | Evidence | Location | Relationship | Confidence | Caveat |
|---|---|---|---|---|---|
| Không cần query-document pair | Retriever được update qua NTP cross-document attention | pp. 2–5 | Direct alternative cho AlignCoder/RLCoder labels | **Source fact or data** | Batch vẫn quyết định implicit positives/negatives |
| Retriever đạt kết quả mạnh trên CoIR/BEIR | Paper báo cáo REVELA-3B vượt E5-Mistral trên CoIR và các model cùng scale | pp. 6–10 | Support cho NTP retrieval objective | **Source fact or data** | Không có RepoEval/CrossCodeEval |
| Domain code giúp code retrieval | Controlled setup cho thấy Revela-code cải thiện CoIR so với wiki/Contriever ở cùng backbone | pp. 9–11 | Tín hiệu domain matters | **Source fact or data** | Chunk code theo sentence/words |
| Batch scaling có ích | Batch 4 → 8 → 16 thường cải thiện | pp. 11–12 | In-batch relation quan trọng | **Source fact or data** | Tốn memory/compute |
| Chi phí tốt hơn pairwise cross-document forward | Appendix nêu REPLUG pairwise có cost quadratic hơn, REVELA tái dùng batch attention | appendix | Motivation efficiency | **Author's stated position** | So sánh implementation-dependent |

## 6. Điểm mạnh và giới hạn

### Điểm mạnh

- Loss chính rõ ràng, không pairwise và không cần preference labels.
- Retriever học từ tín hiệu LM trực tiếp.
- Có thể domain-adapt code retriever chỉ bằng raw repository corpus.
- Cross-document attention tạo soft relation thay vì hard positive.

### Giới hạn

- Relatedness trong batch có thể là topical/lexical, không phải target utility.
- Không có repository-level completion evaluation.
- Random batch có thể tạo false positives và embedding collapse.
- Joint LM/retriever forward phức tạp hơn offline embedding retrieval.
- Causal document attention có thể làm model học shortcut từ chunk co-occurrence.

## 7. Hàm ý cho framework hiện tại

1. Có thể dùng NTP loss làm loss chính cho pretraining retriever trên AST chunks:
   - mask một AST target node;
   - cho LM dự đoán continuation;
   - cross-chunk attention dùng similarity của các node khác.
2. Dùng AST parent/dependency-aware batch để tạo context relation tự nhiên, không tạo pairwise labels.
3. Sau self-supervised pretraining, có thể thêm KD scalar utility như auxiliary loss, nhưng không cần RL.
4. Cần hard negatives implicit bằng chunks cùng file khác function hoặc cùng identifier nhưng target-irrelevant.
5. Eval phải kiểm tra target PPL, retrieval recall và end-to-end completion riêng.

## 8. Kết luận theo mức độ chắc chắn

- **Author's stated position:** NTP có thể học dense retriever mà không cần pairs.
- **Source fact or data:** REVELA có kết quả retrieval mạnh trên CoIR/BEIR/BRIGHT trong setup paper.
- **Reasoned inference:** REVELA cung cấp candidate cho loss chính mới nếu muốn rời RL/pairwise; AST-aware batch là điểm thích nghi tự nhiên.
- **Unverified:** NTP-only retriever có đủ target alignment để vượt AlignCoder hay vẫn cần KD utility auxiliary.

## 9. Câu hỏi recall/transfer

1. Retriever similarity đi vào NTP forward ở đâu?
2. Vì sao REVELA không phải knowledge distillation?
3. Làm thế nào tạo batch AST để relation gần code completion hơn?
4. Loss chính NTP nên kết hợp với target utility KD ra sao mà không tạo pairwise?

