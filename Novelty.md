# Đề xuất novelty: Graph-Structured Multi-hop RL Retrieval with Quantized Semantic States cho Repository-Level Code Completion

## 1. Bối cảnh

Bài toán **repository-level code completion** yêu cầu mô hình sinh code dựa trên ngữ cảnh không chỉ trong file hiện tại mà còn từ nhiều file khác trong cùng repository. Các phương pháp RAG hiện tại thường lấy phần code chưa hoàn thành làm query, retrieve các code snippets liên quan trong repo, rồi đưa các snippets đó vào prompt cho code LLM.

Hai công trình gần nhất là **RLCoder** và **AlignCoder** đều cải thiện retrieval bằng reinforcement learning, nhưng cách tiếp cận của chúng vẫn chủ yếu dựa trên **semantic chunk retrieval**. Vì vậy, chúng chưa khai thác đầy đủ cấu trúc tự nhiên của repository như AST, call graph, import graph, inheritance graph hoặc dependency graph.

Đề tài đề xuất một hướng mới:

> **Học chính sách retrieval như một quá trình traversal nhiều bước trên repository graph, kết hợp quantized semantic states để ổn định RL và cải thiện khả năng reasoning theo dependency.**

---

## 2. Tóm tắt RLCoder

### 2.1 Mục tiêu

RLCoder đề xuất framework reinforcement learning cho repository-level code completion. Thay vì cần nhãn ground-truth candidate, RLCoder dùng feedback từ generator/evaluator để train retriever.

### 2.2 Thành phần chính

RLCoder gồm các ý chính:

1. **Split-Aggregate candidate construction**
   - Chia code theo blank lines thành các mini-blocks.
   - Ghép các mini-blocks thành natural candidates.
   - Mục tiêu là tránh việc fixed-window cắt đứt semantic continuity của code.

2. **RL-based RLRetriever**
   - Retriever được train bằng reinforcement learning.
   - Reward dựa trên perplexity của target code khi dùng retrieved candidate làm context.

3. **Weighted Perplexity Reward**
   - Gán trọng số cao hơn cho các token quan trọng, đặc biệt là identifier/API tokens.
   - Giúp giảm lỗi hallucination API hoặc identifier không tồn tại.

4. **Stop Signal Mechanism**
   - Retriever có thể quyết định khi nào nên dừng retrieve.
   - Nếu retrieval không cần thiết hoặc candidate gây nhiễu, stop signal giúp loại bỏ context không hữu ích.

### 2.3 Hạn chế của RLCoder

RLCoder giải quyết tốt bài toán train retriever không cần labeled data, nhưng vẫn có một số hạn chế:

- Retrieval chủ yếu là **single-step retrieval**.
- Candidate vẫn là **flat code chunks**, chưa phải graph traversal.
- Query chủ yếu dựa trên unfinished code.
- Chưa trực tiếp modeling các quan hệ cấu trúc như:
  - call chain
  - import chain
  - inheritance
  - data/control dependency
- Không có cơ chế semantic state abstraction bằng discrete/quantized representation.
- Stop signal giúp loại bỏ context không cần thiết, nhưng chưa học đường đi reasoning qua dependency graph.

Nói ngắn gọn, RLCoder học **what to retrieve**, nhưng chưa học rõ **how to traverse repository structure**.

---

## 3. Tóm tắt AlignCoder

### 3.1 Mục tiêu

AlignCoder tập trung giải quyết vấn đề **semantic gap** giữa query và target code. Nếu chỉ dùng unfinished code làm query, query có thể thiếu key tokens xuất hiện trong target completion, làm retriever khó tìm đúng snippets.

### 3.2 Thành phần chính

AlignCoder gồm các ý chính:

1. **Query Enhancement**
   - Dùng lightweight LLM sampler sinh nhiều candidate completions.
   - Ghép unfinished code với multiple candidate completions để tạo enhanced query.
   - Enhanced query có xác suất chứa key tokens gần với target code cao hơn.

2. **Coarse-to-Fine Retrieval**
   - Coarse-grained retrieval ban đầu lấy một số snippets bằng sparse retriever.
   - Sau đó sampler sinh candidate completions.
   - Enhanced query được dùng cho fine-grained retrieval.

3. **AlignRetriever trained by RL**
   - Retriever học cách tận dụng inference information trong enhanced query.
   - Reward dựa trên PPL của target code khi dùng retrieved snippets.

4. **Dependency Context**
   - Ngoài base snippets, AlignCoder xây dựng dependency snippets từ import statements và entities liên quan.

### 3.3 Hạn chế của AlignCoder

AlignCoder cải thiện đáng kể semantic alignment, nhưng vẫn còn các hạn chế:

- Retrieval vẫn chủ yếu là **ranking snippets** thay vì graph traversal.
- Query enhancement phụ thuộc vào sampled completions, có thể sinh nhiễu hoặc chứa completion sai.
- Multiple sampling tăng khả năng xuất hiện key tokens, nhưng cũng tăng compute cost và có thể sinh redundant/noisy candidates.
- Dependency context được xây dựng từ import/entity extraction, nhưng chưa học chính sách traversal nhiều bước trên heterogeneous repository graph.
- RL vẫn tối ưu retriever trong không gian embedding/query-enhanced retrieval, chưa khai thác discrete graph state hoặc semantic role state.

Nói ngắn gọn, AlignCoder học cách **align query với target intent**, nhưng chưa học **dependency-aware multi-hop reasoning trên graph cấu trúc repo**.

---

## 4. Khoảng trống nghiên cứu

Từ RLCoder và AlignCoder, có thể thấy 3 khoảng trống chính:

### 4.1 Thiếu graph-structured retrieval

RLCoder và AlignCoder đều retrieve trên code snippets. Dù AlignCoder có dependency snippets, repository vẫn chưa được biểu diễn đầy đủ như một graph gồm nhiều loại node và edge.

Trong thực tế, code completion cấp repo thường cần truy vết qua:

```text
Current Function
    ↓ calls
Service Function
    ↓ imports
Repository Class
    ↓ uses
Database Utility
```

Flat semantic retrieval khó học được chuỗi reasoning như vậy.

### 4.2 Thiếu multi-hop retrieval policy

RLCoder và AlignCoder chủ yếu retrieve một lần hoặc qua cơ chế query enhancement. Nhưng nhiều completion target cần retrieval theo nhiều bước:

```text
Hop 1: tìm file/module liên quan
Hop 2: tìm class/function được gọi
Hop 3: tìm method/API thực sự cần completion
Hop 4: lấy usage pattern hoặc signature liên quan
```

Đây là bài toán phù hợp để mô hình hóa bằng **RL graph traversal policy**.

### 4.3 Thiếu quantized semantic states

Dense embeddings có thể bị:

- semantic drift
- lexical bias
- retrieval instability
- noisy similarity

Với code, nhiều function có tên giống nhau nhưng vai trò khác nhau; ngược lại, nhiều function tên khác nhau nhưng cùng semantic role. Vì vậy, cần một cơ chế semantic abstraction tốt hơn.

Quantized semantic states giúp nhóm node/function theo vai trò, ví dụ:

- Controller
- Service
- Repository
- Validator
- Parser
- Utility
- API Wrapper

Từ đó RL policy học traversal trên semantic role states thay vì embedding liên tục nhiễu.

---

## 5. Novelty đề xuất

### Tên hướng đề xuất

**GraphCoder-RL: Multi-hop Graph Retrieval with Quantized Semantic States for Repository-Level Code Completion**

### Core idea

Thay vì xem retrieval là bài toán chọn top-k snippets theo similarity, đề tài xem retrieval là:

> **Một quá trình agent traversal nhiều bước trên heterogeneous repository graph để tìm dependency path hữu ích nhất cho code completion.**

Pipeline tổng thể:

```text
Repository
 ↓
Tree-sitter Parsing
 ↓
AST + Call Graph + Import Graph + Inheritance Graph
 ↓
Heterogeneous Repository Graph
 ↓
Graph Neural Encoder
 ↓
Vector Quantization
 ↓
Quantized Semantic States
 ↓
Multi-hop RL Retrieval Policy
 ↓
Adaptive Context Construction
 ↓
Code Completion
```

---

## 6. Các đóng góp chính

### Contribution 1: Heterogeneous Repository Graph Retrieval

Đề tài mô hình hóa repository thành graph thay vì tập snippets phẳng.

Node có thể gồm:

- repository
- package/module
- file
- class
- function/method
- variable/entity
- API usage

Edge có thể gồm:

- contains
- imports
- calls
- inherits
- overrides
- uses
- data dependency
- control dependency

Điểm mới so với RLCoder và AlignCoder:

| Phương pháp | Biểu diễn repo | Hạn chế |
|---|---|---|
| RLCoder | Natural candidates / flat snippets | Không có graph traversal |
| AlignCoder | Base snippets + dependency snippets | Có dependency context nhưng chưa là learned graph policy |
| Đề tài đề xuất | Heterogeneous repository graph | Retrieval dựa trên structural reasoning |

---

### Contribution 2: Multi-hop RL Graph Traversal Policy

Retrieval được mô hình hóa như sequential decision making.

Tại mỗi bước, agent chọn node tiếp theo hoặc dừng:

```text
state_t = current query state + visited graph nodes + quantized semantic state

action_t = chọn neighbor node / retrieve node / stop

reward = completion quality improvement - context cost - noise penalty
```

Ví dụ traversal:

```text
Unfinished code
 → current class
 → imported service
 → called repository method
 → database utility signature
```

Điểm mới:

- RLCoder học chọn candidate hữu ích.
- AlignCoder học dùng enhanced query để retrieve tốt hơn.
- Đề tài đề xuất học **đường đi retrieval** trên graph.

Đây là khác biệt quan trọng nhất.

---

### Contribution 3: Quantized Semantic States for Stable RL Retrieval

Sau khi encode graph bằng GNN, embedding liên tục được ánh xạ vào codebook rời rạc:

```math
z \in \mathbb{R}^{d}
```

```math
q(z) = e_k, \quad k \in \{1, 2, ..., K\}
```

Trong đó:

- `z`: graph embedding của node/function/class
- `e_k`: codebook vector
- `k`: semantic state id

Các semantic states có thể biểu diễn vai trò như:

```text
Controller → Service → Repository → Utility
```

Lợi ích:

- giảm nhiễu của dense embedding
- giúp RL policy học dễ hơn
- tăng interpretability
- gom các function có vai trò tương tự vào cùng semantic state
- hỗ trợ traversal theo semantic role sequence

So với AlignCoder:

| Thành phần | AlignCoder | Đề tài đề xuất |
|---|---|---|
| Query improvement | Multiple sampled completions | Graph state + semantic role states |
| Embedding space | Continuous dense embedding | Quantized/discrete graph states |
| RL target | Retrieve snippets phù hợp enhanced query | Learn graph traversal policy |
| Noise handling | Dựa vào nhiều samples và RL retriever | Discrete state abstraction + reward penalty |

---

### Contribution 4: Adaptive Context Construction dựa trên retrieval path

Thay vì concatenate top-k snippets đơn giản, context được xây dựng từ path đã traverse.

Ví dụ:

```text
Path 1: current function → imported service → repository method
Path 2: current class → parent class → overridden method
Path 3: API call → wrapper function → usage examples
```

Context đưa vào LLM có thể gồm:

- signature
- docstring
- usage examples
- minimal body
- import statement
- type hints

Điểm mới:

- không chỉ retrieve code chunks
- retrieve dependency path có cấu trúc
- context có thể giải thích được

---

## 7. So sánh tổng thể với RLCoder và AlignCoder

| Tiêu chí | RLCoder | AlignCoder | Đề tài đề xuất |
|---|---|---|---|
| Mục tiêu chính | Train retriever không cần labeled data | Align query với target intent | Học graph traversal policy cho retrieval |
| Query | Unfinished code | Unfinished code + sampled completions | Unfinished code + graph state + traversal history |
| Retrieval unit | Natural candidates | Base/dependency snippets | Graph nodes, entities, paths |
| Repository structure | Dependency analysis để tạo data/candidates | Dependency snippets từ import/entity extraction | Heterogeneous repository graph |
| RL role | Chọn useful candidate | Học tận dụng enhanced query | Học multi-hop traversal policy |
| Reward | Weighted PPL | PPL-based reward | Completion quality + path quality + uncertainty + token cost |
| Retrieval style | Mostly single-step | Coarse-to-fine, enhanced query | Multi-hop graph traversal |
| Stop mechanism | Có stop signal | Không phải trọng tâm | Stop action trong policy |
| Semantic abstraction | Dense candidate embedding | Dense enhanced query embedding | Quantized semantic graph states |
| Interpretability | Trung bình | Trung bình | Cao: visualize được retrieval path |
| Novelty gap | RL retriever | Query enhancement + RL | Structure-aware, multi-hop, quantized graph RL retrieval |

---

## 8. Công thức reward đề xuất

Reward tổng quát:

```math
R = \alpha \cdot Q_{completion}
  - \beta \cdot C_{token}
  - \gamma \cdot U_{retrieval}
  + \lambda \cdot S_{path}
```

Trong đó:

- `Q_completion`: chất lượng completion, có thể đo bằng EM, edit similarity hoặc PPL reduction.
- `C_token`: chi phí token context.
- `U_retrieval`: độ bất định/nhiễu của retrieval.
- `S_path`: điểm hợp lý của graph path, ví dụ path có call/import/inheritance relation trực tiếp.

Một reward thực dụng khi training:

```math
R = PPL_{no\_ctx} - PPL_{with\_retrieved\_path}
```

Thêm penalty:

```math
R' = R - \beta \cdot \text{TokenCost} - \gamma \cdot \text{IrrelevantNodePenalty}
```

---

## 9. Điểm khác biệt có thể claim trong paper

### Claim 1

> Existing RL-based retrievers optimize snippet selection, while our method learns repository graph traversal policies for dependency-aware retrieval.

### Claim 2

> Unlike query-enhancement methods that rely on sampled completions to bridge the semantic gap, our method exploits explicit repository structures and semantic role transitions.

### Claim 3

> Quantized semantic graph states provide a discrete abstraction of code roles, reducing retrieval instability in continuous embedding spaces and improving RL policy learning.

### Claim 4

> The proposed retrieval path is interpretable and can be visualized as dependency reasoning chains across repository entities.

---

## 10. Experimental design đề xuất

### 10.1 Benchmarks

Có thể dùng cùng benchmark với RLCoder và AlignCoder:

- CrossCodeEval Python
- CrossCodeEval Java
- RepoEval Line
- RepoEval API

### 10.2 Baselines

Nên so sánh với:

- No Retrieval
- BM25
- UniXcoder
- RepoCoder
- RLCoder
- AlignCoder
- Graph retrieval without RL
- Graph retrieval without quantization
- Graph retrieval without multi-hop

### 10.3 Metrics

- Exact Match (EM)
- Edit Similarity (ES)
- Perplexity (PPL)
- Retrieval Recall@k
- Path relevance score
- Context token cost
- Inference latency

### 10.4 Ablation study

Các biến thể cần ablate:

| Variant | Mục tiêu |
|---|---|
| w/o Graph | Kiểm tra vai trò của repository graph |
| w/o Multi-hop | Kiểm tra vai trò traversal nhiều bước |
| w/o Quantization | Kiểm tra vai trò quantized semantic states |
| w/o RL | Kiểm tra vai trò policy learning |
| w/o Path Reward | Kiểm tra reward theo dependency path |
| w/o Token Penalty | Kiểm tra adaptive context budget |

---

## 11. Rủi ro và cách giảm rủi ro

### Rủi ro 1: Graph construction tốn thời gian

Giải pháp:

- cache graph theo repository
- chỉ parse AST/call/import trước inference
- dùng incremental graph update

### Rủi ro 2: GNN training khó scale

Giải pháp:

- dùng subgraph sampling
- dùng GraphSAGE mini-batch
- freeze code embedding, chỉ train GNN projection + policy head

### Rủi ro 3: Quantization làm mất thông tin

Giải pháp:

- dùng soft vector quantization
- residual quantization
- combine continuous embedding + quantized state

### Rủi ro 4: RL khó hội tụ

Giải pháp:

- pretrain retriever bằng contrastive learning
- warm-start policy bằng heuristic graph traversal
- sau đó fine-tune bằng PPO hoặc policy gradient

---

## 12. Phiên bản phương pháp khả thi để triển khai

### Phase 1: Build graph

```text
Tree-sitter
 ↓
Extract AST nodes, functions, classes, imports, calls
 ↓
Build heterogeneous graph
```

### Phase 2: Train graph encoder

```text
Node text embedding + structural features
 ↓
GraphSAGE/GAT
 ↓
Graph-aware node embeddings
```

### Phase 3: Quantization

```text
Graph embeddings
 ↓
VQ codebook
 ↓
Semantic state ids
```

### Phase 4: RL retrieval

```text
State: query + current node + visited path + semantic state
Action: move to neighbor / select node / stop
Reward: PPL reduction + path relevance - token cost
```

### Phase 5: Completion

```text
Retrieved path context
 ↓
Prompt construction
 ↓
Code LLM completion
```

---

## 13. Tên đề tài gợi ý

### Option 1

**GraphCoder-RL: Multi-hop Graph Retrieval with Quantized Semantic States for Repository-Level Code Completion**

### Option 2

**Learning Repository Graph Traversal Policies for Code Completion**

### Option 3

**Quantized Graph Reinforcement Retrieval for Repository-Level Code Completion**

### Option 4

**Structure-aware Reinforcement Retrieval over Repository Graphs for Code Completion**

---

## 14. Tóm tắt novelty cuối cùng

Novelty mạnh nhất của đề tài là:

> Chuyển repository-level retrieval từ bài toán ranking code snippets sang bài toán học chính sách traversal nhiều bước trên heterogeneous repository graph.

So với RLCoder:

- RLCoder dùng RL để chọn useful snippets.
- Đề tài dùng RL để học dependency-aware graph traversal path.

So với AlignCoder:

- AlignCoder dùng sampled completions để enhance query.
- Đề tài dùng repository graph + quantized semantic states để biểu diễn intent và dependency reasoning.

Đóng góp nổi bật:

1. **Graph-structured retrieval** thay cho flat chunk retrieval.
2. **Multi-hop RL traversal policy** thay cho single-step retriever.
3. **Quantized semantic states** để ổn định retrieval và biểu diễn semantic role.
4. **Interpretable dependency paths** cho repository-level code completion.
5. **Adaptive context construction** dựa trên graph path và token budget.

Một câu mô tả ngắn gọn:

> Đề tài đề xuất một framework retrieval mới cho repository-level code completion, trong đó retriever không chỉ học chọn snippets hữu ích như RLCoder hoặc align query như AlignCoder, mà học cách di chuyển qua graph cấu trúc của repository để tìm chuỗi dependency liên quan nhất cho completion.
