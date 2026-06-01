# Đề xuất novelty cập nhật: GraphCoder-RL với Semantic Entity Chunking, Local Context Anchoring và Graph-Structured Multi-hop Retrieval

## 1. Bối cảnh

Bài toán **repository-level code completion** yêu cầu mô hình sinh code dựa trên cả:

- **left context** trong file hiện tại;
- **cross-file context** từ các file khác trong repository;
- các quan hệ cấu trúc như import, call, inheritance, type usage, data/control dependency.

Các phương pháp RAG hiện tại thường dùng phần code chưa hoàn thành làm query, retrieve các code snippets liên quan trong repository, rồi đưa các snippets đó vào prompt cho code LLM.

Hai công trình gần nhất là **RLCoder** và **AlignCoder** đều cải thiện retrieval bằng reinforcement learning, nhưng retrieval của chúng vẫn chủ yếu dựa trên **ranking flat snippets/candidates**. Chúng chưa biến repository thành một không gian reasoning có cấu trúc để agent có thể học cách đi qua các dependency path.

Đề tài đề xuất một hướng mới:

> **Học chính sách retrieval như một quá trình traversal nhiều bước trên heterogeneous repository graph, trong đó left context đóng vai trò query anchor, semantic chunks/entities là node, dependency relations là edge, và quantized semantic states giúp ổn định RL policy.**

---

## 2. Tóm tắt RLCoder

### 2.1 Mục tiêu

RLCoder đề xuất một reinforcement learning framework cho repository-level code completion. Thay vì cần nhãn ground-truth candidate, RLCoder dùng feedback từ generator/evaluator để train retriever.

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

- Retrieval chủ yếu là **single-step candidate selection**.
- Candidate vẫn là **flat code chunks**, chưa phải graph traversal.
- Query chủ yếu dựa trên unfinished code.
- Split-Aggregate tốt hơn fixed-window, nhưng vẫn là rule-based chunking.
- Chưa trực tiếp modeling các quan hệ cấu trúc như:
  - call chain;
  - import chain;
  - inheritance;
  - data/control dependency;
  - semantic role transition.
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

4. **Base snippets + dependency snippets**
   - Ngoài base snippets, AlignCoder xây dựng dependency snippets từ import statements và entities liên quan.

### 3.3 Hạn chế của AlignCoder

AlignCoder cải thiện semantic alignment, nhưng vẫn còn các hạn chế:

- Retrieval vẫn chủ yếu là **ranking snippets** thay vì graph traversal.
- Query enhancement phụ thuộc vào sampled completions, có thể sinh nhiễu hoặc chứa completion sai.
- Multiple sampling tăng khả năng xuất hiện key tokens, nhưng cũng tăng compute cost và có thể sinh redundant/noisy candidates.
- Dependency context được xây dựng từ import/entity extraction, nhưng chưa học chính sách traversal nhiều bước trên heterogeneous repository graph.
- RL vẫn tối ưu retriever trong không gian embedding/query-enhanced retrieval, chưa khai thác discrete graph state hoặc semantic role state.
- Left context được dùng để tạo enhanced query, nhưng chưa được mô hình hóa như **anchor node/state** trong graph traversal.

Nói ngắn gọn, AlignCoder học cách **align query với target intent**, nhưng chưa học **dependency-aware multi-hop reasoning trên graph cấu trúc repo**.

---

## 4. Khoảng trống nghiên cứu

Từ RLCoder và AlignCoder, có thể thấy 4 khoảng trống chính.

### 4.1 Chunking chưa đủ semantic và chưa gắn chặt với graph

RLCoder dùng Split-Aggregate để tạo natural candidates. AlignCoder dùng base snippets và dependency snippets. Tuy nhiên, các cách này vẫn có thể gặp vấn đề:

- fixed window dễ cắt gãy logic;
- function-level chunking có thể thiếu class/import/helper context;
- class-level chunking có thể quá dài nếu class có nhiều method;
- file-level context quá dài và nhiễu;
- dependency snippets phụ thuộc vào static parsing, có thể thiếu semantic continuity.

Vì vậy cần một bước **semantic entity chunking** tốt hơn: chunk không chỉ là đoạn text, mà là node có metadata rõ ràng trong graph.

### 4.2 Thiếu graph-structured retrieval

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

### 4.3 Thiếu multi-hop retrieval policy

RLCoder và AlignCoder chủ yếu retrieve một lần hoặc qua cơ chế query enhancement. Nhưng nhiều completion target cần retrieval theo nhiều bước:

```text
Hop 1: xác định current scope / current symbol
Hop 2: tìm imported module hoặc called service
Hop 3: tìm method/API thực sự cần completion
Hop 4: tìm usage pattern, signature hoặc type liên quan
```

Đây là bài toán phù hợp để mô hình hóa bằng **RL graph traversal policy**.

### 4.4 Thiếu quantized semantic states

Dense embeddings có thể bị:

- semantic drift;
- lexical bias;
- retrieval instability;
- noisy similarity.

Với code, nhiều function có tên giống nhau nhưng vai trò khác nhau; ngược lại, nhiều function tên khác nhau nhưng cùng semantic role. Vì vậy, cần một cơ chế semantic abstraction tốt hơn.

Quantized semantic states giúp nhóm node/function theo vai trò, ví dụ:

- Controller;
- Service;
- Repository;
- Validator;
- Parser;
- Utility;
- API Wrapper.

Từ đó RL policy học traversal trên semantic role states thay vì embedding liên tục nhiễu.

---

## 5. Novelty đề xuất

### Tên hướng đề xuất

**GraphCoder-RL: PPL-guided Semantic Entity Chunking and Quantized Multi-hop Graph Retrieval for Repository-Level Code Completion**

### Core idea

Thay vì xem retrieval là bài toán chọn top-k snippets theo similarity, đề tài xem retrieval là:

> **Một quá trình agent traversal nhiều bước trên heterogeneous chunk-entity repository graph để tìm dependency path hữu ích nhất cho code completion.**

Điểm cập nhật quan trọng:

- **Left context** không bị retrieve như cross-file context, mà được dùng làm **query anchor** và **state representation**.
- **Chunking** không thay graph; chunk/entity trở thành **node trong graph**.
- **PPL-guided Semantic Entity Chunking** giúp tạo node retrieval có ý nghĩa hơn, đặc biệt khi class/function quá dài.
- **GraphSAGE/GAT/GGNN** encode quan hệ giữa chunks/entities.
- **Vector Quantization** tạo semantic role states rời rạc.
- **RL policy** học multi-hop traversal và STOP action.

Pipeline tổng thể:

```text
Repository
 ↓
Tree-sitter Parsing
 ↓
PPL-guided Semantic Entity Chunking
 ↓
Extract metadata:
  file path, parent class/function, defined symbols, used symbols, imports
 ↓
Build Heterogeneous Chunk-Entity Graph:
  contains, imports, calls, inherits, uses_type, data/control dependency
 ↓
Local Context Extraction:
  left context, current scope, imports, local variables, cursor position
 ↓
Query Anchor Construction
 ↓
Coarse Retrieval top-N
 ↓
Local Dependency Subgraph Construction
 ↓
Graph Neural Encoder
 ↓
Vector Quantization
 ↓
Quantized Semantic States
 ↓
Multi-hop RL Retrieval Policy
 ↓
Context Composer:
  preserved left context + retrieved cross-file graph context
 ↓
Code Completion
```

---

## 6. Xử lý left context

### 6.1 Vai trò của left context

Left context là nguồn quan trọng nhất vì nó chứa phần code ngay trước cursor. Không nên xử lý left context giống cross-file retrieval.

Left context có 3 vai trò:

1. **Prompt context bắt buộc**
   - Immediate left context gần cursor phải luôn được giữ.

2. **Query representation**
   - Dùng để tạo query cho coarse retrieval và graph traversal.

3. **Graph anchor state**
   - Tạo điểm bắt đầu cho agent đi trên repository graph.

### 6.2 Thành phần left context nên trích xuất

```text
Q = file_path
  + imports
  + current_class_signature
  + current_function_signature
  + local_variables
  + recent_left_context
  + cursor_position
```

Nên chia left context thành 3 lớp:

| Lớp | Cách xử lý |
|---|---|
| Immediate left context | 20–50 dòng gần cursor, luôn giữ nguyên |
| Scope context | function/class signature, decorators, docstring, local variables, luôn giữ |
| Distant left context | phần xa hơn trong cùng file, chọn bằng symbol overlap/PPL/entropy nếu token budget hạn chế |

### 6.3 Left context trong graph

Tạo các node đặc biệt:

- `CursorNode`
- `CurrentFileNode`
- `CurrentFunctionNode`
- `CurrentClassNode`
- `ImportBlockNode`
- `ImmediateLeftContextNode`
- `LocalVariableStateNode`

Các edge ví dụ:

```text
CursorNode --inside--> CurrentFunctionNode
CurrentFunctionNode --inside--> CurrentClassNode
CurrentFileNode --imports--> ExternalFileNode
ImmediateLeftContextNode --mentions--> SymbolNode
CurrentFunctionNode --calls/uses--> CandidateNode
```

Như vậy:

```text
left context = điểm xuất phát / state
cross-file context = thứ cần retrieve
```

---

## 7. PPL-guided Semantic Entity Chunking

### 7.1 Vì sao cần chunking?

Graph cần node. Nếu node quá nhỏ, graph quá lớn và context bị vụn. Nếu node quá lớn, retrieval bị nhiễu và tốn token.

Các cách chunking thông thường có vấn đề:

| Cách chunk | Vấn đề |
|---|---|
| Fixed window | Cắt gãy logic |
| Function-level | Có thể thiếu class/import/helper context |
| Class-level | Quá dài nếu class có nhiều method |
| File-level | Quá dài và nhiễu |
| Split-Aggregate | Tốt hơn fixed-window nhưng vẫn rule-based |
| Dependency snippet | Phụ thuộc static parsing, có thể thiếu semantic continuity |

### 7.2 Ý tưởng

Không dùng PPL để chunk bừa toàn bộ code. Dùng **PPL-guided Semantic Entity Chunking** có ràng buộc bởi AST/entity.

Quy trình:

```text
1. Tree-sitter parse code thành file/class/function/method/block.
2. Nếu entity ngắn → giữ nguyên.
3. Nếu entity quá dài → dùng PPL/entropy để tìm điểm cắt semantic boundary.
4. Mỗi chunk vẫn lưu metadata:
   - file path
   - parent class/function
   - defined symbols
   - used symbols
   - imports
   - start/end line
5. Build graph giữa các chunk/entity bằng metadata đó.
```

### 7.3 Chunk không làm mất graph

Chunk không thay graph. Chunk trở thành node trong graph.

Ví dụ:

```text
Chunk A: routes/users.py::update_user_route
Chunk B: repository/users.py::update_user
Chunk C: models/user.py::User
Chunk D: database/db.py::get_db
Chunk E: services/auth.py::auth_service
```

Graph:

```text
A --calls--> B
A --uses_type--> C
A --depends_on--> D
A --imports--> E
B --uses_type--> C
```

Vì vậy:

```text
Chunking = tạo node tốt hơn
Graph = nối các node theo dependency
GNN/RL = học đi trên graph để retrieve context
```

### 7.4 Vị trí novelty của chunking

PPL-guided chunking nên là **contribution phụ nhưng quan trọng**, không thay thế novelty chính.

Contribution chính vẫn là:

> Learning repository graph traversal policies for dependency-aware retrieval.

Chunking giúp trả lời câu hỏi:

> Graph node lấy từ đâu và tại sao node đó hợp lý hơn fixed-window/function/class chunking?

---

## 8. Heterogeneous Chunk-Entity Repository Graph

Đề tài mô hình hóa repository thành graph thay vì tập snippets phẳng.

### 8.1 Node types

Node có thể gồm:

- file;
- class;
- function/method;
- semantic chunk;
- variable/entity;
- import block;
- API usage;
- current cursor/left-context anchor.

### 8.2 Edge types

Edge có thể gồm:

- `contains`;
- `imports`;
- `calls`;
- `inherits`;
- `overrides`;
- `uses_type`;
- `defines`;
- `mentions`;
- `data_dependency`;
- `control_dependency`;
- `adjacent_chunk`.

### 8.3 So sánh với RLCoder và AlignCoder

| Phương pháp | Biểu diễn repo | Hạn chế |
|---|---|---|
| RLCoder | Natural candidates / flat snippets | Không có graph traversal |
| AlignCoder | Base snippets + dependency snippets | Có dependency context nhưng chưa là learned graph policy |
| Đề tài đề xuất | Heterogeneous chunk-entity graph | Retrieval dựa trên structural reasoning và dependency path |

---

## 9. Multi-hop RL Graph Traversal Policy

Retrieval được mô hình hóa như sequential decision making.

### 9.1 State

```text
state_t = {
  left_context_embedding,
  current_file,
  current_scope,
  imported_symbols,
  cursor_position,
  current_graph_node,
  visited_nodes,
  retrieval_path_so_far,
  quantized_semantic_state
}
```

### 9.2 Action

```text
action_t = {
  move_to_neighbor_node,
  select_node_as_context,
  stop
}
```

### 9.3 Reward

```text
reward = completion_quality_improvement
       + path_relevance_bonus
       - token_cost_penalty
       - redundancy_penalty
       - irrelevant_node_penalty
```

Ví dụ traversal:

```text
CursorNode
 → CurrentFunction
 → imported service
 → called repository method
 → database utility signature
 → STOP
```

Điểm mới:

- RLCoder học chọn candidate hữu ích.
- AlignCoder học dùng enhanced query để retrieve tốt hơn.
- Đề tài đề xuất học **đường đi retrieval** trên graph.

Đây là khác biệt quan trọng nhất.

---

## 10. Quantized Semantic States for Stable RL Retrieval

Sau khi encode graph bằng GNN, embedding liên tục được ánh xạ vào codebook rời rạc:

```math
z \in \mathbb{R}^{d}
```

```math
q(z) = e_k, \quad k \in \{1, 2, ..., K\}
```

Trong đó:

- `z`: graph embedding của node/function/class/chunk;
- `e_k`: codebook vector;
- `k`: semantic state id.

Các semantic states có thể biểu diễn vai trò như:

```text
Controller → Service → Repository → Utility
```

Lợi ích:

- giảm nhiễu của dense embedding;
- giúp RL policy học dễ hơn;
- tăng interpretability;
- gom các function có vai trò tương tự vào cùng semantic state;
- hỗ trợ traversal theo semantic role sequence.

So với AlignCoder:

| Thành phần | AlignCoder | Đề tài đề xuất |
|---|---|---|
| Query improvement | Multiple sampled completions | Left-context anchor + graph state + semantic role states |
| Embedding space | Continuous dense embedding | Continuous graph embedding + quantized/discrete states |
| RL target | Retrieve snippets phù hợp enhanced query | Learn graph traversal policy |
| Noise handling | Dựa vào nhiều samples và RL retriever | Discrete state abstraction + reward penalty |

---

## 11. Context Composer

Prompt cuối nên tách rõ **local left context** và **retrieved cross-file context**.

Không nên trộn retrieved context vào giữa immediate left context gần cursor.

Prompt assembly đề xuất:

```text
# Retrieved cross-file graph context
[File: repository/users.py]
def update_user(...):
    ...

[File: models/user.py]
class User:
    ...

# Current file context
[Path] src/routes/users.py

[Imports]
from src.repository import users as repository_users
from src.database.models import User

[Current scope]
async def update_user_route(...):

[Immediate left context]
user = await repository_users.

# Complete here
```

Context Composer có nhiệm vụ:

- giữ immediate left context;
- giữ imports/current scope quan trọng;
- thêm retrieved dependency path context;
- loại bỏ node trùng/lặp;
- tối ưu token budget;
- ưu tiên signature/docstring/usage pattern thay vì full body nếu token budget nhỏ.

---

## 12. Training strategy cho retrieval model

Không nên train RL từ đầu. Nên train theo 3 pha.

### Phase 1: Contrastive pretraining / SFT retriever

Mục tiêu: cho model biết query nào gần context nào trước.

Dữ liệu:

```text
q = left context quanh cursor
positive = chunk/function chứa gold dependency
negative = chunk/function không liên quan trong cùng repo
```

Loss:

```math
L_{contrastive}
=
-\log
\frac{\exp(sim(q,c^+)/\tau)}
{\exp(sim(q,c^+)/\tau)+\sum_{c^-}\exp(sim(q,c^-)/\tau)}
```

Initial feature có thể dùng:

- UniXcoder;
- CodeBERT;
- GraphCodeBERT;
- CodeT5 encoder.

### Phase 2: Train graph retriever/reranker

Quy trình:

```text
BM25 / UniXcoder lấy top-50 hoặc top-100 candidates
 ↓
Build local dependency subgraph quanh candidates
 ↓
GraphSAGE/GAT/GGNN encode subgraph
 ↓
Score từng node/chunk
```

Node feature:

```text
node_feature = code_text_embedding + structural_feature + type_embedding
```

Score:

```math
score(q, v) = q^T h_v
```

hoặc:

```math
score(q, v) = MLP([q; h_v; q \odot h_v])
```

### Phase 3: RL fine-tuning bằng completion reward

Action không phải chọn flat chunk nữa mà là chọn node/path trên graph.

```text
State:
  left context embedding
  current graph node
  retrieved path so far
  quantized semantic state

Action:
  chọn node kế tiếp trong graph
  select node as context
  STOP

Reward:
  PPL reduction của target code
  + bonus nếu hit gold dependency
  - penalty nếu context dài/nhiễu/trùng
```

Reward thực dụng:

```math
R = [PPL(target | q) - PPL(target | q, retrieved\_context)]
  + \lambda \cdot HitGold
  - \beta \cdot TokenCost
  - \gamma \cdot Redundancy
```

Policy loss đơn giản:

```math
L_{RL} = - R \sum_t \log \pi(a_t | s_t)
```

Có thể bắt đầu bằng REINFORCE/policy gradient, sau đó thử PPO.

### Action space giới hạn

Không cho RL chọn trên toàn repo.

```text
1. Coarse retrieval lấy top-N chunks.
2. Build subgraph 1-hop/2-hop quanh top-N theo import/call/contains edges.
3. RL chỉ chọn node trong subgraph này.
```

Điều này giúp action space nhỏ hơn và train ổn định hơn.

---

## 13. So sánh tổng thể với RLCoder và AlignCoder

| Tiêu chí | RLCoder | AlignCoder | Đề tài đề xuất |
|---|---|---|---|
| Mục tiêu chính | Train retriever không cần labeled data | Align query với target intent | Học graph traversal policy cho retrieval |
| Query | Unfinished code | Unfinished code + sampled completions | Left context + current scope + graph anchor state |
| Xử lý left context | Query/prompt context | Query + enhanced query | Preserved local context + query anchor + RL state |
| Chunking | Split-Aggregate natural candidates | Base snippets + dependency snippets | PPL-guided semantic entity chunks |
| Retrieval unit | Natural candidates | Base/dependency snippets | Chunk/entity graph nodes và dependency paths |
| Repository structure | Dependency analysis để tạo data/candidates | Dependency snippets từ import/entity extraction | Heterogeneous chunk-entity repository graph |
| RL role | Chọn useful candidate | Học tận dụng enhanced query | Học multi-hop traversal policy |
| Reward | Weighted PPL | PPL-based reward | PPL reduction + path quality + token cost + redundancy penalty |
| Retrieval style | Mostly single-step | Coarse-to-fine, enhanced query | Multi-hop graph traversal |
| Stop mechanism | Stop signal | Không phải trọng tâm | STOP action trong policy |
| Semantic abstraction | Dense candidate embedding | Dense enhanced query embedding | Quantized semantic graph states |
| Interpretability | Trung bình | Trung bình | Cao: visualize được retrieval path |
| Novelty gap | RL retriever | Query enhancement + RL | Semantic chunking + left-context anchoring + graph RL traversal + quantized states |

---

## 14. Công thức reward đề xuất

Reward tổng quát:

```math
R = \alpha \cdot Q_{completion}
  - \beta \cdot C_{token}
  - \gamma \cdot U_{retrieval}
  + \lambda \cdot S_{path}
```

Trong đó:

- `Q_completion`: chất lượng completion, có thể đo bằng EM, edit similarity hoặc PPL reduction;
- `C_token`: chi phí token context;
- `U_retrieval`: độ bất định/nhiễu của retrieval;
- `S_path`: điểm hợp lý của graph path, ví dụ path có call/import/inheritance relation trực tiếp.

Một reward thực dụng khi training:

```math
R = PPL_{no\_ctx} - PPL_{with\_retrieved\_path}
```

Thêm penalty:

```math
R' = R
   - \beta \cdot \text{TokenCost}
   - \gamma \cdot \text{IrrelevantNodePenalty}
   - \eta \cdot \text{RedundancyPenalty}
```

---

## 15. Điểm khác biệt có thể claim trong paper

### Claim 1

> Existing RL-based retrievers optimize snippet selection, while our method learns repository graph traversal policies for dependency-aware retrieval.

### Claim 2

> Unlike query-enhancement methods that rely on sampled completions to bridge the semantic gap, our method uses left-context anchoring and explicit repository structures to guide multi-hop retrieval.

### Claim 3

> PPL-guided semantic entity chunking creates graph nodes with better semantic continuity than fixed-window, function-level, or rule-based Split-Aggregate candidates.

### Claim 4

> Quantized semantic graph states provide a discrete abstraction of code roles, reducing retrieval instability in continuous embedding spaces and improving RL policy learning.

### Claim 5

> The proposed retrieval path is interpretable and can be visualized as dependency reasoning chains across repository entities.

---

## 16. Experimental design đề xuất

### 16.1 Benchmarks

Có thể dùng cùng benchmark với RLCoder và AlignCoder:

- CrossCodeEval Python;
- CrossCodeEval Java;
- RepoEval Line;
- RepoEval API.

### 16.2 Baselines

Nên so sánh với:

- No Retrieval;
- BM25;
- UniXcoder;
- RepoCoder;
- RLCoder;
- AlignCoder;
- Graph retrieval without RL;
- Graph retrieval without quantization;
- Graph retrieval without multi-hop;
- Graph retrieval without PPL-guided chunking;
- Graph retrieval without left-context anchoring.

### 16.3 Metrics

- Exact Match (EM);
- Edit Similarity (ES);
- Perplexity (PPL);
- Retrieval Recall@k;
- MRR;
- nDCG;
- Gold dependency hit rate;
- Path relevance score;
- Context token cost;
- Inference latency.

### 16.4 Dependency-heavy subset

Vì novelty chính là graph-structured multi-hop retrieval, cần báo cáo riêng trên subset cần dependency reasoning.

Một sample được coi là dependency-heavy nếu thỏa ít nhất 2/4 điều kiện:

```text
1. Target có ít nhất 1 API/function/class được định nghĩa ở file khác.
2. Shortest dependency path từ current file tới target symbol file ≥ 2.
3. Có từ 3 internal imports trở lên trong context.
4. BM25/UniXcoder không retrieve được gold dependency trong top-10.
```

Hoặc dùng dependency score:

```math
D(s) =
external\_target\_symbols
+ dep\_path\_len
+ involved\_files
+ baseline\_miss
```

Sau đó lấy top 25% samples theo `D(s)`.

Cần báo cáo:

| Subset | RLCoder | AlignCoder | Ours |
|---|---:|---:|---:|
| Easy | ... | ... | ... |
| Medium | ... | ... | ... |
| Dependency-heavy | ... | ... | ... |

Nếu overall tăng vừa phải nhưng dependency-heavy subset tăng mạnh, đây là bằng chứng trực tiếp cho novelty.

### 16.5 Ablation study

Các biến thể cần ablate:

| Variant | Mục tiêu |
|---|---|
| w/o PPL-guided chunking | Kiểm tra vai trò semantic chunking |
| w/o Left-context anchoring | Kiểm tra vai trò anchor/state từ left context |
| w/o Graph | Kiểm tra vai trò repository graph |
| w/o Multi-hop | Kiểm tra vai trò traversal nhiều bước |
| w/o Quantization | Kiểm tra vai trò quantized semantic states |
| w/o RL | Kiểm tra vai trò policy learning |
| w/o Path Reward | Kiểm tra reward theo dependency path |
| w/o Token Penalty | Kiểm tra adaptive context budget |
| GraphSAGE vs GAT vs GGNN | So sánh graph encoder |
| Hard VQ vs Soft VQ | So sánh cách quantization |

---

## 17. Rủi ro và cách giảm rủi ro

### Rủi ro 1: PPL-guided chunking tốn compute

Giải pháp:

- chỉ áp dụng PPL-based splitting cho entity dài;
- cache PPL/entropy offline;
- chỉ chạy trong candidate regions sau coarse retrieval;
- dùng model nhỏ như DeepSeekCoder-1B/CodeT5-small để tính PPL.

### Rủi ro 2: Graph construction tốn thời gian

Giải pháp:

- cache graph theo repository;
- parse AST/import/call offline;
- dùng incremental graph update;
- inference chỉ build local subgraph quanh top-N candidates.

### Rủi ro 3: GNN training khó scale

Giải pháp:

- dùng subgraph sampling;
- dùng GraphSAGE mini-batch;
- freeze code embedding;
- chỉ train GNN projection + VQ + policy head.

### Rủi ro 4: Quantization làm mất thông tin

Giải pháp:

- dùng soft vector quantization;
- residual quantization;
- combine continuous embedding + quantized state;
- ablate hard vs soft VQ.

### Rủi ro 5: RL khó hội tụ

Giải pháp:

- pretrain retriever bằng contrastive learning;
- warm-start policy bằng heuristic graph traversal;
- giới hạn action space bằng coarse retrieval;
- sau đó fine-tune bằng PPO hoặc policy gradient.

---

## 18. Phiên bản phương pháp khả thi để triển khai

### Phase 1: Parse và chunk

```text
Tree-sitter
 ↓
Extract file/class/function/method/block
 ↓
PPL-guided splitting cho entity dài
 ↓
Semantic chunks/entities + metadata
```

### Phase 2: Build graph

```text
Semantic chunks/entities
 ↓
Extract imports, calls, symbols, inheritance, type usage
 ↓
Build heterogeneous chunk-entity graph
```

### Phase 3: Local context extraction

```text
Left context
 ↓
Extract file path, imports, current scope, local variables, cursor position
 ↓
Create query anchor nodes/states
```

### Phase 4: Train graph encoder

```text
Initial code embeddings + structural features
 ↓
GraphSAGE/GAT/GGNN
 ↓
Graph-aware node embeddings
```

### Phase 5: Quantization

```text
Graph embeddings
 ↓
VQ / soft VQ / residual VQ
 ↓
Semantic state ids
```

### Phase 6: RL retrieval

```text
State: left context + current node + visited path + semantic state
Action: move to neighbor / select node / stop
Reward: PPL reduction + path relevance - token cost - redundancy
```

### Phase 7: Completion

```text
Preserved left context + retrieved cross-file graph context
 ↓
Prompt construction
 ↓
Code LLM completion
```

---

## 19. Tên đề tài gợi ý

### Option 1

**GraphCoder-RL: PPL-guided Semantic Chunking and Multi-hop Graph Retrieval for Repository-Level Code Completion**

### Option 2

**Learning Repository Graph Traversal Policies for Code Completion**

### Option 3

**Quantized Graph Reinforcement Retrieval for Repository-Level Code Completion**

### Option 4

**Structure-aware Reinforcement Retrieval over Repository Graphs for Code Completion**

### Option 5

**Left-Context Anchored Graph Retrieval for Repository-Level Code Completion**

---

## 20. Tóm tắt novelty cuối cùng

Novelty mạnh nhất của đề tài là:

> Chuyển repository-level retrieval từ bài toán ranking code snippets sang bài toán học chính sách traversal nhiều bước trên heterogeneous chunk-entity repository graph, trong đó left context là query anchor và semantic chunks/entities là graph nodes.

So với RLCoder:

- RLCoder dùng RL để chọn useful snippets.
- Đề tài dùng RL để học dependency-aware graph traversal path.
- RLCoder dùng Split-Aggregate rule-based candidates.
- Đề tài dùng PPL-guided semantic entity chunking để tạo graph nodes tốt hơn.

So với AlignCoder:

- AlignCoder dùng sampled completions để enhance query.
- Đề tài dùng left-context anchoring + repository graph + quantized semantic states để biểu diễn intent và dependency reasoning.
- AlignCoder retrieve snippets; đề tài retrieve dependency paths.

Đóng góp nổi bật:

1. **PPL-guided Semantic Entity Chunking** để tạo node retrieval có semantic continuity.
2. **Left-context anchoring** để biến local context thành state/anchor cho graph traversal.
3. **Heterogeneous chunk-entity repository graph** thay cho flat snippet set.
4. **Multi-hop RL traversal policy** thay cho single-step retriever/reranker.
5. **Quantized semantic states** để ổn định retrieval và biểu diễn semantic role.
6. **Interpretable dependency paths** cho repository-level code completion.
7. **Adaptive context construction** kết hợp preserved left context và retrieved cross-file graph context.

Một câu mô tả ngắn gọn:

> Đề tài đề xuất một framework retrieval mới cho repository-level code completion, trong đó retriever không chỉ học chọn snippets hữu ích như RLCoder hoặc align query như AlignCoder, mà học cách bắt đầu từ left context, di chuyển qua graph cấu trúc của repository, và lấy chuỗi dependency liên quan nhất cho completion.
