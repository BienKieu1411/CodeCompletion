# RLCoder Retrieval Candidate Analysis — Insight Results

## Tổng quan hiệu suất

| Dataset | Samples | Exact Match Rate | Prefix Match Rate | Avg Edit Similarity | Correct |
|---|---|---|---|---|---|
| **cceval_java** | 2139 | 0.0% | 27.49% | 0.2886 | 0 |
| **cceval_python** | 2665 | 0.038% | 31.14% | 0.2578 | 1 |
| **repoeval_line** | 1600 | 0.0% | 50.06% | 0.3032 | 0 |
| **repoeval_api** | 1600 | 0.25% | 43.56% | 0.4288 | 4 |

> **CẢNH BÁO:** Tất cả dataset đều có exact match rate gần bằng 0. Model gần như không bao giờ sinh ra output chính xác hoàn toàn so với target.

---

## 1. CCEval Java — Phân tích 20 sample đầu tiên (điểm thấp nhất)

### Patterns chung quan sát được

**Nhận xét tổng quát:** Trong 20 sample đầu tiên (sắp theo edit_similarity tăng dần), phần lớn có edit_similarity = 0.0, nghĩa là model sinh ra output hoàn toàn sai lệch so với target.

#### Pattern 1: Model sinh ra chuỗi rỗng / newlines
- **Ví dụ:** `project_cc_java/181` (edit_sim=0.0), `project_cc_java/923` (edit_sim=0.0)
- **Pred:** 64 dòng trống `"\n\n\n...\n"`
- **Target:** `"= translator.detectLanguage(questionInput);"` và `"targetFilings = fullIndexRepository.findByFormType(filingType);"`
- **Candidates:** Rất relevant — `IBardTranslator.java` chứa chính xác method `detectLanguage()`, `FullIndexRepository.java` chứa các query methods.
- **Tại sao sai?** Model không generate được gì cả mặc dù candidates cung cấp đúng API. Có vẻ model bị "mất phương hướng" khi input quá dài hoặc context window bị overwhelm bởi nhiều candidates không liên quan trực tiếp.

#### Pattern 2: Model hallucinate code hoàn toàn khác
- **Ví dụ:** Nhiều task Java có prediction là code từ domain hoàn toàn khác
- **Candidates:** Relevant nhưng quá nhiều thông tin context, model không biết lọc ra phần cần thiết.

#### Pattern 3: Target rất ngắn nhưng model sinh quá nhiều
- **Nhiều target chỉ là 1 dòng** như `"= translator.detectLanguage(questionInput);"` nhưng model sinh ra 64+ dòng trống hoặc nhiều dòng code không liên quan.
- Model không nhận biết được rằng completion chỉ cần 1 statement ngắn.

#### Đánh giá chất lượng retrieval (Java):
- **Retrieval quality: TRUNG BÌNH - TỐT.** Hầu hết candidates đều chứa code từ cùng project, có class/method signatures liên quan.
- **Vấn đề chính:** Candidates thường là code "xung quanh" (sibling classes, test files, related utilities) nhưng không trực tiếp chứa đáp án.
- **Ví dụ tốt:** `IBardTranslator.java` chứa đúng `detectLanguage()` method — candidate này cực kỳ relevant.
- **Ví dụ kém:** Nhiều candidates chứa boilerplate code (test methods, controller endpoints) không giúp model tìm ra completion đúng.

---

## 2. CCEval Python — Phân tích 20 sample đầu tiên

### Patterns chung quan sát được

#### Pattern 1: Model hiểu sai context và sinh code cho domain khác
- **Ví dụ:** `project_cc_python/1007` (edit_sim=0.02)
  - Target: `"save(self.save_path)"` — gọi method save trên HeteroGraph object
  - Pred: `"y = truth_edges_mask\nhg.y = truth_edges_mask\nhg.y = truth_edges_mask..."` — lặp lại assignment
  - Candidates: Code from `hgraph.py`, `lightning_base.py` — relevant nhưng quá phức tạp, thuộc deep learning framework.
  - **Tại sao?** Model bị kẹt trong repetition loop, lặp lại pattern vừa thấy trong input thay vì tìm completion mới.

#### Pattern 2: Model sinh ra continuation dài thay vì single-line completion
- **Ví dụ:** `project_cc_python/7997` (edit_sim=0.026)
  - Target: `"create_process().run"` — chỉ cần 1 method chain ngắn
  - Pred: Sinh ra cả function definition `_run`, imports, file path comments...
  - Candidates: `test_TerminalTool.py` chứa đúng pattern `create_process()` → `run()` — rất relevant!
  - **Model thấy pattern đúng trong candidate nhưng không biết extract chỉ phần cần thiết.**

#### Pattern 3: Model không hiểu Python-specific patterns
- **Ví dụ:** `project_cc_python/9436` (edit_sim=0.028)
  - Target: `"__args__):"` — Python type checking pattern `isinstance(x, PathLike.__args__)`
  - Pred: Sinh ra `str):` rồi viết cả file reading logic
  - Candidates: Chứa file I/O patterns nhưng không có `PathLike.__args__` pattern cụ thể.
  - **Retrieval không tìm được example chính xác cho pattern hiếm này.**

#### Pattern 4: Deep Learning code quá chuyên biệt
- **Ví dụ:** `project_cc_python/2381` (edit_sim=0.037), `project_cc_python/1637` (edit_sim=0.038)
  - Target: `"db_k):"` — constructor parameter name, `"sizes])"` — SortDataset argument
  - Pred: Model sinh ra continuation code với sai class names hoặc sai arguments
  - Candidates: Code cùng domain (neural network modules, dataset loaders) nhưng từ files khác
  - **Retrieval tìm được code cùng domain nhưng từ file/project khác, không giúp model hiểu project-specific naming.**

#### Pattern 5: Model copy code structure từ candidate nhưng sai semantics
- **Ví dụ:** `project_cc_python/4878` (edit_sim=0.04)
  - Target: `"clear()"` — method call trên BulletList object
  - Pred: `"also_next())\n self.play(bullets.also_next())..."` — lặp lại pattern trước đó
  - Candidates: `bulletlist.py` chứa **cả** `also_next()` lẫn `clear()` methods — cực kỳ relevant!
  - **Model nhìn thấy `also_next()` pattern lặp lại trong input và tiếp tục lặp thay vì chuyển sang `clear()`.**

#### Đánh giá chất lượng retrieval (Python):
- **Retrieval quality: TRUNG BÌNH.** Candidates thường đúng domain (ML, NLP, etc.) nhưng từ files khác, không chứa đáp án trực tiếp.
- **Vấn đề chính:** Python code diversity rất lớn → retrieval dễ trả về code "tương tự nhưng khác project" thay vì code đúng project.
- **Điểm sáng:** Một số candidates chứa chính xác method/API cần dùng nhưng model vẫn không extract được.

---

## 3. RepoEval Line — Phân tích 20 sample đầu tiên

### Patterns chung quan sát được

**Đặc điểm dataset:** RepoEval line-level yêu cầu complete 1 dòng code. Exact match = 0.0% nhưng prefix match = 50.06% → model sinh ra prefix đúng nhưng không hoàn chỉnh dòng.

#### Pattern 1: Model sinh ra empty output
- **Ví dụ:** `huggingface_evaluate/194` (edit_sim=0.0)
  - Target: `"N\tCoreference\tCoreference chain information encoded in a parenthesis structure."`
  - Pred: 64 dòng trống `"\n\n\n...\n"` — model sinh ra content rỗng hoàn toàn
  - Candidates: Metric documentation files (`poseval.py`, `charcut_mt.py`) — **hoàn toàn không liên quan** đến CoNLL format documentation.
  - **Retrieval failure rõ ràng:** Không có candidate nào chứa coreference annotation format.

#### Pattern 2: Model sinh đúng prefix nhưng over-generate
- **Ví dụ:** `huggingface_evaluate/25` (edit_sim=0.01, prefix_match=1)
  - Target: `"}"` — chỉ cần đóng ngoặc citation block
  - Pred: `"\n}\n\"\"\"\n\nclass F1(Metric):..."` — sinh đúng `}` rồi tiếp tục viết class definition
  - Candidates: Các citation blocks tương tự từ `mse.py`, `mae.py`, etc. — **rất relevant** và giúp model biết format.
  - **Vấn đề:** Model không biết dừng lại ở completion point.

#### Pattern 3: Model hiểu sai task type — sinh class thay vì complete statement
- **Ví dụ:** `alibaba_FederatedScope/16` (edit_sim=0.028)
  - Input: `"from abc import ABC, abstractmethod\nimport numpy as np"` — chỉ 2 dòng import
  - Target: `"try:"` — bắt đầu try-except block
  - Pred: Sinh ra class definition `SecretSharing(ABC)` — hallucinate hoàn toàn
  - Candidates: Import statements từ các file khác — **không relevant** vì chỉ match trên import patterns.
  - **Vấn đề retrieval:** Input quá ngắn/generic → retrieval không thể tìm context hữu ích.

#### Pattern 4: Target là documentation/config nhưng retrieval trả về code
- **Ví dụ:** `huggingface_evaluate/83` (edit_sim=0.04)
  - Target: Mô tả dài về ROC AUC metric
  - Pred: Sinh mô tả ngắn rồi jump sang `_KWARGS_DESCRIPTION`
  - Candidates: License headers, metric files — relevant ở mức format nhưng không chứa ROC AUC description.
  - **Model không có đủ domain knowledge để viết documentation chính xác.**

#### Pattern 5: Retrieval trả về cùng template nhưng nhầm giá trị
- **Ví dụ:** `google_vizier/134` (edit_sim=0.051)
  - Target: `"    5,"` — giá trị cho flag `suggestion_count`
  - Pred: `"\n    1,\n    'Number of suggestions...'` — format đúng nhưng giá trị sai (1 thay vì 5)
  - Candidates: Algorithm configuration code — relevant nhưng không chứa default values.
  - **Model hiểu đúng pattern nhưng không biết giá trị cụ thể → "đoán" sai.**

#### Pattern 6: Model sinh tiếp tục ngoài completion cần thiết
- **Ví dụ:** `alibaba_FederatedScope/150` (edit_sim=0.054)
  - Target: `c.ServerApp.ip = "0.0.0.0"` — Jupyter config
  - Pred: Sinh config cho JupyterHub/JupyterLab thay vì ServerApp
  - Candidates: FederatedScope config files — **hoàn toàn không liên quan** đến Jupyter.
  - **Retrieval failure: Candidates từ sai domain hoàn toàn.**

#### Đánh giá chất lượng retrieval (Line):
- **Retrieval quality: YẾU - TRUNG BÌNH.** Nhiều candidates hoàn toàn không liên quan đến task cần complete.
- **Prefix match cao (50%)** cho thấy model có khả năng bắt đầu đúng nhưng sinh quá nhiều, chứng tỏ vấn đề chính là **stopping criterion** chứ không hoàn toàn là retrieval.

---

## 4. RepoEval API — Phân tích 20 sample đầu tiên

### Patterns chung quan sát được

**Đặc điểm dataset:** API-level completion yêu cầu sinh multi-line code blocks. Edit similarity trung bình cao nhất (0.4288) nhưng exact match vẫn rất thấp (0.25%).

#### Pattern 1: Model sinh ra code từ cùng repo nhưng wrong API call
- **Ví dụ:** `huggingface_diffusers/196` (edit_sim=0.004), `huggingface_diffusers/55` (edit_sim=0.004)
  - Target: `AttentionBlock(out_channels, num_head_channels=..., rescale_output_factor=..., eps=..., norm_num_groups=...)`
  - Pred: `DualTransformer2DModel(attn_num_head_channels, out_channels // attn_num_head_channels, ...)` — đúng domain nhưng sai API class
  - Candidates: Chứa cả `DualTransformer2DModel` (từ `modeling_text_unet.py`) — **retrieval trả về code misleading** khiến model chọn sai API.
  - **Vấn đề nghiêm trọng:** Candidate chứa code tương tự nhưng **từ version/variant khác** của model architecture, dẫn model đến API sai.

#### Pattern 2: Model copy code từ candidate thay vì compose đúng API
- **Ví dụ:** `huggingface_diffusers/129`, `huggingface_diffusers/113`, `huggingface_diffusers/179`
  - Target: `return self.__call__(prompt=prompt, ..., **kwargs)` — delegate call pattern
  - Pred: `self.check_inputs(prompt, height, width, callback_steps)` — copied từ candidate `sd_text2img_k_diffusion.py`
  - Candidates: Chứa **rất nhiều files tương tự** (10 pipeline files) — tất cả đều có pattern gần giống nhau.
  - **Model bị confused bởi quá nhiều candidates tương tự.** Nó pick sai pattern (implementation thay vì delegation).

#### Pattern 3: Multi-line targets quá dài và specific
- **Ví dụ:** `google_vizier/3` (edit_sim=0.023)
  - Target: 22 dòng `@parameterized.parameters(...)` decorator — rất specific test configuration
  - Pred: Sinh ra 1 test function thay vì decorator
  - Candidates: Có test files tương tự nhưng không chứa đúng parameterized test pattern cần thiết.
  - **Target quá dài cho code completion** → model không thể "đoán" chính xác.

#### Pattern 4: API-level completion cần hiểu semantic, không chỉ syntactic
- **Ví dụ:** `huggingface_evaluate/180` (edit_sim=0.008)
  - Target: `super().compute(model_or_pipeline=..., data=..., ...)` — cần gọi parent class method
  - Pred: Viết validation logic (`if model_or_pipeline is None: raise ValueError(...)`)
  - Candidates: Chứa evaluator classes khác nhưng không chứa `.compute()` delegation pattern.
  - **Model không hiểu OOP pattern (inheritance delegation).**

#### Pattern 5: Candidates quá tương tự nhau gây "confusion effect"
- **Ví dụ:** `huggingface_diffusers/113`, `huggingface_diffusers/179`
  - Cả 10 candidates đều là pipeline files với **gần như cùng docstring và structure**
  - Model thấy `self.check_inputs(...)` trong candidate và copy thay vì viết `return self.__call__(...)`
  - **Khi tất cả candidates đều giống nhau, model bị over-fit vào pattern phổ biến nhất trong candidates thay vì pattern đúng cho context hiện tại.**

#### Pattern 6: Model sinh ra file path comments trong predictions
- **Ví dụ:** `huggingface_diffusers/113`, `huggingface_diffusers/179`
  - Pred chứa: `"# file path: examples/community/sd_text2img_k_diffusion.py"` — model "rò rỉ" metadata từ candidate vào output 
  - **Model copy cả metadata (file path, line numbers) từ candidate format → chứng tỏ nó đang "copy" chứ không "tổng hợp" thông tin.**

#### Đánh giá chất lượng retrieval (API):
- **Retrieval quality: TRUNG BÌNH - YẾU trên API-level tasks.**
- **Vấn đề lớn nhất:** Quá nhiều candidates tương tự tạo "confusion effect" — model nhìn 10 pipeline files gần giống nhau và chọn sai variant.
- **Mặc dù edit similarity cao nhất** (0.43), model vẫn thường sinh ra API call sai hoặc arguments sai.

---

## 5. Tổng hợp nguyên nhân gốc rễ

### 5.1. Nguyên nhân từ Model

| Nguyên nhân | Tần suất | Mức độ ảnh hưởng |
|---|---|---|
| **Empty/newline generation** | Rất phổ biến (Java) | 🔴 Nghiêm trọng |
| **Repetition loop** | Phổ biến (Python) | 🔴 Nghiêm trọng |
| **Không biết dừng (over-generation)** | Rất phổ biến (Line, API) | 🔴 Nghiêm trọng |
| **Copy từ candidate thay vì compose** | Phổ biến (API) | 🟡 Trung bình |
| **Hallucinate class/method names** | Phổ biến | 🟡 Trung bình |
| **Không hiểu OOP patterns** | Thỉnh thoảng | 🟠 Trung bình |

### 5.2. Nguyên nhân từ Retrieval

| Nguyên nhân | Tần suất | Mức độ ảnh hưởng |
|---|---|---|
| **Candidates tương tự nhưng sai variant** | Rất phổ biến (API) | 🔴 Nghiêm trọng |
| **Candidates đúng domain nhưng sai project** | Phổ biến (Python) | 🟡 Trung bình |
| **Candidates hoàn toàn không liên quan** | Thỉnh thoảng (Line) | 🟠 Trung bình |
| **Input quá ngắn → retrieval không hiệu quả** | Thỉnh thoảng | 🟠 Trung bình |
| **Thiếu candidate chứa đúng API/pattern cần thiết** | Phổ biến | 🟡 Trung bình |

### 5.3. Nguyên nhân từ Task Design

| Nguyên nhân | Tần suất | Mức độ ảnh hưởng |
|---|---|---|
| **Target quá dài/specific (>5 lines)** | Phổ biến (API) | 🟡 Trung bình |
| **Target là non-code content (docs, configs)** | Thỉnh thoảng (Line) | 🟠 Thấp |
| **Target chứa project-specific naming** | Rất phổ biến | 🔴 Nghiêm trọng |

---

## 6. Hướng khắc phục đề xuất

### 6.1. Cải thiện Model Generation

#### A. Thêm Stop Token / Length Control
- **Vấn đề:** Model thường sinh quá nhiều output (đặc biệt line-level tasks).
- **Giải pháp:**
  - Train model với explicit stop tokens cho line-level vs. block-level completion
  - Sử dụng length penalty trong decoding để ưu tiên completions ngắn hơn
  - Post-processing: cắt output ở newline đầu tiên cho line-level tasks

#### B. Giải quyết Empty Generation
- **Vấn đề:** Model sinh ra chuỗi newlines trống (Java tasks).
- **Giải pháp:**
  - Thêm minimum length constraint
  - Fallback mechanism: nếu output toàn whitespace → retry hoặc dùng candidates trực tiếp
  - Kiểm tra temperature/sampling parameters — có thể model confidence quá thấp

#### C. Anti-Repetition
- **Vấn đề:** Model bị kẹt lặp lại patterns (Python tasks).
- **Giải pháp:**
  - Repetition penalty trong decoding
  - N-gram blocking để ngăn lặp lại chuỗi >3 tokens

### 6.2. Cải thiện Retrieval Quality

#### A. Context-Aware Retrieval Filtering
- **Vấn đề:** Quá nhiều candidates tương tự gây confusion (API tasks).
- **Giải pháp:**
  - Diversity filtering: đảm bảo candidates đến từ files khác nhau, không quá trùng lặp
  - Rank candidates theo relevance score và chỉ lấy top-K diverse candidates
  - Ưu tiên candidates từ **cùng file** hoặc **cùng class** với completion point

#### B. Cải thiện Candidate Ranking
- **Giải pháp:**
  - Reranking candidates dựa trên semantic similarity thay vì chỉ lexical matching
  - Fine-tune retriever trên code completion tasks với RL reward signal (đúng ý tưởng RLCoder!)
  - Penalize candidates từ test files khi completing production code (và ngược lại)

#### C. Xử lý Input Ngắn
- **Vấn đề:** Input quá ngắn (2 dòng import) → retrieval không hiệu quả.
- **Giải pháp:**
  - Expand context: đưa thêm context từ file hiện tại (trước và sau completion point)
  - Sử dụng file-level context (filename, class hierarchy) để improve retrieval
  - Fallback: khi input ngắn, skip retrieval và dùng model's parametric knowledge

### 6.3. Cải thiện Reward Signal (RL-specific)

#### A. Multi-Granularity Rewards
- **Hiện tại:** Chỉ dùng exact match (binary reward).
- **Đề xuất:**
  - Dùng edit similarity làm continuous reward
  - Prefix match reward: bonus nếu model sinh đúng prefix
  - Partial credit: reward dựa trên số tokens đúng

#### B. Candidate Selection Reward
- **Đề xuất:**
  - Train RL agent để **chọn subset candidates tốt nhất** thay vì dùng tất cả
  - Reward signal: so sánh performance với vs. không có candidate → candidate nào improve prediction → reward cao

#### C. Length-Aware Reward
- **Đề xuất:**
  - Penalty cho output quá dài so với target
  - Bonus cho stopping ở đúng boundary
  - Separate reward cho line-level vs. block-level tasks

### 6.4. Post-Processing Pipeline

```
Raw Output → Truncation (by task type) → Deduplication → Whitespace cleanup → Final Output
```

1. **Line-level:** Cắt tại dấu `\n` đầu tiên
2. **API-level:** Cắt tại dấu kết thúc block (matching brackets/parentheses)
3. **Remove** redundant empty lines
4. **Strip** trailing whitespace

---

## 7. Kết luận

> **QUAN TRỌNG:** Vấn đề chính KHÔNG phải là retrieval quality (candidates thường đủ relevant). Vấn đề nằm ở:
> 1. **Model không biết SỬ DỤNG candidates** — thấy đáp án trong context nhưng không extract được
> 2. **Model không biết DỪNG** — sinh ra quá nhiều content khi chỉ cần 1 dòng
> 3. **Model bị CONFUSED bởi nhiều candidates tương tự** — pick sai variant/pattern
> 4. **Model bị "câm" (empty generation)** trên một số tasks, đặc biệt Java

**Ưu tiên khắc phục:**
1. 🔴 **P0:** Fix empty generation + stop criterion → tăng ngay lập tức edit_similarity
2. 🟡 **P1:** Improve candidate diversity → giảm confusion effect
3. 🟠 **P2:** Multi-granularity RL reward → train model sử dụng context tốt hơn
