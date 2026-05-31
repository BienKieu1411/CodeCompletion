"""
Dataset Loader cho Github Repos (Train) & Eval Sets.
"""

import os
import json
import random
import logging
import pandas as pd
from typing import List, Dict, Optional, Literal, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Language config ───────────────────────────────────────────────────────────

EXTENSION_TO_LANGUAGE = {
    ".py": "python", ".java": "java", ".js": "javascript",
    ".ts": "typescript", ".cpp": "cpp", ".c": "c",
    ".go": "go", ".rb": "ruby",
}

# line  → câu lệnh đơn, luôn nằm trên 1 dòng
# block → thân if/for/def, có thể multiline
# mixed → gộp cả 2 
VALID_LINE_TYPES: Dict[str, Dict[str, set]] = {
    "python": {
        "line": {
            "expression_statement", "assignment", "augmented_assignment",
            "return_statement", "import_statement", "import_from_statement",
            "assert_statement", "delete_statement", "raise_statement",
            "pass_statement", "break_statement", "continue_statement",
            "global_statement", "nonlocal_statement",
            "call", "argument_list", "binary_operator", "comparison_operator",
            "boolean_operator", "conditional_expression", "subscript", "attribute",
            "string_content", "comment", "parameters",
            "dictionary", "list", "string", "tuple",
            "parenthesized_expression", "concatenated_string",
            "ERROR",
        },
        "block": {
            "if_statement", "for_statement", "while_statement",
            "with_statement", "try_statement",
            "except_clause", "else_clause", "elif_clause",
            "function_definition", "class_definition",
            "block",
            "list_comprehension", "dict_comprehension", "set_comprehension",
            "dictionary_comprehension",
            "generator_expression",
        },
        "mixed": set(),
    },
    "java": {
        "line": {
            "local_variable_declaration", "expression_statement",
            "return_statement", "throw_statement", "break_statement",
            "continue_statement", "assert_statement",
            "method_invocation", "argument_list", "variable_declarator",
            "binary_expression", "field_declaration",
            "explicit_constructor_invocation",
            "object_creation_expression",
            "switch_block_statement_group",
        },
        "block": {
            "if_statement", "for_statement", "enhanced_for_statement",
            "while_statement", "try_statement", "catch_clause",
            "switch_expression", "try_with_resources_statement",
            "block",
        },
        "mixed": set(),
    },
    "javascript": {
        "line": {
            "expression_statement", "variable_declaration", "return_statement",
            "throw_statement", "break_statement", "continue_statement",
            "import_statement", "export_statement",
            "call_expression", "assignment_expression", "await_expression",
        },
        "block": {
            "if_statement", "for_statement", "while_statement",
            "arrow_function", "function_declaration", "class_declaration",
            "try_statement", "switch_statement", "statement_block",
        },
        "mixed": set(),
    },
    "typescript": {
        "line": {
            "expression_statement", "variable_declaration", "return_statement",
            "throw_statement", "break_statement", "continue_statement",
            "import_statement", "call_expression",
        },
        "block": {
            "if_statement", "for_statement", "while_statement",
            "function_declaration", "class_declaration",
            "arrow_function", "statement_block",
        },
        "mixed": set(),
    },
    "go": {
        "line": {
            "expression_statement", "short_var_declaration", "var_declaration",
            "return_statement", "break_statement", "continue_statement",
            "inc_statement", "dec_statement", "send_statement",
        },
        "block": {
            "if_statement", "for_statement", "switch_statement",
            "select_statement", "go_statement", "defer_statement",
            "block",
        },
        "mixed": set(),
    },
    "cpp": {
        "line": {
            "expression_statement", "declaration", "return_statement",
            "break_statement", "continue_statement", "throw_statement",
        },
        "block": {
            "if_statement", "for_statement", "while_statement",
            "do_statement", "switch_statement", "try_statement",
            "compound_statement",
        },
        "mixed": set(),
    },
    "c": {
        "line": {
            "expression_statement", "declaration", "return_statement",
            "break_statement", "continue_statement",
        },
        "block": {
            "if_statement", "for_statement", "while_statement",
            "do_statement", "switch_statement", "compound_statement",
        },
        "mixed": set(),
    },
    "ruby": {
        "line": {
            "assignment", "return", "break", "next",
            "call", "method_call", "expression_statement",
        },
        "block": {
            "if", "unless", "while", "for", "until",
            "method", "class", "module", "do_block",
        },
        "mixed": set(),
    },
}

for _lang, _levels in VALID_LINE_TYPES.items():
    _levels["mixed"] = _levels["line"] | _levels["block"]


# ── Stratified sampling config ────────────────────────────────────────────────

DEFAULT_CUT_DISTRIBUTION: Dict[str, float] = {
    "line":  0.70,
    "block": 0.20,
    "mixed": 0.10,  
}

CutDistribution = Dict[str, float]  # {"line": p, "block": p, "mixed": p}


def _validate_distribution(dist: CutDistribution) -> CutDistribution:
    keys = {"line", "block", "mixed"}
    missing = keys - dist.keys()
    if missing:
        raise ValueError(f"cut_distribution thiếu key: {missing}")
    total = sum(dist[k] for k in keys)
    if total <= 0:
        raise ValueError("Tổng xác suất phải > 0.")
    return {k: dist[k] / total for k in keys}


def _sample_cut_level(dist: CutDistribution) -> Literal["line", "block", "mixed"]:
    r = random.random()
    cumulative = 0.0
    for level in ("line", "block", "mixed"):
        cumulative += dist[level]
        if r < cumulative:
            return level 
    return "mixed"


# ── Parser ────────────────────────────────────────────────────────────────────

_parsers: Dict[str, any] = {}


def _load_parser(language: str):
    try:
        import tree_sitter_languages
        return tree_sitter_languages.get_parser(language)
    except Exception:
        pass
    lang_modules = {
        "python": "tree_sitter_python", "java": "tree_sitter_java",
        "javascript": "tree_sitter_javascript", "typescript": "tree_sitter_typescript",
        "go": "tree_sitter_go", "cpp": "tree_sitter_cpp",
        "c": "tree_sitter_c", "ruby": "tree_sitter_ruby",
    }
    mod_name = lang_modules.get(language)
    if not mod_name:
        return None
    try:
        import importlib, tree_sitter
        mod = importlib.import_module(mod_name)
        ts_ver = tuple(int(x) for x in tree_sitter.__version__.split(".")[:2])
        if ts_ver >= (0, 21):
            lang_obj = tree_sitter.Language(mod.language())
            return tree_sitter.Parser(lang_obj)
        else:
            so_path = f"/tmp/ts_{language}.so"
            tree_sitter.Language.build_library(so_path, [os.path.dirname(mod.__file__)])
            lang_obj = tree_sitter.Language(so_path, language)
            p = tree_sitter.Parser()
            p.set_language(lang_obj)
            return p
    except Exception:
        return None


def get_parser(language: str):
    if language not in _parsers:
        p = _load_parser(language)
        if p is None:
            logger.warning(f"[!] No parser for '{language}', using random cut.")
        _parsers[language] = p
    return _parsers[language]


def detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "python")

def _normalize_newlines(text: str) -> str:
    # Chuẩn hoá CRLF/CR -> LF để tránh '\r' dính vào cuối dòng (Windows line endings)
    return text.replace("\r\n", "\n").replace("\r", "\n")

def _nonempty_lines(text: str) -> List[str]:
    return [ln for ln in (text or "").split("\n") if ln.strip() != ""]

def _is_comment_only(text: str, language: str) -> bool:
    lines = _nonempty_lines(text)
    if not lines:
        return False
    # Treat '#' as comment line too (robust across language detection / noisy data)
    if all(ln.lstrip().startswith("#") for ln in lines):
        return True
    # best-effort for C-like / others
    def is_comment_line(ln: str) -> bool:
        s = ln.lstrip()
        return s.startswith("//") or s.startswith("/*") or s.startswith("*") or s.startswith("*/")
    return all(is_comment_line(ln) for ln in lines)

def _is_import_only(text: str, language: str) -> bool:
    lines = _nonempty_lines(text)
    if not lines:
        return False
    lang = (language or "").lower()
    if lang == "python":
        return all(
            ln.lstrip().startswith("import ") or ln.lstrip().startswith("from ")
            for ln in lines
        )
    if lang in ("java",):
        return all(ln.lstrip().startswith("import ") for ln in lines)
    if lang in ("javascript", "typescript"):
        return all(ln.lstrip().startswith("import ") for ln in lines)
    return False

def _eligible_file_content(content: str, min_file_lines: int, min_file_chars: int) -> bool:
    if content is None:
        return False
    if len(content) < min_file_chars:
        return False
    if content.count("\n") + 1 < min_file_lines:
        return False
    return True


# ── AST cut ───────────────────────────────────────────────────────────────────

def _ast_cut(content: str, language: str,
             level: Literal["line", "block", "mixed"]) -> Optional[Tuple[int, int]]:
    parser = get_parser(language)
    if parser is None:
        return None

    valid_types = VALID_LINE_TYPES.get(language, {}).get(level, set())
    if not valid_types:
        return None

    try:
        tree = parser.parse(bytes(content, "utf8"))
        candidates = []

        total_lines = tree.root_node.end_point[0] + 1

        # Mục tiêu kép:
        #   1. Tránh left_context quá ngắn → ít nhất MIN_ABS_LINES dòng
        #   2. Đảm bảo model thấy import/class header → ít nhất MIN_REL_RATIO * total
        # Lấy MAX của hai để thoả cả hai điều kiện.
        # Giới hạn trên MAX_REL_RATIO để không loại quá nhiều candidates ở file dài.
        MIN_ABS_LINES  = 5      # tuyệt đối: luôn cần ít nhất 5 dòng context
        MIN_REL_RATIO  = 0.10   # tương đối: ít nhất 10% đầu file đã đi qua
        MAX_REL_RATIO  = 0.75   # tương đối: để lại ~25% cuối làm right_context

        min_start = max(MIN_ABS_LINES, int(total_lines * MIN_REL_RATIO))
        max_start = int(total_lines * MAX_REL_RATIO)

        # File quá ngắn: nới lỏng, chỉ giữ MIN_ABS_LINES
        if min_start >= max_start:
            min_start = MIN_ABS_LINES
            max_start = max(MIN_ABS_LINES + 1, total_lines - 1)

        def traverse(node):
            s = node.start_point[0]
            n_lines = node.end_point[0] - s + 1
            type_ok = node.type in valid_types

            if min_start <= s <= max_start:
                if level == "line":
                    if type_ok and n_lines == 1:
                        candidates.append(node)
                elif level == "block":
                    if type_ok and n_lines > 1:   # fix: không lấy block 1 dòng
                        candidates.append(node)
                else:  # mixed
                    if type_ok:
                        candidates.append(node)

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)

        if candidates:
            node = random.choice(candidates)
            return node.start_point[0], node.end_point[0]

    except Exception as e:
        logger.debug(f"AST cut lỗi ({language}): {e}")

    return None

# ── DataLoader ────────────────────────────────────────────────────────────────

class GraphFRLDataLoader:
    """
    Tải dữ liệu train từ github_repos và test từ các eval set.

    cut_distribution:
      Dict{"line": float, "block": float, "mixed": float} — xác suất chọn từng bucket khi completion_level="mixed" (hoặc không set).
      Nếu completion_level là "line"/"block" cụ thể, distribution bị bỏ qua và level cố định được dùng.
    """

    def __init__(
        self,
        dataset_path: str = "data/github_repos/python/train.parquet",
        max_crossfile_chars: int = 8000,
        use_fim: bool = True,
        completion_level: Literal["line", "block", "mixed"] = "mixed",
        cut_distribution: Optional[CutDistribution] = None,
        # Fixed train dataset (RAM-only, AlignCoder-style)
        fixed_train: bool = True,
        fixed_train_size: int = 2000,
        fixed_train_max_attempts: int = 20000,
        # Strict filtering / long-file preference
        min_file_lines: int = 200,
        min_file_chars: int = 2000,
        min_left_context_lines: int = 30,
        reject_comment_groundtruth: bool = True,
        reject_import_groundtruth: bool = True,
    ):
        self.dataset_path = dataset_path
        self.max_crossfile_chars = max_crossfile_chars
        self.use_fim = use_fim
        self.completion_level = completion_level
        self.all_repos: Optional[List[List[Dict]]] = None
        self.fixed_train = fixed_train
        self.fixed_train_size = fixed_train_size
        self.fixed_train_max_attempts = fixed_train_max_attempts
        self.min_file_lines = min_file_lines
        self.min_file_chars = min_file_chars
        self.min_left_context_lines = min_left_context_lines
        self.reject_comment_groundtruth = reject_comment_groundtruth
        self.reject_import_groundtruth = reject_import_groundtruth
        self.fixed_samples: Optional[List[Dict]] = None

        # Xử lý cut_distribution ─────────────────────────────────────────────
        if cut_distribution is not None:
            self.cut_distribution = _validate_distribution(cut_distribution)
        else:
            self.cut_distribution = dict(DEFAULT_CUT_DISTRIBUTION)

        if self.completion_level in ("line", "block"):
            self.cut_distribution = {
                "line":  1.0 if self.completion_level == "line"  else 0.0,
                "block": 1.0 if self.completion_level == "block" else 0.0,
                "mixed": 0.0,
            }
            logger.info(
                f"[DataLoader] completion_level='{self.completion_level}' "
                f"→ cut_distribution bị override thành fixed-level."
            )
        else:
            logger.info(
                f"[DataLoader] Stratified sampling: "
                f"line={self.cut_distribution['line']:.0%}, "
                f"block={self.cut_distribution['block']:.0%}, "
                f"mixed={self.cut_distribution['mixed']:.0%}"
            )

    # ── Load train ────────────────────────────────────────────────────────────

    def load_github_repos(self) -> List[List[Dict]]:
        """Nạp train.parquet, gom file thành repo theo cột 'first'."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Không tìm thấy: {self.dataset_path}")

        print(f"[+] Đang tải {self.dataset_path} ...")
        df = pd.read_parquet(self.dataset_path)

        all_repos, current = [], []
        for _, row in df.iterrows():
            if row.get("first", False):
                if len(current) >= 2:
                    all_repos.append(current)
                current = []
            current.append({"path": row["path"], "content": row["content"]})

        if len(current) >= 2:
            all_repos.append(current)

        print(f"[+] Đã gom {len(all_repos)} repos (>= 2 files).")
        return all_repos

    # ── Construct sample ──────────────────────────────────────────────────────

    def construct_train_sample_safe(
        self,
        repo_files: List[Dict],
        cut_distribution: Optional[CutDistribution] = None,
    ) -> Optional[Dict]:
        """
        Tạo 1 train sample từ repo với Stratified Mixed Sampling.

        Mỗi lần gọi, tự chọn completion level theo xác suất trong
        cut_distribution (instance hoặc override per-call), phản ánh
        phân phối thực tế của các tập test.

        Args:
            repo_files: Danh sách file trong repo.
            cut_distribution: Override distribution cho lần gọi này.
                Nếu None, dùng self.cut_distribution.

        Returns:
            Dict sample hoặc None nếu không tạo được.
        """
        dist = (
            _validate_distribution(cut_distribution)
            if cut_distribution is not None
            else self.cut_distribution
        )

        # Ưu tiên file dài để có context tốt (strict mode)
        eligible_repo_files = [
            f for f in repo_files
            if _eligible_file_content(
                _normalize_newlines(f.get("content", "")),
                min_file_lines=self.min_file_lines,
                min_file_chars=self.min_file_chars,
            )
        ]
        if eligible_repo_files:
            selected = random.choice(eligible_repo_files)
        else:
            selected = random.choice(repo_files)
        language = detect_language(selected["path"])
        content = _normalize_newlines(selected["content"])
        lines = content.split("\n")

        if len(lines) < 10:
            return None

        chosen_level = _sample_cut_level(dist)

        cut = _ast_cut(content, language, chosen_level)

        # Fallback: thử lần lượt các level khác nếu AST không ra kết quả
        if cut is None:
            fallback_order = [l for l in ("line", "block", "mixed") if l != chosen_level]
            for fallback_level in fallback_order:
                cut = _ast_cut(content, language, fallback_level)
                if cut:
                    logger.debug(
                        f"[DataLoader] AST fallback: {chosen_level} → {fallback_level} "
                        f"({language}, {selected['path']})"
                    )
                    break

        if cut:
            start_line, end_line = cut
        else:
            # Fallback cuối: random 1 dòng (Từ 50% đến 80% file)
            start_line = int(len(lines) * random.uniform(0.5, 0.8))
            end_line = start_line

        end_line = min(end_line, len(lines) - 1)

        sample = {
            "id":                selected["path"],
            "left_context":      "\n".join(lines[:start_line]),
            "right_context":     "\n".join(lines[end_line + 1:]) if self.use_fim else "",
            "ground_truth":      "\n".join(lines[start_line: end_line + 1]),
            "crossfile_context": {
                f["path"]: _normalize_newlines(f["content"])
                for f in repo_files if f["path"] != selected["path"]
            },
            # Metadata để debug/phân tích phân phối sau train
            "_cut_level":        chosen_level,
            "_n_lines":          end_line - start_line + 1,
        }

        if sample["_cut_level"] == "block" and sample["_n_lines"] == 1:
            logger.warning(f"[DataLoader] Block sample nhưng chỉ có 1 dòng (Fallback?): {selected['path']}")

        # Strict filters: loại sample xấu (comment/import/blank, left_context ngắn)
        gt = sample.get("ground_truth", "")
        if not gt or gt.strip() == "":
            return None
        if self.reject_comment_groundtruth and _is_comment_only(gt, language):
            return None
        if self.reject_import_groundtruth and _is_import_only(gt, language):
            return None
        if sample.get("left_context", "").count("\n") + 1 < self.min_left_context_lines:
            return None

        return sample

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def prepare_dataset(self):
        if self.all_repos is None:
            self.all_repos = self.load_github_repos()
            # Warmup parsers để phát hiện lỗi sớm
            for lang in VALID_LINE_TYPES:
                get_parser(lang)
            print(f"[+] Sẵn sàng: {len(self.all_repos)} repos.")

        if self.fixed_train and self.fixed_samples is None:
            # Pre-generate fixed train samples (RAM-only)
            assert self.all_repos is not None
            repos = self.all_repos

            fixed: List[Dict] = []
            attempts = 0

            # Pre-filter repos that have at least 2 files; also prefer repos with at least 1 eligible long file
            candidate_repos = []
            for repo_files in repos:
                if len(repo_files) < 2:
                    continue
                has_eligible = any(
                    _eligible_file_content(
                        _normalize_newlines(f.get("content", "")),
                        min_file_lines=self.min_file_lines,
                        min_file_chars=self.min_file_chars,
                    )
                    for f in repo_files
                )
                if has_eligible:
                    candidate_repos.append(repo_files)
            if not candidate_repos:
                candidate_repos = repos

            while len(fixed) < self.fixed_train_size and attempts < self.fixed_train_max_attempts:
                attempts += 1
                repo_files = random.choice(candidate_repos)
                sample = self.construct_train_sample_safe(repo_files)
                if sample is None:
                    continue
                fixed.append(sample)

            self.fixed_samples = fixed
            print(
                f"[+] Fixed-train: generated {len(self.fixed_samples)}/{self.fixed_train_size} samples "
                f"(attempts={attempts}, strict=True)"
            )

    def get_epoch_batches(self, batch_size: int = 4):
        """Yield batch theo epoch, shuffle đầu mỗi epoch."""
        if self.all_repos is None or (self.fixed_train and self.fixed_samples is None):
            self.prepare_dataset()

        if self.fixed_train:
            if not self.fixed_samples:
                return
            samples = list(self.fixed_samples)
            random.shuffle(samples)
            batch: List[Dict] = []
            for sample in samples:
                batch.append(sample)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
            return

        if not self.all_repos:
            return

        random.shuffle(self.all_repos)
        batch = []

        for repo_files in self.all_repos:
            sample = self.construct_train_sample_safe(repo_files)
            if sample:
                batch.append(sample)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    # ── Load test ─────────────────────────────────────────────────────────────

    def load_test_samples(
        self,
        dataset_name: str = "repoeval",
        language: str = "python",
        max_samples: int = 100,
    ) -> List[Dict]:
        """Nạp test/eval set. max_samples=0 để lấy toàn bộ."""
        if dataset_name == "repoeval_update":
            test_path = f"data/repoeval_update/{language}/test.parquet"
        else:
            test_path = f"data/{dataset_name}/test.parquet"

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Không tìm thấy: {test_path}")

        print(f"[+] Đang tải {test_path} ...")
        df = pd.read_parquet(test_path)
        samples = []

        for idx, row in df.iterrows():
            samples.append({
                "id":                row.get("task_id", f"task_{idx}"),
                "left_context":      row.get("left_context", ""),
                "right_context":     row.get("right_context", ""),
                "ground_truth":      row.get("groundtruth", row.get("target_code", "")),
                "crossfile_context": self._parse_crossfile(
                                         row.get("crossfile_context", {})),
            })
            if max_samples > 0 and len(samples) >= max_samples:
                break

        print(f"[+] Đã nạp {len(samples)} test samples.")
        return samples

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_crossfile(raw) -> Dict[str, str]:
        if isinstance(raw, str):
            try: raw = json.loads(raw)
            except: return {}
        if isinstance(raw, list):
            return {x["path"]: x["text"] for x in raw
                    if isinstance(x, dict) and "path" in x and "text" in x}
        if isinstance(raw, dict):
            return raw
        return {}

    def process_crossfile_context(
        self,
        crossfile_dict: Dict[str, str],
        max_chars: Optional[int] = None,
    ) -> str:
        """Đóng gói crossfile context thành chuỗi prompt, ưu tiên file ngắn."""
        max_chars = max_chars or self.max_crossfile_chars
        parts, total = [], 0

        for filename, content in sorted(crossfile_dict.items(), key=lambda kv: len(kv[1])):
            block = f"### File: {filename} ###\n{content}\n\n"
            if total + len(block) > max_chars:
                remaining = max_chars - total
                header = f"### File: {filename} ###\n"
                if remaining > len(header):
                    parts.append(f"{header}{content[:remaining - len(header)]}\n\n")
                break
            parts.append(block)
            total += len(block)

        return "".join(parts).strip()