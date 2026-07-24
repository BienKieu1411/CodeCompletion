import pytest

pandas = pytest.importorskip("pandas")
pd = pandas

import co_retrieval.data.repository_dataset_loader as loader_module
from co_retrieval.data.repository_dataset_loader import DatasetLoader
from co_retrieval.runner import _sample_to_training_sample, _split_by_repository
from co_retrieval.training import TrainingSample


def _sample(repo_id: str, file_path: str) -> TrainingSample:
    return TrainingSample(
        left_context="x",
        target="y",
        file_path=file_path,
        repo_id=repo_id,
    )


def test_loader_preserves_repository_boundaries_as_ids(tmp_path, monkeypatch):
    dataset_path = tmp_path / "train.parquet"
    dataset_path.touch()
    frame = pd.DataFrame(
        [
            {"first": True, "path": "src/a.py", "content": "a"},
            {"first": False, "path": "src/b.py", "content": "b"},
            {"first": True, "path": "pkg/c.py", "content": "c"},
            {"first": False, "path": "pkg/d.py", "content": "d"},
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    repos = DatasetLoader(dataset_path=str(dataset_path)).load_github_repos()

    assert len(repos) == 2
    assert {item["repo_id"] for item in repos[0]} == {"github_repo_000000"}
    assert {item["repo_id"] for item in repos[1]} == {"github_repo_000001"}


def test_loader_records_actual_cut_level_after_ast_fallback(monkeypatch, caplog):
    def fake_ast_cut(_content, _language, level):
        if level == "block":
            return None
        if level == "line":
            return (5, 5)
        raise AssertionError(f"unexpected level: {level}")

    monkeypatch.setattr(loader_module, "_sample_cut_level", lambda _dist: "block")
    monkeypatch.setattr(loader_module, "_ast_cut", fake_ast_cut)
    monkeypatch.setattr(loader_module.random, "choice", lambda items: items[0])

    loader = DatasetLoader(
        completion_level="mixed",
        min_file_lines=1,
        min_file_chars=1,
        min_left_context_lines=1,
    )
    repo_files = [
        {
            "path": "src/service.py",
            "content": "\n".join(f"line_{index}" for index in range(20)),
            "repo_id": "repo-a",
        },
        {"path": "src/helper.py", "content": "def helper(): pass"},
    ]

    sample = loader.construct_train_sample_safe(repo_files)

    assert sample is not None
    assert sample["_requested_cut_level"] == "block"
    assert sample["_cut_level"] == "line"
    assert sample["_n_lines"] == 1
    assert "Block sample nhưng chỉ có 1 dòng" not in caplog.text


def test_sample_conversion_turns_nan_fields_into_empty_text():
    class DummyChunker:
        def chunk_source(self, file_path, content):
            assert file_path == ""
            assert content == ""
            return []

    converted = _sample_to_training_sample(
        {
            "id": float("nan"),
            "task_id": float("nan"),
            "repo_id": float("nan"),
            "left_context": float("nan"),
            "ground_truth": float("nan"),
            "crossfile_context": {float("nan"): float("nan")},
        },
        DummyChunker(),
    )

    assert converted.left_context == ""
    assert converted.target == ""
    assert converted.file_path == ""
    assert converted.repo_id == ""
    assert converted.task_id == ""


def test_sample_conversion_prefixes_candidate_chunks_with_repo_id():
    class RecordingChunker:
        def __init__(self):
            self.paths = []

        def chunk_source(self, file_path, content):
            self.paths.append(file_path)
            return []

    chunker = RecordingChunker()
    _sample_to_training_sample(
        {
            "id": "current.py",
            "repo_id": "repo-a",
            "left_context": "x",
            "ground_truth": "y",
            "crossfile_context": {"src/utils.py": "def helper(): pass"},
        },
        chunker,
    )

    assert chunker.paths == ["repo-a/src/utils.py"]


def test_repository_split_is_disjoint_and_deterministic():
    samples = [
        _sample("repo-a", "src/a.py"),
        _sample("repo-a", "src/b.py"),
        _sample("repo-b", r"src\c.py"),
        _sample("repo-c", "pkg/d.py"),
    ]

    first = _split_by_repository(samples, 0.34, 100, 7)
    second = _split_by_repository(samples, 0.34, 100, 7)

    assert first == second
    train, eval_ = first
    assert {sample.repo_id for sample in train}.isdisjoint(
        {sample.repo_id for sample in eval_}
    )
    assert train and eval_


def test_repository_split_single_or_missing_repo_has_no_eval():
    one_repo = [_sample("repo-a", "a.py"), _sample("repo-a", "b.py")]
    assert _split_by_repository(one_repo, 0.5, 10, 1) == (one_repo, [])

    missing = [_sample("", "a.py"), _sample("repo-b", "b.py")]
    assert _split_by_repository(missing, 0.5, 10, 1) == (missing, [])


def test_repository_split_caps_eval_without_leaking_repositories():
    samples = [
        *[_sample("repo-a", f"a/{index}.py") for index in range(5)],
        _sample("repo-b", "b/one.py"),
        _sample("repo-c", "c/one.py"),
    ]
    train, eval_ = _split_by_repository(samples, 0.67, 1, 3)

    assert len(eval_) == 1
    assert {sample.repo_id for sample in train}.isdisjoint(
        {sample.repo_id for sample in eval_}
    )
