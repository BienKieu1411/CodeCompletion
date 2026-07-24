import sys
import types
import importlib

import pytest

from co_retrieval.cli.co_retrieval_cli import build_parser


def test_train_cli_defaults_respect_max_samples_and_neural_learning_rates(monkeypatch):
    captured = {}
    fake_runner = types.ModuleType("co_retrieval.runner")

    def train(config):
        captured.update(config)
        return {"status": "ok"}

    fake_runner.train = train
    monkeypatch.setitem(sys.modules, "co_retrieval.runner", fake_runner)

    args = build_parser().parse_args(["train", "--use-neural"])
    args.func(args)

    assert captured["fixed_train_size"] == 50
    assert captured["max_train_samples"] == 50
    assert captured["retriever_lr"] == 2e-5
    assert captured["gate_lr"] == 1e-4
    assert captured["soft_prompt_lr"] == 5e-3
    assert captured["eval_ratio"] == 0.1
    assert captured["max_eval_samples"] == 100
    assert captured["completion_level"] == "mixed"
    assert captured["max_pairs_per_sample"] == 4
    assert captured["experiment_mode"] == "intent_main"
    assert captured["intent_mode"] == "static"
    assert captured["gate_mode"] == "learned"
    assert captured["adapter_type"] == "soft_prompt"
    assert captured["generator_name"] == "deepseek-ai/deepseek-coder-6.7b-base"
    assert captured["utility_margin"] == 0.05
    assert captured["leave_one_out_analysis_samples"] == 25
    assert captured["gate_quality_tolerance"] == 0.01
    assert captured["gate_retrieval_reduction_target"] == 0.20
    assert captured["retriever_loss"] == "lipo"
    assert captured["lipo_tau"] == 1.0
    assert captured["skip_train_eval"] is False
    assert captured["train_epochs"] == 1
    assert captured["epoch_budget_mode"] is False
    assert captured["gate_use_retrieval_features"] is True
    assert captured["gate_context_cost_weight"] == 0.01
    assert captured["gate_context_cost_token_unit"] == 512
    assert captured["gate_decision_threshold"] == 0.5
    assert captured["gate_calibrate_threshold"] is True
    assert captured["gate_calibration_samples"] == 128
    assert captured["gate_calibration_retrieval_penalty"] == 0.05
    assert captured["steps_per_round_retriever"] is None
    assert captured["query_entropy_threshold"] == 0.8
    assert captured["query_num_drafts"] == 2


def test_train_cli_accepts_reviewer_blocker_neural_options(monkeypatch):
    captured = {}
    fake_runner = types.ModuleType("co_retrieval.runner")

    def train(config):
        captured.update(config)
        return {"status": "ok"}

    fake_runner.train = train
    monkeypatch.setitem(sys.modules, "co_retrieval.runner", fake_runner)

    args = build_parser().parse_args(
        [
            "train",
            "--use-neural",
            "--experiment-mode",
            "sequential_retriever_first",
            "--leave-one-out-analysis-samples",
            "9",
            "--gate-quality-tolerance",
            "0.02",
            "--gate-retrieval-reduction-target",
            "0.3",
            "--skip-train-eval",
            "--num-epochs",
            "10",
            "--train-epochs",
            "10",
            "--epoch-budget-mode",
            "--disable-gate-retrieval-features",
            "--gate-context-cost-weight",
            "0.02",
            "--gate-context-cost-token-unit",
            "256",
            "--gate-decision-threshold",
            "0.61",
            "--disable-gate-calibration",
            "--gate-calibration-samples",
            "33",
            "--gate-calibration-retrieval-penalty",
            "0.1",
        ]
    )
    args.func(args)

    assert captured["experiment_mode"] == "sequential_retriever_first"
    assert captured["leave_one_out_analysis_samples"] == 9
    assert captured["gate_quality_tolerance"] == 0.02
    assert captured["gate_retrieval_reduction_target"] == 0.3
    assert captured["skip_train_eval"] is True
    assert captured["train_epochs"] == 10
    assert captured["epoch_budget_mode"] is True
    assert captured["gate_use_retrieval_features"] is False
    assert captured["gate_context_cost_weight"] == 0.02
    assert captured["gate_context_cost_token_unit"] == 256
    assert captured["gate_decision_threshold"] == 0.61
    assert captured["gate_calibrate_threshold"] is False
    assert captured["gate_calibration_samples"] == 33
    assert captured["gate_calibration_retrieval_penalty"] == 0.1


def test_train_cli_accepts_lipo_and_cost_aware_options(monkeypatch):
    captured = {}
    fake_runner = types.ModuleType("co_retrieval.runner")

    def train(config):
        captured.update(config)
        return {"status": "ok"}

    fake_runner.train = train
    monkeypatch.setitem(sys.modules, "co_retrieval.runner", fake_runner)
    args = build_parser().parse_args(
        [
            "train",
            "--use-neural",
            "--intent-mode",
            "cost_aware",
            "--retriever-loss",
            "dpo",
            "--lipo-tau",
            "0.5",
            "--steps-per-round-retriever",
            "9",
            "--query-entropy-threshold",
            "0.7",
        ]
    )

    args.func(args)

    assert captured["intent_mode"] == "cost_aware"
    assert captured["retriever_loss"] == "dpo"
    assert captured["lipo_tau"] == 0.5
    assert captured["steps_per_round_retriever"] == 9
    assert captured["query_entropy_threshold"] == 0.7


def test_evaluate_cli_dispatches_neural_checkpoint_options(monkeypatch):
    captured = {}
    fake_runner = types.ModuleType("co_retrieval.runner")

    def evaluate(config):
        captured.update(config)
        return {"status": "ok"}

    fake_runner.evaluate = evaluate
    monkeypatch.setitem(sys.modules, "co_retrieval.runner", fake_runner)

    args = build_parser().parse_args(
        [
            "evaluate",
            "--dataset-path",
            "data/test.parquet",
            "--checkpoint-dir",
            "checkpoints/icar",
            "--output-dir",
            "results/icar_eval",
            "--max-samples",
            "17",
            "--gate-mode",
            "always_retrieve",
            "--gate-decision-threshold",
            "0.4",
            "--include-policy-variants",
        ]
    )

    args.func(args)

    assert captured["use_neural"] is True
    assert captured["dataset_path"] == "data/test.parquet"
    assert captured["checkpoint_dir"] == "checkpoints/icar"
    assert captured["output_dir"] == "results/icar_eval"
    assert captured["fixed_train_size"] == 17
    assert captured["max_train_samples"] == 17
    assert captured["max_eval_samples"] == 17
    assert captured["gate_mode"] == "always_retrieve"
    assert captured["gate_decision_threshold"] == 0.4
    assert captured["include_policy_variants"] is True


def test_runner_splits_comma_separated_dataset_paths():
    pytest.importorskip("pandas")
    runner = importlib.import_module("co_retrieval.runner")

    assert runner._split_dataset_paths("python/train.parquet,java/train.parquet") == [
        "python/train.parquet",
        "java/train.parquet",
    ]
    assert runner._split_dataset_paths(["a.parquet", "", "b.parquet"]) == [
        "a.parquet",
        "b.parquet",
    ]
    assert (
        runner._dataset_repo_prefix("/data/github_repos/java/train.parquet")
        == "github_repos_java"
    )


def test_train_cli_allows_explicit_proxy_overrides(monkeypatch):
    captured = {}
    fake_runner = types.ModuleType("co_retrieval.runner")

    def train(config):
        captured.update(config)
        return {"status": "ok"}

    fake_runner.train = train
    monkeypatch.setitem(sys.modules, "co_retrieval.runner", fake_runner)

    args = build_parser().parse_args(
        [
            "train",
            "--max-samples",
            "7",
            "--fixed-train-size",
            "11",
            "--max-train-samples",
            "13",
            "--retriever-lr",
            "0.3",
        ]
    )
    args.func(args)

    assert captured["fixed_train_size"] == 11
    assert captured["max_train_samples"] == 13
    assert captured["retriever_lr"] == 0.3
    assert captured["gate_lr"] == 0.1
    assert captured["soft_prompt_lr"] == 0.05
