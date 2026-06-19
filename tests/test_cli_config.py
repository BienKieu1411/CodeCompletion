import sys
import types

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
    assert captured["max_pairs_per_sample"] == 4
    assert captured["experiment_mode"] == "intent_main"
    assert captured["intent_mode"] == "static"
    assert captured["gate_mode"] == "learned"
    assert captured["adapter_type"] == "soft_prompt"
    assert captured["utility_margin"] == 0.05
    assert captured["leave_one_out_analysis_samples"] == 25
    assert captured["gate_quality_tolerance"] == 0.01
    assert captured["gate_retrieval_reduction_target"] == 0.20


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
        ]
    )
    args.func(args)

    assert captured["experiment_mode"] == "sequential_retriever_first"
    assert captured["leave_one_out_analysis_samples"] == 9
    assert captured["gate_quality_tolerance"] == 0.02
    assert captured["gate_retrieval_reduction_target"] == 0.3


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
