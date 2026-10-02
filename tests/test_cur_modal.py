import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.train import modal_online_utility as launch


class ModalLauncherTests(unittest.TestCase):
    def test_gain24_profile_uses_four_epoch_request_and_likelihood_prefill(self):
        args = launch.trainer_command(Path("/pool"), Path("/out"), 4, 10., Path("/stop"),
                                      False, Path("/valid"), "gain24")
        self.assertEqual(args[args.index("--algorithm")+1], "gain24")
        self.assertEqual(args[args.index("--epochs")+1], "4")
        self.assertIn("--valid", args)
        self.assertEqual(args[args.index("--valid-every-epochs")+1], "0")
        server = launch.vllm_command("gain24")
        self.assertIn("--no-enable-prefix-caching", server)
        self.assertNotIn("--enable-prefix-caching", server)
        self.assertEqual(server[server.index("--max-num-batched-tokens")+1], "8192")
        self.assertEqual(server[server.index("--max-num-seqs")+1], "128")
        self.assertEqual(server[server.index("--gpu-memory-utilization")+1], "0.35")

    def test_profile_bounds_nested_threads_and_reduces_cpu_reservation(self):
        source = Path(launch.__file__).read_text()
        self.assertIn('"TOKENIZERS_PARALLELISM": "true"', source)
        self.assertIn('"RAYON_NUM_THREADS": "2"', source)
        self.assertIn('"OMP_NUM_THREADS": "1"', source)
        self.assertIn('"MKL_NUM_THREADS": "1"', source)
        self.assertIn('gpu="A100-80GB", cpu=(2, 4)', source)
        self.assertIn('memory=(16384, 24576)', source)

    def test_validation_interval_zero_keeps_only_initial_baseline(self):
        from src.train.train_online_utility import should_validate_epoch
        self.assertTrue(should_validate_epoch(0, 0))
        for epoch in (1, 2, 4, 100):
            self.assertFalse(should_validate_epoch(epoch, 0))
        self.assertFalse(should_validate_epoch(1, 2))
        self.assertTrue(should_validate_epoch(2, 2))

    def test_image_includes_candidate_mining_dependency(self):
        import ast
        source = ast.parse(Path(launch.__file__).read_text())
        packages = [arg.value for node in ast.walk(source)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "pip_install"
                    for arg in node.args if isinstance(arg, ast.Constant)]
        self.assertIn("rank-bm25==0.2.2", packages)

    def test_default_pool_uses_moved_project_dataset(self):
        expected = Path(__file__).resolve().parents[1] / "dataset/data_prepared/train.parquet"
        self.assertEqual(launch.DEFAULT_POOL, expected)
        self.assertTrue(launch.DEFAULT_POOL.is_absolute())

    def test_profile_keeps_both_models_resident(self):
        serving = launch.vllm_command()
        self.assertEqual(serving[serving.index("--gpu-memory-utilization") + 1], "0.55")
        self.assertEqual(serving[serving.index("--max-num-seqs") + 1], "64")
        self.assertEqual(serving[serving.index("--host") + 1], "127.0.0.1")
        self.assertNotIn("--enable-sleep-mode", serving)
        training = launch.trainer_command(Path("/pool.parquet"), Path("/run"), 2, 123., Path("/stop"), True)
        self.assertIn("src.train.train_online_utility", training)
        self.assertEqual(training[training.index("--encoder-lr") + 1], "5e-5")
        self.assertEqual(training[training.index("--batch-size") + 1], "16")
        self.assertEqual(training[training.index("--encoder-microbatch") + 1], "64")
        self.assertEqual(training[training.index("--candidate-pool-size") + 1], "100")
        self.assertEqual(training[training.index("--generator-batch-size") + 1], "128")
        self.assertEqual(training[training.index("--resume") + 1], "/run/latest.pt")
        self.assertIn("--deadline-unix", training)
        self.assertIn("--checkpoint-volume", training)
        self.assertEqual(launch.HARD_TIMEOUT, 39600)
        self.assertEqual(launch.MAX_SOFT_HOURS, 10.75)
        self.assertEqual(float(training[training.index("--max-minutes") + 1]), 645.)

    def test_local_entrypoint_spawns_independent_remote_input(self):
        class Call:
            object_id = "fc-test-independent"

            def get(self):
                return {"status": "stopped", "checkpoint_sha256": None}

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(launch, "validate_pool", return_value={"rows": 1, "sha256": "a" * 64}), \
             patch.object(launch.train, "spawn", return_value=Call()) as spawn, \
             patch.object(launch.train, "remote") as remote:
            pool = Path(directory) / "train.parquet"
            pool.write_bytes(b"prepared")
            upload = Mock()
            upload_context = Mock()
            upload_context.__enter__ = Mock(return_value=upload)
            upload_context.__exit__ = Mock(return_value=False)
            with patch.object(launch.volume, "batch_upload", return_value=upload_context):
                launch.main("spawn-test", str(pool), 6, True, 1., directory)
            spawn.assert_called_once_with(
                "spawn-test", "a" * 64, 6, True, 1.,
                "7658522ea93f31ce86ee50b90162cbeae9e18bb32bec5256c001e977364a8582")
            remote.assert_not_called()

    def test_five_epoch_defaults_match_trainer_and_launcher(self):
        import ast
        online = ast.parse((launch.SOURCE_ROOT / "src/train/train_online_utility.py").read_text())
        defaults = {}
        for node in ast.walk(online):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant):
                    for keyword in node.keywords:
                        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                            defaults[node.args[0].value] = keyword.value.value
        self.assertEqual(defaults["--epochs"], 5)
        self.assertEqual(defaults["--batch-size"], 16)
        self.assertEqual(defaults["--encoder-microbatch"], 64)
        self.assertEqual(defaults["--candidate-pool-size"], 100)
        launcher = ast.parse(Path(launch.__file__).read_text())
        for node in launcher.body:
            if isinstance(node, ast.FunctionDef) and node.name in {"main", "train"}:
                pairs = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):], node.args.defaults))
                self.assertEqual(ast.literal_eval(pairs["epochs"]), 5)

    def test_bad_paths_and_runtime_rejected(self):
        for run_id in ("", "../other", "/", "hello/world", "has space"):
            with self.assertRaises(ValueError):
                launch.check_arguments(run_id, 2, 8.75)
        for hours in (0, -1, 11, 12, float("nan")):
            with self.assertRaises(ValueError):
                launch.check_arguments("test", 2, hours)
        launch.check_arguments("cur-online-001", 2, 8.75)

    def test_preflight_rejects_old_data_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pq.write_table(pa.table({"left_context": ["old"], "target": ["old"]}), path)
            with self.assertRaisesRegex(ValueError, "new ast_repo_pool_v3"):
                launch.validate_pool(path)

    def test_streamed_checkpoint_verified_and_not_overwritten(self):
        data = b"test checkpoint stream"
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(launch.volume, "read_file", return_value=iter([data[:5], data[5:]])) as read:
            path = launch.download_checkpoint("test-run", directory, digest)
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(path.suffix, ".pt")
            self.assertEqual(launch.download_checkpoint("test-run", directory, digest), path)
            self.assertEqual(read.call_count, 1)
            path.write_bytes(b"corrupt existing file")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                launch.download_checkpoint("test-run", directory, digest)

    def test_bad_download_retains_partial_without_publishing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(launch.volume, "read_file", return_value=iter([b"bad"])):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                launch.download_checkpoint("test-run", directory, "0" * 64)
            self.assertEqual(list(Path(directory).rglob("*.pt")), [])
            self.assertEqual(len(list(Path(directory).rglob("*.partial"))), 1)

    def test_shutdown_targets_only_owned_process_group(self):
        process = SimpleNamespace(pid=12345, wait=Mock(return_value=0))
        with patch.object(launch.os, "killpg") as kill:
            launch.stop_group(process)
        self.assertEqual([call.args[0] for call in kill.call_args_list], [12345, 12345])

    def test_server_start_failure_cleans_up_and_commits_status(self):
        server = SimpleNamespace(poll=lambda: 1)
        thread = SimpleNamespace(join=lambda **kwargs: None)
        digest = "1" * 64
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(launch, "MOUNT", Path(directory)), \
             patch.object(launch, "validate_pool", return_value={"sha256": digest}), \
             patch.object(launch, "gpu_snapshot", return_value="fake GPU"), \
             patch.object(launch.subprocess, "run", return_value=SimpleNamespace(stdout="pinned packages")), \
             patch.object(launch, "start_logged", return_value=(server, thread)), \
             patch.object(launch, "stop_group") as stop, \
             patch.object(launch.volume, "commit") as commit:
            result = launch.train.local("test", digest, soft_hours=.1)
            self.assertEqual(result["status"], "failed")
            self.assertIn("vLLM exited during startup", result["error"])
            self.assertIn(server, [call.args[0] for call in stop.call_args_list])
            self.assertGreaterEqual(commit.call_count, 2)
            status = json.loads((Path(directory) / "runs/test/session_result.json").read_text())
            self.assertIsNone(status["checkpoint_sha256"])

    def test_completed_run_stops_both_processes_and_returns_checkpoint_digest(self):
        server, worker = SimpleNamespace(poll=lambda: None), SimpleNamespace(poll=lambda: 0)
        thread = SimpleNamespace(join=lambda **kwargs: None)
        digest = "1" * 64
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs/test"

            def start(command, log_path, label):
                if label == "train":
                    (output / "latest.pt").write_bytes(b"checkpoint")
                    (output / "train_status.json").write_text(json.dumps({"status": "complete", "epoch": 3}))
                return (server if label == "vllm" else worker), thread

            with patch.object(launch, "MOUNT", Path(directory)), \
                 patch.object(launch, "validate_pool", return_value={"sha256": digest}), \
                 patch.object(launch, "gpu_snapshot", return_value="fake GPU"), \
                 patch.object(launch.subprocess, "run", return_value=SimpleNamespace(stdout="pinned packages")), \
                 patch.object(launch, "start_logged", side_effect=start), \
                 patch("requests.Session") as client, \
                 patch.object(launch, "stop_group") as stop, patch.object(launch.volume, "commit"):
                client.return_value.__enter__.return_value.get.return_value.ok = True
                result = launch.train.local("test", digest, soft_hours=.1)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["checkpoint_sha256"], hashlib.sha256(b"checkpoint").hexdigest())
            self.assertEqual([call.args[0] for call in stop.call_args_list], [worker, server])


if __name__ == "__main__":
    unittest.main()
