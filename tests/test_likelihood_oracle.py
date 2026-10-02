from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.train.context_contract import ContextConfig
from src.train.likelihood_oracle import LikelihoodOracle


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def response_for(request, value=-.25):
    return SimpleNamespace(ok=True, json=lambda: {"choices": [
        {"index": i, "prompt_logprobs": [None] + [
            {str(token): {"logprob": value}} for token in ids[1:]]}
        for i, ids in enumerate(request["prompt"])]})


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.oracle = LikelihoodOracle("http://localhost", None, "test", ContextConfig(), 2)

    def tearDown(self):
        self.oracle.session.close()

    def test_target_only_and_fixed_continuation_ids(self):
        def respond(url, json, timeout):
            result = response_for(json).json()
            for item, ids in zip(result["choices"], json["prompt"]):
                for pos in range(1, len(ids)-2):
                    item["prompt_logprobs"][pos][str(ids[pos])]["logprob"] = -99.
            return SimpleNamespace(ok=True, json=lambda: result)
        with patch.object(self.oracle.session, "post", side_effect=respond) as post:
            values = self.oracle.score(["prefix", "other-prefix"], [" x", " x"], Tokenizer())
        self.assertEqual(values, [-.25, -.25])
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["prompt"][0][-2:], request["prompt"][1][-2:])
        self.assertEqual(request["prompt_logprobs"], 0)
        self.assertEqual(request["max_tokens"], 1)

    def test_only_in_batch_dedup_no_persistent_label_cache(self):
        with patch.object(self.oracle.session, "post", side_effect=lambda u, json, timeout: response_for(json)) as post:
            self.assertEqual(self.oracle.score(["p", "p"], ["y", "y"], Tokenizer()), [-.25]*2)
            self.oracle.score(["p"], ["y"], Tokenizer())
            self.assertEqual(post.call_count, 2)

    def test_bad_response_and_limits_fail_loud(self):
        for data in ({"choices": []}, {"choices": [{"index": 0, "prompt_logprobs": [None, {}]}]}):
            with patch.object(self.oracle.session, "post", return_value=SimpleNamespace(ok=True, json=lambda: data)):
                with self.assertRaises(ValueError):
                    self.oracle.score(["p"], ["y"], Tokenizer())
        with self.assertRaises(ValueError):
            self.oracle.score(["prefix"], ["target"], Tokenizer(), max_model_len=5)
        with self.assertRaises(ValueError):
            self.oracle.score(["p"], [""], Tokenizer())
        with patch.object(self.oracle.session, "post", return_value=SimpleNamespace(ok=False, status_code=503, text="unavailable")):
            with self.assertRaisesRegex(RuntimeError, "503"):
                self.oracle.score(["p"], ["y"], Tokenizer())


if __name__ == "__main__":
    unittest.main()
