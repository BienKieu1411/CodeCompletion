import unittest

from src.train.conditional_probes import matched_probe_graph
from src.train.context_contract import ContextConfig, ContextRenderer
from tests.test_cur_training import ToyTokenizer, task


def make_task(candidate_count, *, long_candidates=()):
    item = task()
    item["candidates"] = []
    for i in range(candidate_count):
        size = 500 if i in long_candidates else 24
        item["candidates"].append({
            "path": f"helper{i:03}.py", "start": 0, "end": size,
            "text": f"def helper{i}(): return " + ("x" * size),
            "bm25_score": float(candidate_count - i),
        })
    return item


class ConditionalProbeTests(unittest.TestCase):
    def test_growth_does_not_tokenize_whole_pool_at_every_pick(self):
        renderer = self.renderer(make_task(100))
        original = renderer.feasible_many
        multi_set_calls = []
        def counted(sets):
            if len(sets) > 1:
                multi_set_calls.append(len(sets))
            return original(sets)
        renderer.feasible_many = counted
        matched_probe_graph(renderer, 41, lambda state, choices: sorted(choices))
        # Only the four final-state eligibility scans inspect the full pool.
        self.assertEqual(len(multi_set_calls), 4)

    def renderer(self, item, *, prompt_tokens=5000, cross_file_tokens=4500,
                 max_snippets=10):
        return ContextRenderer(item, ToyTokenizer(),
                               ContextConfig(prompt_tokens, cross_file_tokens,
                                             max_snippets=max_snippets))

    def test_cap_exact_add_edges_and_shared_candidates(self):
        renderer = self.renderer(make_task(32))

        def rank(state, choices):
            return sorted(choices, key=lambda i: (i + len(state)) % 32)

        nodes, edges, audit = matched_probe_graph(renderer, 41, rank)
        self.assertEqual(len(nodes), 24)
        self.assertEqual(len(edges), 20)
        self.assertLessEqual(len(nodes), 24)
        self.assertEqual(len(nodes), len({tuple(node["indices"]) for node in nodes}))
        self.assertTrue(all(set(node) == {"indices", "cost"} for node in nodes))
        self.assertTrue(all(renderer.feasible(node["indices"]) for node in nodes))
        self.assertEqual(len(edges), len({(edge["source"], edge["target"]) for edge in edges}))
        self.assertLessEqual(len(edges), 20)
        for edge in edges:
            source = set(nodes[edge["source"]]["indices"])
            target = set(nodes[edge["target"]]["indices"])
            self.assertEqual(target - source, {edge["candidate"]})
            self.assertFalse(source - target)
            self.assertEqual(edge["state"], nodes[edge["source"]]["indices"])
            self.assertEqual(edge["group"], "add")
        self.assertEqual([state["goal_size"] for state in audit["states"]], [0, 3, 6, 9])
        self.assertEqual([state["actual_size"] for state in audit["states"]], [0, 3, 6, 9])
        self.assertEqual(audit["unique_contexts"], len(nodes))
        self.assertEqual(audit["unique_edges"], len(edges))
        self.assertEqual(audit["stop_edges"], 0)
        self.assertNotIn("stop", {edge["group"] for edge in edges})
        self.assertTrue(any(len(item["state_ids"]) >= 2
                            for item in audit["shared_probe_candidates"]))
        for state in audit["states"]:
            sources = {source for probe in state["probe_sources"]
                       for source in probe["sources"]}
            self.assertTrue({"model", "bm25", "random"}.issubset(sources))
            self.assertLessEqual(len(state["probe_candidates"]), 5)
        self.assertEqual({state["mode"] for state in audit["states"]}
                         & {"model", "bm25", "random"},
                         {"model", "bm25", "random"})

    def test_seed_determinism_and_target_independence(self):
        item = make_task(18)

        def rank(state, choices):
            return sorted(choices, key=lambda i: (-((i * 7 + len(state)) % 19), i))

        first = matched_probe_graph(self.renderer(item), 7, rank)
        changed = {**item, "target_code": "DO_NOT_READ", "right_context": "DO_NOT_READ"}
        second = matched_probe_graph(self.renderer(changed), 7, rank)
        self.assertEqual(first, second)
        self.assertEqual(first, matched_probe_graph(self.renderer(item), 7, rank))

    def test_short_pool_and_exact_budget_are_graceful(self):
        for count in range(5):
            renderer = self.renderer(make_task(count), prompt_tokens=700,
                                     cross_file_tokens=100)
            nodes, edges, audit = matched_probe_graph(renderer, 3)
            self.assertLessEqual(len(nodes), 24)
            self.assertLessEqual(len(edges), 20)
            self.assertTrue(all(renderer.feasible(node["indices"]) for node in nodes))
            self.assertEqual(audit["unique_contexts"], len(nodes))
            self.assertEqual(audit["unique_edges"], len(edges))
            self.assertEqual(audit["stop_edges"], 0)
            self.assertTrue(all(state["actual_size"] <= max(0, count - 1)
                                for state in audit["states"]))
            for state in audit["states"]:
                if state["feasible_additions"]:
                    self.assertTrue(any("random" in probe["sources"]
                                        for probe in state["probe_sources"]))

        renderer = self.renderer(make_task(12, long_candidates=range(4)),
                                 prompt_tokens=700, cross_file_tokens=100)
        nodes, edges, _ = matched_probe_graph(renderer, 3)
        self.assertTrue(all(renderer.feasible(node["indices"]) for node in nodes))
        self.assertFalse(any(any(i < 4 for i in node["indices"]) for node in nodes))


if __name__ == "__main__":
    unittest.main()
