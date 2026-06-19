from co_retrieval.chunking import CodeChunk
from co_retrieval.context_utility import ContextCandidate, ContextUtilityScorer
from co_retrieval.intent import IntentSketcher


class FakeLoss:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeGenerator:
    def teacher_forcing_nll(
        self,
        left_context,
        target,
        retrieved_chunks=None,
        use_soft_prompt=True,
    ):
        if not retrieved_chunks:
            return FakeLoss(10.0)
        if any("helpful" in chunk.defined_symbols for chunk in retrieved_chunks):
            return FakeLoss(4.0)
        return FakeLoss(12.0)


def _chunk(symbol):
    return CodeChunk(
        file_path="service.py",
        start_line=1,
        end_line=1,
        chunk_type="function",
        text=f"def {symbol}(): pass",
        defined_symbols=[symbol],
    )


def test_intent_sketcher_extracts_member_and_symbol_hints():
    left_context = """from services import UserService

service = UserService()
result = service.fetch_"""

    sketch = IntentSketcher().build(left_context)

    assert sketch.prefix == "fetch_"
    assert sketch.member_owner == "service"
    assert sketch.member_prefix == "fetch_"
    assert "UserService" in sketch.class_hints
    assert "services" in sketch.import_hints
    assert "member_owner: service" in sketch.query


def test_context_utility_scores_nll_improvement_over_stop():
    scorer = ContextUtilityScorer(FakeGenerator())
    helpful = _chunk("helpful")
    noisy = _chunk("noisy")

    scores = scorer.score(
        "result = service.fetch_",
        "fetch_user()",
        [
            ContextCandidate("stop", [], is_stop=True),
            ContextCandidate("helpful", [helpful]),
            ContextCandidate("noisy", [noisy]),
        ],
    )

    by_name = {score.name: score for score in scores}
    assert by_name["stop"].utility == 0.0
    assert by_name["helpful"].utility == 6.0
    assert by_name["noisy"].utility == -2.0
    assert scores[0].name == "helpful"
