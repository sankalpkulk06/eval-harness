import pytest

from core.base import BaseGenerator, BaseMetric, BaseRetriever, ResultRecord


class TestBaseRetriever:
    """Test BaseRetriever ABC enforcement."""

    def test_cannot_instantiate(self):
        """Verify BaseRetriever cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRetriever({})

    def test_concrete_impl(self):
        """Verify a concrete implementation works."""

        class ConcreteRetriever(BaseRetriever):
            def retrieve(self, question: str) -> list[str]:
                return ["context"]

        r = ConcreteRetriever({"key": "value"})
        assert r.retrieve("q") == ["context"]


class TestBaseGenerator:
    """Test BaseGenerator ABC enforcement."""

    def test_cannot_instantiate(self):
        """Verify BaseGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseGenerator({})

    def test_concrete_impl(self):
        """Verify a concrete implementation works."""

        class ConcreteGenerator(BaseGenerator):
            def generate(self, question: str, context: list[str]) -> str:
                return "answer"

        g = ConcreteGenerator({"key": "value"})
        assert g.generate("q", []) == "answer"


class TestBaseMetric:
    """Test BaseMetric ABC enforcement."""

    def test_cannot_instantiate(self):
        """Verify BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric({})

    def test_concrete_impl(self):
        """Verify a concrete implementation works."""

        class ConcreteMetric(BaseMetric):
            def score(self, question, answer, context, ground_truth):
                return 0.9

        m = ConcreteMetric({"key": "value"})
        assert m.score("q", "a", [], "gt") == 0.9


class TestResultRecord:
    """Test ResultRecord dataclass."""

    def test_creation(self):
        """Test ResultRecord can be created with defaults."""
        r = ResultRecord(
            question="q",
            ground_truth="gt",
            retrieved_context=["c1"],
            generated_answer="a",
        )
        assert r.question == "q"
        assert r.ground_truth == "gt"
        assert r.retrieved_context == ["c1"]
        assert r.generated_answer == "a"
        assert r.scores == {}
        assert r.latency_ms == 0.0

    def test_with_scores(self):
        """Test ResultRecord with scores."""
        r = ResultRecord(
            question="q",
            ground_truth="gt",
            retrieved_context=["c1"],
            generated_answer="a",
            scores={"metric1": 0.5},
            latency_ms=100.0,
        )
        assert r.scores == {"metric1": 0.5}
        assert r.latency_ms == 100.0
