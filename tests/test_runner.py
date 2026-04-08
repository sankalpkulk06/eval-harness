import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.runner import EvalRunner


class FakeRetriever:
    """Fake retriever for testing."""

    def __init__(self, config):
        self.config = config

    def retrieve(self, question: str) -> list[str]:
        return ["context from fake retriever"]


class FakeGenerator:
    """Fake generator for testing."""

    def __init__(self, config):
        self.config = config

    def generate(self, question: str, context: list[str]) -> str:
        return "fake generated answer"


class FakeMetric:
    """Fake metric for testing."""

    def __init__(self, config):
        self.config = config

    def score(self, question, answer, context, ground_truth):
        return 0.75


class TestEvalRunner:
    """Test EvalRunner orchestration."""

    def test_run_integration(self):
        """Test end-to-end runner with fake components."""
        # Create a test config file
        config = {
            "retriever": {"type": "fake", "setting": "value"},
            "generator": {"type": "fake"},
            "metrics": [{"type": "fake"}],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            import yaml

            yaml.dump(config, f)
            config_path = f.name

        # Create a test dataset file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write(
                json.dumps({"question": "q1", "ground_truth": "gt1"})
                + "\n"
            )
            dataset_path = f.name

        try:
            with patch("core.registry.RETRIEVER_REGISTRY", {"fake": FakeRetriever}), patch(
                "core.registry.GENERATOR_REGISTRY", {"fake": FakeGenerator}
            ), patch("core.registry.METRIC_REGISTRY", {"fake": FakeMetric}):
                runner = EvalRunner(config_path)
                results = runner.run(dataset_path)

                assert len(results) == 1
                r = results[0]
                assert r.question == "q1"
                assert r.ground_truth == "gt1"
                assert r.retrieved_context == ["context from fake retriever"]
                assert r.generated_answer == "fake generated answer"
                assert "fake" in r.scores
                assert r.scores["fake"] == 0.75
                assert r.latency_ms > 0
        finally:
            Path(config_path).unlink()
            Path(dataset_path).unlink()
