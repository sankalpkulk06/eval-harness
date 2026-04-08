import asyncio
import concurrent.futures
from typing import Any

from ragas import SingleTurnSample
from ragas.metrics import AnswerRelevancy, Faithfulness

from core.base import BaseMetric
from core.registry import register_metric


def _run_async(coro):
    """Run an async coroutine in a new event loop (safe for non-async callers)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@register_metric("ragas_faithfulness")
class RagasFaithfulnessMetric(BaseMetric):
    """RAGAS faithfulness metric - measures how faithful the answer is to context."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._metric = Faithfulness()

    def score(self, question: str, answer: str, context: list[str], ground_truth: str) -> float:
        """Score faithfulness using RAGAS metric."""
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=context,
            reference=ground_truth,
        )
        result = _run_async(self._metric.single_turn_ascore(sample))
        return float(result) if result is not None else 0.0


@register_metric("ragas_answer_relevancy")
class RagasAnswerRelevancyMetric(BaseMetric):
    """RAGAS answer relevancy metric - measures how relevant the answer is to the question."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._metric = AnswerRelevancy()

    def score(self, question: str, answer: str, context: list[str], ground_truth: str) -> float:
        """Score answer relevancy using RAGAS metric."""
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=context,
            reference=ground_truth,
        )
        result = _run_async(self._metric.single_turn_ascore(sample))
        return float(result) if result is not None else 0.0
