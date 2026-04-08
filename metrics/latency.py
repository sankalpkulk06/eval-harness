from typing import Any

from core.base import BaseMetric
from core.registry import register_metric


@register_metric("latency")
class LatencyMetric(BaseMetric):
    """Metric that measures latency (retrieve + generate time)."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._latency_ms: float = 0.0

    def set_latency(self, ms: float) -> None:
        """Set the latency in milliseconds (called by runner before score)."""
        self._latency_ms = ms

    def score(self, question: str, answer: str, context: list[str], ground_truth: str) -> float:
        """Return latency in seconds as the score."""
        return self._latency_ms / 1000.0
