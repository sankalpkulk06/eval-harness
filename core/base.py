from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def retrieve(self, question: str) -> list[str]:
        """Return a list of context strings for the question."""
        ...


class BaseGenerator(ABC):
    """Abstract base class for generators."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self, question: str, context: list[str]) -> str:
        """Return the generated answer string."""
        ...


class BaseMetric(ABC):
    """Abstract base class for metrics."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def score(
        self,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str,
    ) -> float:
        """Return a float score, normalised to [0, 1]."""
        ...


@dataclass
class ResultRecord:
    """Result record for a single question."""

    question: str
    ground_truth: str
    retrieved_context: list[str]
    generated_answer: str
    scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
