import time
from typing import Any

import yaml

from core import registry  # noqa: F401 - triggers registry initialization
from core.base import ResultRecord
from datasets.loader import DatasetLoader

# Force registration of all concrete classes
import generators.anthropic_generator  # noqa: F401
import generators.openai_generator  # noqa: F401
import metrics.latency  # noqa: F401
import metrics.llm_judge  # noqa: F401
import metrics.ragas_metrics  # noqa: F401
import retrievers.pgvector_retriever  # noqa: F401
import retrievers.pinecone_retriever  # noqa: F401


class EvalRunner:
    """Orchestrates evaluation runs against a RAG pipeline."""

    def __init__(self, config_path: str):
        """Initialize runner with YAML config.

        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def _build_retriever(self):
        """Build retriever from config."""
        cfg = self.config["retriever"]
        cls = registry.RETRIEVER_REGISTRY[cfg["type"]]
        return cls(cfg)

    def _build_generator(self):
        """Build generator from config."""
        cfg = self.config["generator"]
        cls = registry.GENERATOR_REGISTRY[cfg["type"]]
        return cls(cfg)

    def _build_metrics(self):
        """Build metrics from config."""
        instances = []
        for m_cfg in self.config.get("metrics", []):
            cls = registry.METRIC_REGISTRY[m_cfg["type"]]
            instances.append((m_cfg["type"], cls(m_cfg)))
        return instances

    def run(self, dataset_path: str) -> list[ResultRecord]:
        """Run evaluation on dataset.

        Args:
            dataset_path: Path to JSONL dataset file

        Returns:
            List of ResultRecord objects with scores
        """
        rows = DatasetLoader.load(dataset_path)
        retriever = self._build_retriever()
        generator = self._build_generator()
        metrics = self._build_metrics()
        results = []

        for row in rows:
            question = row["question"]
            ground_truth = row["ground_truth"]

            t0 = time.perf_counter()
            context = retriever.retrieve(question)
            answer = generator.generate(question, context)
            latency_ms = (time.perf_counter() - t0) * 1000

            scores = {}
            for metric_name, metric in metrics:
                # Special handling for latency metric
                if metric_name == "latency":
                    metric.set_latency(latency_ms)
                scores[metric_name] = metric.score(
                    question, answer, context, ground_truth
                )

            results.append(
                ResultRecord(
                    question=question,
                    ground_truth=ground_truth,
                    retrieved_context=context,
                    generated_answer=answer,
                    scores=scores,
                    latency_ms=latency_ms,
                )
            )

        return results
