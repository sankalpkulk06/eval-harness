import os
from typing import Any

from openai import OpenAI

from core.base import BaseMetric
from core.registry import register_metric

JUDGE_SYSTEM = """You are an evaluation judge. Given a question, a ground truth answer,
and a generated answer, score the generated answer on a scale of 1 to 5 where:
1 = completely wrong, 5 = perfect. Respond with only a single integer."""


@register_metric("llm_judge")
class LLMJudgeMetric(BaseMetric):
    """Metric that uses GPT-4o as a judge to score answers."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "gpt-4o")
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def score(self, question: str, answer: str, context: list[str], ground_truth: str) -> float:
        """Score an answer using LLM judge, normalized to [0, 1]."""
        prompt = (
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"Generated answer: {answer}\n"
            "Score (1-5):"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=5,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            score = int(raw)
            score = max(1, min(5, score))  # clamp to [1, 5]
        except ValueError:
            score = 1
        return score / 5.0
