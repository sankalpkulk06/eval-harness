import os
from typing import Any

from openai import OpenAI

from core.base import BaseGenerator
from core.registry import register_generator

SYSTEM_PROMPT = "Answer using only the provided context."


@register_generator("openai")
class OpenAIGenerator(BaseGenerator):
    """Generator using OpenAI's chat completion API."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "gpt-4o")
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, question: str, context: list[str]) -> str:
        """Generate an answer based on question and context."""
        context_block = "\n\n".join(context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {question}",
            },
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()
