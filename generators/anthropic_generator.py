import os
from typing import Any

import anthropic

from core.base import BaseGenerator
from core.registry import register_generator

SYSTEM_PROMPT = "Answer using only the provided context."


@register_generator("anthropic")
class AnthropicGenerator(BaseGenerator):
    """Generator using Anthropic's Claude API."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "claude-3-5-sonnet-20241022")
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, question: str, context: list[str]) -> str:
        """Generate an answer based on question and context."""
        context_block = "\n\n".join(context)
        user_msg = f"Context:\n{context_block}\n\nQuestion: {question}"
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()
