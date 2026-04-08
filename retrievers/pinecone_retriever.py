import os
from typing import Any

from openai import OpenAI
from pinecone import Pinecone

from core.base import BaseRetriever
from core.registry import register_retriever

EMBED_MODEL = "text-embedding-3-small"


@register_retriever("pinecone")
class PineconeRetriever(BaseRetriever):
    """Retriever using Pinecone vector database."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = pc.Index(config["index_name"])
        self.top_k = config.get("top_k", 3)
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _embed(self, text: str) -> list[float]:
        """Embed text using OpenAI embedding model."""
        resp = self._openai.embeddings.create(input=text, model=EMBED_MODEL)
        return resp.data[0].embedding

    def retrieve(self, question: str) -> list[str]:
        """Retrieve context documents for a question."""
        vector = self._embed(question)
        result = self.index.query(
            vector=vector, top_k=self.top_k, include_metadata=True
        )
        return [match.metadata.get("text", "") for match in result.matches]
