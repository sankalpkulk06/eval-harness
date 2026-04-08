import os
from typing import Any

import psycopg2
from openai import OpenAI

from core.base import BaseRetriever
from core.registry import register_retriever

EMBED_MODEL = "text-embedding-3-small"


@register_retriever("pgvector")
class PgvectorRetriever(BaseRetriever):
    """Retriever using PostgreSQL with pgvector extension."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.top_k = config.get("top_k", 3)
        self.table = config.get("table", "documents")
        self._conn = psycopg2.connect(os.environ["PGVECTOR_DSN"])
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _embed(self, text: str) -> list[float]:
        """Embed text using OpenAI embedding model."""
        resp = self._openai.embeddings.create(input=text, model=EMBED_MODEL)
        return resp.data[0].embedding

    def retrieve(self, question: str) -> list[str]:
        """Retrieve context documents for a question using cosine similarity."""
        vector = self._embed(question)
        # pgvector cosine distance operator <=>
        sql = f"""
            SELECT content
            FROM {self.table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (vector, self.top_k))
            rows = cur.fetchall()
        return [row[0] for row in rows]
