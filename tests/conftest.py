import pytest
from unittest.mock import MagicMock, patch

from core.base import ResultRecord


@pytest.fixture
def sample_result():
    """Sample ResultRecord for testing."""
    return ResultRecord(
        question="What is X?",
        ground_truth="X is Y.",
        retrieved_context=["X is Y because Z."],
        generated_answer="X is Y.",
        scores={"latency": 0.05, "llm_judge": 0.8},
        latency_ms=50.0,
    )


@pytest.fixture
def mock_openai_embed():
    """Mock OpenAI embeddings client."""
    with patch("openai.OpenAI") as mock:
        instance = mock.return_value
        instance.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        yield instance


@pytest.fixture
def mock_openai_chat():
    """Mock OpenAI chat completions client."""
    with patch("openai.OpenAI") as mock:
        instance = mock.return_value
        instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Generated answer"))]
        )
        yield instance


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    with patch("anthropic.Anthropic") as mock:
        instance = mock.return_value
        instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Generated answer")]
        )
        yield instance
