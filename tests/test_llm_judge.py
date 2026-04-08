from unittest.mock import MagicMock, patch

from metrics.llm_judge import LLMJudgeMetric


class TestLLMJudgeMetric:
    """Test LLMJudgeMetric."""

    @patch("metrics.llm_judge.OpenAI")
    def test_score_valid(self, mock_openai_class):
        """Test scoring with valid response."""
        mock_instance = MagicMock()
        mock_openai_class.return_value = mock_instance

        # Mock response with score 4
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="4"))]
        )

        metric = LLMJudgeMetric({})
        score = metric.score("q", "a", ["c"], "gt")
        assert score == 0.8  # 4/5

    @patch("metrics.llm_judge.OpenAI")
    def test_score_clamp_high(self, mock_openai_class):
        """Test clamping of high scores."""
        mock_instance = MagicMock()
        mock_openai_class.return_value = mock_instance

        # Mock response with score 9 (out of range)
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="9"))]
        )

        metric = LLMJudgeMetric({})
        score = metric.score("q", "a", ["c"], "gt")
        assert score == 1.0  # clamped to 5, then /5

    @patch("metrics.llm_judge.OpenAI")
    def test_score_clamp_low(self, mock_openai_class):
        """Test clamping of low scores."""
        mock_instance = MagicMock()
        mock_openai_class.return_value = mock_instance

        # Mock response with score -5 (out of range)
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="-5"))]
        )

        metric = LLMJudgeMetric({})
        score = metric.score("q", "a", ["c"], "gt")
        assert score == 0.2  # clamped to 1, then /5

    @patch("metrics.llm_judge.OpenAI")
    def test_score_parse_error(self, mock_openai_class):
        """Test handling of non-numeric response."""
        mock_instance = MagicMock()
        mock_openai_class.return_value = mock_instance

        # Mock response with non-numeric content
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="bad"))]
        )

        metric = LLMJudgeMetric({})
        score = metric.score("q", "a", ["c"], "gt")
        assert score == 0.2  # default to 1, then /5

    @patch("metrics.llm_judge.OpenAI")
    def test_model_config(self, mock_openai_class):
        """Test that model config is respected."""
        mock_instance = MagicMock()
        mock_openai_class.return_value = mock_instance
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="3"))]
        )

        metric = LLMJudgeMetric({"model": "custom-model"})
        metric.score("q", "a", ["c"], "gt")

        # Verify the model was used
        call_args = mock_instance.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "custom-model"
