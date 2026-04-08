import json
import tempfile
from pathlib import Path

from core.base import ResultRecord
from core.report import ReportBuilder


class TestReportBuilder:
    """Test ReportBuilder."""

    def test_to_json(self, sample_result):
        """Test JSON report generation."""
        results = [sample_result]
        builder = ReportBuilder(results, config_name="test_config")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = builder.to_json(tmpdir)
            assert Path(path).exists()

            data = json.loads(Path(path).read_text())
            assert data["config"] == "test_config"
            assert "run_id" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["question"] == "What is X?"
            assert data["summary"]["total_questions"] == 1
            assert data["summary"]["avg_latency_ms"] == 50.0

    def test_to_html(self, sample_result):
        """Test HTML report generation."""
        results = [sample_result]
        builder = ReportBuilder(results, config_name="test_config")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = builder.to_html(tmpdir)
            assert Path(path).exists()

            html = Path(path).read_text()
            assert "What is X?" in html
            assert "test_config" in html
            assert "50.0" in html  # latency

    def test_aggregate_metrics(self):
        """Test metric aggregation."""
        results = [
            ResultRecord(
                question="q1",
                ground_truth="gt1",
                retrieved_context=["c1"],
                generated_answer="a1",
                scores={"metric1": 0.8},
                latency_ms=100.0,
            ),
            ResultRecord(
                question="q2",
                ground_truth="gt2",
                retrieved_context=["c2"],
                generated_answer="a2",
                scores={"metric1": 0.6},
                latency_ms=200.0,
            ),
        ]
        builder = ReportBuilder(results, config_name="test")
        agg = builder._aggregate()

        assert agg["total_questions"] == 2
        assert agg["avg_latency_ms"] == 150.0
        assert agg["avg_scores"]["metric1"] == 0.7

    def test_multiple_metrics(self):
        """Test aggregation with multiple metrics."""
        results = [
            ResultRecord(
                question="q1",
                ground_truth="gt1",
                retrieved_context=["c1"],
                generated_answer="a1",
                scores={"metric1": 0.5, "metric2": 0.8},
                latency_ms=100.0,
            ),
        ]
        builder = ReportBuilder(results, config_name="test")
        agg = builder._aggregate()

        assert len(agg["avg_scores"]) == 2
        assert agg["avg_scores"]["metric1"] == 0.5
        assert agg["avg_scores"]["metric2"] == 0.8
