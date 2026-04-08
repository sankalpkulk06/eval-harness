import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from core.base import ResultRecord


class ReportBuilder:
    """Generates JSON and HTML reports from evaluation results."""

    def __init__(self, results: list[ResultRecord], config_name: str):
        """Initialize report builder.

        Args:
            results: List of ResultRecord objects
            config_name: Name of the config used for this run
        """
        self.results = results
        self.config_name = config_name
        self.run_id = (
            f"{config_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
            f"_{uuid.uuid4().hex[:6]}"
        )

    def _aggregate(self) -> dict:
        """Compute aggregated metrics."""
        all_scores: dict[str, list[float]] = {}
        latencies = []
        for r in self.results:
            latencies.append(r.latency_ms)
            for k, v in r.scores.items():
                all_scores.setdefault(k, []).append(v)
        return {
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "avg_scores": {
                k: sum(v) / len(v) for k, v in all_scores.items()
            },
            "total_questions": len(self.results),
        }

    def to_json(self, output_dir: str = "reports") -> str:
        """Generate JSON report.

        Args:
            output_dir: Directory to write report to

        Returns:
            Path to generated JSON file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "config": self.config_name,
            "summary": self._aggregate(),
            "results": [asdict(r) for r in self.results],
        }
        path = Path(output_dir) / f"{self.run_id}.json"
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    def to_html(self, output_dir: str = "reports") -> str:
        """Generate HTML report.

        Args:
            output_dir: Directory to write report to

        Returns:
            Path to generated HTML file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary = self._aggregate()
        rows_html = ""

        metric_names = (
            list(self.results[0].scores.keys()) if self.results else []
        )

        for r in self.results:
            score_cells = "".join(f"<td>{v:.3f}</td>" for v in r.scores.values())
            rows_html += (
                f"<tr><td>{r.question}</td>"
                f"<td>{r.generated_answer[:80]}…</td>"
                f"<td>{r.latency_ms:.1f}</td>"
                f"{score_cells}</tr>\n"
            )

        metric_headers = "".join(f"<th>{k}</th>" for k in metric_names)

        html = f"""<!DOCTYPE html><html><head><title>{self.run_id}</title>
<style>body{{font-family:sans-serif}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:6px 10px}}th{{background:#f0f0f0}}</style>
</head><body>
<h1>Run: {self.run_id}</h1>
<h2>Summary</h2>
<p>Questions: {summary['total_questions']} | Avg latency: {summary['avg_latency_ms']:.1f}ms</p>
<h2>Results</h2><table>
<tr><th>Question</th><th>Answer (truncated)</th><th>Latency ms</th>
{metric_headers}
</tr>
{rows_html}
</table></body></html>"""
        path = Path(output_dir) / f"{self.run_id}.html"
        path.write_text(html)
        return str(path)
