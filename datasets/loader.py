import json
from pathlib import Path


class DatasetLoader:
    """Load evaluation datasets from JSONL files."""

    @staticmethod
    def load(path: str) -> list[dict]:
        """Load JSONL dataset and validate required fields.

        Args:
            path: Path to JSONL file

        Returns:
            List of dictionaries with 'question' and 'ground_truth' keys
        """
        rows = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                assert (
                    "question" in row and "ground_truth" in row
                ), f"Each JSONL row must have 'question' and 'ground_truth'. Got: {row.keys()}"
                rows.append(row)
        return rows
