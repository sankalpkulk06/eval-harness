import json
import tempfile
from pathlib import Path

import pytest

from datasets.loader import DatasetLoader


class TestDatasetLoader:
    """Test DatasetLoader."""

    def test_load_valid_jsonl(self):
        """Test loading valid JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {"question": "q1", "ground_truth": "gt1"}
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {"question": "q2", "ground_truth": "gt2"}
                )
                + "\n"
            )
            f.flush()
            path = f.name

        try:
            rows = DatasetLoader.load(path)
            assert len(rows) == 2
            assert rows[0]["question"] == "q1"
            assert rows[1]["question"] == "q2"
        finally:
            Path(path).unlink()

    def test_load_with_extra_fields(self):
        """Test loading JSONL with extra fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "question": "q1",
                        "ground_truth": "gt1",
                        "extra_field": "extra_value",
                    }
                )
                + "\n"
            )
            f.flush()
            path = f.name

        try:
            rows = DatasetLoader.load(path)
            assert len(rows) == 1
            assert rows[0]["extra_field"] == "extra_value"
        finally:
            Path(path).unlink()

    def test_load_missing_question(self):
        """Test loading JSONL missing question field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"ground_truth": "gt1"}) + "\n")
            f.flush()
            path = f.name

        try:
            with pytest.raises(AssertionError):
                DatasetLoader.load(path)
        finally:
            Path(path).unlink()

    def test_load_missing_ground_truth(self):
        """Test loading JSONL missing ground_truth field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"question": "q1"}) + "\n")
            f.flush()
            path = f.name

        try:
            with pytest.raises(AssertionError):
                DatasetLoader.load(path)
        finally:
            Path(path).unlink()

    def test_load_empty_lines(self):
        """Test loading JSONL with empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"question": "q1", "ground_truth": "gt1"}) + "\n")
            f.write("\n")
            f.write(json.dumps({"question": "q2", "ground_truth": "gt2"}) + "\n")
            f.flush()
            path = f.name

        try:
            rows = DatasetLoader.load(path)
            assert len(rows) == 2
        finally:
            Path(path).unlink()
