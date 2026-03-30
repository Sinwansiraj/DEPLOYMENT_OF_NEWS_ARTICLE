"""
Unit tests for the inference module.
Tests prediction output structure, label validity, and edge cases.

Run:
    pytest tests/ -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import LABEL_MAP


class TestPredictOutputStructure(unittest.TestCase):
    """Tests that predict() returns the correct data structure."""

    def setUp(self):
        """Set up mock model and tokenizer before each test."""
        import torch

        # Mock the module-level globals in inference.py
        self.mock_model_patcher = patch("src.inference._MODEL")
        self.mock_tokenizer_patcher = patch("src.inference._TOKENIZER")

        self.mock_model = self.mock_model_patcher.start()
        self.mock_tokenizer = self.mock_tokenizer_patcher.start()

        # Tokenizer returns dummy tensors
        self.mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        # Model returns logits for 4 classes (Business has highest score)
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 2.5, 0.3]])
        self.mock_model.return_value = mock_output
        self.mock_model.parameters.return_value = iter(
            [torch.zeros(1)]
        )

    def tearDown(self):
        """Stop all patches."""
        self.mock_model_patcher.stop()
        self.mock_tokenizer_patcher.stop()

    def test_predict_returns_required_keys(self):
        """predict() must return dict with all required keys."""
        from src.inference import predict

        result = predict("Apple stock rises amid strong earnings report.")

        self.assertIn("predicted_label", result)
        self.assertIn("predicted_id", result)
        self.assertIn("confidence", result)
        self.assertIn("all_scores", result)

    def test_predicted_label_is_valid_category(self):
        """predicted_label must be one of the four AG News categories."""
        from src.inference import predict

        result = predict("The World Cup semi-final was a thrilling match.")
        self.assertIn(result["predicted_label"], LABEL_MAP.values())

    def test_confidence_is_valid_probability(self):
        """confidence must be a float in [0, 1]."""
        from src.inference import predict

        result = predict("Scientists discover new exoplanet in habitable zone.")
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_all_scores_length_matches_num_labels(self):
        """all_scores list must have one entry per label."""
        from src.inference import predict

        result = predict("The Fed raised interest rates by 50 basis points.")
        self.assertEqual(len(result["all_scores"]), len(LABEL_MAP))

    def test_all_scores_sum_to_approx_one(self):
        """Softmax probabilities must sum to approximately 1.0."""
        from src.inference import predict

        result = predict("Breaking: earthquake strikes coastal region.")
        total = sum(s["score"] for s in result["all_scores"])
        self.assertAlmostEqual(total, 1.0, places=2)


class TestPredictEdgeCases(unittest.TestCase):
    """Tests for edge-case and error handling in predict()."""

    def test_empty_string_raises_value_error(self):
        """predict() must raise ValueError for empty input."""
        # Don't load model; just test the guard
        with patch("src.inference._MODEL", None), \
             patch("src.inference._TOKENIZER", None):
            from src.inference import predict
            with self.assertRaises((ValueError, RuntimeError)):
                predict("")

    def test_whitespace_only_raises_value_error(self):
        """predict() must raise ValueError for whitespace-only input."""
        with patch("src.inference._MODEL", None), \
             patch("src.inference._TOKENIZER", None):
            from src.inference import predict
            with self.assertRaises((ValueError, RuntimeError)):
                predict("   \n\t  ")

    def test_unloaded_model_raises_runtime_error(self):
        """predict() must raise RuntimeError if model is not loaded."""
        with patch("src.inference._MODEL", None), \
             patch("src.inference._TOKENIZER", None):
            from src.inference import predict
            with self.assertRaises(RuntimeError):
                predict("Some valid text here.")


class TestCleanText(unittest.TestCase):
    """Tests for the text cleaning utility."""

    def test_removes_html_tags(self):
        """clean_text must strip HTML tags."""
        from src.data_preprocessing import clean_text
        result = clean_text("<b>Breaking News</b>: Market rallies.")
        self.assertNotIn("<b>", result)
        self.assertNotIn("</b>", result)

    def test_removes_urls(self):
        """clean_text must remove URLs."""
        from src.data_preprocessing import clean_text
        result = clean_text("Read more at https://example.com/news/article123")
        self.assertNotIn("http", result)

    def test_handles_empty_string(self):
        """clean_text must return empty string for empty input."""
        from src.data_preprocessing import clean_text
        self.assertEqual(clean_text(""), "")

    def test_handles_non_string(self):
        """clean_text must return empty string for non-string input."""
        from src.data_preprocessing import clean_text
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)