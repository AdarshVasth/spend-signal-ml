import pandas as pd
import unittest

from spend_signal import InputError, analyze_transactions, validate_transactions


def sample_frame(n=40):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "amount": [120.0 + (i % 5) for i in range(n)],
        "merchant": ["Corner Shop" if i % 2 else "Cafe" for i in range(n)],
        "category": ["Food"] * n,
        "channel": ["Card"] * n,
    })


class SpendSignalTests(unittest.TestCase):
    def test_analyze_returns_sorted_scores_and_flags(self):
        frame = sample_frame()
        frame.loc[39, ["amount", "merchant", "category", "channel"]] = [9000, "New Vendor", "Travel", "Online"]
        result = analyze_transactions(frame, contamination=0.05)
        self.assertGreaterEqual(result.flagged_count, 1)
        self.assertTrue(result.results.anomaly_score.is_monotonic_increasing)
        self.assertIn("review_note", result.results)


    def test_required_columns_are_case_and_space_tolerant(self):
        frame = sample_frame().rename(columns={"timestamp": " Timestamp "})
        self.assertIn("timestamp", validate_transactions(frame).columns)


    def test_rejects_short_file(self):
        with self.assertRaisesRegex(InputError, "at least 20"):
            analyze_transactions(sample_frame(5))


    def test_rejects_invalid_amounts(self):
        frame = sample_frame()
        frame.loc[0, "amount"] = -2
        with self.assertRaisesRegex(InputError, "greater than zero"):
            analyze_transactions(frame)


    def test_rejects_bad_contamination(self):
        with self.assertRaisesRegex(InputError, "between 1% and 20%"):
            analyze_transactions(sample_frame(), contamination=0.5)


if __name__ == "__main__":
    unittest.main()
