import unittest

import pandas as pd

from tree_demo import train_tree


class DecisionTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = pd.read_csv("data/sample_decisions.csv")

    def test_tree_returns_rules_and_score(self):
        result = train_tree(self.dataset, "activity", max_depth=2)
        self.assertIn("sky_", result["rules"])
        self.assertGreaterEqual(result["accuracy"], 0)
        self.assertLessEqual(result["accuracy"], 1)
        self.assertIn("walk", result["classes"])

    def test_depth_is_bounded(self):
        with self.assertRaises(ValueError):
            train_tree(self.dataset, "activity", max_depth=0)

    def test_unknown_target_is_reported(self):
        with self.assertRaises(ValueError):
            train_tree(self.dataset, "missing")

    def test_target_requires_multiple_classes(self):
        one_class = self.dataset.assign(activity="walk")
        with self.assertRaises(ValueError):
            train_tree(one_class, "activity")

    def test_csv_categorical_columns_are_encoded(self):
        result = train_tree(self.dataset, "activity")
        self.assertTrue(any("sky_" in feature for feature in result["feature_names"]))


if __name__ == "__main__":
    unittest.main()
