import unittest

from tree_demo import train_tree


class DecisionTreeTests(unittest.TestCase):
    def test_tree_returns_rules_and_score(self):
        result = train_tree(2)
        self.assertIn("petal", result["rules"])
        self.assertGreaterEqual(result["accuracy"], 0)
        self.assertLessEqual(result["accuracy"], 1)

    def test_depth_is_bounded(self):
        with self.assertRaises(ValueError):
            train_tree(0)


if __name__ == "__main__":
    unittest.main()
