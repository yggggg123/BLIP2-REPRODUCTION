"""Pure-stdlib checks for comparable budgets, data/votes, and LR schedules."""

import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "train_vqa_improvement.py"
spec = importlib.util.spec_from_file_location("vqa_train", path)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class PlanTests(unittest.TestCase):
    def test_same_plan_independent_of_global_rng_or_variant(self):
        import random
        rows = [{"answers": ["yes"] * 7 + ["no"] * 3} for _ in range(20)]
        first = train.make_sample_plan(rows, 30, 4, 42)
        random.seed(1)
        for _ in range(100):
            random.random()
        self.assertEqual(first, train.make_sample_plan(rows, 30, 4, 42))
        self.assertNotEqual(first, train.make_sample_plan(rows, 30, 4, 43))
        self.assertEqual(120, len(first))
        self.assertEqual(set(range(20)), {item[0] for item in first[:20]})
        self.assertTrue(all(0 <= answer < 10 for _, answer in first))
        # Resumption uses exactly the original global sample positions.
        self.assertEqual(first[40:], train.make_sample_plan(rows, 30, 4, 42)[10 * 4:])

    def test_votes_preserved_and_invalid_rows_rejected(self):
        rows = [{"answers": ["yes"] * 9 + ["no"]}]
        plan = train.make_sample_plan(rows, 1000, 1, 42)
        yes = sum(rows[index]["answers"][answer] == "yes" for index, answer in plan)
        self.assertTrue(850 < yes < 950)
        for invalid in ([], [{"answers": []}]):
            with self.assertRaises(ValueError):
                train.make_sample_plan(invalid, 10, 1, 42)

    def test_both_parameter_groups_share_bounded_schedule(self):
        factors = [train.schedule_factor(i, 1000, 50) for i in range(1000)]
        self.assertTrue(all(0.1 <= factor <= 1.0 for factor in factors))
        self.assertAlmostEqual(1.0, factors[49])
        self.assertAlmostEqual(1.0, factors[50])
        self.assertAlmostEqual(0.1, factors[-1])
        self.assertTrue(all(a >= b for a, b in zip(factors[50:], factors[51:])))


if __name__ == "__main__":
    unittest.main()
