import unittest

from fine_tuning.reward_socrer import _length_reward, reward_function_factory


def words(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


class StubScorer:
    def score_components(self, originals, rewrites):
        return [(0.8, 0.9) for _ in originals]


class RewardFunctionTests(unittest.TestCase):
    def test_length_component_uses_the_same_normalized_scale(self):
        valid_score, valid, _, _ = _length_reward(10, 10)
        short_score, short_valid, _, _ = _length_reward(10, 6)
        long_score, long_valid, _, _ = _length_reward(10, 16)

        self.assertEqual((valid_score, valid), (1.0, True))
        self.assertEqual((short_score, short_valid), (0.0, False))
        self.assertEqual((long_score, long_valid), (0.0, False))

    def test_enforces_both_length_bounds(self):
        reward = reward_function_factory(StubScorer())
        values = reward(
            prompts=[],
            completions=[words(6), words(7), words(15), words(16)],
            target_clause=[words(10)] * 4,
        )

        self.assertLessEqual(values[0], -1.5)
        self.assertGreater(values[1], 0.0)
        self.assertGreater(values[2], 0.0)
        self.assertLessEqual(values[3], -1.5)

    def test_reduces_length_weight_after_stable_compliance(self):
        reward = reward_function_factory(
            StubScorer(),
            adaptation_window=4,
            weight_step=0.1,
        )

        for _ in range(4):
            reward([], [words(10)], [words(10)])

        self.assertAlmostEqual(reward.reward_weights.length, 1 / 3 - 0.1)
        self.assertAlmostEqual(reward.reward_weights.memory, 1 / 3 + 0.05)
        self.assertAlmostEqual(reward.reward_weights.similarity, 1 / 3 + 0.05)

    def test_restores_length_weight_after_repeated_violations(self):
        reward = reward_function_factory(
            StubScorer(),
            adaptation_window=4,
            weight_step=0.1,
        )

        for _ in range(4):
            reward([], [words(10)], [words(10)])
        for _ in range(4):
            reward([], [words(16)], [words(10)])

        self.assertAlmostEqual(reward.reward_weights.length, 1 / 3)
        self.assertAlmostEqual(reward.reward_weights.memory, 1 / 3)
        self.assertAlmostEqual(reward.reward_weights.similarity, 1 / 3)

    def test_empty_completion_is_rejected(self):
        reward = reward_function_factory(StubScorer())
        value = reward([], [""], [words(10)])[0]

        self.assertEqual(value, -2.5)


if __name__ == "__main__":
    unittest.main()
