import unittest

from eval.t3.memory_conditions import TextMemoryLedger


class TextMemoryLedgerTest(unittest.TestCase):
    def setUp(self):
        self.ledger = TextMemoryLedger(
            ["open_drawer", "turn_on_led", "move_slider_left"],
            ["drawer", "led", "slider"],
        )

    def test_trial_one_preserves_frozen_feedback(self):
        raw = "first trial. episode objects: drawer, led, slider"
        self.assertEqual(self.ledger.render("local", 1, raw), raw)
        self.assertEqual(self.ledger.render("cumulative", 1, raw), raw)

    def test_local_contains_only_last_trial_outcome(self):
        self.ledger.observe(1, ["open_drawer"], 0.0)
        text = self.ledger.render("local", 2, "ignored")
        self.assertIn("pull the handle to open the drawer", text)
        self.assertIn("Known rewarding tasks so far: information not provided", text)
        self.assertNotIn("turn on the led was", text)

    def test_cumulative_tracks_only_observed_facts(self):
        self.ledger.observe(1, ["open_drawer"], 0.0)
        self.ledger.observe(2, ["turn_on_led"], 1.0)
        text = self.ledger.render("cumulative", 3, "ignored")
        self.assertIn("Known rewarding tasks so far: turn on the led", text)
        self.assertIn(
            "Known unrewarding tasks so far: pull the handle to open the drawer",
            text,
        )
        self.assertIn("Untested tasks: move the door to the left", text)
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot["known_rewarding_tasks"], ["turn_on_led"])
        self.assertEqual(snapshot["known_unrewarding_tasks"], ["open_drawer"])

    def test_unknown_completed_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the frozen candidate"):
            self.ledger.observe(1, ["hidden_target_not_in_candidates"], 1.0)

    def test_ambiguous_reward_identity_is_rejected_without_using_hidden_target(self):
        with self.assertRaisesRegex(ValueError, "exactly one completed task"):
            self.ledger.observe(1, ["open_drawer", "turn_on_led"], 1.0)


if __name__ == "__main__":
    unittest.main()
