import json
import tempfile
import unittest
from pathlib import Path

from eval.t2.interventions import (
    TASK_FEEDBACK_TEXT,
    FeedbackIntervention,
    counterbalanced_order,
)


class InterventionTest(unittest.TestCase):
    def setUp(self):
        candidates = [
            [
                "turn_on_led",
                "move_slider_left",
                "lift_blue_block_table",
                "open_drawer",
            ],
            [
                "turn_off_led",
                "move_slider_right",
                "lift_blue_block_slider",
                "close_drawer",
            ],
        ]
        self.episodes = []
        for index in range(2):
            episode = {
                "episode_id": f"validation_{index:05d}",
                "episode_index": index,
                "combination_index": 0,
                "feedback_state": ["led", "slider", "blue_block", "drawer"],
                "candidate_tasks": candidates[index],
                "sampled_task_instructions": {
                    task: f"diverse phrase for {task}" for task in TASK_FEEDBACK_TEXT
                },
                "feedback_templates": {
                    "first_trial_template": (
                        "This is the initial trial. Explore these objects: {}. "
                        "Previously explored objects: none."
                    )
                },
            }
            self.episodes.append(episode)
        self.feedback = (
            "Trial 1 completed. Completed rewarding tasks last trial: None. "
            "Completed unrewarding tasks last trial: move the led. "
            "Episode objects: led, slider, blue block, drawer"
        )

    def test_masked_preserves_first_trial_and_removes_outcomes(self):
        intervention = FeedbackIntervention("masked", self.episodes)
        first = intervention.apply(
            "first trial. episode objects: led, slider, blue block, drawer.",
            episode=self.episodes[0],
            trial_index=1,
            stage="trial_start",
        )
        self.assertTrue(first.startswith("first trial"))
        masked = intervention.apply(
            self.feedback,
            episode=self.episodes[0],
            trial_index=2,
            stage="trial_start",
        )
        self.assertNotIn("move the led", masked)
        self.assertIn("information withheld", masked)

    def test_alt_masked_reuses_frozen_initial_template(self):
        intervention = FeedbackIntervention(
            "masked", self.episodes, feedback_type="alt_feedback"
        )
        masked = intervention.apply(
            "There was no reward in the previous trial. Objects to be explored: "
            "led, slider, blue block, drawer. Previously explored objects: led.",
            episode=self.episodes[0],
            trial_index=2,
            stage="trial_start",
        )
        self.assertEqual(
            masked,
            "This is the initial trial. Explore these objects: led, slider, "
            "blue block, drawer. Previously explored objects: none.",
        )
        self.assertNotIn("no reward", masked.lower())

    def test_shuffled_uses_seeded_donor_and_target_candidate_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trials.jsonl"
            records = []
            for episode in self.episodes:
                for trial_index in (1, 2):
                    reward_task = episode["candidate_tasks"][0]
                    records.append(
                        {
                            "episode_id": episode["episode_id"],
                            "trial_index": trial_index,
                            "input_feedback": (
                                f"{TASK_FEEDBACK_TEXT[reward_task]} "
                                "was rewarding last trial;"
                                if trial_index == 2
                                else "first trial. episode objects: "
                                "led, slider, blue block, drawer"
                            ),
                        }
                    )
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            intervention = FeedbackIntervention("shuffled", self.episodes, path)
            output = intervention.apply(
                self.feedback,
                episode=self.episodes[0],
                trial_index=2,
                stage="trial_start",
            )
            target_id = self.episodes[0]["episode_id"]
            donor_id = intervention.donor_map[target_id]
            donor = next(
                episode for episode in self.episodes if episode["episode_id"] == donor_id
            )
            donor_reward_task = donor["candidate_tasks"][0]
            translated_task = intervention.donor_task_maps[target_id][donor_reward_task]
            self.assertEqual(
                output,
                f"{TASK_FEEDBACK_TEXT[translated_task]} was rewarding last trial;",
            )
            self.assertIn(translated_task, self.episodes[0]["candidate_tasks"])
            self.assertTrue(
                all(
                    source != destination
                    for source, destination in intervention.donor_object_maps[
                        target_id
                    ].items()
                )
            )
            metadata = intervention.metadata()
            self.assertEqual(metadata["donor_seed"], 42)
            self.assertIn(target_id, metadata["donor_task_maps"])

    def test_counterbalanced_order_always_changes(self):
        source = self.episodes[0]["feedback_state"]
        self.assertNotEqual(source, counterbalanced_order(source, 0))
        intervention = FeedbackIntervention("counterbalanced", self.episodes)
        output = intervention.apply(
            "first trial. episode objects: led, slider, blue block, drawer.",
            episode=self.episodes[0],
            trial_index=1,
            stage="trial_start",
        )
        self.assertIn("episode objects: slider, blue block, drawer, led", output)
        alt_output = intervention.apply(
            "This is the initial trial. Explore these objects: led, slider, "
            "blue block, drawer. Previously explored objects: none.",
            episode=self.episodes[0],
            trial_index=1,
            stage="trial_start",
        )
        self.assertIn(
            "Explore these objects: slider, blue block, drawer, led", alt_output
        )
        self.assertEqual(
            intervention.input_candidate_order(
                {**self.episodes[0], "candidate_tasks": ["a", "b", "c", "d"]}
            ),
            ["b", "c", "d", "a"],
        )


if __name__ == "__main__":
    unittest.main()
