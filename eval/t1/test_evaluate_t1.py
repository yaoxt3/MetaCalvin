import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.t1.controllers import DirectController
from eval.t1.evaluate_t1 import aggregate_results, rollout_episode


class FakePolicy:
    def reset(self):
        pass

    def step(self, obs, goal):
        return np.zeros(7, dtype=np.float32)


class FakeEnv:
    num_trials = 2
    trial_steps = 2

    def __init__(self):
        self.trial = 1
        self.task_success_dict = {"open_drawer": False, "turn_on_led": False}

    def _bundle(self):
        return {
            "train_feedback": f"trial {self.trial}",
            "alt_feedback": f"alternative trial {self.trial}",
            "lm_feedback": {"role": "user", "content": f"trial {self.trial}"},
        }

    def episode_reset_from_manifest(self, episode):
        self.trial = 1
        self.task_success_dict = {"open_drawer": False, "turn_on_led": False}
        return {"rgb_obs": {}, "robot_obs": np.zeros(15)}, self._bundle(), [], {}

    def trial_reset_from_manifest(self, instruction):
        self.trial += 1
        self.task_success_dict = {"open_drawer": False, "turn_on_led": False}
        return {"rgb_obs": {}, "robot_obs": np.zeros(15)}, self._bundle(), [], {}

    def step(self, action, instruction=None):
        if self.trial == 1:
            self.task_success_dict["open_drawer"] = True
            reward = 0
        else:
            self.task_success_dict["turn_on_led"] = True
            reward = 1
        return {}, reward, None, self.trial - 1, True, {}


class EvaluateTests(unittest.TestCase):
    def test_direct_rollout_writes_common_records(self):
        args = argparse.Namespace(
            controller="student",
            feedback="train_feedback",
            run_name="test",
            split="validation",
            training_seed=42,
            evaluation_seed=7,
            planner_seed=9,
            save_actions=False,
        )
        episode = {
            "episode_id": "validation_00000",
            "episode_index": 0,
            "candidate_tasks": ["open_drawer", "turn_on_led"],
            "meta_task": "turn_on_led",
            "feedback_template_ids": {
                "initial": "diverse:initial:1",
                "failure": "diverse:failure:2",
                "reward": "diverse:reward:3",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "trials": root / "trials.jsonl",
                "decisions": root / "decisions.jsonl",
                "episodes": root / "episodes.jsonl",
            }
            record = rollout_episode(
                args, FakeEnv(), DirectController(FakePolicy(), "student"), episode, paths
            )
            self.assertEqual(record["total_reward"], 1)
            self.assertEqual(record["first_reward_trial"], 2)
            self.assertEqual(record["search_regret"], 1)
            self.assertIsNone(record["task_selection_accuracy"])
            self.assertEqual(len(paths["trials"].read_text().splitlines()), 2)
            self.assertEqual(len(paths["decisions"].read_text().splitlines()), 2)
            summary = aggregate_results([record])
            self.assertEqual(summary["average_return"], 1)
            self.assertEqual(summary["reward_frequency"], 0.5)


if __name__ == "__main__":
    unittest.main()

