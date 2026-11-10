import argparse
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eval.t1.controllers import DirectController
from eval.t3.evaluate_t3 import aggregate, rollout_episode


class FakePolicy:
    def __init__(self):
        self.goals = []
        self.img_queue = deque(maxlen=12)
        self.gripper_queue = deque(maxlen=12)
        self.state_queue = deque(maxlen=12)
        self.mask_queue = deque(maxlen=12)
        self.text_queue = deque(maxlen=12)
        self.feature_cache = None
        self.dt_feat_cache = []
        self.head = SimpleNamespace(hidden_state=None, history_memory=[], rnn_out=None)
        self.model = SimpleNamespace(
            lang_encoder=SimpleNamespace(lm_head=self.head), sep_lm_head=False
        )

    def reset(self):
        for queue in (
            self.img_queue,
            self.gripper_queue,
            self.state_queue,
            self.mask_queue,
            self.text_queue,
        ):
            queue.clear()
        self.head.hidden_state = None
        self.head.history_memory = []
        self.head.rnn_out = None

    def step(self, obs, goal):
        self.goals.append(goal)
        self.img_queue.append(1)
        self.text_queue.append(1)
        self.head.hidden_state = (object(), object())
        self.head.history_memory.append(goal)
        self.head.rnn_out = object()
        return np.zeros(7, dtype=np.float32)


class FakeEnv:
    num_trials = 3
    trial_steps = 2

    def __init__(self):
        self.trial = 1
        self.step_in_trial = 0
        self.task_success_dict = {
            "open_drawer": False,
            "turn_on_led": False,
            "move_slider_left": False,
        }

    def _bundle(self):
        return {
            "train_feedback": (
                "first trial. episode objects: drawer, led, slider"
                if self.trial == 1
                else f"legacy raw feedback for trial {self.trial}"
            )
        }

    def episode_reset_from_manifest(self, episode):
        self.trial = 1
        self.step_in_trial = 0
        return {}, self._bundle(), None, None

    def trial_reset_from_manifest(self, instruction):
        self.trial += 1
        self.step_in_trial = 0
        self.task_success_dict = {key: False for key in self.task_success_dict}
        return {}, self._bundle(), None, None

    def step(self, action, instruction=None):
        self.step_in_trial += 1
        task = ("open_drawer", "turn_on_led", "move_slider_left")[self.trial - 1]
        self.task_success_dict[task] = True
        if self.step_in_trial == 1:
            return (
                {},
                0.0,
                {"train_feedback": "immediate feedback retained"},
                self.trial - 1,
                False,
                {},
            )
        return {}, float(self.trial == 3), None, self.trial - 1, True, {}


class EvaluateT3Test(unittest.TestCase):
    def test_rollout_preserves_immediate_feedback_and_logs_state(self):
        args = argparse.Namespace(
            experiment="E0",
            text_memory="local",
            recurrent_state="persistent",
            split="validation",
            training_seed=42,
            evaluation_seed=42,
            feedback="train_feedback",
            save_actions=False,
        )
        episode = {
            "episode_id": "validation_00000",
            "episode_index": 0,
            "combination_index": 0,
            "feedback_state": ["drawer", "led", "slider"],
            "candidate_tasks": ["open_drawer", "turn_on_led", "move_slider_left"],
            "meta_task": "move_slider_left",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                name: root / f"{name}.jsonl"
                for name in (
                    "trials",
                    "decisions",
                    "episodes",
                    "feedback_events",
                    "state_events",
                )
            }
            policy = FakePolicy()
            record = rollout_episode(
                args,
                FakeEnv(),
                DirectController(policy, "E0_local_persistent"),
                episode,
                paths,
                {},
            )
            self.assertIn("immediate feedback retained", policy.goals)
            trials = [json.loads(line) for line in paths["trials"].read_text().splitlines()]
            self.assertIn("information not provided", trials[1]["input_feedback"])
            self.assertIn("open the drawer", trials[1]["input_feedback"])
            states = [json.loads(line) for line in paths["state_events"].read_text().splitlines()]
            self.assertEqual(len(states), 3)
            self.assertTrue(
                states[1]["after"]["heads"]["lang_encoder.lm_head"]["hidden_present"]
            )
            self.assertFalse(states[1]["recurrent_reset_applied"])
            metrics = aggregate([record])
            self.assertEqual(metrics["average_return"], 1.0)
            self.assertEqual(metrics["known_negative_repetitions"], 0)
            self.assertEqual(metrics["positive_retention_opportunities"], 0)


if __name__ == "__main__":
    unittest.main()
