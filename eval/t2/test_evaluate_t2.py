import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.t1.controllers import DirectController
from eval.t1.evaluate_t1 import aggregate_results as aggregate_t1
from eval.t1.evaluate_t1 import rollout_episode as rollout_t1_episode
from eval.t2.evaluate_t2 import aggregate, rollout_episode
from eval.t2.interventions import FeedbackIntervention


class FakePolicy:
    def __init__(self):
        self.goals = []

    def reset(self):
        pass

    def step(self, obs, goal):
        self.goals.append(goal)
        return np.zeros(7, dtype=np.float32)


class FakeEnv:
    num_trials = 2
    trial_steps = 2

    def __init__(self):
        self.trial = 1
        self.step_in_trial = 0
        self.task_success_dict = {"open_drawer": False, "turn_on_led": False}

    def _bundle(self):
        return {
            "train_feedback": (
                "first trial. episode objects: drawer, led."
                if self.trial == 1
                else "Trial 1 completed. Completed rewarding tasks last trial: None. "
                "Completed unrewarding tasks last trial: open drawer. "
                "Episode objects: drawer, led"
            )
        }

    def episode_reset_from_manifest(self, episode):
        self.trial = 1
        self.step_in_trial = 0
        self.task_success_dict = {"open_drawer": False, "turn_on_led": False}
        return {}, self._bundle(), None, None

    def trial_reset_from_manifest(self, instruction):
        self.trial += 1
        self.step_in_trial = 0
        self.task_success_dict = {"open_drawer": False, "turn_on_led": False}
        return {}, self._bundle(), None, None

    def step(self, action, instruction=None):
        self.step_in_trial += 1
        task = "open_drawer" if self.trial == 1 else "turn_on_led"
        self.task_success_dict[task] = True
        if self.step_in_trial == 1:
            intra_feedback = {"train_feedback": "intra-trial feedback retained"}
            return {}, 0.0, intra_feedback, self.trial - 1, False, {}
        return {}, float(self.trial == 2), None, self.trial - 1, True, {}


class EvaluateT2Test(unittest.TestCase):
    def test_full_rollout_records_observed_task_sequence(self):
        args = argparse.Namespace(
            experiment="E0",
            intervention="full",
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
            "feedback_state": ["drawer", "led"],
            "candidate_tasks": ["open_drawer", "turn_on_led"],
            "meta_task": "turn_on_led",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "trials": root / "trials.jsonl",
                "decisions": root / "decisions.jsonl",
                "episodes": root / "episodes.jsonl",
                "feedback_events": root / "feedback_events.jsonl",
            }
            policy = FakePolicy()
            record = rollout_episode(
                args,
                FakeEnv(),
                DirectController(policy, "E0_full"),
                episode,
                FeedbackIntervention("full", [episode]),
                paths,
            )
            self.assertEqual(record["first_reward_trial"], 2)
            self.assertEqual(record["first_completed_candidate_positions"], [1, 2])
            self.assertEqual(len(paths["trials"].read_text().splitlines()), 2)
            self.assertEqual(len(paths["decisions"].read_text().splitlines()), 2)
            self.assertIn("intra-trial feedback retained", policy.goals)
            events = [
                json.loads(line)
                for line in paths["feedback_events"].read_text().splitlines()
            ]
            self.assertEqual(
                [event["stage"] for event in events],
                ["trial_start", "intra_trial", "trial_start", "intra_trial"],
            )
            self.assertEqual(record["protocol_scope"], "registered")
            metrics = aggregate([record])
            self.assertEqual(metrics["average_return"], 1.0)
            self.assertEqual(metrics["first_completed_candidate_position_counts"], {"1": 1, "2": 1})
            self.assertEqual(metrics["switch_after_negative_rate"], 1.0)
            self.assertIsNone(metrics["history_sensitive_behavior_change_rate_vs_full"])

    def test_masked_is_paired_to_full_and_measures_behavior_change(self):
        episode = {
            "episode_id": "validation_00000",
            "episode_index": 0,
            "combination_index": 0,
            "feedback_state": ["drawer", "led"],
            "candidate_tasks": ["open_drawer", "turn_on_led"],
            "meta_task": "turn_on_led",
        }
        full_args = argparse.Namespace(
            experiment="E0",
            intervention="full",
            split="validation",
            training_seed=42,
            evaluation_seed=42,
            feedback="train_feedback",
            save_actions=False,
        )
        masked_args = argparse.Namespace(**{**vars(full_args), "intervention": "masked"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def paths(prefix):
                return {
                    "trials": root / f"{prefix}_trials.jsonl",
                    "decisions": root / f"{prefix}_decisions.jsonl",
                    "episodes": root / f"{prefix}_episodes.jsonl",
                    "feedback_events": root / f"{prefix}_feedback_events.jsonl",
                }

            full_paths = paths("full")
            rollout_episode(
                full_args,
                FakeEnv(),
                DirectController(FakePolicy(), "E0_full"),
                episode,
                FeedbackIntervention("full", [episode]),
                full_paths,
            )
            masked_record = rollout_episode(
                masked_args,
                FakeEnv(),
                DirectController(FakePolicy(), "E0_masked"),
                episode,
                FeedbackIntervention(
                    "masked", [episode], full_paths["trials"]
                ),
                paths("masked"),
            )
            metrics = aggregate([masked_record])
            self.assertEqual(metrics["paired_full_trial_count"], 2)
            self.assertEqual(metrics["paired_full_input_feedback_change_rate"], 0.5)
            self.assertEqual(metrics["history_sensitive_behavior_change_rate_vs_full"], 0.0)
            self.assertEqual(
                metrics["trial1_negative_control_behavior_change_rate_vs_full"], 0.0
            )

    def test_full_core_outcomes_match_t1_direct_flow(self):
        episode = {
            "episode_id": "validation_00000",
            "episode_index": 0,
            "combination_index": 0,
            "feedback_state": ["drawer", "led"],
            "candidate_tasks": ["open_drawer", "turn_on_led"],
            "meta_task": "turn_on_led",
        }
        t1_args = argparse.Namespace(
            controller="student",
            feedback="train_feedback",
            run_name="t1_reference",
            split="validation",
            training_seed=42,
            evaluation_seed=42,
            planner_seed=42,
            save_actions=False,
        )
        t2_args = argparse.Namespace(
            experiment="E0",
            intervention="full",
            split="validation",
            training_seed=42,
            evaluation_seed=42,
            feedback="train_feedback",
            save_actions=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            t1_paths = {
                "trials": root / "t1_trials.jsonl",
                "decisions": root / "t1_decisions.jsonl",
                "episodes": root / "t1_episodes.jsonl",
            }
            t2_paths = {
                "trials": root / "t2_trials.jsonl",
                "decisions": root / "t2_decisions.jsonl",
                "episodes": root / "t2_episodes.jsonl",
                "feedback_events": root / "t2_feedback_events.jsonl",
            }
            t1_record = rollout_t1_episode(
                t1_args,
                FakeEnv(),
                DirectController(FakePolicy(), "student"),
                episode,
                t1_paths,
            )
            t2_record = rollout_episode(
                t2_args,
                FakeEnv(),
                DirectController(FakePolicy(), "E0_full"),
                episode,
                FeedbackIntervention("full", [episode]),
                t2_paths,
            )
            for key in (
                "total_reward",
                "first_reward_trial",
                "episode_steps",
                "trials_completed",
                "failure_counts",
            ):
                self.assertEqual(t2_record[key], t1_record[key])
            t1_metrics = aggregate_t1([t1_record])
            t2_metrics = aggregate([t2_record])
            for key in (
                "average_return",
                "reward_frequency",
                "reward_found_rate",
                "mean_episode_steps",
            ):
                self.assertEqual(t2_metrics[key], t1_metrics[key])


if __name__ == "__main__":
    unittest.main()
