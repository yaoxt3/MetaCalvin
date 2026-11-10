import unittest

from eval.t1.manifest import PROTOCOL, SCHEMA_VERSION, validate_rollout_manifest


def valid_manifest():
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "split": "validation",
        "num_trials": 1,
        "num_episodes": 1,
        "episodes": [
            {
                "episode_id": "validation_00000",
                "candidate_tasks": ["open_drawer"],
                "candidate_order": ["open_drawer"],
                "meta_task": "open_drawer",
                "trial_reset_states": [
                    {"robot_obs": [0.0] * 15, "scene_obs": [0.0] * 24}
                ],
                "feedback_templates": {
                    "first_trial_template": "{}",
                    "further_trial_template": "{} {}",
                    "reward_template": "{}",
                },
                "feedback_template_ids": {
                    "initial": "diverse:initial:1",
                    "failure": "diverse:failure:2",
                    "reward": "diverse:reward:3",
                },
            }
        ],
    }


class ManifestTests(unittest.TestCase):
    def test_valid_manifest(self):
        validate_rollout_manifest(valid_manifest(), expected_split="validation")

    def test_rejects_split_mismatch(self):
        with self.assertRaises(ValueError):
            validate_rollout_manifest(valid_manifest(), expected_split="test")

    def test_rejects_missing_trial_reset(self):
        payload = valid_manifest()
        payload["episodes"][0]["trial_reset_states"] = []
        with self.assertRaises(ValueError):
            validate_rollout_manifest(payload)


if __name__ == "__main__":
    unittest.main()
