import json
import tempfile
import unittest
from pathlib import Path

from eval.t2.combined_manifest import (
    PROTOCOL,
    create_rollout_manifest,
    load_rollout_manifest,
)


class CombinedManifestTest(unittest.TestCase):
    def _source(self, directory: Path) -> Path:
        source = directory / "combined_validation.json"
        source.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol": "MetaCALVIN-RA-L-P0-Combined-v1",
                    "split": "validation",
                    "split_seed": 20260866,
                    "num_meta_task_pairs": 1,
                    "meta_task_pairs": [["move_slider_left", "open_drawer"]],
                }
            ),
            encoding="utf-8",
        )
        return source

    def test_small_manifest_is_deterministic_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._source(directory)
            first = directory / "first.json"
            second = directory / "second.json"
            kwargs = {
                "split_manifest_path": source,
                "split": "validation",
                "seed": 1,
                "num_resets": 2,
                "num_trials": 4,
                "trial_steps": 500,
            }
            create_rollout_manifest(output_path=first, **kwargs)
            create_rollout_manifest(output_path=second, **kwargs)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            payload = load_rollout_manifest(first, expected_split="validation")
            self.assertEqual(payload["protocol"], PROTOCOL)
            self.assertEqual(payload["num_episodes"], 2)
            self.assertEqual(len(payload["episodes"][0]["trial_reset_states"]), 4)
            self.assertTrue(
                set(payload["episodes"][0]["meta_task"])
                <= set(payload["episodes"][0]["candidate_tasks"])
            )

    def test_refuses_to_overwrite_frozen_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._source(directory)
            output = directory / "frozen.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_rollout_manifest(
                    split_manifest_path=source,
                    split="validation",
                    output_path=output,
                    seed=1,
                    num_resets=1,
                    num_trials=4,
                    trial_steps=500,
                )


if __name__ == "__main__":
    unittest.main()
