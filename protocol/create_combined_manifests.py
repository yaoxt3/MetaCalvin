#!/usr/bin/env python3
"""Create leakage-free Combined-Explore train/validation task manifests.

The legacy Combined environment contains 24 valid unordered task pairs: choose
two of the four objects, then choose one of two state-changing actions for each
object.  A legacy list-order bug accidentally included one held-out pair in the
training data, so the source directory reports 21 training keys even though
there are only 20 non-test pairs.  This script quarantines every source episode
from the four held-out pairs before selecting validation groups.

The old data did not save an explicit episode-to-meta-task index.  We recover it
from the rewarding-task objects in the trial feedback and the initial scene
state.  The state is authoritative for the action direction because one saved
LLM instruction contradicts the LED transition that was actually executed.
All recovered counts must exactly match meta_task_stats.npz.
"""

import argparse
import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_SPLIT_SEED = 20260866
TASK_ORDER = (
    "move_slider_left",
    "move_slider_right",
    "open_drawer",
    "close_drawer",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
)
TEST_PAIRS = (
    ("move_slider_right", "turn_off_led"),
    ("close_drawer", "turn_on_lightbulb"),
    ("turn_off_lightbulb", "turn_on_led"),
    ("move_slider_left", "open_drawer"),
)
OBJECT_ORDER = ("slider", "drawer", "lightbulb", "led")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_pair(pair):
    return tuple(sorted(pair, key=TASK_ORDER.index))


def task_object(task):
    for obj in OBJECT_ORDER:
        if obj in task:
            return obj
    raise ValueError(f"Unknown task object: {task}")


def valid_pairs():
    return tuple(
        pair
        for pair in itertools.combinations(TASK_ORDER, 2)
        if task_object(pair[0]) != task_object(pair[1])
    )


def feedback_object(text):
    normalized = " ".join(text.lower().split())
    if "door" in normalized or "slider" in normalized:
        return "slider"
    if "drawer" in normalized:
        return "drawer"
    if "lightbulb" in normalized:
        return "lightbulb"
    if re.search(r"\bled\b", normalized):
        return "led"
    raise ValueError(f"Cannot identify task object from rewarding feedback: {text!r}")


def task_from_initial_state(obj, scene_obs):
    if obj == "slider":
        return "move_slider_right" if scene_obs[0] > 0.1 else "move_slider_left"
    if obj == "drawer":
        return "close_drawer" if scene_obs[1] > 0.1 else "open_drawer"
    if obj == "lightbulb":
        return "turn_off_lightbulb" if scene_obs[4] > 0.5 else "turn_on_lightbulb"
    if obj == "led":
        return "turn_off_led" if scene_obs[5] > 0.5 else "turn_on_led"
    raise ValueError(f"Unknown object: {obj}")


def task_from_instruction(text):
    normalized = " ".join(text.lower().strip(" .").split())
    obj = feedback_object(normalized)
    if obj == "slider":
        if "left" in normalized:
            return "move_slider_left"
        if "right" in normalized:
            return "move_slider_right"
    elif obj == "drawer":
        if "open" in normalized:
            return "open_drawer"
        if "close" in normalized:
            return "close_drawer"
    else:
        if re.search(r"\bon\b", normalized):
            return f"turn_on_{obj}"
        if re.search(r"\boff\b", normalized):
            return f"turn_off_{obj}"
    raise ValueError(f"Cannot infer task direction from instruction: {text!r}")


def expected_stats_keys():
    return {"_".join(pair): pair for pair in valid_pairs()}


def recover_episode_pairs(dataset_dir, episode_ranges):
    @lru_cache(maxsize=4096)
    def read_frame(frame_index):
        path = dataset_dir / f"episode_{frame_index:06d}.npz"
        with np.load(path, allow_pickle=True) as frame:
            return (
                int(frame["trial_done"]),
                str(frame["train_feedback"]),
                np.asarray(frame["scene_obs"]).copy(),
            )

    def first_frame_of_trial(start, end, trial_index):
        left, right = start, end
        while left < right:
            middle = (left + right) // 2
            if read_frame(middle)[0] >= trial_index:
                right = middle
            else:
                left = middle + 1
        trial, feedback, _ = read_frame(left)
        if trial < trial_index:
            raise ValueError(
                f"Episode {start}:{end} has no frame for trial {trial_index}"
            )
        return left, feedback

    groups = defaultdict(list)
    text_state_disagreements = []
    rewarding_pattern = re.compile(
        r"Completed rewarding tasks last trial:\s*(.*?)\.\s*"
        r"Completed unrewarding tasks last trial:",
        re.IGNORECASE,
    )

    for episode_index, (start, end) in enumerate(episode_ranges):
        start, end = int(start), int(end)
        _, _, initial_scene = read_frame(start)
        max_trial, _, _ = read_frame(end)
        recovered = None

        for trial_index in range(1, max_trial + 1):
            feedback_frame, feedback = first_frame_of_trial(
                start, end, trial_index
            )
            match = rewarding_pattern.search(feedback)
            if match is None:
                continue
            instructions = [
                item.strip()
                for item in match.group(1).split(";")
                if item.strip() and item.strip().lower() != "none"
            ]
            if len(instructions) != 2:
                continue

            objects = tuple(feedback_object(item) for item in instructions)
            if len(set(objects)) != 2:
                raise ValueError(
                    f"Episode {episode_index} has duplicate rewarding objects: "
                    f"{instructions}"
                )
            state_pair = canonical_pair(
                task_from_initial_state(obj, initial_scene) for obj in objects
            )
            text_pair = canonical_pair(
                task_from_instruction(item) for item in instructions
            )
            if text_pair != state_pair:
                text_state_disagreements.append(
                    {
                        "episode_index": episode_index,
                        "feedback_frame": feedback_frame,
                        "instructions": instructions,
                        "text_pair": list(text_pair),
                        "state_pair": list(state_pair),
                    }
                )
            recovered = state_pair
            break

        if recovered is None:
            raise ValueError(
                f"Could not recover two rewarding tasks for episode {episode_index} "
                f"({start}:{end})"
            )
        if recovered not in valid_pairs():
            raise ValueError(f"Invalid recovered pair: {recovered}")
        groups[recovered].append(
            {
                "episode_index": episode_index,
                "start": start,
                "end": end,
                "meta_task_pair": list(recovered),
            }
        )

    return groups, text_state_disagreements


def build_manifests(dataset_dir, split_seed):
    index_path = dataset_dir / "ep_start_end_ids.npy"
    stats_path = dataset_dir / "meta_task_stats.npz"
    episode_ranges = np.load(index_path)
    groups, disagreements = recover_episode_pairs(dataset_dir, episode_ranges)

    expected_source_groups = set(valid_pairs()) - {
        canonical_pair(pair) for pair in TEST_PAIRS
    }
    # The legacy reverse-order comparison leaked exactly this final-test pair
    # into the source directory.
    expected_source_groups.add(
        canonical_pair(("turn_off_lightbulb", "turn_on_led"))
    )
    if set(groups) != expected_source_groups:
        raise ValueError(
            f"Unexpected Combined source groups; missing={sorted(expected_source_groups-set(groups))}, "
            f"unexpected={sorted(set(groups)-expected_source_groups)}"
        )

    stats_keys = expected_stats_keys()
    with np.load(stats_path, allow_pickle=True) as saved_stats:
        raw_stats = {key: int(saved_stats[key]) for key in saved_stats.files}
    unknown_stats = sorted(set(raw_stats) - set(stats_keys))
    recovered_counts = {"_".join(pair): len(records) for pair, records in groups.items()}
    count_mismatches = {
        key: {"saved": raw_stats.get(key), "recovered": recovered_counts.get(key, 0)}
        for key in sorted(set(raw_stats) | set(recovered_counts))
        if raw_stats.get(key) != recovered_counts.get(key, 0)
    }
    if unknown_stats or count_mismatches:
        raise ValueError(
            f"Combined stats audit failed; unknown={unknown_stats}, "
            f"count_mismatches={count_mismatches}"
        )

    test_groups = tuple(sorted({canonical_pair(pair) for pair in TEST_PAIRS}))
    train_val_groups = sorted(set(valid_pairs()) - set(test_groups))
    if len(valid_pairs()) != 24 or len(train_val_groups) != 20:
        raise ValueError("Expected 24 valid pairs: 20 train/validation + 4 test")

    val_groups = sorted(random.Random(split_seed).sample(train_val_groups, 4))
    validation_coverage = set().union(*(set(group) for group in val_groups))
    if validation_coverage != set(TASK_ORDER):
        raise ValueError(
            f"Split seed {split_seed} does not cover all eight subtasks in validation; "
            "select and predeclare a coverage-valid seed before training"
        )
    train_groups = sorted(set(train_val_groups) - set(val_groups))

    source_index_sha256 = sha256_file(index_path)
    common = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "MetaCALVIN-RA-L-P0-Combined-v1",
        "split_seed": split_seed,
        "grouping_unit": "unordered_meta_task_pair",
        "source_dataset_name": dataset_dir.name,
        "source_index_sha256": source_index_sha256,
        "selection_rule": (
            "test: the four legacy held-out unordered task pairs; validation: "
            "Python random.Random(split_seed).sample over the sorted 20 remaining "
            "pairs, with all eight subtasks covered; train: all remaining pairs; "
            "all source episodes belonging to a test pair are quarantined"
        ),
    }
    manifests = {}
    for split, selected_groups in (
        ("train", train_groups),
        ("validation", val_groups),
    ):
        episodes = [record for group in selected_groups for record in groups[group]]
        episodes.sort(key=lambda record: record["episode_index"])
        manifests[split] = {
            **common,
            "split": split,
            "meta_task_pairs": [list(group) for group in selected_groups],
            "num_meta_task_pairs": len(selected_groups),
            "num_episodes": len(episodes),
            "num_frames": sum(r["end"] - r["start"] + 1 for r in episodes),
            "num_windows_ws12": sum(r["end"] - r["start"] + 1 - 12 for r in episodes),
            "episodes": episodes,
        }

    # Final test is simulator-generated from a separate frozen common episode
    # manifest.  Do not expose the leaked legacy source episodes as test data.
    manifests["test_tasks"] = {
        **common,
        "split": "test",
        "meta_task_pairs": [list(group) for group in test_groups],
        "num_meta_task_pairs": len(test_groups),
        "num_episodes": 0,
        "episodes": [],
        "note": (
            "Task-only split definition. Final rollout episodes must be created "
            "and frozen separately; source training episodes are never test data."
        ),
    }

    quarantined = [record for group in test_groups for record in groups.get(group, [])]
    audit = {
        "schema_version": 1,
        "protocol": common["protocol"],
        "source_dataset_name": dataset_dir.name,
        "source_index_sha256": source_index_sha256,
        "source_meta_task_stats_sha256": sha256_file(stats_path),
        "split_seed": split_seed,
        "mathematical_pair_count": {
            "valid_unordered_pairs": 24,
            "train_validation_candidates": 20,
            "validation": 4,
            "train": 16,
            "final_test": 4,
        },
        "legacy_protocol_error": {
            "description": (
                "The legacy eval pair ['turn_on_led','turn_off_lightbulb'] was "
                "compared in reverse order against itertools.combinations and "
                "therefore also appeared in the training source."
            ),
            "leaked_pair": list(
                canonical_pair(("turn_on_led", "turn_off_lightbulb"))
            ),
            "quarantined_source_episode_count": len(quarantined),
            "quarantined_source_episode_indices": sorted(
                record["episode_index"] for record in quarantined
            ),
        },
        "episode_recovery": {
            "source_episode_count": len(episode_ranges),
            "recovered_episode_count": sum(len(records) for records in groups.values()),
            "counts_match_meta_task_stats": True,
            "text_state_disagreement_count": len(disagreements),
            "text_state_disagreements": disagreements,
        },
        "splits": {
            "train": {
                "pairs": [list(group) for group in train_groups],
                "episode_count": manifests["train"]["num_episodes"],
            },
            "validation": {
                "pairs": [list(group) for group in val_groups],
                "episode_count": manifests["validation"]["num_episodes"],
                "subtask_coverage": sorted(validation_coverage, key=TASK_ORDER.index),
            },
            "test": {
                "pairs": [list(group) for group in test_groups],
                "source_episodes_used": 0,
            },
        },
    }
    return manifests, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    manifests, audit = build_manifests(args.dataset_dir.resolve(), args.split_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, manifest in manifests.items():
        output = args.output_dir / f"combined_{split}.json"
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"{split}: pairs={manifest['num_meta_task_pairs']}, "
            f"episodes={manifest['num_episodes']}, sha256={sha256_file(output)}, "
            f"path={output}"
        )

    audit_path = args.output_dir / "combined_split_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"audit: sha256={sha256_file(audit_path)}, path={audit_path}")


if __name__ == "__main__":
    main()
