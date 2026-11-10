#!/usr/bin/env python3
"""Create leakage-free 7-Object train/validation/test manifests.

The split is based only on object configurations, never on rewards, losses, or
evaluation outcomes.  The ten legacy held-out configurations (slider + LED)
remain final-test-only.  Four of the remaining 21 configurations are sampled
with a predeclared seed for validation; the other 17 are used for training.
"""

import argparse
import hashlib
import itertools
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_SPLIT_SEED = 20260813
ALL_OBJECTS = (
    "slider",
    "drawer",
    "led",
    "lightbulb",
    "red_block",
    "pink_block",
    "blue_block",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_combination(feedback):
    match = re.search(r"episode objects:\s*(.+?)(?:\.|$)", feedback, re.I)
    if match is None:
        raise ValueError(f"Cannot parse object combination from: {feedback!r}")
    return tuple(
        sorted(item.strip().replace(" ", "_") for item in match.group(1).split(","))
    )


def expected_legacy_groups():
    groups = []
    for combination in itertools.combinations(ALL_OBJECTS, 4):
        combination = tuple(sorted(combination))
        if {"red_block", "pink_block", "blue_block"} <= set(combination):
            continue
        groups.append(combination)
    return sorted(groups)


def build_manifests(dataset_dir, split_seed):
    index_path = dataset_dir / "ep_start_end_ids.npy"
    episode_ranges = np.load(index_path)
    groups = defaultdict(list)

    for episode_index, (start, end) in enumerate(episode_ranges):
        start, end = int(start), int(end)
        frame_path = dataset_dir / f"episode_{start:06d}.npz"
        with np.load(frame_path, allow_pickle=True) as frame:
            combination = canonical_combination(str(frame["train_feedback"]))
        groups[combination].append(
            {
                "episode_index": episode_index,
                "start": start,
                "end": end,
                "object_combination": list(combination),
            }
        )

    expected = expected_legacy_groups()
    if sorted(groups) != expected:
        missing = sorted(set(expected) - set(groups))
        unexpected = sorted(set(groups) - set(expected))
        raise ValueError(
            f"Unexpected object configurations; missing={missing}, unexpected={unexpected}"
        )

    test_groups = sorted(
        group for group in groups if {"slider", "led"} <= set(group)
    )
    train_val_groups = sorted(set(groups) - set(test_groups))
    if len(test_groups) != 10 or len(train_val_groups) != 21:
        raise ValueError(
            f"Expected 21 train/validation and 10 test groups, got "
            f"{len(train_val_groups)} and {len(test_groups)}"
        )

    val_groups = sorted(random.Random(split_seed).sample(train_val_groups, 4))
    if set().union(*(set(group) for group in val_groups)) != set(ALL_OBJECTS):
        raise ValueError(
            "The seeded validation split does not cover all seven objects; "
            "choose and document a new split seed before training."
        )
    train_groups = sorted(set(train_val_groups) - set(val_groups))

    source_index_sha256 = sha256_file(index_path)
    split_groups = {
        "train": train_groups,
        "validation": val_groups,
        "test": test_groups,
    }
    manifests = {}
    for split, selected_groups in split_groups.items():
        episodes = [record for group in selected_groups for record in groups[group]]
        episodes.sort(key=lambda record: record["episode_index"])
        frame_count = sum(record["end"] - record["start"] + 1 for record in episodes)
        manifests[split] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": "MetaCALVIN-RA-L-P0-7Object-v1",
            "split": split,
            "split_seed": split_seed,
            "grouping_unit": "object_combination",
            "source_dataset_name": dataset_dir.name,
            "source_index_sha256": source_index_sha256,
            "selection_rule": (
                "test: all valid 4-object configurations containing slider and led; "
                "validation: Python random.Random(split_seed).sample over the sorted "
                "remaining configurations; train: all remaining configurations"
            ),
            "object_combinations": [list(group) for group in selected_groups],
            "num_object_combinations": len(selected_groups),
            "num_episodes": len(episodes),
            "num_frames": frame_count,
            "num_windows_ws12": sum(
                record["end"] - record["start"] + 1 - 12 for record in episodes
            ),
            "episodes": episodes,
        }
    return manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    manifests = build_manifests(args.dataset_dir.resolve(), args.split_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {}
    for split, manifest in manifests.items():
        output = args.output_dir / f"7object_{split}.json"
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_hashes[split] = sha256_file(output)
        print(
            f"{split}: groups={manifest['num_object_combinations']}, "
            f"episodes={manifest['num_episodes']}, "
            f"windows_ws12={manifest['num_windows_ws12']}, "
            f"sha256={manifest_hashes[split]}, path={output}"
        )

    group_sets = {
        split: {tuple(group) for group in manifest["object_combinations"]}
        for split, manifest in manifests.items()
    }
    episode_sets = {
        split: {record["episode_index"] for record in manifest["episodes"]}
        for split, manifest in manifests.items()
    }
    pairwise_group_overlap = {}
    pairwise_episode_overlap = {}
    for left, right in itertools.combinations(("train", "validation", "test"), 2):
        key = f"{left}__{right}"
        pairwise_group_overlap[key] = len(group_sets[left] & group_sets[right])
        pairwise_episode_overlap[key] = len(episode_sets[left] & episode_sets[right])

    audit = {
        "schema_version": 1,
        "protocol": "MetaCALVIN-RA-L-P0-7Object-v1",
        "source_dataset_name": args.dataset_dir.resolve().name,
        "source_index_sha256": manifests["train"]["source_index_sha256"],
        "split_seed": args.split_seed,
        "observed_total_object_combinations": sum(
            manifest["num_object_combinations"] for manifest in manifests.values()
        ),
        "observed_total_episodes": sum(
            manifest["num_episodes"] for manifest in manifests.values()
        ),
        "source_contains_legacy_test_combinations": True,
        "leakage_check": {
            "pairwise_group_overlap": pairwise_group_overlap,
            "pairwise_episode_overlap": pairwise_episode_overlap,
            "all_pairwise_overlaps_are_zero": not any(pairwise_group_overlap.values())
            and not any(pairwise_episode_overlap.values()),
        },
        "splits": {
            split: {
                "manifest_sha256": manifest_hashes[split],
                "num_object_combinations": manifest["num_object_combinations"],
                "num_episodes": manifest["num_episodes"],
                "num_frames": manifest["num_frames"],
                "num_windows_ws12": manifest["num_windows_ws12"],
                "object_coverage": sorted(
                    set().union(*(set(group) for group in group_sets[split]))
                ),
            }
            for split, manifest in manifests.items()
        },
    }
    audit_path = args.output_dir / "7object_split_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"audit: sha256={sha256_file(audit_path)}, path={audit_path}")


if __name__ == "__main__":
    main()
