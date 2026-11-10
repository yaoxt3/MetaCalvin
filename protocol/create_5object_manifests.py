#!/usr/bin/env python3
"""Create leakage-free 5-Object train/validation/test manifests.

The legacy source contains all ten three-object configurations, including the
three configurations containing both slider and LED that the environment marks
as held-out evaluation groups.  Those three groups are quarantined as final
test.  Validation is selected from the remaining seven groups using only
configuration membership and episode counts, never rewards, losses, or model
outputs.
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
DEFAULT_VALIDATION_FRACTION = 0.20
ALL_OBJECTS = (
    "slider",
    "drawer",
    "led",
    "lightbulb",
    "red_block",
)
PROTOCOL = "MetaCALVIN-RA-L-P0-5Object-T4-v1"


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


def expected_groups():
    return sorted(
        tuple(sorted(combination))
        for combination in itertools.combinations(ALL_OBJECTS, 3)
    )


def choose_validation_groups(groups, eligible_groups, split_seed, target_fraction):
    """Choose two full-coverage groups closest to the target episode fraction.

    Requiring full object coverage makes validation useful for every object.
    Balancing by episode counts avoids wasting a disproportionate part of this
    small seven-group training pool.  A seeded choice is used only if multiple
    candidates have exactly the same metadata-only score.
    """
    candidates = [
        tuple(candidate)
        for candidate in itertools.combinations(sorted(eligible_groups), 2)
        if set().union(*(set(group) for group in candidate)) == set(ALL_OBJECTS)
    ]
    if not candidates:
        raise ValueError("No two-group validation split covers all five objects")

    eligible_episode_count = sum(len(groups[group]) for group in eligible_groups)
    target_episode_count = target_fraction * eligible_episode_count

    def score(candidate):
        count = sum(len(groups[group]) for group in candidate)
        return abs(count - target_episode_count)

    best_score = min(score(candidate) for candidate in candidates)
    tied = sorted(
        candidate for candidate in candidates if score(candidate) == best_score
    )
    selected = random.Random(split_seed).choice(tied)
    return sorted(selected), {
        "candidate_rule": "exactly two groups whose union covers all five objects",
        "target_validation_episode_fraction": target_fraction,
        "eligible_episode_count": eligible_episode_count,
        "target_validation_episode_count": round(target_episode_count, 10),
        "num_valid_candidates": len(candidates),
        "num_best_score_ties": len(tied),
        "selected_validation_episode_count": sum(
            len(groups[group]) for group in selected
        ),
        "absolute_episode_count_error": round(best_score, 10),
        "tie_break_seed": split_seed,
    }


def build_manifests(dataset_dir, split_seed, validation_fraction):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")

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

    expected = expected_groups()
    if sorted(groups) != expected:
        missing = sorted(set(expected) - set(groups))
        unexpected = sorted(set(groups) - set(expected))
        raise ValueError(
            f"Unexpected object configurations; missing={missing}, "
            f"unexpected={unexpected}"
        )

    test_groups = sorted(
        group for group in groups if {"slider", "led"} <= set(group)
    )
    eligible_groups = sorted(set(groups) - set(test_groups))
    if len(test_groups) != 3 or len(eligible_groups) != 7:
        raise ValueError(
            f"Expected 7 train/validation and 3 test groups, got "
            f"{len(eligible_groups)} and {len(test_groups)}"
        )

    validation_groups, validation_selection = choose_validation_groups(
        groups,
        eligible_groups,
        split_seed,
        validation_fraction,
    )
    train_groups = sorted(set(eligible_groups) - set(validation_groups))

    source_index_sha256 = sha256_file(index_path)
    split_groups = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    selection_rule = (
        "test: all 3-object configurations containing slider and led; "
        "validation: among the other seven groups, select two whose union covers "
        f"all five objects and whose episode count is closest to "
        f"{validation_fraction:.6g} of eligible "
        "episodes, with split_seed used only for exact-score ties; train: all "
        "remaining groups"
    )

    manifests = {}
    for split, selected_groups in split_groups.items():
        episodes = [record for group in selected_groups for record in groups[group]]
        episodes.sort(key=lambda record: record["episode_index"])
        frame_count = sum(record["end"] - record["start"] + 1 for record in episodes)
        manifests[split] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "split": split,
            "split_seed": split_seed,
            "grouping_unit": "object_combination",
            "source_dataset_name": dataset_dir.name,
            "source_index_sha256": source_index_sha256,
            "selection_rule": selection_rule,
            "object_combinations": [list(group) for group in selected_groups],
            "num_object_combinations": len(selected_groups),
            "num_episodes": len(episodes),
            "num_frames": frame_count,
            # Match DiskCalvinDataset._build_file_indices for min/window size 12.
            "num_windows_ws12": sum(
                record["end"] - record["start"] + 1 - 12 for record in episodes
            ),
            "episodes": episodes,
        }
    return manifests, validation_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    args = parser.parse_args()

    manifests, validation_selection = build_manifests(
        args.dataset_dir.resolve(),
        args.split_seed,
        args.validation_fraction,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {}
    for split, manifest in manifests.items():
        output = args.output_dir / f"5object_{split}.json"
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
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
        "validation_selection": validation_selection,
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
    audit_path = args.output_dir / "5object_split_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"audit: sha256={sha256_file(audit_path)}, path={audit_path}")


if __name__ == "__main__":
    main()
