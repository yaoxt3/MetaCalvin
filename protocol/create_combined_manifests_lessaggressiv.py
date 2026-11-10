#!/usr/bin/env python3
"""Build and audit manifests for the matched Combined Minimal dataset."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROTOCOL = "MetaCALVIN-RA-L-F1-Combined-Minimal-lessaggressiv-v1"
STATE_FILENAME = "collection_state_lessaggressiv.json"
METADATA_FILENAME = "episode_metadata_lessaggressiv.jsonl"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_metadata(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Invalid metadata JSON on line %d: %s" % (line_number, exc)
                    )
    return records


def reference_summary(path, expected_split):
    manifest = load_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("split") != expected_split:
        raise ValueError("Invalid %s reference manifest: %s" % (expected_split, path))
    pair_frames = Counter()
    pair_episodes = Counter()
    episode_targets = {}
    for record in manifest["episodes"]:
        pair = tuple(record["meta_task_pair"])
        length = int(record["end"]) - int(record["start"]) + 1
        reference_episode_index = int(record["episode_index"])
        if reference_episode_index in episode_targets:
            raise ValueError("Duplicate reference episode index")
        episode_targets[reference_episode_index] = (pair, length)
        pair_frames[pair] += length
        pair_episodes[pair] += 1
    return manifest, {
        "sha256": sha256_file(path),
        "num_episodes": int(manifest["num_episodes"]),
        "num_frames": int(manifest["num_frames"]),
        "num_windows_ws12": int(manifest["num_windows_ws12"]),
        "pair_frames": pair_frames,
        "pair_episodes": pair_episodes,
        "episode_targets": episode_targets,
    }


def validate_dataset(dataset_dir, records, state):
    index_path = dataset_dir / "ep_start_end_ids.npy"
    episode_ranges = np.load(str(index_path))
    if len(episode_ranges) != len(records):
        raise ValueError("Index and metadata episode counts differ")
    expected_next = 0
    for episode_index, (record, saved_range) in enumerate(zip(records, episode_ranges)):
        if int(record["episode_index"]) != episode_index:
            raise ValueError("Non-sequential metadata episode index")
        actual = tuple(map(int, saved_range))
        expected = (int(record["start"]), int(record["end"]))
        if actual != expected:
            raise ValueError("Range mismatch at episode %d" % episode_index)
        if actual[0] != expected_next:
            raise ValueError("Dataset frame IDs are not contiguous")
        saved_length = actual[1] - actual[0] + 1
        if saved_length != int(record["saved_length"]):
            raise ValueError("Saved-length mismatch at episode %d" % episode_index)
        if int(record["first_completion_frames"]) > saved_length:
            raise ValueError(
                "Episode %d was clipped before its first completed subtask"
                % episode_index
            )
        expected_next = actual[1] + 1

    frame_files = list(dataset_dir.glob("episode_*.npz"))
    if len(frame_files) != expected_next:
        raise ValueError(
            "Physical frame-file count mismatch: expected %d, found %d"
            % (expected_next, len(frame_files))
        )
    if int(state["next_frame_index"]) != expected_next:
        raise ValueError("Collection state frame count does not match the index")
    return index_path, episode_ranges


def build(dataset_dir, train_reference_path, validation_reference_path):
    dataset_dir = Path(dataset_dir).resolve()
    state = load_json(dataset_dir / STATE_FILENAME)
    if not state.get("complete"):
        raise ValueError("Collection is not marked complete")
    records = load_metadata(dataset_dir / METADATA_FILENAME)
    if len(records) != int(state["next_target_index"]):
        raise ValueError("Collection state and metadata record counts differ")
    index_path, _ = validate_dataset(dataset_dir, records, state)

    references = {}
    for split, path in (
        ("train", Path(train_reference_path).resolve()),
        ("validation", Path(validation_reference_path).resolve()),
    ):
        manifest, summary = reference_summary(path, split)
        frozen = state["reference_manifests"][split]
        if summary["sha256"] != frozen["sha256"]:
            raise ValueError("Reference %s manifest changed after collection" % split)
        references[split] = (manifest, summary, path)

    source_index_sha256 = sha256_file(index_path)
    output_manifests = {}
    split_audit = {}
    for split in ("train", "validation"):
        reference, reference_stats, reference_path = references[split]
        selected = [record for record in records if record["split"] == split]
        selected.sort(key=lambda record: int(record["episode_index"]))
        pair_frames = Counter()
        pair_episodes = Counter()
        episodes = []
        for record in selected:
            pair = tuple(record["meta_task_pair"])
            length = int(record["saved_length"])
            reference_episode_index = int(record["reference_episode_index"])
            expected_pair, expected_length = reference_stats["episode_targets"].get(
                reference_episode_index, (None, None)
            )
            if pair != expected_pair or length != expected_length:
                raise ValueError(
                    "%s episode %d does not match its declared strict reference"
                    % (split, int(record["episode_index"]))
                )
            pair_frames[pair] += length
            pair_episodes[pair] += 1
            episodes.append(
                {
                    "episode_index": int(record["episode_index"]),
                    "start": int(record["start"]),
                    "end": int(record["end"]),
                    "meta_task_pair": list(pair),
                    "first_completion_frames": int(record["first_completion_frames"]),
                    "reference_episode_index": int(
                        record["reference_episode_index"]
                    ),
                }
            )
        num_frames = sum(item["end"] - item["start"] + 1 for item in episodes)
        num_windows = sum(item["end"] - item["start"] + 1 - 12 for item in episodes)
        actual = {
            "num_episodes": len(episodes),
            "num_frames": num_frames,
            "num_windows_ws12": num_windows,
        }
        for key in actual:
            if actual[key] != reference_stats[key]:
                raise ValueError("%s %s is not reference-matched" % (split, key))
        if pair_frames != reference_stats["pair_frames"]:
            raise ValueError("%s per-pair frame distribution differs" % split)
        if pair_episodes != reference_stats["pair_episodes"]:
            raise ValueError("%s per-pair episode distribution differs" % split)

        output_manifests[split] = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "split": split,
            "split_seed": reference["split_seed"],
            "grouping_unit": "unordered_meta_task_pair",
            "source_dataset_name": dataset_dir.name,
            "source_index_sha256": source_index_sha256,
            "filtering": "minimal_at_least_one_completed_subtask",
            "matching_rule": (
                "Each Minimal episode is paired with one frozen Strict reference "
                "episode from the same task pair and clipped to exactly the same "
                "length; the retained prefix must contain a completed subtask."
            ),
            "reference_manifest": str(reference_path),
            "reference_manifest_sha256": reference_stats["sha256"],
            "meta_task_pairs": reference["meta_task_pairs"],
            "num_meta_task_pairs": reference["num_meta_task_pairs"],
            **actual,
            "episodes": episodes,
        }
        split_audit[split] = {
            **actual,
            "reference_num_episodes": reference_stats["num_episodes"],
            "reference_num_frames": reference_stats["num_frames"],
            "reference_num_windows_ws12": reference_stats["num_windows_ws12"],
            "per_pair_episode_counts_match": True,
            "per_pair_frame_counts_match": True,
        }

    rejection_counts = dict(state.get("rejections", {}))
    audit = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_dataset_name": dataset_dir.name,
        "source_index_sha256": source_index_sha256,
        "filtering": "minimal_at_least_one_completed_subtask",
        "exact_reference_matching": True,
        "splits": split_audit,
        "collection": {
            "attempts": int(state["attempts"]),
            "accepted": len(records),
            "rejections": rejection_counts,
            "retention_rate": (
                len(records) / float(state["attempts"]) if state["attempts"] else 0.0
            ),
        },
    }
    return output_manifests, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-train-manifest", type=Path, required=True)
    parser.add_argument("--reference-validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifests, audit = build(
        args.dataset_dir,
        args.reference_train_manifest,
        args.reference_validation_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, manifest in manifests.items():
        path = args.output_dir / ("combined_%s_lessaggressiv.json" % split)
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "%s: episodes=%d frames=%d windows_ws12=%d sha256=%s path=%s"
            % (
                split,
                manifest["num_episodes"],
                manifest["num_frames"],
                manifest["num_windows_ws12"],
                sha256_file(path),
                path,
            )
        )
    audit_path = args.output_dir / "combined_split_audit_lessaggressiv.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("audit: sha256=%s path=%s" % (sha256_file(audit_path), audit_path))


if __name__ == "__main__":
    main()
