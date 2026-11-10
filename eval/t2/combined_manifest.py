#!/usr/bin/env python3
"""Generate frozen Combined-Explore simulator rollout manifests.

The P0 Combined split files intentionally contain task pairs rather than
simulator episodes.  This module expands one validation or test task-pair
split into exact semantic states, candidate orders, feedback templates, and
per-trial robot/scene reset observations.  It does not load a policy and does
not train or modify a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


T2_DIR = Path(__file__).resolve().parent
ROBOTEST_ROOT = T2_DIR.parents[1]
for dependency_path in (
    ROBOTEST_ROOT,
    ROBOTEST_ROOT / "calvin" / "calvin_models",
    ROBOTEST_ROOT / "calvin" / "calvin_env",
    ROBOTEST_ROOT / "calvin" / "calvin_env" / "tacto",
):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from env.multi_env import MultiEnv  # noqa: E402
from env.multi_env_utils import (  # noqa: E402
    TASK_INSTRUCTIONS,
    build_template_completed_task,
    build_template_first_trial,
    build_template_further_trial,
)
from eval.recording import sha256_file, write_json_atomic  # noqa: E402
from eval.utils import get_environment_states_for_task  # noqa: E402
from protocol.create_combined_manifests import (  # noqa: E402
    canonical_pair,
    valid_pairs,
)


SCHEMA_VERSION = 1
PROTOCOL = "MetaCALVIN-RA-L-P0-Combined-Rollout-v1"
DEFAULT_MANIFEST_SEED = 1
DEFAULT_NUM_TRIALS = 4
DEFAULT_TRIAL_STEPS = 500
DEFAULT_NUM_RESETS = 25
STATIC_OBJECTS = ("slider", "drawer", "led", "lightbulb")


def _template_id(stage: str, template: str) -> str:
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
    return f"diverse:{stage}:{digest}"


def _json_array(value: Any) -> list:
    return np.asarray(value, dtype=np.float64).tolist()


def _semantic_env() -> MultiEnv:
    """Create only the state-generation portion of MultiEnv.

    Combined reset-state generation uses no PyBullet state.  Avoiding the
    normal constructor keeps manifest creation CPU-only and independent of a
    renderer while preserving MultiEnv's canonical state-to-task logic.
    """

    env = object.__new__(MultiEnv)
    env.objects = list(STATIC_OBJECTS)
    env.static_objects = list(STATIC_OBJECTS)
    return env


def load_task_pairs(path: Path, split: str) -> tuple[Dict[str, Any], list[list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "MetaCALVIN-RA-L-P0-Combined-v1":
        raise ValueError(f"Unexpected Combined split protocol: {payload.get('protocol')!r}")
    if payload.get("split") != split:
        raise ValueError(
            f"Combined task split is {payload.get('split')!r}, expected {split!r}"
        )
    raw_pairs = payload.get("meta_task_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("Combined task split contains no meta_task_pairs")

    allowed = set(valid_pairs())
    pairs: list[list[str]] = []
    seen = set()
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise ValueError(f"Invalid Combined task pair: {raw_pair!r}")
        pair = canonical_pair(raw_pair)
        if pair not in allowed:
            raise ValueError(f"Invalid or same-object Combined task pair: {raw_pair!r}")
        if pair in seen:
            raise ValueError(f"Duplicate Combined task pair: {raw_pair!r}")
        seen.add(pair)
        pairs.append(list(pair))
    if payload.get("num_meta_task_pairs") != len(pairs):
        raise ValueError("num_meta_task_pairs does not match meta_task_pairs")
    return payload, pairs


def create_rollout_manifest(
    *,
    split_manifest_path: Path,
    split: str,
    output_path: Path,
    seed: int,
    num_resets: int,
    num_trials: int,
    trial_steps: int,
    max_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported Combined rollout split: {split!r}")
    if num_resets <= 0 or num_trials <= 0 or trial_steps <= 0:
        raise ValueError("num_resets, num_trials, and trial_steps must be positive")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {output_path}")

    source, task_pairs = load_task_pairs(split_manifest_path, split)
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        task_pairs = task_pairs[:max_pairs]

    random.seed(seed)
    np.random.seed(seed)
    env = _semantic_env()
    helper_args = argparse.Namespace()
    episodes = []

    for task_pair_index, task_pair in enumerate(task_pairs):
        for reset_index in range(num_resets):
            environment_state, feedback_state = get_environment_states_for_task(
                helper_args, task_pair, env
            )
            env.state_to_tasks(environment_state, feedback_state)
            candidate_tasks = list(env.sampled_subtasks)
            if len(candidate_tasks) != 4 or len(set(candidate_tasks)) != 4:
                raise ValueError(
                    f"Expected four unique Combined candidates, got {candidate_tasks!r}"
                )
            if not all(task in candidate_tasks for task in task_pair):
                raise ValueError(
                    f"Target pair {task_pair!r} is infeasible under {candidate_tasks!r}"
                )

            trial_reset_states = []
            for trial_index in range(num_trials):
                robot_obs, scene_obs = env.get_env_state_for_initial_condition(
                    environment_state
                )
                trial_reset_states.append(
                    {
                        "trial_index": trial_index + 1,
                        "robot_obs": _json_array(robot_obs),
                        "scene_obs": _json_array(scene_obs),
                    }
                )

            templates = {
                "first_trial_template": build_template_first_trial(eval=True),
                "end_trial_template": build_template_further_trial(eval=True),
                "completed_task_template": build_template_completed_task(eval=True),
            }
            template_ids = {
                "initial": _template_id("initial", templates["first_trial_template"]),
                "end_trial": _template_id("end_trial", templates["end_trial_template"]),
                "intra_trial": _template_id(
                    "intra_trial", templates["completed_task_template"]
                ),
            }
            sampled_instructions = {
                task: random.choice(TASK_INSTRUCTIONS[task]) for task in TASK_INSTRUCTIONS
            }
            episode_index = len(episodes)
            episodes.append(
                {
                    "episode_id": f"combined_{split}_{episode_index:05d}",
                    "episode_index": episode_index,
                    "task_pair_index": task_pair_index,
                    "reset_index": reset_index,
                    "feedback_state": list(feedback_state),
                    "candidate_tasks": candidate_tasks,
                    "candidate_order": candidate_tasks,
                    "meta_task": list(task_pair),
                    "canonical_meta_task_pair": list(canonical_pair(task_pair)),
                    "environment_state": dict(environment_state),
                    "trial_reset_states": trial_reset_states,
                    "feedback_templates": templates,
                    "feedback_template_ids": template_ids,
                    "sampled_subtask_instructions": sampled_instructions,
                }
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "split": split,
        "generation_seed": seed,
        "source_split_manifest": str(split_manifest_path.resolve()),
        "source_split_manifest_sha256": sha256_file(split_manifest_path),
        "source_split_seed": source.get("split_seed"),
        "num_meta_task_pairs": len(task_pairs),
        "num_resets_per_pair": num_resets,
        "num_trials": num_trials,
        "trial_steps": trial_steps,
        "num_episodes": len(episodes),
        "physical_reset_policy": "exact robot_obs and scene_obs for every trial",
        "semantic_sampling_policy": (
            "legacy Combined get_environment_states_for_task with independent "
            "Python and NumPy RNGs seeded by generation_seed"
        ),
        "episodes": episodes,
    }
    validate_rollout_manifest(payload, expected_split=split)
    write_json_atomic(output_path, payload)
    return payload


def validate_rollout_manifest(
    payload: Mapping[str, Any], *, expected_split: Optional[str] = None
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported Combined rollout schema: {payload.get('schema_version')}")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected Combined rollout protocol: {payload.get('protocol')!r}")
    if expected_split is not None and payload.get("split") != expected_split:
        raise ValueError(
            f"Combined rollout split is {payload.get('split')!r}, expected {expected_split!r}"
        )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Combined rollout manifest contains no episodes")
    if payload.get("num_episodes") != len(episodes):
        raise ValueError("num_episodes does not match the Combined episode list")
    expected_count = int(payload["num_meta_task_pairs"]) * int(
        payload["num_resets_per_pair"]
    )
    if len(episodes) != expected_count:
        raise ValueError(
            f"Expected {expected_count} Combined episodes, found {len(episodes)}"
        )

    valid = set(valid_pairs())
    num_trials = int(payload["num_trials"])
    seen_ids = set()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if not episode_id or episode_id in seen_ids:
            raise ValueError(f"Missing or duplicate episode_id: {episode_id!r}")
        seen_ids.add(episode_id)
        feedback_state = episode.get("feedback_state")
        if sorted(feedback_state or []) != sorted(STATIC_OBJECTS):
            raise ValueError(f"Invalid Combined feedback_state in {episode_id}")
        candidates = episode.get("candidate_tasks")
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError(f"Invalid Combined candidates in {episode_id}")
        if len(set(candidates)) != 4 or episode.get("candidate_order") != candidates:
            raise ValueError(f"Candidate order mismatch in {episode_id}")
        pair = canonical_pair(episode.get("meta_task", []))
        if pair not in valid or not all(task in candidates for task in pair):
            raise ValueError(f"Infeasible Combined meta-task pair in {episode_id}")
        if episode.get("canonical_meta_task_pair") != list(pair):
            raise ValueError(f"Canonical pair mismatch in {episode_id}")

        resets = episode.get("trial_reset_states", [])
        if len(resets) != num_trials:
            raise ValueError(f"Episode {episode_id} has {len(resets)} reset states")
        for state in resets:
            if np.asarray(state.get("robot_obs", [])).shape != (15,):
                raise ValueError(f"Invalid robot_obs in {episode_id}")
            if np.asarray(state.get("scene_obs", [])).shape != (24,):
                raise ValueError(f"Invalid scene_obs in {episode_id}")
        for key in (
            "first_trial_template",
            "end_trial_template",
            "completed_task_template",
        ):
            if key not in episode.get("feedback_templates", {}):
                raise ValueError(f"Missing {key} in {episode_id}")
        for key in ("initial", "end_trial", "intra_trial"):
            if key not in episode.get("feedback_template_ids", {}):
                raise ValueError(f"Missing {key} template ID in {episode_id}")


def load_rollout_manifest(
    path: Path, *, expected_split: Optional[str] = None
) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_rollout_manifest(payload, expected_split=expected_split)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a frozen Combined-Explore rollout manifest"
    )
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-seed", type=int, default=DEFAULT_MANIFEST_SEED)
    parser.add_argument("--num-resets", type=int, default=DEFAULT_NUM_RESETS)
    parser.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument("--trial-steps", type=int, default=DEFAULT_TRIAL_STEPS)
    parser.add_argument("--max-pairs", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.split_manifest.is_file():
        parser.error(f"Combined split manifest does not exist: {args.split_manifest}")
    try:
        payload = create_rollout_manifest(
            split_manifest_path=args.split_manifest.resolve(),
            split=args.split,
            output_path=args.output.resolve(),
            seed=args.manifest_seed,
            num_resets=args.num_resets,
            num_trials=args.num_trials,
            trial_steps=args.trial_steps,
            max_pairs=args.max_pairs,
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(
        f"split={payload['split']} pairs={payload['num_meta_task_pairs']} "
        f"episodes={payload['num_episodes']} seed={payload['generation_seed']} "
        f"sha256={sha256_file(args.output.resolve())} path={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
