"""Creation and validation of frozen T1 rollout episode manifests."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION = 1
PROTOCOL = "MetaCALVIN-RA-L-T1-7Object-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_split_combinations(path: Path, expected_split: str) -> List[List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != expected_split:
        raise ValueError(
            f"Split manifest says {payload.get('split')!r}, expected {expected_split!r}"
        )
    combinations = payload.get("object_combinations")
    if not combinations:
        raise ValueError(f"Split manifest has no object_combinations: {path}")
    return [list(combination) for combination in combinations]


def _json_array(value: Any) -> list:
    return np.asarray(value, dtype=np.float64).tolist()


def _template_id(stage: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"diverse:{stage}:{digest}"


def create_rollout_manifest(
    env: Any,
    *,
    split: str,
    split_manifest_path: Path,
    output_path: Path,
    seed: int,
    num_resets: int,
    num_trials: int,
    trial_steps: int,
    max_combinations: Optional[int] = None,
) -> Dict[str, Any]:
    """Freeze semantic states, feedback templates, and every physical reset."""

    if num_resets <= 0 or num_trials <= 0 or trial_steps <= 0:
        raise ValueError("num_resets, num_trials, and trial_steps must be positive")
    combinations = load_split_combinations(split_manifest_path, split)
    if max_combinations is not None:
        combinations = combinations[:max_combinations]

    # get_environment_states is retained to preserve the legacy 7-object state
    # and balanced meta-task sampling semantics. Its global RNG consumption is
    # contained here; every resulting determinant is serialized below.
    from env.env_utils import (
        REWARD_TEMPLATES_EVAL,
        TASK_INSTRUCTIONS,
        build_template_first_trial,
        build_template_further_trial,
    )
    from eval.utils import get_environment_states

    random.seed(seed)
    np.random.seed(seed)
    helper_args = SimpleNamespace(
        num_objs=7,
        num_resets=num_resets,
        num_trials=num_trials,
    )
    episodes = []
    for combination_index, source_combination in enumerate(combinations):
        exploration_limits = {index: 0 for index in range(num_trials - 1)}
        for reset_index in range(num_resets):
            combination = list(source_combination)
            random.shuffle(combination)
            environment_state, meta_task, candidate_tasks, meta_task_index = get_environment_states(
                helper_args,
                env,
                combination,
                exploration_limits,
            )
            feedback_state = list(combination)
            # Verify the exact candidate order instead of trusting a duplicated
            # value returned by the legacy helper.
            computed_tasks = env.state_to_tasks(environment_state, feedback_state)
            if list(candidate_tasks) != list(computed_tasks):
                raise ValueError("Candidate task generation is internally inconsistent")

            trial_reset_states = []
            for trial_index in range(num_trials):
                robot_obs, scene_obs = env.get_env_state_for_initial_condition(
                    environment_state, False
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
                "further_trial_template": build_template_further_trial(eval=True),
                "reward_template": random.choice(REWARD_TEMPLATES_EVAL),
            }
            template_ids = {
                "initial": _template_id("initial", templates["first_trial_template"]),
                "failure": _template_id("failure", templates["further_trial_template"]),
                "reward": _template_id("reward", templates["reward_template"]),
            }
            sampled_task_instructions = {
                task: random.choice(TASK_INSTRUCTIONS[task]) for task in TASK_INSTRUCTIONS
            }
            episode_index = len(episodes)
            episodes.append(
                {
                    "episode_id": f"{split}_{episode_index:05d}",
                    "episode_index": episode_index,
                    "combination_index": combination_index,
                    "reset_index": reset_index,
                    "feedback_state": feedback_state,
                    "candidate_tasks": list(candidate_tasks),
                    "candidate_order": list(candidate_tasks),
                    "meta_task": meta_task,
                    "meta_task_index": meta_task_index,
                    "environment_state": dict(environment_state),
                    "trial_reset_states": trial_reset_states,
                    "feedback_templates": templates,
                    "feedback_template_ids": template_ids,
                    "sampled_task_instructions": sampled_task_instructions,
                }
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "split": split,
        "generation_seed": seed,
        "source_split_manifest": str(split_manifest_path.resolve()),
        "source_split_manifest_sha256": sha256_file(split_manifest_path),
        "num_object_combinations": len(combinations),
        "num_resets_per_combination": num_resets,
        "num_trials": num_trials,
        "trial_steps": trial_steps,
        "num_episodes": len(episodes),
        "physical_reset_policy": "exact robot_obs and scene_obs for every trial",
        "episodes": episodes,
    }
    validate_rollout_manifest(payload, expected_split=split)
    write_json_atomic(output_path, payload)
    return payload


def validate_rollout_manifest(
    payload: Mapping[str, Any], *, expected_split: Optional[str] = None
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported T1 manifest schema: {payload.get('schema_version')}")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected T1 protocol: {payload.get('protocol')}")
    if expected_split is not None and payload.get("split") != expected_split:
        raise ValueError(
            f"Rollout manifest split is {payload.get('split')!r}, expected {expected_split!r}"
        )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("T1 rollout manifest contains no episodes")
    if payload.get("num_episodes") != len(episodes):
        raise ValueError("T1 rollout manifest num_episodes does not match its episode list")
    num_trials = int(payload["num_trials"])
    seen_ids = set()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if not episode_id or episode_id in seen_ids:
            raise ValueError(f"Missing or duplicate episode_id: {episode_id!r}")
        seen_ids.add(episode_id)
        candidates = episode.get("candidate_tasks")
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f"Invalid candidates in episode {episode_id}")
        if episode.get("candidate_order") != candidates:
            raise ValueError(f"candidate_order mismatch in episode {episode_id}")
        if episode.get("meta_task") not in candidates:
            raise ValueError(f"Infeasible meta_task in episode {episode_id}")
        reset_states = episode.get("trial_reset_states", [])
        if len(reset_states) != num_trials:
            raise ValueError(f"Episode {episode_id} has {len(reset_states)} reset states")
        for state in reset_states:
            if np.asarray(state.get("robot_obs", [])).shape != (15,):
                raise ValueError(f"Invalid robot_obs in episode {episode_id}")
            if np.asarray(state.get("scene_obs", [])).shape != (24,):
                raise ValueError(f"Invalid scene_obs in episode {episode_id}")
        for key in ("first_trial_template", "further_trial_template", "reward_template"):
            if key not in episode.get("feedback_templates", {}):
                raise ValueError(f"Missing {key} in episode {episode_id}")
        for key in ("initial", "failure", "reward"):
            if key not in episode.get("feedback_template_ids", {}):
                raise ValueError(f"Missing {key} feedback template ID in episode {episode_id}")


def load_rollout_manifest(path: Path, *, expected_split: Optional[str] = None) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_rollout_manifest(payload, expected_split=expected_split)
    return payload
