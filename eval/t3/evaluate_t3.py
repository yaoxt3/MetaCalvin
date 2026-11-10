#!/usr/bin/env python3
"""Frozen-policy T3 external-text-memory by recurrent-state evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

T3_DIR = Path(__file__).resolve().parent
ROBOTEST_ROOT = T3_DIR.parents[1]
for dependency_path in (
    ROBOTEST_ROOT,
    ROBOTEST_ROOT / "models",
    ROBOTEST_ROOT / "open_flamingo",
    ROBOTEST_ROOT / "calvin" / "calvin_models",
    ROBOTEST_ROOT / "calvin" / "calvin_env",
    ROBOTEST_ROOT / "calvin" / "calvin_env" / "tacto",
):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import torch

from eval.recording import (
    action_summary,
    append_jsonl,
    latency_summary,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from eval.t1.controllers import DirectController, action_to_list
from eval.t1.evaluate_t1 import (
    aggregate_results as aggregate_t1_results,
    completed_tasks,
    create_environment,
    failure_type,
    load_policy,
    runtime_metadata,
    seed_everything,
)
from eval.t1.manifest import load_rollout_manifest
from eval.t3.memory_conditions import TEXT_MEMORY_MODES, TextMemoryLedger
from eval.t3.recurrent_state import (
    RECURRENT_STATE_MODES,
    metadata as recurrent_metadata,
    prepare_trial_boundary,
    recurrent_snapshot,
)


PROTOCOL = "MetaCALVIN-RA-L-T3-7Object-v1"
REFERENCE_CONDITION = ("local", "persistent")
ACTION_PREFIX_STEPS = 10
ACTION_CHANGE_TOLERANCE = 1e-5


def condition_name(text_memory: str, recurrent_state: str) -> str:
    return f"{text_memory}_{recurrent_state}"


def evaluation_run_name(args: argparse.Namespace) -> str:
    return (
        f"T3_{args.experiment}_{condition_name(args.text_memory, args.recurrent_state)}_"
        f"{args.split}_train_seed_{args.training_seed}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--experiment", choices=("E0", "E3", "E4", "E7"), required=True
    )
    result.add_argument("--text_memory", choices=TEXT_MEMORY_MODES, required=True)
    result.add_argument("--recurrent_state", choices=RECURRENT_STATE_MODES, required=True)
    result.add_argument("--split", choices=("validation", "test"), required=True)
    result.add_argument("--episode_manifest", type=Path, required=True)
    result.add_argument("--student_checkpoint", type=Path, required=True)
    result.add_argument("--training_seed", type=int, required=True)
    result.add_argument("--output_dir", type=Path, required=True)
    result.add_argument("--reference_trials_path", type=Path)
    result.add_argument("--expected_checkpoint_sha256")
    result.add_argument("--final_test_ack", action="store_true")
    result.add_argument("--max_episodes", type=int)
    result.add_argument("--save_actions", action="store_true")
    result.add_argument("--evaluation_seed", type=int, default=42)
    result.add_argument(
        "--feedback",
        choices=("train_feedback", "alt_feedback"),
        default="train_feedback",
    )
    result.add_argument("--num_trials", type=int, required=True)
    result.add_argument("--trial_steps", type=int, required=True)
    result.add_argument("--path_to_env", type=Path, required=True)
    result.add_argument("--path_to_oracle", type=Path, required=True)
    result.add_argument("--disable_egl", action="store_true")
    result.add_argument("--vision_encoder_path", default="ViT-L-14")
    result.add_argument("--vision_encoder_pretrained", default="openai")
    result.add_argument("--lm_path", default="anas-awadalla/mpt-1b-redpajama-200b-dolly")
    result.add_argument("--tokenizer_path", default="anas-awadalla/mpt-1b-redpajama-200b-dolly")
    result.add_argument("--cross_attn_every_n_layers", type=int, default=1)
    result.add_argument("--disable_gripper", action="store_false", dest="use_gripper")
    result.set_defaults(use_gripper=True)
    result.add_argument("--fusion_mode", default="post")
    result.add_argument("--llm_name", default="mpt_dolly_3b")
    result.add_argument("--decoder_type", choices=("lstm", "lstm_seq"), required=True)
    result.add_argument("--action_tokens", action="store_true")
    result.add_argument("--window_size", type=int, required=True)
    return result


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.episode_manifest, "episode manifest"),
        (args.student_checkpoint, "student checkpoint"),
        (args.path_to_oracle, "task oracle"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if not args.path_to_env.is_dir():
        raise ValueError(f"CALVIN environment directory does not exist: {args.path_to_env}")
    if args.split == "test" and not args.final_test_ack:
        raise ValueError("Formal T3 test requires --final_test_ack")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max_episodes must be positive")
    expected_decoder = {
        "E0": "lstm",
        "E3": "lstm_seq",
        "E4": "lstm",
        "E7": "lstm_seq",
    }[args.experiment]
    if args.decoder_type != expected_decoder:
        raise ValueError(
            f"{args.experiment} requires decoder_type={expected_decoder}, got {args.decoder_type}"
        )
    is_reference = (args.text_memory, args.recurrent_state) == REFERENCE_CONDITION
    if not is_reference and (
        args.reference_trials_path is None or not args.reference_trials_path.is_file()
    ):
        raise ValueError(
            "Non-reference T3 conditions require --reference_trials_path from "
            "the matching local_persistent run"
        )


def load_reference_trials(path: Path) -> Dict[tuple[str, int], Dict[str, Any]]:
    result: Dict[tuple[str, int], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["episode_id"]), int(record["trial_index"]))
            if key in result:
                raise ValueError(f"Duplicate T3 reference record at line {line_number}: {key}")
            if "action_prefix" not in record or "input_feedback" not in record:
                raise ValueError(f"Incomplete T3 reference record at line {line_number}")
            result[key] = record
    return result


def paired_behavior_comparison(
    *,
    input_feedback: str,
    action_prefix: Sequence[Sequence[float]],
    first_completed_task: str | None,
    reward: float,
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    current = np.asarray(action_prefix, dtype=np.float64)
    baseline = np.asarray(reference["action_prefix"], dtype=np.float64)
    if current.ndim != 2 or baseline.ndim != 2 or current.shape[1:] != baseline.shape[1:]:
        raise ValueError(
            f"Invalid paired action prefixes: current={current.shape}, reference={baseline.shape}"
        )
    compared_steps = min(len(current), len(baseline))
    if compared_steps:
        difference = current[:compared_steps] - baseline[:compared_steps]
        prefix_rmse = float(np.sqrt(np.mean(np.square(difference))))
        first_action_l2 = float(np.linalg.norm(difference[0]))
    else:
        prefix_rmse = None
        first_action_l2 = None
    action_changed = len(current) != len(baseline) or (
        prefix_rmse is not None and prefix_rmse > ACTION_CHANGE_TOLERANCE
    )
    first_task_changed = first_completed_task != reference.get("first_completed_task")
    return {
        "reference_episode_id": str(reference["episode_id"]),
        "reference_trial_index": int(reference["trial_index"]),
        "input_feedback_changed": input_feedback != str(reference["input_feedback"]),
        "action_prefix_steps_compared": compared_steps,
        "action_prefix_rmse": prefix_rmse,
        "first_action_l2": first_action_l2,
        "action_changed": action_changed,
        "first_completed_task_changed": first_task_changed,
        "reward_changed": float(reward) != float(reference["reward"]),
        "behavior_changed": action_changed or first_task_changed,
    }


def rollout_episode(
    args: argparse.Namespace,
    env: Any,
    controller: DirectController,
    episode: Mapping[str, Any],
    paths: Mapping[str, Path],
    reference_trials: Mapping[tuple[str, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    started = time.perf_counter()
    obs, feedback_bundle, environment_description, _ = env.episode_reset_from_manifest(episode)
    controller.reset(episode, environment_description)
    ledger = TextMemoryLedger(episode["candidate_tasks"], episode["feedback_state"])
    history: list[Dict[str, Any]] = []
    trial_records: list[Dict[str, Any]] = []
    episode_actions: list[list] = []
    control_latencies: list[float] = []
    total_reward = 0.0
    total_steps = 0
    first_reward_trial = None
    first_completed_positions: list[int] = []

    for trial_index in range(1, env.num_trials + 1):
        if trial_index == 1:
            state_event = {
                "trial_index": 1,
                "state_mode": args.recurrent_state,
                "episode_reset_applied": True,
                "input_queues_cleared_in_both_modes": True,
                "recurrent_reset_applied": True,
                "after": recurrent_snapshot(controller.policy),
            }
        else:
            state_event = prepare_trial_boundary(
                controller.policy, args.recurrent_state, trial_index
            )
        append_jsonl(
            paths["state_events"],
            {
                "record_type": "state_event",
                "experiment": args.experiment,
                "condition": condition_name(args.text_memory, args.recurrent_state),
                "episode_id": episode["episode_id"],
                **state_event,
            },
        )

        raw_feedback = feedback_bundle[args.feedback]
        if raw_feedback is None:
            raise ValueError(f"Missing feedback at the start of trial {trial_index}")
        memory_before = ledger.snapshot()
        input_feedback = ledger.render(
            args.text_memory, trial_index, raw_feedback
        )
        decision = controller.begin_trial(input_feedback, history)
        feedback_events = [
            {
                "step_in_trial": 0,
                "stage": "trial_start",
                "raw_feedback": str(raw_feedback),
                "rendered_feedback": input_feedback,
            }
        ]
        trial_started = time.perf_counter()
        trial_actions: list[list] = []
        trial_latencies: list[float] = []
        completed_sequence: list[str] = []
        seen_completed: set[str] = set()
        reward = 0.0
        step_count = 0

        for _ in range(env.trial_steps):
            timed_action = controller.act(obs)
            action_values = action_to_list(timed_action.action)
            trial_actions.append(action_values)
            episode_actions.append(action_values)
            trial_latencies.append(timed_action.latency_seconds)
            control_latencies.append(timed_action.latency_seconds)
            obs, step_reward, feedback_new, _, trial_done, _ = env.step(
                timed_action.action, None
            )
            reward = max(reward, float(step_reward))
            step_count += 1
            total_steps += 1
            if feedback_new is not None:
                immediate = feedback_new.get(args.feedback)
                if immediate is not None:
                    controller.goal = str(immediate)
                    feedback_events.append(
                        {
                            "step_in_trial": step_count,
                            "stage": "intra_trial",
                            "raw_feedback": str(immediate),
                            "rendered_feedback": str(immediate),
                        }
                    )
            current = set(completed_tasks(env))
            for task in sorted(current - seen_completed):
                completed_sequence.append(task)
            seen_completed = current
            if trial_done:
                break

        completed = completed_tasks(env)
        first_completed = completed_sequence[0] if completed_sequence else None
        position = (
            list(episode["candidate_tasks"]).index(first_completed) + 1
            if first_completed in episode["candidate_tasks"]
            else None
        )
        if position is not None:
            first_completed_positions.append(position)
        known_negative_repeat = bool(
            set(completed) & set(memory_before["known_unrewarding_tasks"])
        )
        positive_retention_opportunity = bool(memory_before["known_rewarding_tasks"])
        positive_retained = (
            first_completed in memory_before["known_rewarding_tasks"]
            if positive_retention_opportunity
            else None
        )
        previous_facts = ledger.trials[-1] if ledger.trials else None
        negative_transition_opportunity = bool(
            previous_facts
            and previous_facts.reward <= 0
            and previous_facts.completed_tasks
        )
        switched_after_negative = (
            first_completed not in previous_facts.completed_tasks
            if negative_transition_opportunity and first_completed is not None
            else None
        )
        facts = ledger.observe(trial_index, completed, reward)
        memory_after = ledger.snapshot()
        action_prefix = trial_actions[:ACTION_PREFIX_STEPS]
        paired_reference = None
        if reference_trials:
            key = (str(episode["episode_id"]), trial_index)
            if key not in reference_trials:
                raise ValueError(f"Missing T3 local_persistent reference: {key}")
            paired_reference = paired_behavior_comparison(
                input_feedback=input_feedback,
                action_prefix=action_prefix,
                first_completed_task=first_completed,
                reward=reward,
                reference=reference_trials[key],
            )
        outcome = {
            "trial_index": trial_index,
            "selected_task": None,
            "completed_tasks": completed,
            "selected_task_completed": None,
            "reward": reward,
            "meta_task": episode["meta_task"],
        }
        trial_record = {
            "record_type": "trial",
            "run_name": evaluation_run_name(args),
            "protocol": PROTOCOL,
            "experiment": args.experiment,
            "text_memory": args.text_memory,
            "recurrent_state": args.recurrent_state,
            "condition": condition_name(args.text_memory, args.recurrent_state),
            "split": args.split,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
            "episode_id": episode["episode_id"],
            "episode_index": episode["episode_index"],
            "combination_index": episode["combination_index"],
            "trial_index": trial_index,
            "candidate_tasks": list(episode["candidate_tasks"]),
            "meta_task": episode["meta_task"],
            "input_feedback": input_feedback,
            "raw_environment_feedback": str(raw_feedback),
            "feedback_events": feedback_events,
            "state_event": state_event,
            "memory_before": memory_before,
            "observed_trial_facts": facts.to_dict(),
            "memory_after": memory_after,
            "decision": decision.to_dict(),
            "completed_tasks": completed,
            "completed_task_sequence": completed_sequence,
            "first_completed_task": first_completed,
            "first_completed_candidate_position": position,
            "known_negative_repeat": known_negative_repeat,
            "positive_retention_opportunity": positive_retention_opportunity,
            "positive_retained": positive_retained,
            "negative_transition_opportunity": negative_transition_opportunity,
            "switched_after_negative": switched_after_negative,
            "reward": reward,
            "success": reward > 0,
            "trial_steps": step_count,
            "timeout": step_count >= env.trial_steps and not completed,
            "failure_type": failure_type(
                reward=reward,
                decision=decision,
                completed=completed,
                meta_task=episode["meta_task"],
                timed_out=step_count >= env.trial_steps,
                direct=True,
            ),
            "trial_wall_seconds": time.perf_counter() - trial_started,
            "control_latency": latency_summary(trial_latencies),
            "action_summary": action_summary(trial_actions),
            "action_prefix": action_prefix,
            "paired_local_persistent_comparison": paired_reference,
        }
        if args.save_actions:
            trial_record["actions"] = trial_actions
        append_jsonl(paths["trials"], trial_record)
        append_jsonl(
            paths["decisions"],
            {
                "record_type": "decision",
                "run_name": evaluation_run_name(args),
                "protocol": PROTOCOL,
                "experiment": args.experiment,
                "condition": condition_name(args.text_memory, args.recurrent_state),
                "episode_id": episode["episode_id"],
                "trial_index": trial_index,
                "candidate_tasks": list(episode["candidate_tasks"]),
                "meta_task": episode["meta_task"],
                "decision": decision.to_dict(),
                "outcome": outcome,
            },
        )
        for event in feedback_events:
            append_jsonl(
                paths["feedback_events"],
                {
                    "record_type": "feedback_event",
                    "experiment": args.experiment,
                    "condition": condition_name(args.text_memory, args.recurrent_state),
                    "episode_id": episode["episode_id"],
                    "trial_index": trial_index,
                    **event,
                },
            )
        trial_records.append(trial_record)
        history.append(outcome)
        controller.observe_trial(outcome)
        total_reward += reward
        if reward > 0 and first_reward_trial is None:
            first_reward_trial = trial_index
        if trial_index < env.num_trials:
            obs, feedback_bundle, environment_description, _ = env.trial_reset_from_manifest(None)

    exploitation = (
        sum(record["reward"] for record in trial_records[first_reward_trial:])
        / (env.num_trials - first_reward_trial)
        if first_reward_trial is not None and first_reward_trial < env.num_trials
        else None
    )
    failures = Counter(
        record["failure_type"] for record in trial_records if record["failure_type"]
    )
    paired = [
        record["paired_local_persistent_comparison"]
        for record in trial_records
        if record["paired_local_persistent_comparison"] is not None
    ]
    paired_rmse = [
        float(record["action_prefix_rmse"])
        for record in paired
        if record["action_prefix_rmse"] is not None
    ]
    episode_record = {
        "record_type": "episode",
        "run_name": evaluation_run_name(args),
        "protocol": PROTOCOL,
        "experiment": args.experiment,
        "text_memory": args.text_memory,
        "recurrent_state": args.recurrent_state,
        "condition": condition_name(args.text_memory, args.recurrent_state),
        "split": args.split,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "episode_id": episode["episode_id"],
        "episode_index": episode["episode_index"],
        "candidate_tasks": list(episode["candidate_tasks"]),
        "meta_task": episode["meta_task"],
        "total_reward": total_reward,
        "reward_found": total_reward > 0,
        "reward_frequency": total_reward / env.num_trials,
        "first_reward_trial": first_reward_trial,
        "failure_adjusted_first_reward_trial": first_reward_trial or (env.num_trials + 1),
        "search_regret": (first_reward_trial or (env.num_trials + 1)) - 1,
        "reward_regret_vs_oracle": env.num_trials - total_reward,
        "post_discovery_exploitation_rate": exploitation,
        "episode_steps": total_steps,
        "trials_completed": len(trial_records),
        "task_selection_accuracy": None,
        "selected_task_completion_rate": None,
        "known_negative_repetitions": sum(
            bool(record["known_negative_repeat"]) for record in trial_records
        ),
        "positive_retention_opportunities": sum(
            bool(record["positive_retention_opportunity"]) for record in trial_records
        ),
        "positive_retentions": sum(
            bool(record["positive_retained"])
            for record in trial_records
            if record["positive_retention_opportunity"]
        ),
        "negative_transition_opportunities": sum(
            bool(record["negative_transition_opportunity"]) for record in trial_records
        ),
        "negative_switches": sum(
            bool(record["switched_after_negative"])
            for record in trial_records
            if record["negative_transition_opportunity"]
        ),
        "first_completed_candidate_positions": first_completed_positions,
        "paired_reference_trial_count": len(paired),
        "paired_reference_behavior_changed_count": sum(
            bool(record["behavior_changed"]) for record in paired
        ),
        "paired_reference_trial1_behavior_changed_count": int(
            bool(paired and paired[0]["behavior_changed"])
        ),
        "paired_reference_action_prefix_rmse_sum": sum(paired_rmse),
        "paired_reference_action_prefix_rmse_count": len(paired_rmse),
        "no_task_timeouts": sum(record["timeout"] for record in trial_records),
        "llm_calls": 0,
        "episode_wall_seconds": time.perf_counter() - started,
        "planner_latency": latency_summary([]),
        "control_latency": latency_summary(control_latencies),
        "action_summary": action_summary(episode_actions),
        "failure_counts": dict(sorted(failures.items())),
    }
    append_jsonl(paths["episodes"], episode_record)
    return episode_record


def aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = aggregate_t1_results(records)

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    total_trials = sum(record["trials_completed"] for record in records)
    positive_opportunities = sum(record["positive_retention_opportunities"] for record in records)
    negative_opportunities = sum(record["negative_transition_opportunities"] for record in records)
    paired_trials = sum(record["paired_reference_trial_count"] for record in records)
    paired_rmse_count = sum(record["paired_reference_action_prefix_rmse_count"] for record in records)
    positions = Counter(
        position
        for record in records
        for position in record["first_completed_candidate_positions"]
    )
    metrics.update(
        {
            "known_negative_repeat_rate": rate(
                sum(record["known_negative_repetitions"] for record in records),
                total_trials,
            ),
            "positive_retention_opportunities": positive_opportunities,
            "positive_retention_rate": rate(
                sum(record["positive_retentions"] for record in records),
                positive_opportunities,
            ),
            "negative_transition_opportunities": negative_opportunities,
            "switch_after_negative_rate": rate(
                sum(record["negative_switches"] for record in records),
                negative_opportunities,
            ),
            "first_completed_candidate_position_counts": {
                str(key): value for key, value in sorted(positions.items())
            },
            "paired_local_persistent_trial_count": paired_trials,
            "paired_local_persistent_behavior_change_rate": rate(
                sum(record["paired_reference_behavior_changed_count"] for record in records),
                paired_trials,
            ),
            "trial1_negative_control_behavior_change_rate_vs_local_persistent": rate(
                sum(
                    record["paired_reference_trial1_behavior_changed_count"]
                    for record in records
                ),
                len(records) if paired_trials else 0,
            ),
            "mean_action_prefix_rmse_vs_local_persistent": (
                sum(record["paired_reference_action_prefix_rmse_sum"] for record in records)
                / paired_rmse_count
                if paired_rmse_count
                else None
            ),
            "no_task_timeouts": sum(record["no_task_timeouts"] for record in records),
        }
    )
    return metrics


def run(args: argparse.Namespace) -> None:
    seed_everything(args.evaluation_seed)
    manifest = load_rollout_manifest(args.episode_manifest.resolve(), expected_split=args.split)
    if (manifest["num_trials"], manifest["trial_steps"]) != (
        args.num_trials,
        args.trial_steps,
    ):
        raise ValueError("T3 arguments do not match the frozen rollout manifest")
    episodes = manifest["episodes"]
    if args.max_episodes is not None:
        if args.max_episodes > len(episodes):
            raise ValueError("--max_episodes exceeds the frozen manifest")
        episodes = episodes[: args.max_episodes]
    is_reference = (args.text_memory, args.recurrent_state) == REFERENCE_CONDITION
    reference_trials: Dict[tuple[str, int], Dict[str, Any]] = {}
    if not is_reference:
        reference_trials = load_reference_trials(args.reference_trials_path)
        expected = {
            (str(episode["episode_id"]), trial_index)
            for episode in episodes
            for trial_index in range(1, args.num_trials + 1)
        }
        if set(reference_trials) != expected:
            raise ValueError("T3 reference trials do not exactly cover this evaluation batch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_manifest": args.output_dir / "run_manifest.json",
        "checkpoint_report": args.output_dir / "checkpoint_load_report.json",
        "trials": args.output_dir / "trials.jsonl",
        "decisions": args.output_dir / "decisions.jsonl",
        "episodes": args.output_dir / "episodes.jsonl",
        "feedback_events": args.output_dir / "feedback_events.jsonl",
        "state_events": args.output_dir / "state_events.jsonl",
        "memory_manifest": args.output_dir / "memory_manifest.json",
        "summary": args.output_dir / "summary.json",
        "summary_csv": args.output_dir / "summary.csv",
    }
    collisions = [path for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to append to an existing T3 run: {collisions}")
    checkpoint = args.student_checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    episode_manifest_sha = sha256_file(args.episode_manifest)
    if args.expected_checkpoint_sha256 and checkpoint_sha.lower() != args.expected_checkpoint_sha256.lower():
        raise ValueError(
            f"Checkpoint SHA256 mismatch: expected={args.expected_checkpoint_sha256}, actual={checkpoint_sha}"
        )
    reference_manifest = None
    if not is_reference:
        reference_manifest_path = args.reference_trials_path.parent / "run_manifest.json"
        if not reference_manifest_path.is_file():
            raise ValueError(f"T3 reference run manifest is missing: {reference_manifest_path}")
        reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
        expected_identity = {
            "status": "complete",
            "protocol": PROTOCOL,
            "experiment": args.experiment,
            "text_memory": "local",
            "recurrent_state": "persistent",
        }
        mismatches = {
            key: (reference_manifest.get(key), value)
            for key, value in expected_identity.items()
            if reference_manifest.get(key) != value
        }
        if reference_manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha:
            mismatches["checkpoint_sha256"] = "mismatch"
        if reference_manifest.get("episode_manifest", {}).get("sha256") != episode_manifest_sha:
            mismatches["episode_manifest_sha256"] = "mismatch"
        if mismatches:
            raise ValueError(f"T3 reference run is not matched: {mismatches}")

    memory_record = {
        "protocol": PROTOCOL,
        "condition": condition_name(args.text_memory, args.recurrent_state),
        "text": TextMemoryLedger(
            episodes[0]["candidate_tasks"], episodes[0]["feedback_state"]
        ).metadata(args.text_memory),
        "state": recurrent_metadata(args.recurrent_state),
        "reference_condition": condition_name(*REFERENCE_CONDITION),
        "reference_trials_path": (
            str(args.reference_trials_path.resolve()) if args.reference_trials_path else None
        ),
        "reference_trials_sha256": (
            sha256_file(args.reference_trials_path) if args.reference_trials_path else None
        ),
    }
    write_json_atomic(paths["memory_manifest"], memory_record)
    selection_metadata_path = checkpoint.parent / "best_checkpoint.json"
    selection_metadata = (
        json.loads(selection_metadata_path.read_text(encoding="utf-8"))
        if selection_metadata_path.is_file()
        else None
    )
    run_name = evaluation_run_name(args)
    run_record = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": utc_now(),
        "run_name": run_name,
        "protocol": PROTOCOL,
        "experiment": args.experiment,
        "text_memory": args.text_memory,
        "recurrent_state": args.recurrent_state,
        "condition": condition_name(args.text_memory, args.recurrent_state),
        "arguments": vars(args),
        "command": [sys.executable, *sys.argv],
        "episode_manifest": {
            "path": str(args.episode_manifest.resolve()),
            "sha256": episode_manifest_sha,
            "episode_count_available": int(manifest["num_episodes"]),
            "episode_count_used": len(episodes),
            "is_partial_evaluation": len(episodes) != int(manifest["num_episodes"]),
        },
        "checkpoint": {
            "role": "student",
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint.stat().st_size,
            "selection_metadata": selection_metadata,
        },
        "memory_manifest_sha256": sha256_file(paths["memory_manifest"]),
        "runtime": runtime_metadata(),
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                T3_DIR / "memory_conditions.py",
                T3_DIR / "recurrent_state.py",
                ROBOTEST_ROOT / "eval/t1/evaluate_t1.py",
                ROBOTEST_ROOT / "eval/t1/controllers.py",
                ROBOTEST_ROOT / "eval/t1/manifest.py",
                ROBOTEST_ROOT / "eval/t1/frozen_env.py",
                ROBOTEST_ROOT / "eval/recording.py",
                ROBOTEST_ROOT / "models/agent_model.py",
            )
            if path.is_file()
        },
    }
    write_json_atomic(paths["run_manifest"], run_record)

    try:
        env = create_environment(args, provide_lm_feedback=False)
        policy, checkpoint_report, model_metadata = load_policy(
            args, checkpoint, paths["checkpoint_report"]
        )
        controller = DirectController(
            policy, f"{args.experiment}_{condition_name(args.text_memory, args.recurrent_state)}"
        )
        torch.cuda.reset_peak_memory_stats()
        records = []
        for index, episode in enumerate(episodes, start=1):
            print(
                f"[{index}/{len(episodes)}] {episode['episode_id']} | "
                f"{args.experiment} | {condition_name(args.text_memory, args.recurrent_state)}"
            )
            records.append(
                rollout_episode(args, env, controller, episode, paths, reference_trials)
            )
        summary = {
            "schema_version": 1,
            "run_name": run_name,
            "protocol": PROTOCOL,
            "experiment": args.experiment,
            "text_memory": args.text_memory,
            "recurrent_state": args.recurrent_state,
            "condition": condition_name(args.text_memory, args.recurrent_state),
            "split": args.split,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "episode_manifest_sha256": episode_manifest_sha,
            "episode_count_available": int(manifest["num_episodes"]),
            "episode_count_used": len(records),
            "is_partial_evaluation": len(records) != int(manifest["num_episodes"]),
            "metrics": aggregate(records),
            "models": {"policy": model_metadata, "reasoning": None},
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        write_json_atomic(paths["summary"], summary)
        flat = {
            "run_name": run_name,
            "experiment": args.experiment,
            "text_memory": args.text_memory,
            "recurrent_state": args.recurrent_state,
            "split": args.split,
            **{
                key: value
                for key, value in summary["metrics"].items()
                if not isinstance(value, (dict, list))
            },
            "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
        }
        with paths["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat))
            writer.writeheader()
            writer.writerow(flat)
        run_record.update(
            {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "checkpoint_load_report": checkpoint_report,
                "summary": summary,
            }
        )
        write_json_atomic(paths["run_manifest"], run_record)
        print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    except BaseException as error:
        run_record.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        write_json_atomic(paths["run_manifest"], run_record)
        raise


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
