#!/usr/bin/env python3
"""Frozen-policy T2 history-intervention evaluator for 7-Object Explore."""

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

T2_DIR = Path(__file__).resolve().parent
ROBOTEST_ROOT = T2_DIR.parents[1]
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
from eval.t2.interventions import FeedbackIntervention, INTERVENTIONS


PROTOCOL = "MetaCALVIN-RA-L-T2-7Object-v2"
REGISTERED_EXPERIMENTS = frozenset({"E0", "E3"})
ACTION_PREFIX_STEPS = 10
ACTION_CHANGE_TOLERANCE = 1e-5


def protocol_scope(experiment: str) -> str:
    return "registered" if experiment in REGISTERED_EXPERIMENTS else "exploratory_extension"


def evaluation_run_name(args: argparse.Namespace) -> str:
    return (
        f"T2_{args.experiment}_{args.intervention}_{args.split}_"
        f"train_seed_{args.training_seed}"
    )


def feedback_template_metadata(
    args: argparse.Namespace,
    episode: Mapping[str, Any],
    *,
    trial_index: int,
    previous_reward: float | None,
) -> Dict[str, Any]:
    stage = (
        "initial"
        if trial_index == 1
        else ("reward" if previous_reward and previous_reward > 0 else "failure")
    )
    template_id = (
        f"fixed:{stage}"
        if args.feedback == "train_feedback"
        else episode["feedback_template_ids"][stage]
    )
    return {
        "family": "fixed" if args.feedback == "train_feedback" else "diverse",
        "stage": stage,
        "template_id": template_id,
        "intervention": args.intervention,
    }


def paired_behavior_comparison(
    *,
    input_feedback: str,
    action_prefix: Sequence[Sequence[float]],
    first_completed_task: str | None,
    reward: float,
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    if "action_prefix" not in reference:
        raise ValueError(
            "Full reference was produced by an incompatible T2 evaluator: "
            "missing action_prefix"
        )
    current = np.asarray(action_prefix, dtype=np.float64)
    baseline = np.asarray(reference["action_prefix"], dtype=np.float64)
    if current.ndim != 2 or baseline.ndim != 2 or current.shape[1:] != baseline.shape[1:]:
        raise ValueError(
            f"Invalid paired action prefixes: current={current.shape}, full={baseline.shape}"
        )
    compared_steps = min(len(current), len(baseline))
    if compared_steps:
        difference = current[:compared_steps] - baseline[:compared_steps]
        prefix_rmse = float(np.sqrt(np.mean(np.square(difference))))
        first_action_l2 = float(np.linalg.norm(difference[0]))
    else:
        prefix_rmse = None
        first_action_l2 = None
    length_changed = len(current) != len(baseline)
    first_task_changed = first_completed_task != reference.get("first_completed_task")
    action_changed = length_changed or (
        prefix_rmse is not None and prefix_rmse > ACTION_CHANGE_TOLERANCE
    )
    return {
        "full_episode_id": str(reference["episode_id"]),
        "full_trial_index": int(reference["trial_index"]),
        "input_feedback_changed": input_feedback != str(reference["input_feedback"]),
        "action_prefix_steps_compared": compared_steps,
        "action_prefix_rmse": prefix_rmse,
        "first_action_l2": first_action_l2,
        "action_prefix_length_changed": length_changed,
        "action_changed": action_changed,
        "first_completed_task_changed": first_task_changed,
        "reward_changed": float(reward) != float(reference["reward"]),
        "behavior_changed": action_changed or first_task_changed,
    }


def validate_full_reference(
    args: argparse.Namespace,
    episodes: Sequence[Mapping[str, Any]],
    intervention: FeedbackIntervention,
) -> None:
    if args.intervention == "full":
        return
    expected_keys = {
        (str(episode["episode_id"]), trial_index)
        for episode in episodes
        for trial_index in range(1, args.num_trials + 1)
    }
    actual_keys = set(intervention.full_trials)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:5]
        extra = sorted(actual_keys - expected_keys)[:5]
        raise ValueError(
            "Matched Full trial coverage differs from the current T2 condition: "
            f"missing={missing}, extra={extra}"
        )
    for key, record in intervention.full_trials.items():
        expected = {
            "experiment": args.experiment,
            "intervention": "full",
            "split": args.split,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
        }
        mismatches = {
            field: (record.get(field), value)
            for field, value in expected.items()
            if record.get(field) != value
        }
        if mismatches:
            raise ValueError(f"Incompatible Full reference record {key}: {mismatches}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", choices=("E0", "E3", "E4", "E7"), required=True)
    result.add_argument("--intervention", choices=INTERVENTIONS, required=True)
    result.add_argument("--split", choices=("validation", "test"), required=True)
    result.add_argument("--episode_manifest", type=Path, required=True)
    result.add_argument("--student_checkpoint", type=Path, required=True)
    result.add_argument("--training_seed", type=int, required=True)
    result.add_argument("--output_dir", type=Path, required=True)
    result.add_argument("--full_trials_path", type=Path)
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
    result.add_argument("--num_trials", type=int, default=5)
    result.add_argument("--trial_steps", type=int, default=300)
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
        raise ValueError("Formal T2 test requires --final_test_ack")
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
    expected_feedback = (
        "alt_feedback" if args.experiment in {"E4", "E7"} else "train_feedback"
    )
    if args.feedback != expected_feedback:
        raise ValueError(
            f"{args.experiment} requires feedback={expected_feedback}, got {args.feedback}"
        )
    if args.intervention != "full" and (
        args.full_trials_path is None or not args.full_trials_path.is_file()
    ):
        raise ValueError(
            f"{args.intervention} T2 requires --full_trials_path from the matching Full run"
        )


def rollout_episode(
    args: argparse.Namespace,
    env: Any,
    controller: DirectController,
    episode: Mapping[str, Any],
    intervention: FeedbackIntervention,
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    started = time.perf_counter()
    obs, feedback_bundle, environment_description, _ = env.episode_reset_from_manifest(
        episode
    )
    controller.reset(episode, environment_description)
    history = []
    trial_records = []
    episode_actions = []
    control_latencies = []
    total_reward = 0.0
    total_steps = 0
    first_reward_trial = None
    previous_reward = None
    known_negative_tasks: set[str] = set()
    first_completed_original_positions = []
    first_completed_presented_positions = []
    input_candidate_order = intervention.input_candidate_order(episode)

    for trial_index in range(1, env.num_trials + 1):
        raw_feedback = feedback_bundle[args.feedback]
        if raw_feedback is None:
            raise ValueError(
                f"Missing {args.feedback} at the start of trial {trial_index}"
            )
        input_feedback = intervention.apply(
            raw_feedback,
            episode=episode,
            trial_index=trial_index,
            stage="trial_start",
        )
        input_feedback_template = feedback_template_metadata(
            args,
            episode,
            trial_index=trial_index,
            previous_reward=previous_reward,
        )
        decision = controller.begin_trial(input_feedback, history)
        feedback_events = [
            {
                "step_in_trial": 0,
                "stage": "trial_start",
                "raw_feedback": str(raw_feedback),
                "intervened_feedback": input_feedback,
            }
        ]
        trial_started = time.perf_counter()
        trial_actions = []
        trial_latencies = []
        completed_sequence = []
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
                raw_intra_feedback = feedback_new.get(args.feedback)
                if raw_intra_feedback is not None:
                    intra_feedback = intervention.apply(
                        raw_intra_feedback,
                        episode=episode,
                        trial_index=trial_index,
                        stage="intra_trial",
                    )
                    # T2 uses DirectController exclusively. Preserve immediate
                    # within-trial feedback as the policy goal while keeping
                    # Masked/Shuffled scoped to previous-trial history.
                    controller.goal = intra_feedback
                    feedback_events.append(
                        {
                            "step_in_trial": step_count,
                            "stage": "intra_trial",
                            "raw_feedback": str(raw_intra_feedback),
                            "intervened_feedback": intra_feedback,
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
        original_candidate_order = list(episode["candidate_tasks"])
        original_position = (
            original_candidate_order.index(first_completed) + 1
            if first_completed in original_candidate_order
            else None
        )
        presented_position = (
            input_candidate_order.index(first_completed) + 1
            if first_completed in input_candidate_order
            else None
        )
        if original_position is not None:
            first_completed_original_positions.append(original_position)
        if presented_position is not None:
            first_completed_presented_positions.append(presented_position)
        known_negative_repeat = bool(set(completed) & known_negative_tasks)
        previous_trial = trial_records[-1] if trial_records else None
        negative_history_available = bool(
            previous_trial
            and previous_trial["reward"] <= 0
            and previous_trial["first_completed_task"] is not None
        )
        positive_history_available = bool(
            previous_trial and previous_trial["reward"] > 0
        )
        switched_after_negative = (
            bool(
                first_completed is not None
                and first_completed != previous_trial["first_completed_task"]
            )
            if negative_history_available
            else None
        )
        repeated_target_after_positive = (
            first_completed == episode["meta_task"]
            if positive_history_available
            else None
        )
        action_prefix = trial_actions[:ACTION_PREFIX_STEPS]
        paired_full = None
        if args.intervention != "full":
            reference = intervention.full_reference(
                str(episode["episode_id"]), trial_index
            )
            if reference is None:
                raise ValueError(
                    "Missing matched Full reference for "
                    f"{episode['episode_id']} trial {trial_index}"
                )
            paired_full = paired_behavior_comparison(
                input_feedback=input_feedback,
                action_prefix=action_prefix,
                first_completed_task=first_completed,
                reward=reward,
                reference=reference,
            )
        outcome = {
            "trial_index": trial_index,
            "selected_task": decision.selected_task,
            "completed_tasks": completed,
            "selected_task_completed": None,
            "reward": reward,
            "meta_task": episode["meta_task"],
        }
        trial_record = {
            "record_type": "trial",
            "run_name": evaluation_run_name(args),
            "protocol": PROTOCOL,
            "protocol_scope": protocol_scope(args.experiment),
            "experiment": args.experiment,
            "intervention": args.intervention,
            "split": args.split,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
            "episode_id": episode["episode_id"],
            "episode_index": episode["episode_index"],
            "combination_index": episode["combination_index"],
            "trial_index": trial_index,
            "candidate_tasks": list(episode["candidate_tasks"]),
            "input_candidate_order": input_candidate_order,
            "meta_task": episode["meta_task"],
            "input_feedback": input_feedback,
            "raw_input_feedback": str(raw_feedback),
            "input_feedback_template": input_feedback_template,
            "feedback_events": feedback_events,
            "decision": decision.to_dict(),
            "completed_tasks": completed,
            "completed_task_sequence": completed_sequence,
            "first_completed_task": first_completed,
            "first_completed_candidate_position": presented_position,
            "first_completed_original_candidate_position": original_position,
            "first_completed_presented_candidate_position": presented_position,
            "negative_history_available": negative_history_available,
            "switched_after_negative": switched_after_negative,
            "positive_history_available": positive_history_available,
            "repeated_target_after_positive": repeated_target_after_positive,
            "reward": reward,
            "success": reward > 0,
            "known_negative_repeat": known_negative_repeat,
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
            "paired_full_comparison": paired_full,
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
                "protocol_scope": protocol_scope(args.experiment),
                "experiment": args.experiment,
                "intervention": args.intervention,
                "episode_id": episode["episode_id"],
                "trial_index": trial_index,
                "candidate_tasks": list(episode["candidate_tasks"]),
                "input_candidate_order": input_candidate_order,
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
                    "intervention": args.intervention,
                    "episode_id": episode["episode_id"],
                    "trial_index": trial_index,
                    **event,
                },
            )
        trial_records.append(trial_record)
        history.append(outcome)
        controller.observe_trial(outcome)
        total_reward += reward
        previous_reward = reward
        if reward > 0 and first_reward_trial is None:
            first_reward_trial = trial_index
        known_negative_tasks.update(
            task for task in completed if task != episode["meta_task"]
        )

        if trial_index < env.num_trials:
            obs, feedback_bundle, environment_description, _ = (
                env.trial_reset_from_manifest(None)
            )

    exploitation = (
        sum(record["reward"] for record in trial_records[first_reward_trial:])
        / (env.num_trials - first_reward_trial)
        if first_reward_trial is not None and first_reward_trial < env.num_trials
        else None
    )
    failures = Counter(
        record["failure_type"] for record in trial_records if record["failure_type"]
    )
    paired_records = [
        record["paired_full_comparison"]
        for record in trial_records
        if record["paired_full_comparison"] is not None
    ]
    later_paired_records = [
        record["paired_full_comparison"]
        for record in trial_records[1:]
        if record["paired_full_comparison"] is not None
    ]
    negative_transitions = [
        record for record in trial_records if record["negative_history_available"]
    ]
    positive_transitions = [
        record for record in trial_records if record["positive_history_available"]
    ]
    paired_prefix_rmse = [
        float(record["action_prefix_rmse"])
        for record in paired_records
        if record["action_prefix_rmse"] is not None
    ]
    trial1_paired = paired_records[0] if paired_records else None
    episode_record = {
        "record_type": "episode",
        "run_name": evaluation_run_name(args),
        "protocol": PROTOCOL,
        "protocol_scope": protocol_scope(args.experiment),
        "experiment": args.experiment,
        "intervention": args.intervention,
        "split": args.split,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "episode_id": episode["episode_id"],
        "episode_index": episode["episode_index"],
        "candidate_tasks": list(episode["candidate_tasks"]),
        "input_candidate_order": input_candidate_order,
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
            record["known_negative_repeat"] for record in trial_records
        ),
        "llm_calls": 0,
        "episode_wall_seconds": time.perf_counter() - started,
        "planner_latency": latency_summary([]),
        "control_latency": latency_summary(control_latencies),
        "action_summary": action_summary(episode_actions),
        "failure_counts": dict(sorted(failures.items())),
        "first_completed_candidate_positions": first_completed_presented_positions,
        "first_completed_original_candidate_positions": (
            first_completed_original_positions
        ),
        "first_completed_presented_candidate_positions": (
            first_completed_presented_positions
        ),
        "negative_transition_opportunities": len(negative_transitions),
        "switched_after_negative_count": sum(
            bool(record["switched_after_negative"]) for record in negative_transitions
        ),
        "positive_transition_opportunities": len(positive_transitions),
        "repeated_target_after_positive_count": sum(
            bool(record["repeated_target_after_positive"])
            for record in positive_transitions
        ),
        "paired_full_trial_count": len(paired_records),
        "paired_full_later_trial_count": len(later_paired_records),
        "paired_full_trial1_count": int(trial1_paired is not None),
        "paired_full_trial1_behavior_changed_count": int(
            bool(trial1_paired and trial1_paired["behavior_changed"])
        ),
        "paired_full_feedback_changed_count": sum(
            bool(record["input_feedback_changed"]) for record in paired_records
        ),
        "paired_full_behavior_changed_count": sum(
            bool(record["behavior_changed"]) for record in paired_records
        ),
        "paired_full_later_behavior_changed_count": sum(
            bool(record["behavior_changed"]) for record in later_paired_records
        ),
        "paired_full_first_task_changed_count": sum(
            bool(record["first_completed_task_changed"]) for record in paired_records
        ),
        "paired_full_reward_changed_count": sum(
            bool(record["reward_changed"]) for record in paired_records
        ),
        "paired_full_action_prefix_rmse_sum": sum(paired_prefix_rmse),
        "paired_full_action_prefix_rmse_count": len(paired_prefix_rmse),
        "no_task_timeouts": sum(record["timeout"] for record in trial_records),
    }
    append_jsonl(paths["episodes"], episode_record)
    return episode_record


def aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = aggregate_t1_results(records)
    presented_positions = Counter(
        position
        for record in records
        for position in record["first_completed_presented_candidate_positions"]
    )
    original_positions = Counter(
        position
        for record in records
        for position in record["first_completed_original_candidate_positions"]
    )
    negative_opportunities = sum(
        record["negative_transition_opportunities"] for record in records
    )
    negative_switches = sum(
        record["switched_after_negative_count"] for record in records
    )
    positive_opportunities = sum(
        record["positive_transition_opportunities"] for record in records
    )
    positive_repeats = sum(
        record["repeated_target_after_positive_count"] for record in records
    )
    paired_trials = sum(record["paired_full_trial_count"] for record in records)
    paired_later_trials = sum(
        record["paired_full_later_trial_count"] for record in records
    )
    paired_trial1 = sum(record["paired_full_trial1_count"] for record in records)
    paired_rmse_count = sum(
        record["paired_full_action_prefix_rmse_count"] for record in records
    )

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    metrics.update(
        {
            "first_completed_candidate_position_counts": {
                str(key): value for key, value in sorted(presented_positions.items())
            },
            "first_completed_original_candidate_position_counts": {
                str(key): value for key, value in sorted(original_positions.items())
            },
            "first_completed_presented_candidate_position_counts": {
                str(key): value for key, value in sorted(presented_positions.items())
            },
            "original_order_first_candidate_rate": rate(
                original_positions.get(1, 0), sum(original_positions.values())
            ),
            "presented_order_first_candidate_rate": rate(
                presented_positions.get(1, 0), sum(presented_positions.values())
            ),
            "negative_transition_opportunities": negative_opportunities,
            "switch_after_negative_rate": rate(
                negative_switches, negative_opportunities
            ),
            "positive_transition_opportunities": positive_opportunities,
            "repeat_target_after_positive_rate": rate(
                positive_repeats, positive_opportunities
            ),
            "paired_full_trial_count": paired_trials,
            "paired_full_input_feedback_change_rate": rate(
                sum(record["paired_full_feedback_changed_count"] for record in records),
                paired_trials,
            ),
            "paired_full_behavior_change_rate": rate(
                sum(record["paired_full_behavior_changed_count"] for record in records),
                paired_trials,
            ),
            "history_sensitive_behavior_change_rate_vs_full": rate(
                sum(
                    record["paired_full_later_behavior_changed_count"]
                    for record in records
                ),
                paired_later_trials,
            ),
            "paired_full_first_completed_task_change_rate": rate(
                sum(
                    record["paired_full_first_task_changed_count"]
                    for record in records
                ),
                paired_trials,
            ),
            "paired_full_reward_change_rate": rate(
                sum(record["paired_full_reward_changed_count"] for record in records),
                paired_trials,
            ),
            "trial1_negative_control_behavior_change_rate_vs_full": rate(
                sum(
                    record["paired_full_trial1_behavior_changed_count"]
                    for record in records
                ),
                paired_trial1,
            ),
            "mean_action_prefix_rmse_vs_full": (
                sum(
                    record["paired_full_action_prefix_rmse_sum"]
                    for record in records
                )
                / paired_rmse_count
                if paired_rmse_count
                else None
            ),
            "no_task_timeouts": sum(record["no_task_timeouts"] for record in records),
            "known_negative_repeat_rate": (
                sum(record["known_negative_repetitions"] for record in records)
                / sum(record["trials_completed"] for record in records)
            ),
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
        raise ValueError("T2 arguments do not match the frozen rollout manifest")
    episodes = manifest["episodes"]
    if args.max_episodes is not None:
        if args.max_episodes > len(episodes):
            raise ValueError(
                f"--max_episodes={args.max_episodes} exceeds the frozen manifest "
                f"episode count {len(episodes)}"
            )
        episodes = episodes[: args.max_episodes]
    intervention = FeedbackIntervention(
        args.intervention,
        episodes,
        args.full_trials_path,
        feedback_type=args.feedback,
        donor_seed=args.evaluation_seed,
    )
    validate_full_reference(args, episodes, intervention)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_manifest": args.output_dir / "run_manifest.json",
        "checkpoint_report": args.output_dir / "checkpoint_load_report.json",
        "trials": args.output_dir / "trials.jsonl",
        "decisions": args.output_dir / "decisions.jsonl",
        "episodes": args.output_dir / "episodes.jsonl",
        "feedback_events": args.output_dir / "feedback_events.jsonl",
        "intervention": args.output_dir / "intervention_manifest.json",
        "summary": args.output_dir / "summary.json",
        "summary_csv": args.output_dir / "summary.csv",
    }
    collisions = [path for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to append to an existing T2 run: {collisions}")
    checkpoint = args.student_checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    episode_manifest_sha = sha256_file(args.episode_manifest)
    if args.expected_checkpoint_sha256 and (
        checkpoint_sha.lower() != args.expected_checkpoint_sha256.lower()
    ):
        raise ValueError(
            f"Checkpoint SHA256 mismatch: expected={args.expected_checkpoint_sha256}, "
            f"actual={checkpoint_sha}"
        )
    full_run_reference = None
    if args.intervention != "full":
        full_run_manifest_path = args.full_trials_path.parent / "run_manifest.json"
        if not full_run_manifest_path.is_file():
            raise ValueError(
                f"Full reference run_manifest.json is missing: {full_run_manifest_path}"
            )
        full_run_reference = json.loads(
            full_run_manifest_path.read_text(encoding="utf-8")
        )
        expected_full_identity = {
            "status": "complete",
            "protocol": PROTOCOL,
            "experiment": args.experiment,
            "intervention": "full",
        }
        identity_mismatches = {
            key: (full_run_reference.get(key), expected)
            for key, expected in expected_full_identity.items()
            if full_run_reference.get(key) != expected
        }
        if full_run_reference.get("checkpoint", {}).get("sha256") != checkpoint_sha:
            identity_mismatches["checkpoint_sha256"] = (
                full_run_reference.get("checkpoint", {}).get("sha256"),
                checkpoint_sha,
            )
        if (
            full_run_reference.get("episode_manifest", {}).get("sha256")
            != episode_manifest_sha
        ):
            identity_mismatches["episode_manifest_sha256"] = (
                full_run_reference.get("episode_manifest", {}).get("sha256"),
                episode_manifest_sha,
            )
        if identity_mismatches:
            raise ValueError(
                f"Full reference run is not matched to this condition: {identity_mismatches}"
            )
    selection_metadata_path = checkpoint.parent / "best_checkpoint.json"
    selection_metadata = (
        json.loads(selection_metadata_path.read_text(encoding="utf-8"))
        if selection_metadata_path.is_file()
        else None
    )
    run_name = evaluation_run_name(args)
    intervention_record = {
        **intervention.metadata(),
        "full_trials_path": (
            str(args.full_trials_path.resolve()) if args.full_trials_path else None
        ),
        "full_trials_sha256": (
            sha256_file(args.full_trials_path) if args.full_trials_path else None
        ),
        "full_run_manifest_path": (
            str((args.full_trials_path.parent / "run_manifest.json").resolve())
            if args.full_trials_path
            else None
        ),
        "full_run_manifest_sha256": (
            sha256_file(args.full_trials_path.parent / "run_manifest.json")
            if args.full_trials_path
            else None
        ),
    }
    write_json_atomic(paths["intervention"], intervention_record)
    run_record = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": utc_now(),
        "run_name": run_name,
        "protocol": PROTOCOL,
        "protocol_scope": protocol_scope(args.experiment),
        "experiment": args.experiment,
        "intervention": args.intervention,
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
            "selection_metadata_path": (
                str(selection_metadata_path.resolve())
                if selection_metadata_path.is_file()
                else None
            ),
            "selection_metadata": selection_metadata,
        },
        "intervention_manifest_sha256": sha256_file(paths["intervention"]),
        "runtime": runtime_metadata(),
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                T2_DIR / "interventions.py",
                ROBOTEST_ROOT / "eval/t1/evaluate_t1.py",
                ROBOTEST_ROOT / "eval/t1/controllers.py",
                ROBOTEST_ROOT / "eval/t1/manifest.py",
                ROBOTEST_ROOT / "eval/t1/frozen_env.py",
                ROBOTEST_ROOT / "eval/recording.py",
                ROBOTEST_ROOT / "eval/resource_monitor.py",
                ROBOTEST_ROOT / "env/protoenv.py",
                ROBOTEST_ROOT / "env/env_utils.py",
                ROBOTEST_ROOT / "models/agent_model.py",
                ROBOTEST_ROOT / "models/factory.py",
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
        controller = DirectController(policy, f"{args.experiment}_{args.intervention}")
        torch.cuda.reset_peak_memory_stats()
        records = []
        for index, episode in enumerate(episodes, start=1):
            print(
                f"[{index}/{len(episodes)}] {episode['episode_id']} | "
                f"{args.experiment} | {args.intervention}"
            )
            records.append(
                rollout_episode(args, env, controller, episode, intervention, paths)
            )
        summary = {
            "schema_version": 1,
            "run_name": run_name,
            "protocol": PROTOCOL,
            "protocol_scope": protocol_scope(args.experiment),
            "experiment": args.experiment,
            "intervention": args.intervention,
            "split": args.split,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "episode_manifest_sha256": run_record["episode_manifest"]["sha256"],
            "episode_count_available": run_record["episode_manifest"][
                "episode_count_available"
            ],
            "episode_count_used": len(records),
            "is_partial_evaluation": run_record["episode_manifest"][
                "is_partial_evaluation"
            ],
            "metrics": aggregate(records),
            "models": {"policy": model_metadata, "reasoning": None},
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        write_json_atomic(paths["summary"], summary)
        flat = {
            "run_name": run_name,
            "experiment": args.experiment,
            "intervention": args.intervention,
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
