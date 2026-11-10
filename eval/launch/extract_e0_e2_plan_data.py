#!/usr/bin/env python3
"""Extract plan-facing E0/E2 metrics from completed structured rollouts."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PLAN_USES = [
    {
        "plan_item": "F0",
        "models": "E0,E2 (within E0-E7)",
        "pdf_pages": "2,3,6,7,8,11,12",
        "required": "normal AR/RF plus 3 training seeds and the full 2x2x2 factorial",
        "available": "seed-42 legacy normal rollout for E0 and E2",
        "missing": "E1/E3-E7; seeds 43/44; formal P0 common-manifest evaluation; factorial effects",
    },
    {
        "plan_item": "F1",
        "models": "E0",
        "pdf_pages": "7,10,12",
        "required": "Strict reference versus FE-SO and FE-M",
        "available": "E0 Strict seed-42 legacy normal baseline",
        "missing": "FE-SO/FE-M models, seeds 43/44, matched formal test",
    },
    {
        "plan_item": "T1",
        "models": "E0,E2",
        "pdf_pages": "8",
        "required": "AR, RF, first reward, negative repetition, steps, latency, memory, size, planner baselines",
        "available": "student outcome, control-step latency, total model inference time, checkpoint size",
        "missing": "other planners/Teacher/Oracle; parameter count; peak GPU memory; high-level latency; formal seeds/CI",
    },
    {
        "plan_item": "T2",
        "models": "E0 (with E3)",
        "pdf_pages": "8",
        "required": "Full, Masked, Shuffled, Counterbalanced paired tests",
        "available": "E0 Full-feedback legacy baseline and observed negative-task repetitions",
        "missing": "three interventions, E3, explicit selected-task/next-task logs, formal common manifest",
    },
    {
        "plan_item": "T3",
        "models": "E0 (with E3)",
        "pdf_pages": "9,12",
        "required": "local/cumulative text crossed with reset/persistent LSTM",
        "available": "E0 persistent-LSTM legacy baseline",
        "missing": "LSTM-reset and text-memory interventions, E3, paired formal evaluation",
    },
    {
        "plan_item": "T4",
        "models": "E0 (with E3); E2 can also be reported",
        "pdf_pages": "9,12",
        "required": "first reward, post-discovery exploitation, conditional AR, repetition, executor failure, timeout",
        "available": "first reward, post-discovery reward, conditional AR, completed-task repetition, no-task timeouts",
        "missing": "explicit selected-task intent and executor-failure attribution; E3; formal multi-seed statistics",
    },
    {
        "plan_item": "T6",
        "models": "E0,E2",
        "pdf_pages": "5,9,12",
        "required": "E0 fixed/diverse test and E2 fixed/diverse test: four paired cells",
        "available": "E0/fixed-test and E2/diverse-test diagonal cells",
        "missing": "E0/diverse-test and E2/fixed-test cross cells; held-out template manifest; 3 seeds; paired hierarchical CI",
    },
    {
        "plan_item": "T7",
        "models": "E0 (with E1)",
        "pdf_pages": "5,10,12",
        "required": "original, one-distractor mask, all-distractors mask, target mask, Oracle instruction",
        "available": "E0 original-image legacy normal baseline and action summaries",
        "missing": "E1; all masks; Oracle clear instruction; segmentation-frozen rules; paired action-change analysis",
    },
    {
        "plan_item": "T8",
        "models": "E0 (with FE-SO/FE-M)",
        "pdf_pages": "7,10,12",
        "required": "Normal, injected initial failure, bounded noise across three filtering strengths",
        "available": "E0 Strict/Normal legacy baseline",
        "missing": "FE-SO/FE-M; two perturbations; recovery labels/steps; validation-frozen perturbation rules; 3 seeds",
    },
    {
        "plan_item": "P0/A0/statistics",
        "models": "E0,E2",
        "pdf_pages": "6,10,11",
        "required": "frozen test manifest, simulator seed, 3 training seeds, seed SD, hierarchical paired bootstrap",
        "available": "checkpoint/data metadata, 150 paired symbolic episode setups, trial/episode/action/log/video records",
        "missing": "manifest-driven rollout, explicit simulator seeds, seeds 43/44, hierarchical training-seed uncertainty",
    },
]


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def latest_completed_run(root, experiment):
    candidates = []
    for status_path in root.glob(f"{experiment}/seed_*/eval_seed_*/*/launcher_status.json"):
        if read_json(status_path).get("status") == "complete":
            candidates.append(status_path.parent)
    if not candidates:
        raise FileNotFoundError(f"No completed {experiment} run below {root}")
    return max(candidates, key=lambda path: path.name)


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def checkpoint_inventory(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["model_state_dict"]
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    dtype_counts = Counter(str(value.dtype) for value in tensors)
    return {
        "state_dict_entries": len(state),
        "state_tensor_entries": len(tensors),
        "state_tensor_elements": sum(value.numel() for value in tensors),
        "state_tensor_bytes": sum(value.numel() * value.element_size() for value in tensors),
        "state_tensor_dtypes": dict(sorted(dtype_counts.items())),
        "warning": "State tensor elements include parameters and buffers; use as a checkpoint inventory, not an exact trainable-parameter count.",
    }


def describe_run(experiment, run_dir):
    manifest = read_json(run_dir / "run_manifest.json")
    summary = read_json(run_dir / "summary.json")
    episodes = sorted(read_jsonl(run_dir / "episodes.jsonl"), key=lambda row: row["episode_index"])
    trials = sorted(
        read_jsonl(run_dir / "trials.jsonl"),
        key=lambda row: (row["episode_index"], row["trial_index"]),
    )
    trials_by_episode = defaultdict(list)
    for trial in trials:
        trials_by_episode[trial["episode_index"]].append(trial)

    first_reward = Counter()
    post_attempts = 0
    post_rewards = 0
    negative_events = 0
    negative_repetitions = 0
    timeouts = 0
    no_task_trials = 0
    nonreward_task_trials = 0
    any_task_trials = 0
    inference_steps = 0
    inference_seconds = 0.0
    trial_p95 = []
    max_step_latency = 0.0
    gripper_open_steps = 0
    gripper_closed_steps = 0

    for episode in episodes:
        first = episode["first_reward_trial"]
        first_reward["none" if first is None else str(first)] += 1
        known_negative = set()
        for trial in trials_by_episode[episode["episode_index"]]:
            completed = set(trial["completed_tasks"])
            reward = float(trial["reward"])
            if completed:
                any_task_trials += 1
            if not completed:
                no_task_trials += 1
            if reward == 0 and completed:
                nonreward_task_trials += 1
            if reward == 0 and not completed and int(trial["trial_steps"]) >= 300:
                timeouts += 1
            negative = completed - {episode["meta_task"]}
            negative_events += len(negative)
            negative_repetitions += sum(task in known_negative for task in negative)
            known_negative.update(negative)
            if first is not None and int(trial["trial_index"]) > int(first):
                post_attempts += 1
                post_rewards += int(reward > 0)

            latency = trial["inference_latency"]
            inference_steps += int(latency["count"])
            inference_seconds += float(latency["total_seconds"])
            if latency["p95_seconds"] is not None:
                trial_p95.append(float(latency["p95_seconds"]))
            if latency["max_seconds"] is not None:
                max_step_latency = max(max_step_latency, float(latency["max_seconds"]))
            action = trial["action_summary"]
            gripper_open_steps += int(action["gripper_open_steps"])
            gripper_closed_steps += int(action["gripper_closed_steps"])

    rewards = np.asarray([float(row["reward"]) for row in episodes])
    found = np.asarray([bool(row["reward_found"]) for row in episodes])
    steps = np.asarray([int(row["episode_steps"]) for row in episodes])
    discovered_first = np.asarray(
        [int(row["first_reward_trial"]) for row in episodes if row["first_reward_trial"] is not None]
    )
    checkpoint = manifest["checkpoint"]
    inventory = checkpoint_inventory(checkpoint["path"])
    selection = checkpoint.get("selection_metadata") or {}
    arguments = manifest["arguments"]
    metrics = {
        "experiment": experiment,
        "run_dir": str(run_dir.resolve()),
        "status": manifest["status"],
        "training_seed": int(run_dir.parents[1].name.removeprefix("seed_")),
        "evaluation_seed": int(arguments["seed"]),
        "feedback": arguments["feedback"],
        "protocol_label": arguments["protocol_label"],
        "formal_p0_result": False,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_size_bytes": int(checkpoint["size_bytes"]),
        "checkpoint_size_gib": float(checkpoint["size_bytes"]) / 1024**3,
        "checkpoint_global_step_zero_based": selection.get("global_step"),
        "checkpoint_completed_updates": (
            int(selection["global_step"]) + 1 if selection.get("global_step") is not None else None
        ),
        "checkpoint_epoch_zero_based": selection.get("epoch"),
        "checkpoint_validation_loss": selection.get("validation_loss"),
        "num_episodes": len(episodes),
        "num_trials": len(trials),
        "num_videos": len(list((run_dir / "videos").glob("*.mp4"))),
        "total_reward": float(rewards.sum()),
        "average_reward": float(rewards.mean()),
        "reward_found_episodes": int(found.sum()),
        "reward_found_rate": float(found.mean()),
        "average_episode_steps": float(steps.mean()),
        "total_environment_steps": int(steps.sum()),
        "first_reward_conditional_mean": (
            float(discovered_first.mean()) if discovered_first.size else None
        ),
        "first_reward_conditional_median": (
            float(np.median(discovered_first)) if discovered_first.size else None
        ),
        "average_reward_conditioned_on_discovery": (
            float(rewards[found].mean()) if found.any() else None
        ),
        "post_discovery_trials": post_attempts,
        "post_discovery_rewards": post_rewards,
        "post_discovery_exploitation_rate": (
            post_rewards / post_attempts if post_attempts else None
        ),
        "negative_completed_task_events": negative_events,
        "known_negative_repetitions": negative_repetitions,
        "known_negative_repetition_rate": (
            negative_repetitions / negative_events if negative_events else None
        ),
        "timeout_no_task_trials": timeouts,
        "no_task_completed_trials": no_task_trials,
        "nonrewarding_task_completed_trials": nonreward_task_trials,
        "any_task_completed_trials": any_task_trials,
        "any_task_completion_rate": any_task_trials / len(trials),
        "inference_step_count": inference_steps,
        "total_model_only_inference_seconds": inference_seconds,
        "mean_control_step_latency_ms": 1000 * inference_seconds / inference_steps,
        "mean_trial_p95_control_latency_ms": (
            1000 * float(np.mean(trial_p95)) if trial_p95 else None
        ),
        "maximum_observed_control_latency_ms": 1000 * max_step_latency,
        "gripper_open_steps": gripper_open_steps,
        "gripper_closed_steps": gripper_closed_steps,
        "llm_calls": 0,
        "parameter_count": None,
        "checkpoint_state_inventory": inventory,
        "peak_gpu_memory_bytes": None,
        "high_level_selection_latency_seconds": None,
        "simulator_seed": None,
        "first_reward_distribution": {
            key: first_reward.get(key, 0) for key in ["1", "2", "3", "4", "5", "none"]
        },
    }
    return {
        "manifest": manifest,
        "summary": summary,
        "episodes": episodes,
        "trials": trials,
        "trials_by_episode": trials_by_episode,
        "metrics": metrics,
    }


def episode_rows(experiment, data):
    trial_map = data["trials_by_episode"]
    for episode in data["episodes"]:
        trials = trial_map[episode["episode_index"]]
        first = episode["first_reward_trial"]
        post_trials = [row for row in trials if first is not None and row["trial_index"] > first]
        yield {
            "experiment": experiment,
            "episode_index": episode["episode_index"],
            "combination_index": episode["combination_index"],
            "reset_index": episode["reset_index"],
            "evaluation_seed": episode["evaluation_seed"],
            "simulator_seed": episode["simulator_seed"],
            "feedback_type": episode["feedback_type"],
            "feedback_state": json_cell(episode["feedback_state"]),
            "candidate_tasks": json_cell(episode["candidate_tasks"]),
            "meta_task": episode["meta_task"],
            "meta_task_index": episode["meta_task_index"],
            "initial_environment_state": json_cell(episode["initial_environment_state"]),
            "reward": episode["reward"],
            "reward_found": int(episode["reward_found"]),
            "first_reward_trial": episode["first_reward_trial"],
            "episode_steps": episode["episode_steps"],
            "trials_completed": episode["trials_completed"],
            "post_discovery_trials": len(post_trials),
            "post_discovery_rewards": sum(float(row["reward"]) > 0 for row in post_trials),
            "post_discovery_exploitation_rate": (
                sum(float(row["reward"]) > 0 for row in post_trials) / len(post_trials)
                if post_trials else ""
            ),
            "model_inference_seconds": episode["inference_latency"]["total_seconds"],
            "mean_control_latency_ms": (
                1000 * episode["inference_latency"]["mean_seconds"]
            ),
            "video_path": episode["video_path"],
        }


def trial_rows(experiment, trials):
    for trial in trials:
        action = trial["action_summary"]
        latency = trial["inference_latency"]
        template = trial["input_feedback_template"]
        yield {
            "experiment": experiment,
            "episode_index": trial["episode_index"],
            "trial_index": trial["trial_index"],
            "meta_task": trial["meta_task"],
            "meta_task_index": trial["meta_task_index"],
            "feedback_state": json_cell(trial["feedback_state"]),
            "candidate_tasks": json_cell(trial["candidate_tasks"]),
            "feedback_type": trial["feedback_type"],
            "input_feedback": trial["input_feedback"],
            "feedback_family": template["feedback_family"],
            "feedback_template_id": template["template_id"],
            "trial_steps": trial["trial_steps"],
            "reward": trial["reward"],
            "success": int(trial["success"]),
            "completed_tasks": json_cell(trial["completed_tasks"]),
            "cumulative_completed_tasks": json_cell(trial["cumulative_completed_tasks"]),
            "failure_category": trial["failure_category"],
            "timeout_no_task": int(
                float(trial["reward"]) == 0
                and not trial["completed_tasks"]
                and int(trial["trial_steps"]) >= 300
            ),
            "inference_steps": latency["count"],
            "inference_total_seconds": latency["total_seconds"],
            "inference_mean_ms": 1000 * latency["mean_seconds"],
            "inference_p95_ms": 1000 * latency["p95_seconds"],
            "inference_max_ms": 1000 * latency["max_seconds"],
            "gripper_open_steps": action["gripper_open_steps"],
            "gripper_closed_steps": action["gripper_closed_steps"],
            "action_mean": json_cell(action["mean"]),
            "action_minimum": json_cell(action["minimum"]),
            "action_maximum": json_cell(action["maximum"]),
        }


def task_rows(experiment, episodes):
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["meta_task"]].append(episode)
    for task, rows in sorted(grouped.items()):
        reward = np.asarray([float(row["reward"]) for row in rows])
        found = np.asarray([bool(row["reward_found"]) for row in rows])
        steps = np.asarray([int(row["episode_steps"]) for row in rows])
        first = [int(row["first_reward_trial"]) for row in rows if row["first_reward_trial"] is not None]
        yield {
            "experiment": experiment,
            "meta_task": task,
            "episodes": len(rows),
            "total_reward": float(reward.sum()),
            "average_reward": float(reward.mean()),
            "reward_found_episodes": int(found.sum()),
            "reward_found_rate": float(found.mean()),
            "average_episode_steps": float(steps.mean()),
            "first_reward_conditional_mean": float(np.mean(first)) if first else "",
        }


def grouped_episode_rows(experiment, episodes, key, output_key):
    grouped = defaultdict(list)
    for episode in episodes:
        value = episode[key]
        if key == "feedback_state":
            value = sorted(value)
        grouped[json_cell(value) if isinstance(value, (list, dict)) else str(value)].append(episode)
    for value, rows in sorted(grouped.items()):
        reward = np.asarray([float(row["reward"]) for row in rows])
        found = np.asarray([bool(row["reward_found"]) for row in rows])
        steps = np.asarray([int(row["episode_steps"]) for row in rows])
        first = [int(row["first_reward_trial"]) for row in rows if row["first_reward_trial"] is not None]
        yield {
            "experiment": experiment,
            output_key: value,
            "episodes": len(rows),
            "average_reward": float(reward.mean()),
            "reward_found_rate": float(found.mean()),
            "average_episode_steps": float(steps.mean()),
            "first_reward_conditional_mean": float(np.mean(first)) if first else "",
        }


def feedback_template_rows(experiment, trials):
    grouped = defaultdict(list)
    for trial in trials:
        template = trial["input_feedback_template"]
        key = (template["template_stage"], template["template_id"])
        grouped[key].append(trial)
    for (stage, template_id), rows in sorted(grouped.items()):
        reward = np.asarray([float(row["reward"]) for row in rows])
        steps = np.asarray([int(row["trial_steps"]) for row in rows])
        yield {
            "experiment": experiment,
            "template_stage": stage,
            "feedback_template_id": template_id,
            "trials": len(rows),
            "rewarding_trials": int(np.count_nonzero(reward > 0)),
            "trial_reward_rate": float(np.mean(reward > 0)),
            "average_trial_steps": float(steps.mean()),
            "template_text": rows[0]["input_feedback_template"].get("template_text"),
        }


def failure_rows(experiment, trials):
    grouped = Counter()
    for trial in trials:
        if float(trial["reward"]) > 0:
            category = "rewarding_meta_task_completed"
        elif not trial["completed_tasks"] and int(trial["trial_steps"]) >= 300:
            category = "timeout_no_task_completed"
        elif trial["completed_tasks"]:
            category = "only_nonrewarding_task_completed"
        else:
            category = "no_task_completed_before_early_end"
        grouped[category] += 1
    total = len(trials)
    for category, count in sorted(grouped.items()):
        yield {
            "experiment": experiment,
            "failure_or_outcome_category": category,
            "trial_count": count,
            "trial_fraction": count / total,
        }


def paired_bootstrap(values, rng, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    boot = values[draws].mean(axis=1)
    return {
        "mean_difference_E2_minus_E0": float(values.mean()),
        "episode_paired_bootstrap_95_ci": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "bootstrap_samples": samples,
        "bootstrap_seed": 20260816,
        "warning": "Episode-level CI only; not hierarchical across training seeds.",
    }


def build_paired(e0, e2):
    left = {row["episode_index"]: row for row in e0["episodes"]}
    right = {row["episode_index"]: row for row in e2["episodes"]}
    if set(left) != set(right):
        raise ValueError("E0 and E2 episode indexes do not match")
    setup_fields = [
        "feedback_state",
        "candidate_tasks",
        "meta_task",
        "meta_task_index",
        "initial_environment_state",
    ]
    rows = []
    for index in sorted(left):
        a, b = left[index], right[index]
        setup_match = all(a[field] == b[field] for field in setup_fields)
        rows.append(
            {
                "episode_index": index,
                "paired_symbolic_setup_match": int(setup_match),
                "meta_task": a["meta_task"],
                "feedback_state": json_cell(a["feedback_state"]),
                "E0_reward": a["reward"],
                "E2_reward": b["reward"],
                "reward_difference_E2_minus_E0": float(b["reward"]) - float(a["reward"]),
                "E0_reward_found": int(a["reward_found"]),
                "E2_reward_found": int(b["reward_found"]),
                "reward_found_difference_E2_minus_E0": int(b["reward_found"]) - int(a["reward_found"]),
                "E0_first_reward_trial": a["first_reward_trial"],
                "E2_first_reward_trial": b["first_reward_trial"],
                "E0_episode_steps": a["episode_steps"],
                "E2_episode_steps": b["episode_steps"],
                "steps_difference_E2_minus_E0": int(b["episode_steps"]) - int(a["episode_steps"]),
            }
        )
    rng = np.random.default_rng(20260816)
    comparisons = {
        "paired_episodes": len(rows),
        "symbolic_setup_matches": sum(row["paired_symbolic_setup_match"] for row in rows),
        "reward": paired_bootstrap(
            [row["reward_difference_E2_minus_E0"] for row in rows], rng
        ),
        "reward_found": paired_bootstrap(
            [row["reward_found_difference_E2_minus_E0"] for row in rows], rng
        ),
        "episode_steps": paired_bootstrap(
            [row["steps_difference_E2_minus_E0"] for row in rows], rng
        ),
    }
    return rows, comparisons


def artifact_rows(experiment, run_dir):
    names = [
        "summary.json",
        "summary.csv",
        "run_manifest.json",
        "checkpoint_load_report.json",
        "trials.jsonl",
        "episodes.jsonl",
        "run.log",
        "launcher_status.json",
        "command.sh",
        "environment_freeze.txt",
        "nvidia_smi.txt",
    ]
    for name in names:
        path = run_dir / name
        if path.is_file():
            yield {
                "experiment": experiment,
                "artifact": name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }


def video_rows(experiment, episodes):
    for episode in episodes:
        path = Path(episode["video_path"])
        yield {
            "experiment": experiment,
            "episode_index": episode["episode_index"],
            "meta_task": episode["meta_task"],
            "feedback_state": json_cell(episode["feedback_state"]),
            "reward": episode["reward"],
            "reward_found": int(episode["reward_found"]),
            "first_reward_trial": episode["first_reward_trial"],
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }


def make_report(e0, e2, paired):
    m0, m2 = e0["metrics"], e2["metrics"]
    rf = paired["reward_found"]
    ar = paired["reward"]
    return f"""# E0/E2：整体补充实验计划数据提取报告

## 1. PDF 中使用 E0/E2 的实验

| 项目 | 使用方式 | 当前可用 | 仍缺少 |
|---|---|---|---|
| F0 | E0/E2 属于完整 E0-E7 factorial | seed 42 normal legacy | 其余 cells、seeds 43/44、正式 P0 |
| F1 | E0 是 Strict reference | E0 normal baseline | FE-SO、FE-M、其余 seeds |
| T1 | 预声明报告 E0 reference 与 E2 diverse | outcome、steps、control latency、checkpoint size | planners/Teacher/Oracle、显存、参数量、3 seeds |
| T2 | E0 与 E3 做历史干预 | E0 Full baseline | Masked/Shuffled/Counterbalanced、E3 |
| T3 | E0 与 E3 做文本×LSTM 记忆 | E0 persistent baseline | reset/local/cumulative interventions、E3 |
| T4 | E0/E3 探索利用；E2 也可报告 | first reward、post-discovery、timeout、重复 | selected-task intent、E3、正式多 seed |
| T6 | E0/E2 做 Fixed/Diverse 2×2 | E0/Fixed 与 E2/Diverse | E0/Diverse、E2/Fixed、3 seeds |
| T7 | E0 与 E1 做视觉遮挡 | E0 原图 baseline | E1、三类遮挡、Oracle instruction |
| T8 | E0 是 Strict/Normal reference | E0 Strict normal baseline | FE-SO/FE-M、initial failure、noise、3 seeds |
| P0/A0 | 全部模型共同协议/统计 | checkpoint、日志、150 个成对符号场景 | test manifest 驱动、simulator seed、层次 bootstrap |

## 2. 当前已提取的核心结果

| 指标 | E0 Fixed | E2 Diverse |
|---|---:|---:|
| Episodes / Trials | {m0['num_episodes']} / {m0['num_trials']} | {m2['num_episodes']} / {m2['num_trials']} |
| Average Reward | {m0['average_reward']:.4f} | {m2['average_reward']:.4f} |
| Reward Found | {m0['reward_found_rate']:.1%} | {m2['reward_found_rate']:.1%} |
| Average episode steps | {m0['average_episode_steps']:.2f} | {m2['average_episode_steps']:.2f} |
| Conditional first-reward trial | {m0['first_reward_conditional_mean']:.3f} | {m2['first_reward_conditional_mean']:.3f} |
| Post-discovery exploitation | {m0['post_discovery_exploitation_rate']:.1%} | {m2['post_discovery_exploitation_rate']:.1%} |
| Known-negative repetition | {m0['known_negative_repetition_rate']:.1%} | {m2['known_negative_repetition_rate']:.1%} |
| No-task timeout trials | {m0['timeout_no_task_trials']} | {m2['timeout_no_task_trials']} |
| Mean control latency | {m0['mean_control_step_latency_ms']:.3f} ms | {m2['mean_control_step_latency_ms']:.3f} ms |
| Total model-only inference | {m0['total_model_only_inference_seconds']:.1f} s | {m2['total_model_only_inference_seconds']:.1f} s |

E2−E0 的 episode-paired difference：

- AR：{ar['mean_difference_E2_minus_E0']:.4f}，episode-paired bootstrap 95% CI {ar['episode_paired_bootstrap_95_ci']}。
- RF：{rf['mean_difference_E2_minus_E0']:.4f}，episode-paired bootstrap 95% CI {rf['episode_paired_bootstrap_95_ci']}。
- 150/150 个 episode 的 object/state/candidate/meta-task 符号设置一致。

这些 CI 只覆盖 episode variability。当前每个配置只有 training seed 42，不能替代计划要求的 3-seed hierarchical paired bootstrap。

## 3. 数据文件

- `extracted_metrics.json`：完整聚合指标、first-reward 分布、成对 CI。
- `episode_metrics.csv`：300 条 episode 记录。
- `trial_metrics.csv`：1500 条 trial 记录。
- `task_metrics.csv`：按 14 个 meta task 分组的 AR/RF/steps。
- `combination_metrics.csv`：按 10 个 held-out object combinations 分组。
- `target_position_metrics.csv`：按 meta-task 在候选列表中的位置分组。
- `feedback_template_metrics.csv`：按 feedback template 分组的 trial 表现。
- `failure_metrics.csv`：reward、non-reward completion、timeout 分解。
- `paired_episode_comparison.csv`：E0/E2 的 150 个成对 episode。
- `first_reward_distribution.csv`：首次奖励 trial 分布。
- `plan_data_availability.csv`：逐项列出计划要求、已具备和缺失数据。
- `source_artifacts.csv`：原始日志/manifest/summary 的路径、大小和 SHA256。
- `video_manifest.csv`：300 个视频与 episode/task/reward 的映射及 SHA256。

## 4. 使用限制

当前结果属于 `MetaCALVIN-legacy-7Object-v1`。它没有由 frozen final-test episode manifest 驱动，simulator seed 字段为空；两个 checkpoint 也都只是 epoch 1（13,592 updates）的 seed-42 checkpoint。因此这些文件可用于审计、诊断和生成 provisional 表格，不能直接充当 PDF 计划要求的最终 F0/T1-T8 结果。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/home/yxt/thesis/rebuttal/evaluation_runs/legacy_e0_e2"),
    )
    parser.add_argument("--e0-run", type=Path)
    parser.add_argument("--e2-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    e0_dir = args.e0_run or latest_completed_run(args.run_root, "E0")
    e2_dir = args.e2_run or latest_completed_run(args.run_root, "E2")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    runs = {
        "E0": describe_run("E0", e0_dir),
        "E2": describe_run("E2", e2_dir),
    }
    paired_rows, paired = build_paired(runs["E0"], runs["E2"])

    all_episode_rows = []
    all_trial_rows = []
    all_task_rows = []
    all_combination_rows = []
    all_target_position_rows = []
    all_template_rows = []
    all_failure_rows = []
    all_artifacts = []
    all_videos = []
    first_reward_rows = []
    for experiment, data in runs.items():
        all_episode_rows.extend(episode_rows(experiment, data))
        all_trial_rows.extend(trial_rows(experiment, data["trials"]))
        all_task_rows.extend(task_rows(experiment, data["episodes"]))
        all_combination_rows.extend(
            grouped_episode_rows(experiment, data["episodes"], "feedback_state", "feedback_state")
        )
        all_target_position_rows.extend(
            grouped_episode_rows(experiment, data["episodes"], "meta_task_index", "meta_task_index")
        )
        all_template_rows.extend(feedback_template_rows(experiment, data["trials"]))
        all_failure_rows.extend(failure_rows(experiment, data["trials"]))
        all_artifacts.extend(artifact_rows(experiment, Path(data["metrics"]["run_dir"])))
        all_videos.extend(video_rows(experiment, data["episodes"]))
        for trial, count in data["metrics"]["first_reward_distribution"].items():
            first_reward_rows.append(
                {
                    "experiment": experiment,
                    "first_reward_trial": trial,
                    "episode_count": count,
                    "episode_fraction": count / data["metrics"]["num_episodes"],
                }
            )

    write_json(
        args.output_dir / "extracted_metrics.json",
        {
            "source_pdf": "/home/yxt/thesis/rebuttal/RAL-rebuttal/MetaCALVIN_RA-L_Rebuttal_整体补充实验计划.pdf",
            "result_scope": "legacy seed-42 diagnostic/provisional; not formal P0",
            "runs": {key: value["metrics"] for key, value in runs.items()},
            "paired_E2_minus_E0": paired,
        },
    )
    write_csv(args.output_dir / "episode_metrics.csv", all_episode_rows)
    write_csv(args.output_dir / "trial_metrics.csv", all_trial_rows)
    write_csv(args.output_dir / "task_metrics.csv", all_task_rows)
    write_csv(args.output_dir / "combination_metrics.csv", all_combination_rows)
    write_csv(args.output_dir / "target_position_metrics.csv", all_target_position_rows)
    write_csv(args.output_dir / "feedback_template_metrics.csv", all_template_rows)
    write_csv(args.output_dir / "failure_metrics.csv", all_failure_rows)
    write_csv(args.output_dir / "paired_episode_comparison.csv", paired_rows)
    write_csv(args.output_dir / "first_reward_distribution.csv", first_reward_rows)
    write_csv(args.output_dir / "plan_data_availability.csv", PLAN_USES)
    write_csv(args.output_dir / "source_artifacts.csv", all_artifacts)
    write_csv(args.output_dir / "video_manifest.csv", all_videos)
    (args.output_dir / "REPORT.md").write_text(
        make_report(runs["E0"], runs["E2"], paired), encoding="utf-8"
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
