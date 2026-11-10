#!/usr/bin/env python3
"""Generate frozen episodes and run the unified T1 planner ladder.

Formal use is intentionally two-stage:

1. ``--mode generate-manifest`` freezes every semantic and physical reset.
2. ``--mode evaluate`` runs exactly one controller on that immutable manifest.

The legacy ``eval/evaluate.py`` remains unchanged for historical reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


T1_DIR = Path(__file__).resolve().parent
ROBOTEST_ROOT = T1_DIR.parents[1]
WORKSPACE_ROOT = ROBOTEST_ROOT.parents[2]
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

from eval.recording import (  # noqa: E402
    action_summary,
    append_jsonl,
    jsonable,
    latency_summary,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from eval.t1.controllers import (  # noqa: E402
    DirectController,
    PlannerExecutorController,
    action_to_list,
)
from eval.t1.manifest import (  # noqa: E402
    create_rollout_manifest,
    load_rollout_manifest,
)
from eval.t1.planners import build_planner  # noqa: E402


CONTROLLERS = (
    "student",
    "shared_ambiguous",
    "untuned_ambiguous",
    "random",
    "no_repeat",
    "rule_based",
    "gemma",
    "oracle",
)
DIRECT_CONTROLLERS = {"student", "shared_ambiguous", "untuned_ambiguous"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified frozen-manifest evaluator for the T1 planner comparison"
    )
    parser.add_argument("--mode", choices=("generate-manifest", "evaluate"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--episode_manifest", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path)
    parser.add_argument("--final_test_ack", action="store_true")

    parser.add_argument("--controller", choices=CONTROLLERS)
    parser.add_argument("--student_checkpoint", type=Path)
    parser.add_argument("--executor_checkpoint", type=Path)
    parser.add_argument(
        "--expected_checkpoint_sha256",
        type=str,
        help="optional frozen checkpoint identity; evaluation aborts on mismatch",
    )
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--training_seed", type=int)
    parser.add_argument("--evaluation_seed", type=int, default=42)
    parser.add_argument("--planner_seed", type=int, default=42)
    parser.add_argument("--feedback", choices=("train_feedback", "alt_feedback"), default="train_feedback")
    parser.add_argument("--max_episodes", type=int)
    parser.add_argument("--save_actions", action="store_true")

    parser.add_argument("--num_objs", type=int, default=7)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--trial_steps", type=int, default=300)
    parser.add_argument("--num_resets", type=int, default=15)
    parser.add_argument("--manifest_seed", type=int, default=20260818)
    parser.add_argument("--max_combinations", type=int)

    parser.add_argument(
        "--path_to_env",
        type=Path,
        default=WORKSPACE_ROOT / "calvin-official/dataset/calvin_debug_dataset/validation",
    )
    parser.add_argument(
        "--path_to_oracle",
        type=Path,
        default=ROBOTEST_ROOT / "calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml",
    )
    parser.add_argument(
        "--disable_egl",
        action="store_true",
        help="use PyBullet DIRECT rendering for CPU-only smoke tests",
    )
    parser.add_argument("--vision_encoder_path", default="ViT-L-14")
    parser.add_argument("--vision_encoder_pretrained", default="openai")
    parser.add_argument("--lm_path", default="anas-awadalla/mpt-1b-redpajama-200b-dolly")
    parser.add_argument("--tokenizer_path", default="anas-awadalla/mpt-1b-redpajama-200b-dolly")
    parser.add_argument("--cross_attn_every_n_layers", type=int, default=1)
    parser.add_argument("--disable_gripper", action="store_false", dest="use_gripper")
    parser.set_defaults(use_gripper=True)
    parser.add_argument("--fusion_mode", default="post")
    parser.add_argument("--llm_name", default="mpt_dolly_3b")
    parser.add_argument("--decoder_type", default="lstm")
    parser.add_argument("--action_tokens", action="store_true")
    parser.add_argument("--window_size", type=int, default=12)

    parser.add_argument("--reasoning_model", default="google/gemma-2-27b-it")
    parser.add_argument("--reasoning_tokenizer", default="google/gemma-2-27b-it")
    parser.add_argument("--path_to_hf_cache", type=Path, default=WORKSPACE_ROOT / "cache/huggingface")
    parser.add_argument("--reasoning_temperature", type=float, default=0.8)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_objs != 7:
        parser.error("T1 v1 currently supports only the 7-object ProtoEnv protocol")
    for name in ("num_trials", "trial_steps", "num_resets"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max_episodes must be positive")
    if args.max_combinations is not None and args.max_combinations <= 0:
        parser.error("--max_combinations must be positive")
    if not args.path_to_env.is_dir():
        parser.error(f"CALVIN environment directory does not exist: {args.path_to_env}")
    if not args.path_to_oracle.is_file():
        parser.error(f"task-oracle config does not exist: {args.path_to_oracle}")

    if args.mode == "generate-manifest":
        if args.split_manifest is None or not args.split_manifest.is_file():
            parser.error("--split_manifest must identify an existing P0 split manifest")
        if args.episode_manifest.exists():
            parser.error(f"refusing to overwrite frozen episode manifest: {args.episode_manifest}")
        return

    if args.controller is None:
        parser.error("--controller is required in evaluate mode")
    if args.output_dir is None:
        parser.error("--output_dir is required in evaluate mode")
    if not args.episode_manifest.is_file():
        parser.error(f"episode manifest does not exist: {args.episode_manifest}")
    if args.split == "test" and not args.final_test_ack:
        parser.error("formal test evaluation requires --final_test_ack")
    checkpoint = args.student_checkpoint if args.controller == "student" else args.executor_checkpoint
    checkpoint_flag = "--student_checkpoint" if args.controller == "student" else "--executor_checkpoint"
    if checkpoint is None or not checkpoint.is_file():
        parser.error(f"{checkpoint_flag} must identify an existing checkpoint")
    if args.controller == "student" and args.training_seed is None:
        parser.error("student runs must record --training_seed")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        from transformers import set_seed

        set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_environment(args: argparse.Namespace, *, provide_lm_feedback: bool):
    import hydra
    from omegaconf import OmegaConf

    from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
    from eval.t1.frozen_env import FrozenProtoEnv

    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
    if args.disable_egl:
        render_conf = OmegaConf.load(args.path_to_env / ".hydra/merged_config.yaml")
        for camera_name in list(render_conf.cameras):
            if camera_name not in {"static", "gripper"}:
                del render_conf.cameras[camera_name]
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize(".")
        base_env = hydra.utils.instantiate(
            render_conf.env,
            show_gui=False,
            use_vr=False,
            use_scene_info=True,
            use_egl=False,
        )
    else:
        base_env = get_env(args.path_to_env, obs_space=obs_space, show_gui=False)
    task_cfg = OmegaConf.load(args.path_to_oracle)
    task_oracle = hydra.utils.instantiate(task_cfg)
    env = FrozenProtoEnv(
        base_env,
        task_oracle,
        num_trials=args.num_trials,
        trial_steps=args.trial_steps,
        provide_feedback=True,
        provide_train_feedback=True,
        provide_lm_feedback=provide_lm_feedback,
        reset_robot_pos=False,
        num_objs=7,
    )
    return env


def load_policy(args: argparse.Namespace, checkpoint_path: Path, report_path: Path):
    import torch

    # RoboTest's legacy layout adds ``models/`` itself to sys.path; these are
    # top-level modules rather than an importable ``models`` package.
    from agent_model import ModelWrapper
    from factory import create_model_and_transforms

    window_size = 1 if args.decoder_type == "lstm_seq" else args.window_size
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path or args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_gripper=args.use_gripper,
        fusion_mode=args.fusion_mode,
        llm_name=args.llm_name,
        decoder_type=args.decoder_type,
        action_tokens=args.action_tokens,
        window_size=window_size,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    load_result = model.load_state_dict(state_dict, strict=False)
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "strict": False,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    write_json_atomic(report_path, report)
    model.half().cuda().eval()
    policy = ModelWrapper(model, tokenizer, image_processor, torch.float16, False)
    model_metadata = {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    return policy, report, model_metadata


def build_controller(args: argparse.Namespace, policy: Any, clear_instructions: Mapping[str, str]):
    if args.controller in DIRECT_CONTROLLERS:
        return DirectController(policy, args.controller)

    reasoning_llm = None
    if args.controller == "gemma":
        from agent.reasoning_llm import ReasoningLLM
        from transformers import set_seed

        set_seed(args.planner_seed)
        reasoning_llm = ReasoningLLM(
            args.reasoning_model,
            args.reasoning_tokenizer,
            cache_dir=str(args.path_to_hf_cache),
            temperature=args.reasoning_temperature,
        )
        reasoning_llm.model.eval()
    planner = build_planner(
        args.controller,
        clear_instructions,
        seed=args.planner_seed,
        reasoning_llm=reasoning_llm,
    )
    return PlannerExecutorController(policy, planner)


def controller_model_metadata(controller: Any) -> Optional[Dict[str, int]]:
    reasoning_llm = getattr(getattr(controller, "planner", None), "reasoning_llm", None)
    reasoning_model = getattr(reasoning_llm, "model", None)
    if reasoning_model is None:
        return None
    return {
        "parameter_count": int(
            sum(parameter.numel() for parameter in reasoning_model.parameters())
        ),
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in reasoning_model.parameters()
                if parameter.requires_grad
            )
        ),
    }


def completed_tasks(env: Any) -> List[str]:
    return sorted(task for task, completed in env.task_success_dict.items() if completed)


def failure_type(
    *,
    reward: float,
    decision: Any,
    completed: Sequence[str],
    meta_task: str,
    timed_out: bool,
    direct: bool,
) -> Optional[str]:
    if reward > 0:
        return None
    if not decision.valid:
        return "invalid_planner_output"
    if direct:
        return "direct_timeout" if timed_out and not completed else "direct_execution_failure"
    selected = decision.selected_task
    if selected in completed and selected != meta_task:
        return "wrong_task_selection"
    if selected == meta_task and selected not in completed:
        return "executor_failure"
    if completed:
        return "unintended_task_completion"
    return "executor_timeout" if timed_out else "executor_failure"


def rollout_episode(
    args: argparse.Namespace,
    env: Any,
    controller: Any,
    episode: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    episode_started = time.perf_counter()
    obs, feedback_bundle, environment_description, _ = env.episode_reset_from_manifest(episode)
    controller.reset(episode, environment_description)
    history: List[Dict[str, Any]] = []
    trial_records: List[Dict[str, Any]] = []
    episode_actions: List[list] = []
    control_latencies: List[float] = []
    planner_latencies: List[float] = []
    total_reward = 0.0
    total_steps = 0
    first_reward_trial = None
    previous_reward = None

    for trial_index in range(1, env.num_trials + 1):
        if args.controller == "gemma":
            planner_feedback = feedback_bundle["lm_feedback"]
        else:
            planner_feedback = feedback_bundle[args.feedback]
        template_stage = (
            "initial"
            if trial_index == 1
            else ("reward" if previous_reward and previous_reward > 0 else "failure")
        )
        if args.controller == "gemma":
            feedback_template = {
                "family": "lm_feedback",
                "stage": template_stage,
                "template_id": f"lm_feedback:{template_stage}:v1",
            }
        else:
            feedback_template = {
                "family": "fixed" if args.feedback == "train_feedback" else "diverse",
                "stage": template_stage,
                "template_id": (
                    f"fixed:{template_stage}"
                    if args.feedback == "train_feedback"
                    else episode["feedback_template_ids"][template_stage]
                ),
            }
        decision = controller.begin_trial(planner_feedback, history)
        planner_latency = decision.metadata.get("planner_latency_seconds")
        if planner_latency is not None:
            planner_latencies.append(float(planner_latency))

        trial_started = time.perf_counter()
        trial_actions: List[list] = []
        trial_latencies: List[float] = []
        reward = 0.0
        trial_done = False
        step_count = 0
        for _ in range(env.trial_steps):
            timed_action = controller.act(obs)
            action_values = action_to_list(timed_action.action)
            trial_actions.append(action_values)
            episode_actions.append(action_values)
            trial_latencies.append(timed_action.latency_seconds)
            control_latencies.append(timed_action.latency_seconds)
            instruction_for_env = None if args.controller in DIRECT_CONTROLLERS else decision.instruction
            obs, step_reward, _, _, trial_done, _ = env.step(
                timed_action.action, instruction_for_env
            )
            reward = max(reward, float(step_reward))
            step_count += 1
            total_steps += 1
            if trial_done:
                break

        completed = completed_tasks(env)
        selected_completed = decision.selected_task in completed if decision.selected_task else None
        outcome = {
            "trial_index": trial_index,
            "selected_task": decision.selected_task,
            "completed_tasks": completed,
            "selected_task_completed": selected_completed,
            "reward": reward,
            "meta_task": episode["meta_task"],
        }
        category = failure_type(
            reward=reward,
            decision=decision,
            completed=completed,
            meta_task=episode["meta_task"],
            timed_out=step_count >= env.trial_steps,
            direct=args.controller in DIRECT_CONTROLLERS,
        )
        trial_record = {
            "record_type": "trial",
            "run_name": args.run_name,
            "protocol": "MetaCALVIN-RA-L-T1-7Object-v1",
            "split": args.split,
            "controller": args.controller,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
            "planner_seed": args.planner_seed,
            "episode_id": episode["episode_id"],
            "episode_index": episode["episode_index"],
            "trial_index": trial_index,
            "candidate_tasks": list(episode["candidate_tasks"]),
            "meta_task": episode["meta_task"],
            "input_feedback": planner_feedback,
            "input_feedback_template": feedback_template,
            "decision": decision.to_dict(),
            "selected_task_is_target": (
                decision.selected_task == episode["meta_task"]
                if decision.selected_task is not None
                else None
            ),
            "completed_tasks": completed,
            "selected_task_completed": selected_completed,
            "reward": reward,
            "success": reward > 0,
            "failure_type": category,
            "known_negative_repeat": bool(decision.metadata.get("known_negative_before", False)),
            "trial_steps": step_count,
            "trial_wall_seconds": time.perf_counter() - trial_started,
            "control_latency": latency_summary(trial_latencies),
            "action_summary": action_summary(trial_actions),
        }
        if args.save_actions:
            trial_record["actions"] = trial_actions
        append_jsonl(paths["trials"], trial_record)
        append_jsonl(
            paths["decisions"],
            {
                "record_type": "decision",
                "run_name": args.run_name,
                "controller": args.controller,
                "episode_id": episode["episode_id"],
                "trial_index": trial_index,
                "candidate_tasks": list(episode["candidate_tasks"]),
                "meta_task": episode["meta_task"],
                "decision": decision.to_dict(),
                "outcome": outcome,
            },
        )
        trial_records.append(trial_record)
        history.append(outcome)
        controller.observe_trial(outcome)
        total_reward += reward
        previous_reward = reward
        if reward > 0 and first_reward_trial is None:
            first_reward_trial = trial_index

        if trial_index < env.num_trials:
            reset_instruction = None if args.controller in DIRECT_CONTROLLERS else decision.instruction
            obs, feedback_bundle, environment_description, _ = env.trial_reset_from_manifest(
                reset_instruction
            )

    planner_applicable = args.controller not in DIRECT_CONTROLLERS
    selected = [
        record["decision"]["selected_task"]
        for record in trial_records
        if record["decision"]["selected_task"] is not None
    ]
    episode_record = {
        "record_type": "episode",
        "run_name": args.run_name,
        "protocol": "MetaCALVIN-RA-L-T1-7Object-v1",
        "split": args.split,
        "controller": args.controller,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "planner_seed": args.planner_seed,
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
        "post_discovery_exploitation_rate": (
            sum(record["reward"] for record in trial_records[first_reward_trial:])
            / (env.num_trials - first_reward_trial)
            if first_reward_trial is not None and first_reward_trial < env.num_trials
            else None
        ),
        "episode_steps": total_steps,
        "trials_completed": len(trial_records),
        "task_selection_accuracy": (
            sum(task == episode["meta_task"] for task in selected) / len(trial_records)
            if planner_applicable
            else None
        ),
        "valid_planner_decisions": (
            sum(record["decision"]["valid"] for record in trial_records)
            if planner_applicable
            else None
        ),
        "selected_task_completion_rate": (
            sum(bool(record["selected_task_completed"]) for record in trial_records)
            / len(trial_records)
            if planner_applicable
            else None
        ),
        "known_negative_repetitions": sum(
            record["known_negative_repeat"] for record in trial_records
        ),
        "llm_calls": controller.llm_calls,
        "episode_wall_seconds": time.perf_counter() - episode_started,
        "planner_latency": latency_summary(planner_latencies),
        "control_latency": latency_summary(control_latencies),
        "action_summary": action_summary(episode_actions),
        "failure_counts": {
            category: sum(record["failure_type"] == category for record in trial_records)
            for category in sorted(
                {record["failure_type"] for record in trial_records if record["failure_type"]}
            )
        },
    }
    append_jsonl(paths["episodes"], episode_record)
    return episode_record


def aggregate_results(episode_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not episode_records:
        raise ValueError("Cannot aggregate an empty T1 run")
    planner_accuracies = [
        record["task_selection_accuracy"]
        for record in episode_records
        if record["task_selection_accuracy"] is not None
    ]
    total_trials = sum(record["trials_completed"] for record in episode_records)
    total_reward = sum(record["total_reward"] for record in episode_records)
    exploitation_rates = [
        record["post_discovery_exploitation_rate"]
        for record in episode_records
        if record["post_discovery_exploitation_rate"] is not None
    ]
    completion_rates = [
        record["selected_task_completion_rate"]
        for record in episode_records
        if record["selected_task_completion_rate"] is not None
    ]
    all_failures: Dict[str, int] = {}
    for record in episode_records:
        for key, value in record["failure_counts"].items():
            all_failures[key] = all_failures.get(key, 0) + int(value)
    return {
        "episode_count": len(episode_records),
        "trial_count": total_trials,
        "average_return": mean(record["total_reward"] for record in episode_records),
        "reward_frequency": total_reward / total_trials,
        "reward_found_rate": mean(float(record["reward_found"]) for record in episode_records),
        "mean_first_reward_trial_success_only": (
            mean(
                record["first_reward_trial"]
                for record in episode_records
                if record["first_reward_trial"] is not None
            )
            if any(record["first_reward_trial"] is not None for record in episode_records)
            else None
        ),
        "mean_failure_adjusted_first_reward_trial": mean(
            record["failure_adjusted_first_reward_trial"] for record in episode_records
        ),
        "mean_search_regret": mean(record["search_regret"] for record in episode_records),
        "mean_reward_regret_vs_oracle": mean(
            record["reward_regret_vs_oracle"] for record in episode_records
        ),
        "post_discovery_exploitation_rate": (
            mean(exploitation_rates) if exploitation_rates else None
        ),
        "mean_episode_steps": mean(record["episode_steps"] for record in episode_records),
        "task_selection_accuracy": mean(planner_accuracies) if planner_accuracies else None,
        "selected_task_completion_rate": mean(completion_rates) if completion_rates else None,
        "known_negative_repetitions": sum(
            record["known_negative_repetitions"] for record in episode_records
        ),
        "llm_calls": sum(record["llm_calls"] for record in episode_records),
        "mean_episode_wall_seconds": mean(
            record["episode_wall_seconds"] for record in episode_records
        ),
        "failure_counts": all_failures,
    }


def runtime_metadata() -> Dict[str, Any]:
    import torch

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_manifest_generation(args: argparse.Namespace) -> None:
    seed_everything(args.manifest_seed)
    env = create_environment(args, provide_lm_feedback=True)
    payload = create_rollout_manifest(
        env,
        split=args.split,
        split_manifest_path=args.split_manifest.resolve(),
        output_path=args.episode_manifest.resolve(),
        seed=args.manifest_seed,
        num_resets=args.num_resets,
        num_trials=args.num_trials,
        trial_steps=args.trial_steps,
        max_combinations=args.max_combinations,
    )
    print(
        f"Created {payload['num_episodes']} frozen {args.split} episodes: "
        f"{args.episode_manifest.resolve()}"
    )
    print(f"SHA256: {sha256_file(args.episode_manifest.resolve())}")


def run_evaluation(args: argparse.Namespace) -> None:
    import torch

    seed_everything(args.evaluation_seed)
    manifest = load_rollout_manifest(args.episode_manifest.resolve(), expected_split=args.split)
    if manifest["num_trials"] != args.num_trials or manifest["trial_steps"] != args.trial_steps:
        raise ValueError(
            "Evaluator trial settings must equal the frozen manifest: "
            f"manifest=({manifest['num_trials']}, {manifest['trial_steps']}), "
            f"args=({args.num_trials}, {args.trial_steps})"
        )
    episodes = manifest["episodes"]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": args.output_dir / "run_manifest.json",
        "checkpoint_report": args.output_dir / "checkpoint_load_report.json",
        "decisions": args.output_dir / "decisions.jsonl",
        "trials": args.output_dir / "trials.jsonl",
        "episodes": args.output_dir / "episodes.jsonl",
        "summary": args.output_dir / "summary.json",
        "summary_csv": args.output_dir / "summary.csv",
    }
    collisions = [path for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to append to an existing T1 run: {collisions}")

    if args.run_name is None:
        seed_label = f"train_seed_{args.training_seed}" if args.training_seed is not None else f"planner_seed_{args.planner_seed}"
        args.run_name = f"T1_{args.controller}_{args.split}_{seed_label}"
    checkpoint_path = (
        args.student_checkpoint.resolve()
        if args.controller == "student"
        else args.executor_checkpoint.resolve()
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        args.expected_checkpoint_sha256 is not None
        and checkpoint_sha256.lower() != args.expected_checkpoint_sha256.lower()
    ):
        raise ValueError(
            "Checkpoint SHA256 mismatch: "
            f"expected={args.expected_checkpoint_sha256.lower()}, "
            f"actual={checkpoint_sha256.lower()}, path={checkpoint_path}"
        )
    selection_metadata_path = checkpoint_path.parent / "best_checkpoint.json"
    selection_metadata = (
        json.loads(selection_metadata_path.read_text(encoding="utf-8"))
        if selection_metadata_path.is_file()
        else None
    )
    run_record = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": utc_now(),
        "run_name": args.run_name,
        "protocol": "MetaCALVIN-RA-L-T1-7Object-v1",
        "arguments": vars(args),
        "command": [sys.executable, *sys.argv],
        "episode_manifest": {
            "path": str(args.episode_manifest.resolve()),
            "sha256": sha256_file(args.episode_manifest.resolve()),
            "episode_count_used": len(episodes),
            "episode_count_available": manifest["num_episodes"],
        },
        "checkpoint": {
            "role": "student" if args.controller == "student" else "shared_executor",
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
            "selection_metadata_path": (
                str(selection_metadata_path.resolve())
                if selection_metadata_path.is_file()
                else None
            ),
            "selection_metadata": selection_metadata,
        },
        "runtime": runtime_metadata(),
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                T1_DIR / "controllers.py",
                T1_DIR / "planners.py",
                T1_DIR / "manifest.py",
                T1_DIR / "frozen_env.py",
                ROBOTEST_ROOT / "eval/recording.py",
                ROBOTEST_ROOT / "eval/resource_monitor.py",
                ROBOTEST_ROOT / "env/protoenv.py",
                ROBOTEST_ROOT / "env/env_utils.py",
                ROBOTEST_ROOT / "models/agent_model.py",
                ROBOTEST_ROOT / "models/factory.py",
                ROBOTEST_ROOT / "agent/reasoning_llm.py",
            )
            if path.is_file()
        },
    }
    write_json_atomic(paths["manifest"], run_record)

    try:
        env = create_environment(args, provide_lm_feedback=args.controller == "gemma")
        policy, checkpoint_report, model_metadata = load_policy(
            args, checkpoint_path, paths["checkpoint_report"]
        )
        controller = build_controller(args, policy, env.task_instruction_feedback)
        torch.cuda.reset_peak_memory_stats()
        episode_records = []
        for index, episode in enumerate(episodes, start=1):
            print(f"[{index}/{len(episodes)}] {episode['episode_id']} | {args.controller}")
            episode_records.append(
                rollout_episode(args, env, controller, episode, paths)
            )
        summary = {
            "schema_version": 1,
            "run_name": args.run_name,
            "protocol": "MetaCALVIN-RA-L-T1-7Object-v1",
            "split": args.split,
            "controller": args.controller,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
            "planner_seed": args.planner_seed,
            "checkpoint": str(checkpoint_path),
            "episode_manifest_sha256": run_record["episode_manifest"]["sha256"],
            "metrics": aggregate_results(episode_records),
            "models": {
                "policy": model_metadata,
                "reasoning": controller_model_metadata(controller),
            },
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        write_json_atomic(paths["summary"], summary)
        flat_csv = {
            "run_name": args.run_name,
            "controller": args.controller,
            "split": args.split,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
            "planner_seed": args.planner_seed,
            **{
                key: value
                for key, value in summary["metrics"].items()
                if not isinstance(value, (dict, list))
            },
            "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
        }
        with paths["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_csv))
            writer.writeheader()
            writer.writerow(flat_csv)
        run_record.update(
            {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "checkpoint_load_report": checkpoint_report,
                "summary": summary,
            }
        )
        write_json_atomic(paths["manifest"], run_record)
        print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    except BaseException as error:
        run_record.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        write_json_atomic(paths["manifest"], run_record)
        raise


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.mode == "generate-manifest":
        run_manifest_generation(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
