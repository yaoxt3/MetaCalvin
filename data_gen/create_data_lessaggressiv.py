#!/usr/bin/env python3
"""Collect the single F1 Combined-Explore Minimal comparison dataset."""

import argparse
import os
import random
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTEST_DIR = SCRIPT_DIR.parent
for relative in (
    "calvin/calvin_models",
    "calvin/calvin_env",
    "calvin/calvin_env/tacto",
    "calvin/calvin_models/calvin_agent",
    "open_flamingo",
    "models",
    ".",
):
    sys.path.insert(0, str(ROBOTEST_DIR / relative))
sys.path.insert(0, str(SCRIPT_DIR))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

# urdfpy 0.0.22 (used by CALVIN/TACTO) still refers to the NumPy alias that
# was removed in NumPy 1.24. Keep this compatibility local to this new entry.
if "float" not in np.__dict__:
    np.float = float

from agent.agent import OracleAgent
from agent.reasoning_llm import ReasoningLLM
from agent_model import ModelWrapper
from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
from data_utils_lessaggressiv import collect_data_lessaggressiv
from env.multi_env import MultiEnv
from factory import create_model_and_transforms


def seed_everything_lessaggressiv(seed):
    """Seed collection without importing the training-only Lightning stack."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def none_or_str(value):
    return None if value == "None" else value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect Combined-Explore trajectories with Minimal filtering and "
            "exact pair/episode-length matching to frozen Strict manifests."
        )
    )
    parser.add_argument(
        "--checkpoint_path",
        default=(
            "/data/yaoxt/RoboData/RoboCheckpoint/"
            "checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_"
            "mpt_dolly_3b_2.pth"
        ),
    )
    parser.add_argument("--reasoning_model", default="google/gemma-2-27b-it", type=none_or_str)
    parser.add_argument("--tokenizer", default="google/gemma-2-27b-it", type=none_or_str)
    parser.add_argument("--task_llm", default=None, type=none_or_str)
    parser.add_argument("--vlm", default=None, type=none_or_str)
    parser.add_argument("--cache_dir", default=None, type=none_or_str)
    parser.add_argument("--use_reasoning", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_cuda_and_cpu", action="store_true")

    parser.add_argument("--provide_feedback", action="store_true")
    parser.add_argument("--num_instructions", default=2, type=int)
    parser.add_argument("--reset_robot_pos", action="store_true")
    parser.add_argument("--rollout_steps", default=1600, type=int)
    parser.add_argument("--num_trials", default=4, type=int)
    parser.add_argument(
        "--calvin_dataset_path",
        default=str(ROBOTEST_DIR / "calvin/dataset/calvin_debug_dataset/validation"),
    )
    parser.add_argument(
        "--calvin_task_path",
        default=str(
            ROBOTEST_DIR
            / "calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
        ),
    )

    parser.add_argument(
        "--reference_train_manifest",
        default=str(ROBOTEST_DIR / "protocol/manifests/combined_train.json"),
    )
    parser.add_argument(
        "--reference_validation_manifest",
        default=str(ROBOTEST_DIR / "protocol/manifests/combined_validation.json"),
    )
    parser.add_argument(
        "--save_dir",
        default=(
            "/data/yaoxt/pn36ce/pn36ce-dss-0000/"
            "training_explore_lessaggressiv"
        ),
    )
    parser.add_argument("--report_interval", default=5, type=int)
    parser.add_argument(
        "--max_attempts",
        default=0,
        type=int,
        help="Safety cap over all attempts; 0 means no cap.",
    )
    parser.add_argument(
        "--stop_after_episodes",
        default=0,
        type=int,
        help=(
            "Pause after this many accepted episodes in total; 0 collects all. "
            "Rerunning with a larger value resumes from the saved state."
        ),
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.num_instructions != 2:
        parser.error("Combined-Explore requires exactly two instructions per trial")
    if not args.use_reasoning or args.reasoning_model is None:
        parser.error("This oracle collection requires --use_reasoning and a reasoning model")
    if not args.provide_feedback:
        parser.error("Combined-Explore oracle collection requires --provide_feedback")
    if args.rollout_steps < 13:
        parser.error("--rollout_steps must be at least 13 for ws=12 training")
    if args.num_trials < 1:
        parser.error("--num_trials must be positive")
    if args.report_interval < 1:
        parser.error("--report_interval must be positive")
    if args.max_attempts < 0:
        parser.error("--max_attempts cannot be negative")
    if args.stop_after_episodes < 0:
        parser.error("--stop_after_episodes cannot be negative")
    return args


def load_reasoner(model_name, tokenizer, device_map, cache_dir, load_in_8bit=False):
    if model_name is None:
        return None
    kwargs = {"device_map": device_map}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    return ReasoningLLM(model_name, tokenizer, **kwargs)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Minimal matched dataset: %s" % save_dir)
    seed_everything_lessaggressiv(args.seed)

    base_env = get_env(Path(args.calvin_dataset_path), show_gui=False)
    task_cfg = OmegaConf.load(args.calvin_task_path)
    task_oracle = hydra.utils.instantiate(task_cfg)
    env = MultiEnv(
        base_env,
        task_oracle,
        num_trials=args.num_trials,
        trial_steps=int(args.rollout_steps / args.num_trials),
        num_instructions=args.num_instructions,
        provide_feedback=args.provide_feedback,
        reset_robot_pos=args.reset_robot_pos,
        provide_train_feedback=True,
    )
    print("Combined-Explore environment loaded")

    model, image_processor, policy_tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        llm_name="mpt_dolly_3b",
        cross_attn_every_n_layers=1,
        decoder_type="lstm",
        fusion_mode="post",
        use_gripper=True,
        window_size=12,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", mmap=True)
    state_dict = checkpoint["model_state_dict"]
    for key in list(state_dict.keys()):
        if key.startswith("module."):
            state_dict[key[len("module."):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.eval().half().to(args.device)
    policy_model = ModelWrapper(
        model, policy_tokenizer, image_processor, torch.float16, False
    )
    print("Executor policy loaded")

    if args.use_cuda_and_cpu:
        reasoning_llm = load_reasoner(
            args.reasoning_model,
            args.tokenizer,
            "auto",
            args.cache_dir,
        )
        task_llm = load_reasoner(args.task_llm, args.task_llm, "cpu", args.cache_dir)
        vlm = load_reasoner(args.vlm, args.vlm, "cpu", args.cache_dir)
    else:
        reasoning_llm = load_reasoner(
            args.reasoning_model,
            args.tokenizer,
            "auto",
            args.cache_dir,
            load_in_8bit=True,
        )
        task_llm = load_reasoner(args.task_llm, args.task_llm, "auto", args.cache_dir)
        vlm = load_reasoner(args.vlm, args.vlm, "auto", args.cache_dir)
    print("Reasoning components loaded")

    agent = OracleAgent(
        policy_model,
        reasoning_llm=reasoning_llm,
        vlm=vlm,
        task_llm=task_llm,
    )
    collect_data_lessaggressiv(args, agent, env)


if __name__ == "__main__":
    main()
