import sys
sys.path.append("../calvin/calvin_models/")
sys.path.append("../calvin/calvin_env/")
sys.path.append("../calvin/calvin_env/tacto")
sys.path.append("../calvin/calvin_models/calvin_agent/")
sys.path.append("../open_flamingo/")
sys.path.append("../models/")
sys.path.append("../")


import torch
import hydra
from pathlib import Path
import numpy as np
import re
import argparse
import json
import platform
import socket

import pybullet as p

from pathlib import Path

from agent.agent import OracleAgent
from env.protoenv import ProtoEnv
from env.multi_env import MultiEnv
from agent.reasoning_llm import ReasoningLLM
from agent.agent import OracleAgent


from factory import create_model_and_transforms
from agent_model import ModelWrapper

from calvin.calvin_env.calvin_env.envs.play_table_env import get_env

from omegaconf import DictConfig, OmegaConf

import time

from transformers import set_seed as seed_everything
from eval.utils import eval_explore, eval_combi_explore
from eval.recording import sha256_file, utc_now, write_json_atomic
import csv
import os

def none_or_str(value):
    if value == 'None':
        return None
    return value


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="anas-awadalla/mpt-1b-redpajama-200b-dolly", type=str)
    parser.add_argument("--path_to_env", default="../calvin/dataset/calvin_debug_dataset/validation", type=str)
    parser.add_argument("--path_to_oracle", default="../calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--model_checkpoint", 
        type=str, 
        default='/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint/open_flamingo_checkpoint.pt',
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--use_gripper",
        action="store_false",
        help="use gripper tokens",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="post",
        help="fusion mode for multimodal embeddings",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="mpt_dolly_3b",
        help="name of the language model",
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="lstm",
        help="type of decoder to use",
    )
    parser.add_argument(
        "--action_tokens",
        action = "store_true",
        help="use additional action tokens",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=4,
        help="number of trials",
    )
    parser.add_argument(
        "--trial_steps",
        type=int,
        default=300,
        help="number of steps per trial",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render the environment",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="save rollout observations as MP4 videos without opening a GUI window",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="./videos",
        help="directory used for recorded rollout videos",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=30.0,
        help="frame rate of recorded rollout videos",
    )
    parser.add_argument(
        "--max_combinations",
        type=int,
        default=None,
        help="optionally limit evaluated object combinations for smoke tests or visualization",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default="train_feedback",
        help="type of feedback [train_feedback, alt_feedback]",
    )
    parser.add_argument(
        "--num_objs",
        type=int,
        default=3,
        help="number of objects",
    )
    parser.add_argument(
        "--num_resets",
        type=int,
        default=3,
        help="number of runs per environment state",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="evaluate on train performance",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=12,
        help="window size for model"
    )
    parser.add_argument(
        "--run_name",
        type=none_or_str,
        default=None,
        help="name of the run",
    )
    parser.add_argument(
        "--path_to_results",
        type=none_or_str,
        default=None,
        help="path to results file if it exists",
    )
    parser.add_argument(
        "--summary_json_path",
        type=none_or_str,
        default=None,
        help="optional path for the machine-readable aggregate summary",
    )
    parser.add_argument(
        "--trial_log_path",
        type=none_or_str,
        default=None,
        help="optional append-only JSONL path for one record per completed trial",
    )
    parser.add_argument(
        "--episode_log_path",
        type=none_or_str,
        default=None,
        help="optional append-only JSONL path for one record per completed meta-episode",
    )
    parser.add_argument(
        "--run_manifest_path",
        type=none_or_str,
        default=None,
        help="optional path for run arguments, code hashes, environment, and completion status",
    )
    parser.add_argument(
        "--checkpoint_load_report_path",
        type=none_or_str,
        default=None,
        help="optional path for missing and unexpected checkpoint state-dict keys",
    )
    parser.add_argument(
        "--protocol_label",
        type=str,
        default="legacy-evaluator",
        help="label stored with structured records; does not alter evaluation behavior",
    )
    parser.add_argument(
        "--eval_llm",
        action="store_true",
        help="evaluate the language model",
    )
    parser.add_argument('--reasoning_model', 
        default= 'google/gemma-2-27b-it', 
        type=none_or_str, 
        help='Path to the reasoning model'
    )
    parser.add_argument('--tokenizer', 
        default= 'google/gemma-2-27b-it', 
        type=none_or_str, 
        help='Path to the reasoning model'
    )
    parser.add_argument('--path_to_hf_cache',
        default= '/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/huggingface_cache',
        type=str,
        help='Path to the hugginface model cache')
    parser.add_argument('--combi_explore',
                        action="store_true",
        help='Evaluate the Combi Explore setting')
    args = parser.parse_args()
    if args.video_fps <= 0:
        parser.error("--video_fps must be greater than zero")
    if args.max_combinations is not None and args.max_combinations <= 0:
        parser.error("--max_combinations must be greater than zero")
    seed_everything(args.seed)

    if args.run_name is None:
        explore_setting = 'combi' if args.combi_explore else 'standard'
        run_name = f"eval__{explore_setting}_{args.model_checkpoint.split('/')[-1].split('.')[0]}_{args.num_objs}_{args.feedback}"
    else:
        run_name = args.run_name
    args.run_name = run_name
    
    assert args.feedback in ['train_feedback', 'alt_feedback'], 'Feedback type not supported'

    checkpoint_path = Path(args.model_checkpoint).resolve()
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")

    code_paths = [
        Path(__file__).resolve(),
        (Path(__file__).parent / "utils.py").resolve(),
        (Path(__file__).parent / "recording.py").resolve(),
        (Path(__file__).parent.parent / "env" / "protoenv.py").resolve(),
        (Path(__file__).parent.parent / "models" / "agent_model.py").resolve(),
    ]
    checkpoint_metadata_path = checkpoint_path.parent / "best_checkpoint.json"
    checkpoint_metadata = None
    if checkpoint_metadata_path.is_file():
        with checkpoint_metadata_path.open("r", encoding="utf-8") as metadata_file:
            checkpoint_metadata = json.load(metadata_file)

    print(f"Computing checkpoint SHA256: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    run_manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": utc_now(),
        "run_name": run_name,
        "protocol_label": args.protocol_label,
        "protocol_warning": (
            "This is the legacy evaluator. It does not consume the frozen P0 common-episode "
            "manifest and must not be reported as the formal P0 final-test result."
        ),
        "command": [sys.executable, *sys.argv],
        "arguments": vars(args),
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
            "selection_metadata": checkpoint_metadata,
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "reproducibility": {
            "evaluation_seed": args.seed,
            "seed_function": "transformers.set_seed",
            "simulator_seed": None,
            "simulator_seed_note": (
                "The legacy evaluator does not explicitly call base_env.seed; each sampled "
                "initial environment state is stored in episodes.jsonl."
            ),
        },
        "source_sha256": {
            str(path): sha256_file(path) for path in code_paths
        },
    }
    write_json_atomic(args.run_manifest_path, run_manifest)

    # Load the environment
    base_env = get_env(
        Path(args.path_to_env),
        obs_space={
            "rgb_obs": ["rgb_static", "rgb_gripper"],
            "depth_obs": [],
        },
        show_gui=False,
    )
    task_cfg = OmegaConf.load(Path(args.path_to_oracle))
    task_oracle = hydra.utils.instantiate(task_cfg)

    if args.combi_explore:
        args.num_objs = 4
        base_env = MultiEnv(base_env, task_oracle, num_trials=args.num_trials, trial_steps=args.trial_steps,
                provide_feedback=True, provide_train_feedback=True, provide_lm_feedback=True if args.eval_llm else False,
                reset_robot_pos=False)
    else:
        base_env = ProtoEnv(base_env, task_oracle, num_trials=args.num_trials, trial_steps=args.trial_steps,
                provide_feedback=True, provide_train_feedback=True, provide_lm_feedback=True if args.eval_llm else False,
                reset_robot_pos=False, num_objs=args.num_objs)

    print('Environment loaded')
    # Load the model
    window_size = 1 if args.decoder_type == 'lstm_seq' else args.window_size
    print(window_size)
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_gripper=args.use_gripper,
        fusion_mode=args.fusion_mode,
        llm_name=args.llm_name,
        decoder_type=args.decoder_type,
        action_tokens=args.action_tokens,
        window_size=window_size
    )

    checkpoint = torch.load(args.model_checkpoint, map_location='cpu', mmap=True)

    for key in list(checkpoint['model_state_dict'].keys()):
        #remove module from key name
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    
    load_result = model.load_state_dict(checkpoint['model_state_dict'], False)
    checkpoint_load_report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "strict": False,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    write_json_atomic(args.checkpoint_load_report_path, checkpoint_load_report)
    print(
        "Checkpoint load report: "
        f"{len(load_result.missing_keys)} missing keys, "
        f"{len(load_result.unexpected_keys)} unexpected keys"
    )
    model.half()
    model.cuda()
    
    policy_model = ModelWrapper(model, tokenizer, image_processor, torch.float16, False)
    print('Model loaded')

    if args.eval_llm:
        # Load the reasoning model
        reasoning_llm = ReasoningLLM(args.reasoning_model, args.tokenizer,  cache_dir=args.path_to_hf_cache)
        agent = OracleAgent(policy_model, reasoning_llm)
        print('Oracle Agent Loaded')
    else:
        agent = policy_model

    if args.combi_explore:
        eval_stats = eval_combi_explore(args, base_env, agent)
    else:
        eval_stats = eval_explore(args, base_env, agent)
    
    # Save the evaluation stats first check if a results file exists
    if args.path_to_results is not None:
        results_file = Path(args.path_to_results)
    else:
        results_file = Path(f'./results/results.csv')

    if not results_file.exists():
        # Ensure the parent directory exists
        os.makedirs(results_file.parent, exist_ok=True)
        with open(results_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            columns = ['run_name'] + list(eval_stats.keys())
            writer.writerow(columns)

    # Append the new row
    with open(results_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([run_name] + [eval_stats[key] for key in eval_stats.keys()])

    summary_payload = {
        "schema_version": 1,
        "run_name": run_name,
        "protocol_label": args.protocol_label,
        "feedback": args.feedback,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_stats": eval_stats,
    }
    write_json_atomic(args.summary_json_path, summary_payload)
    run_manifest["status"] = "complete"
    run_manifest["completed_at_utc"] = utc_now()
    run_manifest["checkpoint_load_report"] = checkpoint_load_report
    run_manifest["summary"] = summary_payload
    write_json_atomic(args.run_manifest_path, run_manifest)


if __name__ == '__main__':
    evaluate()
