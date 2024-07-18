import sys
sys.path.append('../')
import torch
import hydra
import cv2
from pathlib import Path
from hydra import initialize, compose
import numpy as np
import re
import argparse

import pybullet as p

from pathlib import Path

from agent.agent import OracleAgent
from env.protoenv import ProtoEnv
from env.env_v1 import Env
from agent.reasoning_llm import ReasoningLLM


from robot_flamingo.models.factory import create_model_and_transforms
from robot_flamingo.eval.eval_utils import ModelWrapper

from calvin.calvin_env.calvin_env.envs.play_table_env import get_env

from omegaconf import DictConfig, OmegaConf

import time

from pytorch_lightning import seed_everything
from eval.utils import eval
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
    parser.add_argument("--path_to_env", default="/home/richard/Documents/GitHub/RoboTest/calvin/dataset/calvin_debug_dataset/validation", type=str)
    parser.add_argument("--path_to_oracle", default="/home/richard/Documents/GitHub/RoboTest/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--model_checkpoint", 
        type=str, 
        default="../RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth",
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
        "--eval_performance",
        action="store_true",
        help="evaluate on train performance",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
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
    args = parser.parse_args()
    seed_everything(args.seed)

    if args.run_name is None:
        run_name = f"eval_{args.model_checkpoint.split('/')[-1].split('.')[0]}_{args.num_objs}_{args.feedback}"
    else:
        run_name = args.run_name
    
    assert args.feedback in ['train_feedback', 'alt_feedback'], 'Feedback type not supported'

    # Load the environment
    base_env = get_env(Path(args.path_to_env), show_gui=False)
    task_cfg = OmegaConf.load(Path(args.path_to_oracle))
    task_oracle = hydra.utils.instantiate(task_cfg)

    base_env = ProtoEnv(base_env, task_oracle, num_trials=args.num_trials, trial_steps=args.trial_steps, instructions_per_trial=1, no_op_steps=0,
                provide_feedback=True, provide_train_feedback=True, provide_lm_feedback=False, reset_robot_pos=False, five_objs=args.num_objs == 5)

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
    
    model.load_state_dict(checkpoint['model_state_dict'], False)
    model.half()
    model.cuda()
    
    policy_model = ModelWrapper(model, tokenizer, image_processor, torch.float16, False)
    print('Model loaded')

    eval_stats = eval(args, base_env, policy_model)
    
    # Save the evaluation stats first check if a results file exists
    if args.path_to_results is not None:
        results_file = Path(args.path_to_results)
        if results_file.exists():
            with open(results_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([run_name] + [eval_stats[key] for key in eval_stats.keys()])
        else:
            with open(results_file, 'w') as file:
                writer = csv.writer(file)
                columns = ['run_name'] + list(eval_stats.keys())
                writer.writerow(columns)
                writer.writerow([run_name] + [eval_stats[key] for key in eval_stats.keys()])
    else:
        results_file = Path(f'./results/results.csv')
        if not results_file.exists():
            os.makedirs(results_file.parent, exist_ok=True)
        with open(results_file, 'w') as file:
            writer = csv.writer(file)
            columns = ['run_name'] + list(eval_stats.keys())
            writer.writerow(columns)
            writer.writerow([run_name] + [eval_stats[key] for key in eval_stats.keys()])




if __name__ == '__main__':
    evaluate()