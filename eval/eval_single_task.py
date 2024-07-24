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
import csv
import os
import itertools
import random
import cv2

def rollout_single(args, task, task_instruction, feedback_state, environment_state, env, model, steps):
    model.reset()
    obs, feedback, _, info = env.episode_reset(feedback_state, environment_state, task, train=True)
    current_trial = 0
    total_reward = 0
    episode_length = 0
    for i in range(steps):
        episode_length += 1
        action = model.step(obs, task_instruction)
        obs, reward, _, _, trial_done, _ = env.step(action)

        if trial_done:
            total_reward += reward
            break
            
        if args.render:
            rgb = env.render(mode="rgb_array")[:,:,::-1]
            cv2.imshow('test',rgb)
            cv2.waitKey(1)

    return total_reward, episode_length

def eval_single_tasks(args, env, model):
    all_successes = 0
    episode_lengths = 0
    results = {}

    all_tasks = env.train_tasks

    for task in all_tasks:
        #task = 'lift_red_block_table'
        task_sucesses = 0
        task_episode_lengths = 0
        task_instruction = env.task_instruction_feedback[task]
        for i in range(args.num_trials):
            # Sample state for the environment
            environment_state, feedback_state = get_env_states_for_task(args, task, env)
            if args.vague_feedback:
                episode_objects = ' ,'.join(feedback_state)
                explored_objects = ' ,'.join([obj for obj in feedback_state if obj not in task])
                task_instruction = 'no reward last trial. episode objects: {0}. explored objects: {1}'.format(episode_objects, explored_objects)

            if i == 0:
                print(f"Task: {task}, Instruction: {task_instruction}")
                
            success, episode_length = rollout_single(args, task, task_instruction, feedback_state, environment_state, env, model, args.trial_steps)
            task_sucesses += success
            task_episode_lengths += episode_length
            all_successes += success
            episode_lengths += episode_length

        results[task] = {'successe rate': task_sucesses/args.num_trials, 'episode_lengths': task_episode_lengths/args.num_trials}
        print(f"Task: {task}, Successe Rate: {task_sucesses/args.num_trials}, Episode Lengths: {task_episode_lengths/args.num_trials}")
    
    results["overall"] = {"successe rate": all_successes/(args.num_trials*len(all_tasks)), "episode_lengths": episode_lengths/(args.num_trials*len(all_tasks))}
    print(f"Overall Successe Rate: {all_successes/(args.num_trials*len(all_tasks))}, Overall Episode Lengths: {episode_lengths/(args.num_trials*len(all_tasks))}")

    return results

def get_env_states_for_task(args, task, env):
    objects = env.objects
    drawer_states = ['open', 'closed']
    slider_states = ['left', 'right']
    lightbulb_states = [0, 1]
    red_block_states = ['table']
    states = drawer_states + slider_states + red_block_states + ['on', 'off']
    state_dict = {"drawer": drawer_states, "slider": slider_states, "lightbulb": lightbulb_states, "led": lightbulb_states, "red_block": red_block_states}

    for obj in objects:
        if obj in task:
            meta_object = obj

    # To account for red block being on the table
    meta_state = None
    for state in states:
        if state in task:
            meta_state = state
    
    environment_state = {"drawer":random.choice(drawer_states), "slider":random.choice(slider_states), "lightbulb":random.choice(lightbulb_states), "led":random.choice(lightbulb_states), 
                         "red_block":"table", "pink_block":"slider_right", "blue_block":"slider_left"}
    
    if args.num_objs == 5:
        feedback_state = [meta_object] + random.sample([obj for obj in objects if obj != meta_object], 2)
    else:
        feedback_state = objects
    
    if meta_object in ["red_block", "pink_block", "blue_block"]:
        environment_state[meta_object] = "table" # Change when we vary positions and colors of blocks
    elif meta_object in ["lightbulb", "led"]:
        # Get the oposite of the meta state
        if meta_state == "off":
            environment_state[meta_object] = 1
        else:
            environment_state[meta_object] = 0   
    else:
        # Get the oposite of the meta state
        environment_state[meta_object] = [state for state in state_dict[meta_object] if state != meta_state][0]
    
    return environment_state, feedback_state

if __name__ == '__main__':

    def none_or_str(value):
        if value == 'None':
            return None
        return value


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
        default=10,
        help="number of evaluation trials for each task",
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
        default="legacy",
        help="type of feedback",
    )
    parser.add_argument(
        "--num_objs",
        type=int,
        default=3,
        help="number of objects",
    )
    parser.add_argument(
        "--vague_feedback",
        action="store_true",
        help="give vague feedback"
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
    args = parser.parse_args()
    seed_everything(args.seed)

    if args.run_name is None:
        run_name = f"eval_{args.model_checkpoint.split('/')[-1].split('.')[0]}_{args.num_objs}_{args.feedback}"
    else:
        run_name = args.run_name

    # Load the environment
    base_env = get_env(Path(args.path_to_env), show_gui=False)
    task_cfg = OmegaConf.load(Path(args.path_to_oracle))
    task_oracle = hydra.utils.instantiate(task_cfg)


    base_env = ProtoEnv(base_env, task_oracle, num_trials=args.num_trials, trial_steps=args.trial_steps,
                   provide_feedback=True, provide_train_feedback=True, provide_lm_feedback=False, reset_robot_pos=True, num_objs=args.num_objs)
        
    print('Environment loaded')
    # Load the model
    window_size = 1 if args.decoder_type == 'lstm_seq' else args.window_size
    print(window_size)
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=1,
        use_gripper=True,
        fusion_mode="post",
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

    eval_stats = eval_single_tasks(args, base_env, policy_model)
    
    # Save the evaluation stats first check if a results file exists
    if args.path_to_results is not None:
        results_file = Path(args.path_to_results)
        if results_file.exists():
            with open(results_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([run_name] + eval_stats)
        else:
            with open(results_file, 'w') as file:
                writer = csv.writer(file)
                writer.writerow([run_name] + eval_stats)
    else:
        results_file = Path(f'./results/single_results.csv')
        os.makedirs(results_file.parent, exist_ok=True)
        with open(results_file, 'w') as file:
            writer = csv.writer(file)
            columns = ['run_name'] + list(eval_stats.keys())
            writer.writerow(columns)
            writer.writerow([run_name] + [eval_stats[key] for key in eval_stats.keys()])
    
    print("Evaluation complete")