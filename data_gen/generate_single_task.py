import sys
sys.path.append("../calvin/calvin_models/")
sys.path.append("../calvin/calvin_env/")
sys.path.append("../calvin/calvin_env/tacto")
sys.path.append("../calvin/calvin_models/calvin_agent/")
sys.path.append("../open_flamingo/")
sys.path.append("../robot_flamingo/")
sys.path.append("../data/")
sys.path.append("../agent/")
sys.path.append("../")

import torch
import hydra
import cv2
from pathlib import Path
import numpy as np
import re
import argparse

import pybullet as p

from pathlib import Path

from env.protoenv import ProtoEnv


from models.factory import create_model_and_transforms
from models.agent_model import ModelWrapper

from calvin.calvin_env.calvin_env.envs.play_table_env import get_env

from omegaconf import DictConfig, OmegaConf


from pytorch_lightning import seed_everything
import os
import random
import cv2

from copy import deepcopy

def rollout_single(args, task, task_instruction, feedback_state, environment_state, env, model, steps):
    model.reset()
    episode_data = []
    obs, feedback, environment_description, info = env.episode_reset(feedback_state, environment_state, task, train=True)
    robot = deepcopy(env.base_env.robot) # Copy the robot to avoid changing the original robot state for computing absolute actions
    total_reward = 0
    episode_length = 0
    for i in range(steps):
        episode_length += 1
        action = model.step(obs, task_instruction)
        action_save = action.copy()
        abs_action = robot.relative_to_absolute(action_save.copy())
        step_dict = create_episode_dict(obs, action_save, abs_action, feedback, task_instruction, None, info, False, i)
        episode_data.append(step_dict)

        obs, reward, _, _, trial_done, step_info = env.step(action)
        info = step_info['info']

        if trial_done:
            total_reward += reward
            break
        
        if args.render:
            rgb = env.render(mode="rgb_array")[:,:,::-1]
            cv2.imshow('test',rgb)
            cv2.waitKey(1)

    if total_reward > 0:
        return episode_data
    else:
        return None

def collect_data(args, env, model):
    total_episodes = 0
    # Check for existing start id
    if os.path.isfile(args.save_dir + "/ep_start_end_ids.npy"):
        ep_start_end_ids = np.load(args.save_dir + "/ep_start_end_ids.npy")
        ep_start_id = ep_start_end_ids[-1][1] + 1
        saved_ep = ep_start_end_ids.shape[0]
        episodes_left = args.data_episodes - saved_ep
        print(f"Found existing episode start id. Starting from episode {saved_ep}. {episodes_left} episodes left.")
    else:
        ep_start_id = 0
        saved_ep = 0
        episodes_left = args.data_episodes
        print(f"No existing episode start id found. Starting from episode 0. {episodes_left} episodes left.")

    all_tasks = env.train_tasks
    collected_task_episodes = {task: 0 for task in all_tasks}

    while total_episodes < args.data_episodes:
        print(collected_task_episodes)
        while True:
            task = random.choice(all_tasks)
            if collected_task_episodes[task] < args.data_episodes // len(all_tasks):
                break
        # Sample state for the environment
        environment_state, feedback_state = get_env_states_for_task(args, task, env)
        task_instruction = env.task_instruction_feedback[task]
        print(task_instruction)
        data = rollout_single(args, task, task_instruction, feedback_state, environment_state, env, model, args.trial_steps)
        if data is not None:
            collected_task_episodes[task] += 1
            total_episodes += 1
            for i in range(len(data)):
                idx = ep_start_id + i
                obs = data[i]
                save_path = args.save_dir + f"/episode_{idx:06}.npz"
                np.savez(save_path, **obs)

            if os.path.isfile(args.save_dir + "/ep_start_end_ids.npy"):
                ep_start_end_ids = np.load(args.save_dir + "/ep_start_end_ids.npy")
                ep_start_end_ids = np.append(ep_start_end_ids, [[ep_start_id, ep_start_id + len(data)- 1]], axis=0)
                np.save(args.save_dir + "/ep_start_end_ids.npy", ep_start_end_ids)
            else:
                print("Episode start end ids file not found. Creating new one.")
                ep_start_end_ids = np.array([[0, ep_start_id + len(data) -1]])
                np.save(args.save_dir + "/ep_start_end_ids.npy", ep_start_end_ids)
            ep_start_id += len(data)

def create_episode_dict(obs, rel_actions, abs_actions, feedback, instruction, output, info, trial_done, idx):
    '''Create episode dictionary for saving to disk.'''
    step_dict = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    step_dict[k2] = v2.detach().cpu().numpy()
                else:
                    step_dict[k2] = v2
        if isinstance(v, torch.Tensor):
            step_dict[k] = v.detach().cpu().numpy()
        else:
            step_dict[k] = v

    # Collect feedback from env and LLM output/instruction
    step_dict['feedback'] = feedback['train_feedback']
    step_dict['env_feedback'] = feedback['lm_feedback']
    step_dict['lm_output'] = output
    step_dict['lm_instruction'] = instruction

    # Collect actions
    step_dict['rel_actions'] = rel_actions
    step_dict['actions'] = abs_actions

    # Collect additional info
    step_dict['trial_done'] = trial_done
    step_dict['idx'] = idx

    return step_dict

        
    
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
        default="/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth",
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
        default=1,
        help="number of evaluation trials for each task",
    )
    parser.add_argument(
        "--trial_steps",
        type=int,
        default=150,
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
        "--window_size",
        type=int,
        default=12,
        help="window size for model"
    )
    parser.add_argument(
        "--save_dir",
        type=none_or_str,
        default=None,
        help="path to data folder",
    )
    parser.add_argument(
        "--data_episodes",
        type=int,
        default=10,
        help="number of episodes to collect",
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    if args.save_dir is None:
        if not os.path.exists("./Data"):
            os.mkdir("./Data")
        
        args.save_dir = "./Data"



    # Load the environment
    base_env = get_env(Path(args.path_to_env), show_gui=False)
    task_cfg = OmegaConf.load(Path(args.path_to_oracle))
    task_oracle = hydra.utils.instantiate(task_cfg)

    base_env = ProtoEnv(base_env, task_oracle, num_trials=args.num_trials, trial_steps=args.trial_steps,
            provide_feedback=True, provide_train_feedback=True, provide_lm_feedback=False, reset_robot_pos=True, five_objs=args.num_objs)
        
    print('Environment loaded')
    # Load the model
    window_size = 1 if args.decoder_type == 'lstm_seq' else args.window_size

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

    collect_data(args, base_env, policy_model)
    print('Data collected')

