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
import os

from agent.agent import OracleAgent
from env.protoenv import ProtoEnv
from env.multi_env import MultiEnv
from agent.reasoning_llm import ReasoningLLM
from data_utils import collect_data
from factory import create_model_and_transforms
from agent_model import ModelWrapper

from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
from pytorch_lightning import seed_everything

from omegaconf import DictConfig, OmegaConf

import argparse

def none_or_str(value):
    if value == 'None':
        return None
    return value

def main():
    parser = argparse.ArgumentParser(description='Script to collect data from the environment using the oracle agent')

    #Agent parameters
    parser.add_argument('--checkpoint_path', default= '/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth', 
                        type=str, help='Path to the policy model')
    parser.add_argument('--reasoning_model', default= 'google/gemma-2-27b-it', type=none_or_str, help='Path to the reasoning model')
    parser.add_argument('--tokenizer', default= 'google/gemma-2-27b-it', type=none_or_str, help='Path to the reasoning model')
    parser.add_argument('--task_llm', default= None, type=none_or_str, help='Path to the task planning model')
    parser.add_argument('--vlm', default= None, type=none_or_str, help='Path to the VLM model to describe the environment')
    parser.add_argument('--use_reasoning', action= 'store_true', help='Whether to use reasoning or not')
    parser.add_argument('--device', default= 'cuda', type=str, help='Device to run the policy model')
    parser.add_argument('--use_cuda_and_cpu', action= 'store_true', help='Whether split up the models between cuda and cpu or not')
  
    #Environment parameters
    parser.add_argument('--provide_feedback', action= 'store_true', help='Whether to provide feedback or not')
    parser.add_argument('--num_instructions', default= 2, type=int, help='Number of instructions to provide')
    parser.add_argument('--reset_robot_pos', action= 'store_true', help='Whether to reset the robot position or not')
    parser.add_argument('--multi_env', action= 'store_true', help='Whether to use the multi_env or not')
    parser.add_argument('--rollout_steps', default= 1000, type=int, help='Number of steps per meta_episode')
    parser.add_argument('--num_trials', default= 4, type=int, help='Number of trials per episode')
    parser.add_argument('--eval', action= 'store_true', help='Generate evaluation data')
    parser.add_argument('--num_objs', default=3, type=int, help='Number of relevant objects')
    parser.add_argument('--calvin_dataset_path', default= "../calvin/dataset/calvin_debug_dataset/validation",
                        type=str, help='Path to the calvin dataset to initialize the environment')
    parser.add_argument('--calvin_task_path', default= "../calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml",
                        type=str, help='Path to calvin new_playtable_tasks.yaml')

    #Data collection parameters
    parser.add_argument('--data_episodes', default= 300, type=int, help='Number of episodes to collect')
    # Hardcoded for 4 trials for now, change later
    parser.add_argument('--zero_trial_limit', default= 5, type=int, help='Number of trials that should have zero trials of exploration before reward is found')
    parser.add_argument('--one_trial_limit', default= 15, type=int, help='Number of trials that should have one trial of exploration before reward is found')
    parser.add_argument('--two_trial_limit', default= 30, type=int, help='Number of trials that should have two trials of exploration before reward is found')
    parser.add_argument('--save_dir', default= None, type=none_or_str, help='Path to save the collected data')
    parser.add_argument('--report_interval', default= 10, type=int, help='Interval to report the number of episodes collected')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--render', action= 'store_true', help='Whether to render the environment or not')

    
    args = parser.parse_args()
    
    
    if args.save_dir is None:
        #Create a directory to save the collected data if it does not exist
        args.save_dir = os.path.join(os.getcwd(), 'data_oracle/training')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    print(args.save_dir)

    seed_everything(args.seed)
    
 
    # Load the base environment
    base_env = get_env(Path(args.calvin_dataset_path), show_gui=False)
    task_cfg = OmegaConf.load(args.calvin_task_path)
    task_oracle = hydra.utils.instantiate(task_cfg)

    if args.multi_env:
        env = MultiEnv(base_env, task_oracle, num_trials=args.num_trials, trial_steps=int(args.rollout_steps/args.num_trials),
                    provide_feedback=args.provide_feedback, reset_robot_pos=args.reset_robot_pos, provide_train_feedback=True)
    else:
        env = ProtoEnv(base_env, task_oracle, num_trials=args.num_trials, trial_steps=int(args.rollout_steps/args.num_trials),
                    provide_feedback=args.provide_feedback, reset_robot_pos=args.reset_robot_pos, num_objs=args.num_objs, provide_train_feedback=True)

    print("Environment loaded")

    
    # Load the policy model
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path='anas-awadalla/mpt-1b-redpajama-200b-dolly',
        tokenizer_path='anas-awadalla/mpt-1b-redpajama-200b-dolly',
        llm_name='mpt_dolly_3b',
        cross_attn_every_n_layers=1,
        decoder_type='lstm',
        fusion_mode='post',
        use_gripper=True,
        window_size=12,

    )

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', mmap=True)
    for key in list(checkpoint['model_state_dict'].keys()):
        #remove module from key name
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    
    model.load_state_dict(checkpoint['model_state_dict'], False)
    
    model.eval()
    model.half()
    model.to(args.device)
    
    policy_model = ModelWrapper(model, tokenizer, image_processor, torch.float16, False)
    print("Policy model loaded")

    # Load reasoning, task planning and VLM models
    reasoning_llm = None
    task_llm = None
    vlm = None

    if args.use_cuda_and_cpu:
        if args.reasoning_model is not None:
            reasoning_llm = ReasoningLLM(args.reasoning_model, args.tokenizer, device_map='auto', cache_dir='/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/huggingface_cache')
        if args.task_llm is not None:
            task_llm = ReasoningLLM(args.task_llm, device_map='cpu')
        if args.vlm is not None:
            vlm = ReasoningLLM(args.vlm, device_map='cpu')
    else:
        if args.reasoning_model is not None:
            reasoning_llm = ReasoningLLM(args.reasoning_model, args.tokenizer, cache_dir='/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/huggingface_cache', load_in_8bit=True)
        if args.task_llm is not None:
            task_llm = ReasoningLLM(args.task_llm)
        if args.vlm is not None:
            vlm = ReasoningLLM(args.vlm)
    print("Reasoning, task planning and VLM models loaded")
    
    # Create the agent
    agent = OracleAgent(policy_model, reasoning_llm, task_llm, vlm)

    # Collect data
    collect_data(args, agent, env)



if __name__ == '__main__':
    main()
