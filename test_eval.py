import sys
sys.path.append('../')

import torch
import hydra
import cv2
from pathlib import Path
from hydra import initialize, compose
import numpy as np
import re

import pybullet as p

from pathlib import Path

from agent.agent import OracleAgent
from protoenv import ProtoEnv
from agent.reasoning_llm import ReasoningLLM

from utils import convert_obs_dict_to_cpu

from robot_flamingo.models.factory import create_model_and_transforms
from robot_flamingo.eval.eval_utils import ModelWrapper

from calvin.calvin_env.calvin_env.envs.play_table_env import get_env

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from calvin_agent.evaluation.utils import (
    collect_plan,
    get_env_state_for_initial_condition,
)
import time

from pytorch_lightning import seed_everything



def rollout(env, agent, steps, use_reasoning=False, annotation=None):
    agent.policy_model.reset()
    obs, feedback, environment_description = env.episode_reset(train=True)
    lm_feedback = feedback['lm_feedback']
    train_feedback = feedback['train_feedback']
    legacy_feedback = feedback['legacy_feedback']
    print(train_feedback)
    print(legacy_feedback)
    current_trial = 0
    instruction_step = True
    for i in range(steps):
        instruction = annotation
        action, instruction = agent.step(obs, lm_feedback, use_reasoning=use_reasoning, env_description=environment_description, annotation=annotation, instruction_step=instruction_step)
        #print(sum([abs(a) for a in list(action[:-1])])/len(action[:-1]))
        obs, reward, feedback, trial, trial_done, _ = env.step(action, instruction)
        
        if feedback is None:
            instruction_step = False
            lm_feedback = None
        else:
            instruction_step = True
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['train_feedback']
            print(train_feedback)
        if trial_done:
            print('trial done')
            #env.base_env.seed(45 + current_trial)
            current_trial += 1
            if current_trial >= env.num_trials:
                break
            obs, feedback, environment_description = env.trial_reset(instruction)
            agent.policy_model.reset()
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['train_feedback']
            print(train_feedback)
            instruction_step = True
            


        rgb = env.render(mode="rgb_array")[:,:,::-1]
        cv2.imshow('test',rgb)
        cv2.waitKey(1)

def rollout_eval(env, model, steps):
    model.reset()
    obs, feedback, environment_description = env.episode_reset(train=True)
    lm_feedback = feedback['lm_feedback']
    train_feedback = feedback['train_feedback']
    #train_feedback = 'Objects: slider, drawer, lightbulb'
    print(train_feedback)
    reset_actions = None
    current_trial = 0
    #train_feedback = 'no reward last trial. episode objects: lightbulb, slider, drawer. explored objects: lightbulb'
    for i in range(steps):
        if reset_actions is None:
            train_feedback = 'move the door to the right was rewarding last trial; '
            action = model.step(obs, train_feedback)
        else:
            action = reset_actions
            reset_actions = None
        #print(sum([abs(a) for a in list(action[:-1])])/len(action[:-1]))
        obs, reward, feedback, trial, trial_done, _ = env.step(action)
        
        if feedback is not None:
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['train_feedback']
            print(train_feedback, 'HERE!!!!')

        if trial_done:
            print('trial done')
            current_trial += 1
            if current_trial >= env.num_trials:
                break
            obs, feedback, environment_description = env.trial_reset()
            #model.reset()
            reset_actions = None
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['train_feedback']
            #if current_trial == 1:
            #    train_feedback = 'no reward last trial. episode objects: lightbulb, slider, drawer. explored objects: lightbulb, slider'
            #else:
            #    train_feedback = 'push the handle to close the drawer was rewarding last trial;'
            print(train_feedback, 'THERE')


        rgb = env.render(mode="rgb_array")[:,:,::-1]
        cv2.imshow('test',rgb)
        cv2.waitKey(1)

if __name__ == '__main__':
    seed_everything(10) # 781
    # Load the base environment
    base_env = get_env(Path("/home/richard/Documents/GitHub/RoboTest/calvin/dataset/calvin_debug_dataset/validation"), show_gui=False)
    #base_env.seed(42)
    
    task_cfg = OmegaConf.load("/home/richard/Documents/GitHub/RoboTest/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    env = ProtoEnv(base_env, task_oracle, num_trials=4, trial_steps=200, instructions_per_trial=1, no_op_steps=0,
                   provide_feedback=True, provide_train_feedback=True, provide_lm_feedback=False)

    # Load the model
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

    
    checkpoint = torch.load('/home/richard/Documents/GitHub/RoboTest/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth', map_location='cpu', mmap=True)
    
    for key in list(checkpoint['model_state_dict'].keys()):
        #remove module from key name
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    
    model.load_state_dict(checkpoint['model_state_dict'], False)
    model.half()
    model.cuda()
    
    policy_model = ModelWrapper(model, tokenizer, image_processor, torch.float16, False)

    
    #Load reasoning model
    #reasoning_llm = ReasoningLLM(model_name="Qwen/Qwen1.5-4B-Chat", tokenizer_name="Qwen/Qwen1.5-4B-Chat", torch_dtype='auto', device_map="cpu", temperature=0.7)

    agent = OracleAgent(policy_model, reasoning_llm=None, task_llm=None, vlm=None)

    # Rollout the environment
    annotation = 'push the handle to close the drawer'
    #annotation = None
    #start = time.time()
    rollout(env, agent, 2000, use_reasoning=False, annotation=annotation)
    #end = time.time()
    #print(end - start)

    #rollout_eval(env, policy_model, 1600)
    