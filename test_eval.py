import sys
sys.path.append('../')

import torch
import hydra
import cv2
from pathlib import Path
from hydra import initialize, compose

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



def rollout(env, agent, steps, use_reasoning=False, annotation=None):
    agent.policy_model.reset()
    obs, feedback, environment_description = env.episode_reset(train=True)
    lm_feedback = feedback['lm_feedback']
    train_feedback = feedback['train_feedback']
    current_trial = 0
    instruction_step = True
    for i in range(steps):
        print(train_feedback)
        print(lm_feedback)
        instruction = annotation
        action, instruction = agent.step(obs, lm_feedback, use_reasoning=use_reasoning, env_description=environment_description, annotation=annotation, instruction_step=instruction_step)
        obs, reward, feedback, trial, trial_done, _ = env.step(action, instruction)
        
        if feedback is None:
            instruction_step = False
            lm_feedback = None
        else:
            instruction_step = True
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['train_feedback']
        if trial_done:
            print('trial done')
            current_trial += 1
            if current_trial >= env.num_trials:
                break
            obs, feedback, environment_description = env.trial_reset(instruction)
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['train_feedback']
            instruction_step = True


        rgb = env.render(mode="rgb_array")[:,:,::-1]
        cv2.imshow('test',rgb)
        cv2.waitKey(1)

if __name__ == '__main__':
    # Load the base environment
    base_env = get_env(Path("/home/richard/Documents/GitHub/RoboTest/calvin/dataset/calvin_debug_dataset/validation"), show_gui=False)
    task_cfg = OmegaConf.load("/home/richard/Documents/GitHub/RoboTest/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    env = ProtoEnv(base_env, task_oracle, num_trials=2, trial_steps=1000, provide_feedback=True, provide_train_feedback=True)

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
    window_size=30,

    )

    checkpoint = torch.load('/home/richard/Documents/Flamingo/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth', map_location='cpu', mmap=True)
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
    annotation = 'grasp the door handle, then slide the door to the left'
    #annotation = None
    start = time.time()
    rollout(env, agent, 2000, use_reasoning=False, annotation=annotation)
    end = time.time()
    print(end - start)