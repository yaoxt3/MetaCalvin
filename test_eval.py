import sys
#sys.path.append("/home/richard/Documents/GitHub/RoboTest/open_flamingo")
sys.path.append("/home/richard/Documents/GitHub/RoboTest/calvin/calvin_models")
sys.path.append("/home/richard/Documents/GitHub/RoboTest/calvin/calvin_env")
sys.path.append("/home/richard/Documents/GitHub/RoboTest/calvin/calvin_env/tacto")
sys.path.append("/home/richard/Documents/GitHub/RoboTest/calvin/calvin_env/")

import torch
import hydra
import cv2
from pathlib import Path
from hydra import initialize, compose

from agent import OracleAgent
from protoenv import ProtoEnv
from reasoning_llm import ReasoningLLM

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


def rollout(env, agent, steps, annotation=None):
    agent.policy_model.reset()
    obs, feedback, environment_description = env.episode_reset()
    current_trial = 0
    instruction_step = True
    for i in range(steps):
        action, instruction = agent.step(obs, feedback, use_reasoning=True, env_description=environment_description, annotation=annotation, instruction_step=instruction_step)
        instruction_step = False
        obs, reward, trial, trial_done = env.step(action, instruction)
        if trial_done:
            current_trial += 1
            if current_trial >= env.num_trials:
                break
            obs, feedback, environment_description = env.trial_reset(instruction)
            instruction_step = True

        rgb = env.render(mode="rgb_array")[:,:,::-1]
        cv2.imshow('test',rgb)
        cv2.waitKey(1)

if __name__ == '__main__':
    # Load the base environment
    base_env = get_env(Path("/home/richard/Documents/GitHub/RoboTest/calvin/dataset/calvin_debug_dataset/validation"), show_gui=False)
    task_cfg = OmegaConf.load("/home/richard/Documents/GitHub/RoboTest/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    env = ProtoEnv(base_env, task_oracle, num_trials=4, trial_steps=250, provide_feedback=True)

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

    checkpoint = torch.load('/home/richard/Documents/Flamingo/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth', map_location='cpu', mmap=True)
    for key in list(checkpoint['model_state_dict'].keys()):
        #remove module from key name
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    
    model.load_state_dict(checkpoint['model_state_dict'], False)
    
    model.half()
    #model.cuda()
    
    policy_model = ModelWrapper(model, tokenizer, image_processor, torch.float16, False)

    
    #Load reasoning model
    reasoning_llm = ReasoningLLM(model_name='stabilityai/stablelm-zephyr-3b', tokenizer_name='stabilityai/stablelm-zephyr-3b', device_map="auto")

    agent = OracleAgent(policy_model, reasoning_llm=reasoning_llm, task_llm=None, vlm=None)
    print('here')

    # Rollout the environment
    #annotation = 'move the slider door to the left'
    annotation = None
    rollout(env, agent, 1000, annotation)