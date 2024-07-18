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
from env.protoenv import ProtoEnv
#from env.env_v1 import ProtoEnv
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

def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        test_line = current_line + " " + word
        text_width, _ = cv2.getTextSize(test_line, font, fontScale=0.3, thickness=1)[0]
        
        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    
    lines.append(current_line)
    # reverse the lines to maintain order
    lines.reverse()
    return lines



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
        # Add small random noise to first 6 actions
        obs, reward, feedback, trial, trial_done, _ = env.step(action, instruction)
        
        if feedback is None:
            instruction_step = False
            lm_feedback = None
        else:
            instruction_step = True
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['legacy_feedback']
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
            train_feedback = feedback['legacy_feedback']
            print(train_feedback)
            instruction_step = True
            


        rgb = env.render(mode="rgb_array")[:,:,::-1]
        cv2.imshow('test',rgb)
        cv2.waitKey(1)

import cv2
import numpy as np

def rollout_eval(env, model, steps, fps=15):
    model.reset()
    obs, feedback, environment_description = env.episode_reset(train=True)
    lm_feedback = feedback['lm_feedback']
    train_feedback = feedback['legacy_feedback']
    print(train_feedback)
    reset_actions = None
    current_trial = 0
    
    # Initialize video writer
    height, width, _ = env.render(mode="rgb_array").shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/richard/Documents/GitHub/RoboTest/Videos/test.mp4', fourcc, fps, (width, height))

    for i in range(steps):
        if reset_actions is None:
            #train_feedback = 'move the door to the left was rewarding last trial; '
            action = model.step(obs, train_feedback)
            #print(sum([abs(a) for a in list(action[:-1])])/len(action[:-1]))
        else:
            action = reset_actions
            reset_actions = None
        obs, reward, feedback, trial, trial_done, _ = env.step(action)
        
        if feedback is not None:
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['legacy_feedback']

        if trial_done:
            print('trial done')
            current_trial += 1
            if current_trial >= env.num_trials:
                break
            obs, feedback, environment_description = env.trial_reset()
            reset_actions = None
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['legacy_feedback']
            print(train_feedback, 'THERE')

        rgb = env.render(mode="rgb_array")
        rgb = np.array(rgb[:, :, ::-1], dtype=np.uint8) 
        
        # Put feedback text on the frame
        feedback_text = f'{train_feedback}'
        max_text_width = int(width * 0.9)  # Limit text width to 90% of frame width
        lines = wrap_text(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, max_text_width)
        
        font_scale = 0.3  # Adjust font size based on scale factor
        font_color = (0, 255, 0)  # Green color
        thickness = 1 # Adjust thickness based on scale factor
        line_type = cv2.LINE_AA  # Anti-aliased line type for smoother text
        
        y = height - 10
        for line in lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            x = int((width - text_size[0]) / 2)  # Center text horizontally
            cv2.putText(rgb, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, line_type)
            y -= int(text_size[1] * 1.5)  # Adjust vertical spacing between lines
        
        # Write the frame to the video file
        out.write(rgb)
        
        cv2.imshow('test', rgb)
        cv2.waitKey(1000)

    # Release video writer and close display window
    out.release()
    cv2.destroyAllWindows()

# Example usage
# rollout_eval(env, model, steps=1000)




if __name__ == '__main__':
    seed_everything(38) # 781
    # Load the base environment
    base_env = get_env(Path("/home/richard/Documents/GitHub/RoboTest/calvin/dataset/calvin_debug_dataset/validation"), show_gui=False)
    #base_env.seed(42)
    
    task_cfg = OmegaConf.load("/home/richard/Documents/GitHub/RoboTest/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    env = ProtoEnv(base_env, task_oracle, num_trials=10, trial_steps=400, instructions_per_trial=1, no_op_steps=0,
                   provide_feedback=True, provide_train_feedback=True, provide_lm_feedback=False, reset_robot_pos=False, num_objs=3)

    # Load the model
    model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path='anas-awadalla/mpt-1b-redpajama-200b-dolly',
    tokenizer_path='anas-awadalla/mpt-1b-redpajama-200b-dolly',
    llm_name='mpt_dolly_3b',
    cross_attn_every_n_layers=1,
    decoder_type='lstm_seq',
    fusion_mode='post',
    use_gripper=True,
    window_size=1,
    action_tokens=False,

    )

    
    checkpoint = torch.load('/home/richard/Documents/GitHub/RoboTest/RoboCheckpoint/lstm_seq_ws32_3_objs.pth', map_location='cpu', mmap=True)

    for key in list(checkpoint['model_state_dict'].keys()):
        #remove module from key name
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    
    model.load_state_dict(checkpoint['model_state_dict'], False)
    # Save the model checkpoint
    
    '''
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad and 'normalizer' not in name:
            del state_dict[name]

    checkpoint_dict = {
        'model_state_dict': state_dict,
    } 
    torch.save(checkpoint_dict, '/home/richard/Documents/GitHub/RoboTest/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth')
    '''
    model.half()
    model.cuda()
    
    policy_model = ModelWrapper(model, tokenizer, image_processor, torch.float16, False)

    
    #Load reasoning model
    #reasoning_llm = ReasoningLLM(model_name="Qwen/Qwen1.5-4B-Chat", tokenizer_name="Qwen/Qwen1.5-4B-Chat", torch_dtype='auto', device_map="cpu", temperature=0.7)

    agent = OracleAgent(policy_model, reasoning_llm=None, task_llm=None, vlm=None)

    # Rollout the environment
    annotation = 'move the slider door to the left'
    #annotation = None
    #start = time.time()
    #rollout(env, agent, 3000, use_reasoning=False, annotation=annotation)
    #end = time.time()
    #print(end - start)

    rollout_eval(env, policy_model, 1600)
    