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

from robot_flamingo.models.factory import create_model_and_transforms
from robot_flamingo.eval.eval_utils import ModelWrapper

from calvin.calvin_env.calvin_env.envs.play_table_env import get_env

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from calvin_agent.evaluation.utils import (
    collect_plan,
    get_env_state_for_initial_condition,
)

def convert_obs_dict_to_cpu(obs):
    #converts tensors in obs dict to cpu. works for nested dicts
    for key in obs.keys():
        if isinstance(obs[key], dict):
            obs[key] = convert_obs_dict_to_cpu(obs[key])
        else:
            obs[key] = obs[key].cpu()



if __name__ == '__main__':
    HYDRA_FULL_ERROR=1
    # Test get env
    #env, cfg = load_env()
    
    env = get_env(Path("/home/richard/Documents/GitHub/RoboTest/calvin/dataset/calvin_debug_dataset/validation"), show_gui=False)
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

    #model.half()
    checkpoint = torch.load('/home/richard/Documents/Flamingo/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth', map_location='cpu', mmap=True)
    for key in list(checkpoint['model_state_dict'].keys()):
        #remove module from key name
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    
    model.load_state_dict(checkpoint['model_state_dict'], False)
    
    model.half()
    model.cuda()
    print('Model loaded')
    policy_model = ModelWrapper(model, tokenizer, image_processor, torch.float16, False)
    print('Policy model loaded')
    initial_state = {'led': 0, 'lightbulb': 0, 'slider': 'left', 'drawer': 'open', 'red_block': 'table', 'blue_block': 'table', 'pink_block': 'slider_left', 'grasped': 0}
    annotation = "turn on the lamp by pushing the handle"
    
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    #obs = convert_obs_dict_to_numpy(obs)
    
    policy_model.reset()
    
    planned_actions = []
    for i in range(1000):
        if policy_model.replan != -1 and i % model.replan == 0:
            if policy_model.model.refresh != -1:
                policy_model.model.lang_encoder.lm_head.hidden_state = None
                policy_model.model.lang_encoder.lm_head.history_memory = policy_model.model.lang_encoder.lm_head.history_memory[-policy_model.refresh:]
            else:
                policy_model.reset()
        
        action = policy_model.step(obs, annotation, (len(planned_actions) == 0))
        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)
        if policy_model.use_diff:
            policy_model.action_hist_queue.append(action)
        
        obs, _, _, current_info = env.step(action)
        #obs = convert_obs_dict_to_numpy(obs)

        rgb = env.render(mode="rgb_array")[:,:,::-1]
        cv2.imshow('test',rgb)
        cv2.waitKey(1)
        