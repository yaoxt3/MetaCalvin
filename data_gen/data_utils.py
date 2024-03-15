import numpy as np
import torch
import os
import time
import gc
from contextlib import contextmanager


def rollout(env, agent, steps, use_reasoning=False, train=True, annotation=None):
    '''Perform rollout of oracle agent in env. Save episode if successful.'''
    episode_data = []
    last_trial_reward = False
    switched = False

    agent.reset()
    obs, feedback, environment_description = env.episode_reset(train=train)
    lm_feedback = feedback['lm_feedback']
    train_feedback = feedback['train_feedback']
    current_trial = 0
    instruction_step = True

    for i in range(steps):
        action, instruction = agent.step(obs, lm_feedback, use_reasoning=use_reasoning, env_description=environment_description, annotation=annotation, instruction_step=instruction_step)
        obs, reward, feedback, trial, trial_done, info = env.step(action, instruction)

        if feedback is None:
            instruction_step = False
            lm_feedback = None
        else:
            instruction_step = True
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['train_feedback']

            if trial_done is False:
                if reward:
                    last_trial_reward = True
                else:
                    if last_trial_reward:
                        switched = True
                        last_trial_reward = False
                        break


        step_dict = {}
        start = time.time()
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

        step_dict['feedback'] = train_feedback
        step_dict['rel_actions'] = action
        episode_data.append(step_dict)

        if trial_done:
            current_trial += 1
            if reward:
                last_trial_reward = True
            else:
                if last_trial_reward:
                    switched = True
                    last_trial_reward = False
                    break

            if info['task_completed'] is False:
                break

            if current_trial >= env.num_trials:
                break
                

            obs, feedback, environment_description = env.trial_reset(instruction)
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['train_feedback']
            instruction_step = True
    
    #Collect tensors and clear them from cuda memory
    gc.collect()
    torch.cuda.empty_cache()  
    
    if last_trial_reward and switched is False:
        yield episode_data
        del episode_data

    else:
        yield None
        del episode_data


def collect_data(args, agent, env):
    '''Collect data from env using agent. Save successful episodes.'''
    total_ep_run = 0
    saved_ep = 0
    ep_start_id = 0
    times = []
    #Check for existing start id
    if os.path.isfile(args.save_dir + "/ep_start_end_ids.npy"):
        ep_start_end_ids = np.load(args.save_dir + "/ep_start_end_ids.npy")
        ep_start_id = ep_start_end_ids[-1][1] + 1
        saved_ep = ep_start_end_ids.shape[0]
        print(f"Found existing episode start id. Starting from episode {ep_start_id}. {args.data_episodes - saved_ep} episodes left.")

    while saved_ep < args.data_episodes:
        start = time.time()

        episode_data = rollout(env, agent, args.rollout_steps, args.use_reasoning)
        episode_data = next(episode_data)
        print(torch.cuda.memory_summary())

        if episode_data is not None:
            save_obs_and_action(episode_data, args.save_dir, ep_start_id)
            ep_start_id = ep_start_id + len(episode_data)
            saved_ep += 1
            print("Episode Saved")
        else:
            print("Episode Discarded")
        end = time.time()
        times.append(end - start)
        total_ep_run += 1

        gc.collect()
        torch.cuda.empty_cache()

        if saved_ep % args.report_interval == 0:
            print(f"Saved {saved_ep} episodes. {saved_ep/args.data_episodes*100}% done. {args.data_episodes - saved_ep} episodes left.")
            print(f"Average time per episode: {np.mean(times)}")
            print(f"Total episodes run: {total_ep_run}. Percentage of successful episodes: {saved_ep/total_ep_run*100}%")
        

def save_obs_and_action(obs, save_dir, ep_start_id):
    '''Takes in list of dict with obs, actions and feedback and saves them to disk. Expects
    numpy arrays.'''
    ep_len = len(obs)
    for i in range(ep_start_id, ep_start_id + ep_len):
        obs_i = obs[i - ep_start_id]
        
        save_path = save_dir + f"/episode_{i:06}.npz"
        np.savez(save_path, **obs_i)

    #Save episode start and end id
    if os.path.isfile(save_dir + "/ep_start_end_ids.npy"):
        ep_start_end_ids = np.load(save_dir + "/ep_start_end_ids.npy")
        ep_start_end_ids = np.append(ep_start_end_ids, [[ep_start_id, ep_start_id + ep_len - 1]], axis=0)
        np.save(save_dir + "/ep_start_end_ids.npy", ep_start_end_ids)
    else:
        print("Episode start end ids file not found. Creating new one.")
        ep_start_end_ids = np.array([[0, ep_start_id + ep_len -1]])
        np.save(save_dir + "/ep_start_end_ids.npy", ep_start_end_ids)
            


        
