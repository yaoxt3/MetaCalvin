import numpy as np
import torch
import os
import time
import gc
from contextlib import contextmanager

from typing import List, Dict, Any, Tuple


def rollout(env, agent, steps, meta_task_stats, exploration_stats, exploration_stats_lmits, total_episodes, use_reasoning=False, train=True, annotation=None):
    '''Perform rollout of oracle agent in env. Save episode if successful.'''
    episode_data = []
    last_trial_reward = False
    switched = False
    meta_task_stats_episode = meta_task_stats.copy()
    exploration_stats_episode = exploration_stats.copy()
    agent.reset()
    #make sure all meta task are sampled equally
    while True:
        obs, feedback, environment_description = env.episode_reset(train=train)
        meta_task = env.meta_task
        print(meta_task, 'Resampled')
        if meta_task_stats[meta_task] <= round(total_episodes/len(env.train_tasks)):
            meta_task_stats_episode[meta_task] += 1
            break

    lm_feedback = feedback['lm_feedback']
    train_feedback = feedback['train_feedback']
    current_trial = 0
    trials_before_reward = 0
    instruction_step = True
    for i in range(steps):
        action, instruction = agent.step(obs, lm_feedback, use_reasoning=use_reasoning, env_description=environment_description, annotation=annotation, instruction_step=instruction_step)
        action_save = action.copy()
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
                    trials_before_reward += 1
                    if last_trial_reward:
                        switched = True
                        last_trial_reward = False
                        break
            
        
        if trial_done is False:
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
            step_dict['feedback'] = train_feedback
            step_dict['rel_actions'] = action_save
            episode_data.append(step_dict)

        else:
            current_trial += 1
            if reward:
                last_trial_reward = True
            else:
                trials_before_reward += 1

                if last_trial_reward:
                    switched = True
                    last_trial_reward = False
                    break

                # Hardcoded for now, since there are 3 tasks sampled, the reward should be found in the third trial at the latest
                if current_trial == 3:
                    break
            
            # Break early if enought episodes sampled for this exploration. For trials before reward = 1 
            # we cant stop early because it could turn into trials before reward = 2
            if trials_before_reward == 0 or trials_before_reward == 2:
                if exploration_stats_episode[meta_task][str(trials_before_reward)] >= exploration_stats_lmits[meta_task][str(trials_before_reward)]:
                    print('Break cause of exploration')
                    break

            if info['task_completed'] is False:
                break

            if current_trial >= env.num_trials:
                break
                

            reset_obs, feedback, environment_description = env.trial_reset(instruction)
            lm_feedback = feedback['lm_feedback']
            train_feedback = feedback['train_feedback']
            instruction_step = True

            if isinstance(reset_obs, list):
                for obs in reset_obs:
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
                    step_dict['feedback'] = train_feedback
                    step_dict['rel_actions'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1])
                    episode_data.append(step_dict)

                obs = reset_obs[-1]

            else:
                obs = reset_obs
            
            # Reset the agent model
            agent.policy_model.reset()
    
    #Collect tensors and clear them from cuda memory
    gc.collect()
    torch.cuda.empty_cache()  
    
    if last_trial_reward and switched is False and exploration_stats_episode[meta_task][str(trials_before_reward)] < exploration_stats_lmits[meta_task][str(trials_before_reward)]:
        #Update statistics
        meta_task_stats[meta_task] += 1
        exploration_stats[meta_task][str(trials_before_reward)] += 1
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
    episodes_left = args.data_episodes
    times = []
    meta_task_stats = {}
    exploration_stats = {} # Hardcoded for now to number of different tasks
    for task in env.train_tasks:
        meta_task_stats[task] = 0
        exploration_stats[task] = {'0': 0, '1': 0, '2': 0}
    
    # Create limits for exploration stats, should be heavily biased towards 2 trials before reward
    exploration_stats_limits = {}
    for task in env.train_tasks:
        exploration_stats_limits[task] = {'0': args.zero_trial_limit, '1': args.one_trial_limit, '2': args.two_trial_limit}
    sum_limits = args.zero_trial_limit + args.one_trial_limit + args.two_trial_limit
    if sum_limits * len(env.train_tasks) > args.data_episodes:
        total_episodes = args.data_episodes
        print(f"Sum of limits larger than data episodes. Total episodes set to {total_episodes}")
    else:
        total_episodes = sum_limits * len(env.train_tasks)
        print(f"Sum of limits smaller than data episodes. Total episodes set to {total_episodes}")

    # Check for existing start id
    if os.path.isfile(args.save_dir + "/ep_start_end_ids.npy"):
        ep_start_end_ids = np.load(args.save_dir + "/ep_start_end_ids.npy")
        ep_start_id = ep_start_end_ids[-1][1] + 1
        saved_ep = ep_start_end_ids.shape[0]
        episodes_left = args.data_episodes - saved_ep
        print(f"Found existing episode start id. Starting from episode {saved_ep}. {episodes_left} episodes left.")
        meta_task_stats_npz = np.load(args.save_dir + "/meta_task_stats.npz")
        meta_task_stats = convert_to_dict(meta_task_stats_npz)
        exploration_stats_npz = np.load(args.save_dir + "/exploration_stats.npz")
        exploration_stats = convert_to_dict(exploration_stats_npz)

    while saved_ep < args.data_episodes:
        start = time.time()
        print(meta_task_stats)
        print(exploration_stats)
        episode_data = rollout(env, agent, args.rollout_steps, meta_task_stats, exploration_stats, exploration_stats_limits,
                                total_episodes, args.use_reasoning)
        episode_data = next(episode_data)

        if episode_data is not None:
            save_obs_and_action(episode_data, args.save_dir, ep_start_id, meta_task_stats, exploration_stats)
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
        

def save_obs_and_action(obs, save_dir, ep_start_id, meta_task_stats, exploration_stats):
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
    
    #Save stats
    save_path_meta = save_dir + "/meta_task_stats.npz"
    save_path_explore = save_dir + "/exploration_stats.npz"
    np.savez(save_path_meta, **meta_task_stats)
    np.savez(save_path_explore, **exploration_stats)
            

def merge_datasets(data_dir1, data_dir2):
    '''Merge two data directories into one. Assumes that the data directories are
    created using the save_obs_and_action function.'''
    #Load episode start and end ids
    ep_start_end_ids1 = np.load(data_dir1 + "/ep_start_end_ids.npy")
    ep_start_end_final_id1 = ep_start_end_ids1[-1][1]
    ep_start_end_ids2 = np.load(data_dir2 + "/ep_start_end_ids.npy")
    
    # Shift episode start and end ids for dataset 2
    for i in range(ep_start_end_ids2.shape[0]):
        ep_start_end_ids2[i][0] += ep_start_end_final_id1 + 1
        ep_start_end_ids2[i][1] += ep_start_end_final_id1 + 1

    ep_start_end_ids = np.append(ep_start_end_ids1, ep_start_end_ids2, axis=0)
    np.save(data_dir1 + "/ep_start_end_ids.npy", ep_start_end_ids)

    # Merge Stats
    meta_task_stats1 = np.load(data_dir1 + "/meta_task_stats.npz")
    meta_task_stats2 = np.load(data_dir2 + "/meta_task_stats.npz")
    meta_task_stats = {}
    for key in meta_task_stats1.keys():
        meta_task_stats[key] = int(meta_task_stats1[key]) + int(meta_task_stats2[key])
    np.savez(data_dir1 + "/meta_task_stats.npz", **meta_task_stats)

    exploration_stats1 = np.load(data_dir1 + "/exploration_stats.npz")
    exploration_stats2 = np.load(data_dir2 + "/exploration_stats.npz")
    exploration_stats = {}
    for key in exploration_stats1.keys():
        exploration_stats[key] = {}
        for sub_key in exploration_stats1[key].keys():
            exploration_stats[key][sub_key] = int(exploration_stats1[key][sub_key]) + int(exploration_stats2[key][sub_key])
    np.savez(data_dir1 + "/exploration_stats.npz", **exploration_stats)

    # Merge episodes
    for i in range(ep_start_end_ids2[0][0], ep_start_end_ids2[-1][1] + 1):
        episode = np.load(data_dir2 + f"/episode_{i - ep_start_end_final_id1 - 1:06}.npz", allow_pickle=True)
        save_path = data_dir1 + f"/episode_{i:06}.npz"
        np.savez(save_path, **episode)

def convert_to_dict(npz_file):
    return_dict = {}
    for key in npz_file.keys():
        if isinstance(npz_file[key], Dict):
            return_dict[key] = convert_to_dict(npz_file[key])    
        else:
            return_dict[key] = int(npz_file[key])
    return return_dict

def reframe_train_feedback(dataset):
    ep_start_end_ids = np.load(dataset + "/ep_start_end_ids.npy")
    len_dataset = ep_start_end_ids[-1][1] + 1
    possible_objects = ['red block', 'pink block', 'slider', 'drawer', 'led', 'lightbulb']
    used_objects = []
    for ep in ep_start_end_ids:
        used_objects = []
        ep_start = ep[0]
        ep_end = ep[1]
        for i in range(ep_start, ep_end+1):
            episode = np.load(dataset + f"/episode_{i:06}.npz", allow_pickle=True)
            episode = dict(episode)
            feedback = np.array2string(episode['feedback'])
            print(feedback)
            
            if 'Trial 1' in feedback:
                for obj in possible_objects:
                    if obj in feedback:
                        used_objects.append(obj)
                feedback_out = '{0}, {1}, {2}'.format(used_objects[0], used_objects[1], used_objects[2])
            else:
                if 'trial 1' in feedback or 'trial 2' in feedback or 'trial 3' in feedback:
                    if 'trial 1' in feedback:
                        trial = 'Trial 2'
                    elif 'trial 2' in feedback:
                        trial = 'Trial 3'
                    elif 'trial 3' in feedback:
                        trial = 'Trial 4'

                    if 'was not rewarding' in feedback and 'was rewarding' not in feedback:
                        feedback = '{0}, {1}, {2}'.format(used_objects[0], used_objects[1], used_objects[2])
                    elif 'was rewarding' in feedback:
                        feedback_split = feedback.split(':')[1].split(';')
                        for split in [feedback_split[0], feedback_split[1]]:
                            if 'was rewarding' in split:
                                feedback_out = split

                else:
                    #if 'Trial 1' or 'trial 1' in feedback:
                    #    trial = 'Trial 1'
                    #elif 'Trial 2' or 'trial 2' in feedback:
                    #    trial = 'Trial 2'
                    #elif 'Trial 3' or 'trial 3' in feedback:
                    #    trial = 'Trial 3'
                    feedback_out = '{0}, {1}, {2}'.format(used_objects[0], used_objects[1], used_objects[2])
                    feedback_out = '{0}, {1}, {2}'.format(used_objects[0], used_objects[1], used_objects[2])
                
            print(feedback_out)
        episode['feedback'] = feedback_out
        save_path = dataset + f"/episode_{i:06}.npz"
        np.savez(save_path, **episode)

def visualize_dataset(dataset):
    ep_start_end_ids = np.load(dataset + "/ep_start_end_ids.npy")
    len_dataset = ep_start_end_ids[-1][1] + 1
    possible_objects = ['red block', 'pink block', 'slider', 'drawer', 'led', 'lightbulb']
    used_objects = []
    for i in range(len_dataset):
        episode = np.load(dataset + f"/episode_{i:06}.npz", allow_pickle=True)
        episode = dict(episode)
        feedback = np.array2string(episode['feedback'])
        action = episode['rel_actions']
        rgb = episode['rgb_static']
        print(feedback)
        cv2.imshow('test',rgb[:,:,::-1])
        cv2.waitKey(60)

def get_data_statistics(dataset):
    feedback_stats = {}
    ep_start_end_ids = np.load(dataset + "/ep_start_end_ids.npy")
    len_dataset = ep_start_end_ids[-1][1] + 1
    for i in range(len_dataset):
        episode = np.load(dataset + f"/episode_{i:06}.npz", allow_pickle=True)
        episode = dict(episode)
        feedback = np.array2string(episode['feedback'])
        if feedback in feedback_stats.keys():
            feedback_stats[feedback] += 1
        else:
            feedback_stats[feedback] = 1
    percentage = {}
    for key in feedback_stats.keys():
        percentage[key] = round(100 * feedback_stats[key]/len_dataset, 2)
    # Save stats
    save_path = dataset + "/feedback_stats.npz"
    np.savez(save_path, **feedback_stats)
    # Sort by percentage
    percentage = dict(sorted(percentage.items(), key=lambda item: item[1], reverse=True))
    for key in percentage.keys():
        print(f"{key}: {percentage[key]}%")
    


if __name__ == '__main__':
    import cv2
    #get_data_statistics('/home/richard/Documents/GitHub/RoboTest/data/test')
    #get_data_statistics('/home/richard/Documents/GitHub/RoboTest/data/validation_3_objs_biased')
    visualize_dataset('/home/richard/Documents/GitHub/RoboTest/data/validation_5_objs_biased')

        
