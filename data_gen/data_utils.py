import sys
sys.path.append('../')

import numpy as np
import torch
import os
import time
import gc
import numpy as np
from contextlib import contextmanager


from typing import List, Dict, Any, Tuple
from copy import deepcopy
from env.env_utils import *
import cv2


def rollout(env, agent, steps, meta_task_stats, exploration_stats, exploration_stats_limits, total_episodes, 
            use_reasoning=False, eval=False, render=False, annotation=None):
    '''Perform rollout of oracle agent in env. Save episode if successful. Expects actions output by agent to be relative actions'''
    episode_data = []
    last_trial_reward = False
    switched = False
    repeated_task = False
    meta_task_stats_episode = meta_task_stats.copy()
    exploration_stats_episode = exploration_stats.copy()
    robot = deepcopy(env.base_env.robot) # Copy the robot to avoid changing the original robot state for computing absolute actions
    #make sure all meta task are sampled equally
    while True:
        meta_task = random.choice(list(meta_task_stats.keys()))
        if meta_task_stats[meta_task] < round(total_episodes/len(env.train_tasks)):
            meta_task_list = [meta_task]
            environment_state, feedback_state = get_env_states_for_task(exploration_stats_episode, exploration_stats_limits, meta_task_list, env)
            meta_task_stats_episode[meta_task] += 1
            break
    print(environment_state, feedback_state)
    obs, feedback, environment_description, info = env.episode_reset(feedback_state, environment_state, meta_task, train=True)
    env_feedback = feedback['lm_feedback']
    print(feedback['train_feedback'])
    current_trial = 0
    trials_before_reward = 0
    instruction_step = True
    no_task_completed = False

    for i in range(steps):
        action, instruction, output = agent.step(obs, env_feedback, use_reasoning=use_reasoning, env_description=deepcopy(environment_description), annotation=annotation, instruction_step=instruction_step)
        
        # Copy rel and  absolute actions
        action_save = action.copy()
        abs_action = robot.relative_to_absolute(action_save.copy())
        # Record step data
        if not instruction_step: 
            feedback['lm_feedback'] = None
            output = None
            if current_trial != 0:
                environment_description = None
        step_dict = create_episode_dict(obs, action_save, abs_action, feedback, instruction, output, environment_description, current_trial, i)
        episode_data.append(step_dict)

        # Next step
        obs, reward, feedback_new, trial, trial_done, step_info = env.step(action, instruction)
        info = step_info['info'] # Contains robot and scene info
        


        if feedback_new is None:
            instruction_step = False
            env_feedback = None
        else:
            instruction_step = True
            feedback = feedback_new
            env_feedback = feedback['lm_feedback']
            print(env_feedback)

        if trial_done:
            current_trial += 1
            if reward:
                last_trial_reward = True
            else:
                trials_before_reward += 1

                if last_trial_reward:
                    switched = True
                    last_trial_reward = False
                    print('HERE')
                    break

                if current_trial == int(env.num_trials - 1):
                    break
            

            # Check that atleast one task has been completed and no task has been repeated
            if step_info['task_completed'] is False:
                no_task_completed = True
                break
            
            if 'repeated_task' in step_info.keys():
                if step_info['repeated_task']:
                    repeated_task =  True
                    break

            if current_trial >= env.num_trials:
                break
                

            obs, feedback, environment_description, info = env.trial_reset(instruction)
            env_feedback = feedback['lm_feedback']
            print(env_feedback)
            instruction_step = True
            
            # Reset the agent model
            agent.policy_model.reset()
        
        if render:
            rgb = env.render(mode="rgb_array")[:,:,::-1]
            cv2.imshow('test',rgb)
            cv2.waitKey(1)
    
    #Collect tensors and clear them from cuda memory
    gc.collect()
    torch.cuda.empty_cache()  
    print(trials_before_reward)
    if last_trial_reward and switched is False and no_task_completed is False and repeated_task is False:
        #Update statistics
        meta_task_stats[meta_task] += 1
        if meta_task in exploration_stats.keys():
            exploration_stats[meta_task][str(trials_before_reward)] += 1
        else:
            exploration_stats[str(trials_before_reward)] += 1
        yield episode_data
        del episode_data

    else:
        yield None
        del episode_data

def get_env_states_for_task(exploration_stats, exploration_stats_limits, task, env):
    # Sample state
    objects = env.objects
    drawer_states = ['open', 'close']
    slider_states = ['left', 'right']
    lightbulb_states = [0, 1]
    block_states = ['table', 'slider']
    states = drawer_states + slider_states + block_states + ['on', 'off']
    state_dict = {"drawer": drawer_states, "slider": slider_states, "lightbulb": lightbulb_states, "led": lightbulb_states, "red_block": block_states,
                  "pink_block": block_states, "blue_block": block_states}
    if env.num_objs == 7:
        environment_state = {}
    elif env.num_objs == 5:
        environment_state = {'pink_block': 'slider_right', 'blue_block': 'slider_left'}
    elif env.num_objs == 4:
        environment_state = {'pink_block': 'slider_right', 'blue_block': 'slider_left', 'red_block': 'table'}
    else:
        environment_state = {'pink_block': 'slider_right', 'blue_block': 'slider_left', 'red_block': 'table', 'led': 0}
    blocks = ['red_block', 'pink_block', 'blue_block']

    meta_object_list = []
    meta_state_list = []
    # Get meta object and its state from the meta task. Handle slider, which can be both state and object
    for subtask in task:
        for state in states:
            if state in subtask:
                if state == 'slider': 
                    if any(block in subtask for block in blocks):
                        meta_state_list.append(state)
                    else:
                        continue
                else:
                    meta_state_list.append(state)

        for obj in objects:
            if obj in subtask:
                if obj not in meta_state_list:
                    meta_object_list.append(obj)
                else:
                    continue

    
    for meta_object, meta_state in zip(meta_object_list, meta_state_list):
        if meta_object in blocks:
            if meta_state == 'slider':
                meta_state = random.choice(['slider_left', 'slider_right'])
            else:
                block_states.remove(meta_state)

            environment_state[meta_object] = meta_state
            random.shuffle(block_states)
            blocks.remove(meta_object)
            slider_states = (['slider_left', 'slider_right'])
            for block in blocks:
                state = random.choice(block_states)
                if state == 'slider':
                    if 'slider' in meta_state:
                        state = 'slider_right' if meta_state == 'slider_left' else 'slider_left'
                        block_states.remove('slider')
                    else:
                        state = random.choice(slider_states)
                        slider_states.remove(state)
                else:
                    block_states.remove(state)
                environment_state[block] = state
                
            if 'slider' in meta_state:
                environment_state['slider'] = 'right' if meta_state == 'slider_left' else 'left'

        elif meta_object in ["lightbulb", "led"]:
            # Get the oposite of the meta state
            if meta_state == "off":
                environment_state[meta_object] = 1
            else:
                environment_state[meta_object] = 0   
        elif meta_object == "drawer":
            # Get the oposite of the meta state
            environment_state[meta_object] = 'closed' if meta_state == 'open' else 'open'
        else:
            environment_state[meta_object] = 'left' if meta_state == 'right' else 'right'
        
    
    assigned_block_slider = []
    for obj in objects:
        if obj not in environment_state.keys():
            if obj in blocks:
                state = random.choice(block_states)
                if state == 'slider' and len(assigned_block_slider) == 1:
                    block_states.remove(state)
                    state = 'slider_right' if 'slider_left' in assigned_block_slider else 'slider_left'
                elif state == 'slider' and len(assigned_block_slider) == 0:
                    state = random.choice(['slider_left', 'slider_right'])
                    assigned_block_slider.append(state)
                else:
                    block_states.remove(state)
                environment_state[obj] = state
            elif obj == 'drawer':
                environment_state[obj] = random.choice(['closed', 'open'])
            else:  
                environment_state[obj] = random.choice(state_dict[obj])

    # Get exploration stats below limit
    # Feedback for simple env
    if len(task) == 1:
        task = task[0]
        meta_object = meta_object_list[0]
        keys = []
        for key in exploration_stats[task].keys():
            if exploration_stats[task][key] < exploration_stats_limits[task][key]:
                keys.append(key)
        meta_object_idx = random.choice(keys)
        # Sample feedback objects
        if env.num_objs == 5:
            feedback_state = random.sample([obj for obj in objects if obj != meta_object], 2)
            feedback_state.insert(int(meta_object_idx), meta_object)
        elif env.num_objs == 7:
            possible_objects = deepcopy(env.static_objects)
            # Ensure that block hidden by slider can not be used as a task object
            for block in blocks:
                block_state = environment_state[block]
                if 'slider' in block_state:
                    block_slider_position = block_state.split('_')[1]
                    if block_slider_position != environment_state['slider']:
                        possible_objects.append(block)
                else:
                    possible_objects.append(block)
            feedback_state = random.sample([obj for obj in possible_objects if obj != meta_object], 3)
            feedback_state.insert(int(meta_object_idx), meta_object)

        else:
            feedback_state = objects
            idx = feedback_state.index(meta_object)
            feedback_state.pop(idx)
            feedback_state.insert(int(meta_object_idx), meta_object)
    # Feedback for multi env
    else:
        last_idx_list = []
        # Adjust feedback state to reflect the expected order of completion to control specific number of trials before reward
        for key in exploration_stats.keys():
            if exploration_stats[str(key)] < exploration_stats_limits[str(key)]:
                last_idx_list.append(key)
        last_idx = int(random.choice(last_idx_list))
        print(last_idx)
        if last_idx == 0:
            meta_object_one_idx = random.choice([0, 1])
            meta_object_two_idx = 0 if meta_object_one_idx == 1 else 0
        elif last_idx == 1:
            meta_object_one_idx = random.choice([0, 1, 2])
            meta_object_two_idx = random.choice([0, 1]) if meta_object_one_idx == 2 else 2
        else:
            meta_object_one_idx = random.choice([0, 1])
            meta_object_two_idx = 3
        meta_idxs = [meta_object_one_idx, meta_object_two_idx]

        feedback_state = random.sample([obj for obj in objects if obj not in meta_object_list], 4 - len(meta_object_list))
        feedback_state = feedback_state + meta_object_list

        original_positions = [feedback_state.index(obj) for obj in meta_object_list]
        new_positions = [meta_idxs[i] for i in range(len(meta_object_list))]

        adjusted_positions = sorted(zip(original_positions, new_positions), key=lambda x: x[0])

        for original_position, new_position in reversed(adjusted_positions):
            feedback_state.pop(original_position)

        for _, new_position in adjusted_positions:
            obj = meta_object_list.pop(0)
            feedback_state.insert(new_position, obj)

        # Now feedback_state should have the elements in the correct positions
        print(feedback_state)
    
    return environment_state, feedback_state

def collect_data(args, agent, env):
    '''Collect data from env using agent. Save successful episodes.'''
    total_ep_run = 0
    saved_ep = 0
    ep_start_id = 0
    episodes_left = args.data_episodes
    times = []
    meta_task_stats = {}
    exploration_stats = {} # Hardcoded for now to number of different tasks
    if args.multi_env:
        exploration_stats = {'0': 0, '1': 0, '2': 0}
    else:
        for task in env.train_tasks:
            meta_task_stats[str(task)] = 0
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
        ep_start_end_ids = np.load(args.save_dir + "/ep_start_end_ids.npy", allow_pickle=True)
        ep_start_id = ep_start_end_ids[-1][1] + 1
        saved_ep = ep_start_end_ids.shape[0]
        episodes_left = args.data_episodes - saved_ep
        print(f"Found existing episode start id. Starting from episode {saved_ep}. {episodes_left} episodes left.")
        meta_task_stats_npz = np.load(args.save_dir + "/meta_task_stats.npz", allow_pickle=True)
        meta_task_stats = convert_to_dict(meta_task_stats_npz)
        exploration_stats_npz = np.load(args.save_dir + "/exploration_stats.npz", allow_pickle=True)
        exploration_stats = convert_to_dict(exploration_stats_npz, True)

    while saved_ep < args.data_episodes:
        start = time.time()
        agent.reset()
        episode_data = rollout(env, agent, args.rollout_steps, meta_task_stats, exploration_stats, exploration_stats_limits,
                                total_episodes, args.use_reasoning, args.render, args.eval)
        episode_data = next(episode_data)
        print(meta_task_stats)
        print(exploration_stats)
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
    step_dict['train_feedback'] = feedback['train_feedback']
    step_dict['alt_feedback'] = feedback['alt_feedback']
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

            

def merge_datasets(data_dir1, data_dir2):
    '''Merge two data directories into one. Assumes that the data directories are
    created using the save_obs_and_action function.'''
    # Load episode start and end ids
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
    meta_task_stats1 = np.load(data_dir1 + "/meta_task_stats.npz", allow_pickle=True)
    meta_task_stats2 = np.load(data_dir2 + "/meta_task_stats.npz", allow_pickle=True)
    meta_task_stats = {}
    for key in meta_task_stats1.keys():
        meta_task_stats[key] = int(meta_task_stats1[key]) + int(meta_task_stats2[key])
    np.savez(data_dir1 + "/meta_task_stats.npz", **meta_task_stats)

    exploration_stats1 = np.load(data_dir1 + "/exploration_stats.npz", allow_pickle=True)
    exploration_stats2 = np.load(data_dir2 + "/exploration_stats.npz", allow_pickle=True)
    exploration_stats = {}
    for key in exploration_stats1.keys():
        exploration_stats[key] = {}
        for sub_key in exploration_stats1[key].item().keys():
            exploration_stats[key][sub_key] = int(exploration_stats1[key].item()[sub_key]) + int(exploration_stats2[key].item()[sub_key])
    np.savez(data_dir1 + "/exploration_stats.npz", **exploration_stats)

    # Merge episodes
    for i in range(ep_start_end_ids2[0][0], ep_start_end_ids2[-1][1] + 1):
        episode = np.load(data_dir2 + f"/episode_{i - ep_start_end_final_id1 - 1:06}.npz", allow_pickle=True)
        save_path = data_dir1 + f"/episode_{i:06}.npz"
        np.savez(save_path, **episode)

def convert_to_dict(npz_file, nested=False):
    return_dict = {}
    for key in npz_file.keys():
        if nested:
            nested_dict = {}
            for n_key in npz_file[key].item().keys():
                nested_dict[n_key] = int(npz_file[key].item()[n_key])
            return_dict[key] = nested_dict    
        else:
            return_dict[key] = int(npz_file[key])
    return return_dict

def add_alternative_language(data_dir):
    '''Add alternative labels to exisiting datasets.'''
    ep_start_end_ids = np.load(data_dir + "/ep_start_end_ids.npy")
    for i in range(len(ep_start_end_ids)):
        ep_start_id, ep_end_id = ep_start_end_ids[i][0], ep_start_end_ids[i][1]

        first_trial_template = build_template_first_trial()
        further_trial_template = build_template_further_trial()
        reward_template = random.choice(REWARD_TEMPLATES)

        sampled_task_instructions = {key: random.choice(TASK_INSTRUCTIONS[key]) for key in TASK_INSTRUCTIONS.keys()} 


        for j in range(ep_start_id, ep_end_id + 1):
            time_step_data = np.load(data_dir + f"/episode_{j:06}.npz", allow_pickle=True)
            time_step_data = dict(time_step_data)
            feedback = str(time_step_data['feedback'])
            alternative_feedback = get_alternative_feedback(feedback, first_trial_template, further_trial_template, reward_template, sampled_task_instructions)
            time_step_data['alternative_feedback'] = alternative_feedback
            save_path = data_dir + f"/episode_{j:06}.npz"
            np.savez(save_path, **time_step_data)

def get_alternative_feedback(feedback, first_trial_template, further_trial_template, reward_template, sampled_task_instructions):
    if "no reward last trial" in feedback:
        episode_objects = get_episode_objects(feedback)
        episode_objects_str = str(', '.join(episode_objects))
        explored_objects = get_explored_objects(feedback)
        explored_objects_str = str(', '.join(explored_objects))
        alternative_feedback = further_trial_template.format(episode_objects_str, explored_objects_str)

    elif "was rewarding last trial" in feedback:
        episode_objects = get_episode_objects(feedback)
        episode_objects_str = str(', '.join(episode_objects))
        reward_task = get_reward_object(feedback)
        reward_task_key = TASK_INSTRUCTION_FEEDBACK[reward_task]
        reward_task_str = sampled_task_instructions[reward_task_key]
        alternative_feedback = reward_template.format(reward_task_str)
    
    else:
        episode_objects = get_episode_objects(feedback)
        episode_objects_str = str(', '.join(episode_objects))
        alternative_feedback = first_trial_template.format(episode_objects_str)

    return alternative_feedback

def get_episode_objects(test_string):
    episode_objects = []
    episode_objects_start = test_string.find("episode objects: ") + len("episode objects: ")
    episode_objects_end = test_string.find(". explored objects")
    episode_objects_string = test_string[episode_objects_start:episode_objects_end]
    episode_objects = episode_objects_string.split(", ")
    return episode_objects

def get_explored_objects(test_string):
    explored_objects = []
    explored_objects_start = test_string.find("explored objects: ") + len("explored objects: ")
    explored_objects_end = len(test_string)
    explored_objects_string = test_string[explored_objects_start:explored_objects_end]
    explored_objects = explored_objects_string.split(", ")
    return explored_objects

def get_reward_object(reward_string):
    reward_task = ""
    reward_task = reward_string.split(" was rewarding")[0]
    return reward_task


def get_data_statistics(dataset):
    feedback_stats = {}
    ep_start_end_ids = np.load(dataset + "/ep_start_end_ids.npy")
    len_dataset = ep_start_end_ids[-1][1] + 1
    print(len(ep_start_end_ids))
    for i in range(len_dataset):
        episode = np.load(dataset + f"/episode_{i:06}.npz", allow_pickle=True)
        episode = dict(episode)
        print(np.mean(episode['rel_actions'][:6]))
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

def visualize_dataset(dataset):
    ep_start_end_ids = np.load(dataset + "/ep_start_end_ids.npy")
    len_dataset = ep_start_end_ids[-1][1] + 1
    possible_objects = ['red block', 'pink block', 'slider', 'drawer', 'led', 'lightbulb']
    used_objects = []
    for i in range(len_dataset):
        episode = np.load(dataset + f"/episode_{i:06}.npz", allow_pickle=True)
        episode = dict(episode)
        feedback = np.array2string(episode['train_feedback'])
        action = episode['rel_actions']
        rgb = episode['rgb_static']
        print(feedback)
        cv2.imshow('test',rgb[:,:,::-1])
        cv2.waitKey(60)

    


if __name__ == '__main__':
    visualize_dataset('/home/richard/Documents/GitHub/RoboTest/data/test')

    
        
