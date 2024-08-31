import cv2
import itertools
import random

def rollout_eval(args, feedback_state, environment_state, env, model, steps, feedback_type='train_feedback', meta_task=None):
    model.reset()
    obs, feedback, _, _ = env.episode_reset(feedback_state, environment_state, train=False if args.eval else True, meta_task=meta_task)
    train_feedback = feedback[feedback_type]
    print(train_feedback)
    current_trial = 0
    total_reward = 0
    episode_length = 0
    for i in range(steps):
        episode_length += 1
        action = model.step(obs, train_feedback)
        obs, reward, feedback, trial, trial_done, _ = env.step(action)
        
        if feedback is not None:
            train_feedback = feedback[feedback_type]
            print(train_feedback)

        if trial_done:
            print('trial done')
            current_trial += 1
            total_reward += reward
            if current_trial >= env.num_trials:
                break
            obs, feedback, _, _ = env.trial_reset()
            train_feedback = feedback[feedback_type]
            print(train_feedback)
            


        if args.render:
            rgb = env.render(mode="rgb_array")[:,:,::-1]
            cv2.imshow('test',rgb)
            cv2.waitKey(1)

    return total_reward, episode_length

def rollout_eval_llm(args, feedback_state, environment_state, env, agent, steps, meta_task=None):
    agent.reset()
    obs, feedback, environment_description, _ = env.episode_reset(feedback_state, environment_state, train=False if args.eval else True, meta_task=meta_task)
    env_feedback = feedback['lm_feedback']
    total_reward = 0
    current_trial = 0
    episode_length = 0
    instruction_step = True
    for i in range(steps):
        episode_length += 1
        action, instruction, _ = agent.step(obs, env_feedback, use_reasoning=True, env_description=environment_description,
                                                  instruction_step=instruction_step)

        # Next step
        obs, reward, feedback_new, trial, trial_done, step_info = env.step(action, instruction)
        
        if feedback_new is None:
            instruction_step = False
            env_feedback = None
        else:
            instruction_step = True
            feedback = feedback_new
            env_feedback = feedback['lm_feedback']

        if trial_done:
            current_trial += 1
            total_reward += reward

            if current_trial >= env.num_trials:
                break
                

            obs, feedback, environment_description, _ = env.trial_reset(instruction)
            env_feedback = feedback['lm_feedback']
            instruction_step = True
            
        
        if args.render:
            rgb = env.render(mode="rgb_array")[:,:,::-1]
            cv2.imshow('test',rgb)
            cv2.waitKey(1)
    
    return total_reward, episode_length


def get_environment_states(args, env, combination, exploration_limits):
    drawer_states = ['open', 'closed']
    slider_states = ['left', 'right']
    lightbulb_states = [0, 1]
    led_states = [0, 1]


    if args.num_objs == 3:
        drawer_state = random.choice(drawer_states)
        slider_state = random.choice(slider_states)
        lightbulb_state = random.choice(lightbulb_states)
        environment_state = {'drawer': drawer_state, 'slider': slider_state, 'lightbulb': lightbulb_state, 'led': 0,
                                    'red_block': 'table', 'pink_block': 'slider_right', 'blue_block': 'slider_left'}
    elif args.num_objs == 5:
        drawer_state = random.choice(drawer_states)
        slider_state = random.choice(slider_states)
        lightbulb_state = random.choice(lightbulb_states)
        led_state = random.choice(led_states)
        environment_state = {'drawer': drawer_state, 'slider': slider_state, 'lightbulb': lightbulb_state, 'led': led_state,
                                    'red_block': 'table', 'pink_block': 'slider_right', 'blue_block': 'slider_left'}
    elif args.num_objs == 7:
        drawer_state = random.choice(drawer_states)
        slider_state = random.choice(slider_states)
        lightbulb_state = random.choice(lightbulb_states)
        led_state = random.choice(led_states)
        block_objects = ['red_block', 'pink_block', 'blue_block']
        block_states = ['slider_left', 'slider_right', 'table']
        environment_state = {}
        
        slider_sampled = False
        for obj in combination:
            if obj in block_objects:
                if slider_sampled:
                    block_state = 'table'
                else:
                    block_state = random.choice(block_states)
                
                if 'slider' in block_state:
                    slider_state = 'left' if 'right' in block_state else 'right'
                    slider_sampled = True
                
                block_objects.remove(obj)
                block_states.remove(block_state)
                environment_state[obj] = block_state

        for block in block_objects:
            block_state = random.choice(block_states)
            block_states.remove(block_state)
            environment_state[block] = block_state

        
        environment_state['drawer'] = drawer_state
        environment_state['lightbulb'] = lightbulb_state
        environment_state['led'] = led_state
        environment_state['slider'] = slider_state
    
    # Generate Feedback
    limit = args.num_resets // (args.num_trials -1 ) + 1
    idxs = []
    for key in exploration_limits.keys():
            if exploration_limits[key] < limit:
                idxs.append(key)
    
    # Sample position for meta-object in feedback
    idx = random.choice(idxs)
    exploration_limits[idx] += 1
    random.shuffle(combination)
    sampled_tasks = env.state_to_tasks(environment_state, combination)
    print(sampled_tasks)
    meta_task = sampled_tasks[idx]
    print(environment_state, meta_task)
    
    return environment_state, meta_task

def get_environment_states_for_task(args, task, env):
    drawer_states = ['open', 'close']
    slider_states = ['left', 'right']
    lamp_states = ['on', 'off']
    states = drawer_states + slider_states + lamp_states
    environment_state = {'blue_block': 'slider_left', 'pink_block': 'slider_right', 'red_block': 'table'}
    
    # Sample the environment state
    meta_objs = []
    for subtask in task:
        for obj in env.objects:
            if obj in subtask:
                meta_obj = obj
        meta_objs.append(meta_obj)
        for state in states:
            if state in subtask:
                meta_state = state
        
        if meta_obj == 'drawer':
            state = 'closed' if meta_state == 'open' else 'open'
        elif meta_obj == 'slider':
            state = 'left' if meta_state == 'right' else 'right'
        elif meta_obj == 'lightbulb' or meta_obj == 'led':
            state = 1 if meta_state == 'off' else 0
        else:
            state = 'table'
        
        environment_state[meta_obj] = state
    
    for obj in env.objects:
        if obj not in environment_state:
            if obj == 'led':
                environment_state[obj] = random.choice([0, 1])
            elif obj == 'lightbulb':
                environment_state[obj] = random.choice([0, 1])
            elif obj == 'slider':
                environment_state[obj] = random.choice(['left', 'right'])
            elif obj == 'drawer':
                environment_state[obj] = random.choice(['open', 'closed'])
            else:
                environment_state[obj] = 'table'
            
    # Sample feedback state
    feedback_state = random.sample([obj for obj in env.objects if obj not in meta_objs], 2)
    feedback_state = feedback_state + meta_objs
    random.shuffle(feedback_state)

    return environment_state, feedback_state




def eval_combi_explore(args, base_env, agent):
    num_steps = base_env.trial_steps * base_env.num_trials

    train_tasks, eval_tasks = base_env.train_tasks, base_env.eval_tasks
    if args.eval:
        tasks = eval_tasks
    else:
        tasks = train_tasks

    total_reward = 0
    total_steps = 0
    total_episode_reward_found = 0
    stats = {}
    for task in tasks:
        task_reward = 0
        task_steps = 0
        task_episode_reward_found = 0
        for reset in range(args.num_resets):
            print("################New Episode#################")
            environment_state, feedback_state = get_environment_states_for_task(args, task, base_env)
            print(environment_state)
            if args.eval_llm:
                reward, episode_length = rollout_eval_llm(args, feedback_state, environment_state, base_env, agent, num_steps, task)
            else:
                reward, episode_length = rollout_eval(args, feedback_state, environment_state, base_env, agent, num_steps, args.feedback, task)
            task_reward += reward
            task_steps += episode_length
            task_episode_reward_found += 1 if reward > 0 else 0
            total_reward += reward
            total_steps += episode_length
            total_episode_reward_found += 1 if reward > 0 else 0
        
        task_str = ' '.join(task)
        stats[task_str] = {'total_reward': task_reward, 'average_reward': task_reward / args.num_resets, \
                           'ratio_reward_found': task_episode_reward_found / args.num_resets, 'episode_length': task_steps / args.num_resets}
        
        print(f"Task: {task_str}, Reward: {task_reward}, Ratio Reward Found: {task_episode_reward_found / args.num_resets}, Steps: {task_steps / args.num_resets}")
    
    stats['overall_reward'] = {'total_reward': total_reward, 'average_reward': total_reward / (len(tasks) * args.num_resets), \
                                'ratio_reward_found': total_episode_reward_found / (len(tasks) * args.num_resets), 'episode_length': total_steps / (len(tasks) * args.num_resets)}
    
    return stats


        
       

def eval_explore(args, base_env, agent):
    num_steps = base_env.trial_steps * base_env.num_trials
    # Get all possible environment states
    train_combinations, eval_combinations = base_env.get_object_combinations()
    if args.eval:
        combinations = eval_combinations
    else:
        combinations = train_combinations
    total_reward = 0
    total_steps = 0
    total_episode_reward_found = 0
    stats = {}
    for combination in combinations:
        feedback_reward = 0
        feedback_steps = 0
        feedback_episode_reward_found = 0
        exploration_limits = {i: 0 for i in range(args.num_trials - 1)}
        for reset in range(args.num_resets):
            #base_env.base_env.seed(args.seed + reset)
            print("################New Episode#################")
            random.shuffle(combination)
            print(exploration_limits)
            state, meta_task = get_environment_states(args, base_env, combination, exploration_limits)
            print(exploration_limits)
            print(state)
            if args.eval_llm:
                reward, episode_length = rollout_eval_llm(args, combination, state, base_env, agent, num_steps, meta_task)
            else:
                reward, episode_length= rollout_eval(args, combination, state, base_env, agent, num_steps, args.feedback, meta_task)
            feedback_reward += reward
            feedback_steps += episode_length
            feedback_episode_reward_found += 1 if reward > 0 else 0
            total_reward += reward
            total_steps += episode_length
            total_episode_reward_found += 1 if reward > 0 else 0

        print(f"Feedback state: {combination}, Reward: {feedback_reward}, Ratio Reward Found: {feedback_episode_reward_found / args.num_resets}, \
                Steps: {feedback_steps /  args.num_resets}")
        feedback_state_str = ' '.join(combination)
        stats[feedback_state_str] = {'total_reward': feedback_reward, 'average_reward': feedback_reward / args.num_resets, \
                                     'ratio_reward_found': feedback_episode_reward_found / args.num_resets, 'episode_length': feedback_steps / args.num_resets}
    
    stats['overall_reward'] = {'total_reward': total_reward, 'average_reward': total_reward / ( len(combinations) * args.num_resets), \
                               'ratio_reward_found':  total_episode_reward_found / (len(combinations) * args.num_resets), 'episode_length': total_steps / (len(combinations) * args.num_resets)}

    return stats 