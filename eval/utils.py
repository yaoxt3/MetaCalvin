import cv2
import itertools
import random

def rollout_eval(args, feedback_state, environment_state, env, model, steps, feedback_type='train_feedback'):
    model.reset()
    obs, feedback, _ = env.episode_reset(feedback_state, environment_state, train=True)
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

        if trial_done:
            print('trial done')
            current_trial += 1
            total_reward += reward
            if current_trial >= env.num_trials:
                break
            obs, feedback, _ = env.trial_reset()
            train_feedback = feedback[feedback_type]
            print(train_feedback)
            


        if args.render:
            rgb = env.render(mode="rgb_array")[:,:,::-1]
            cv2.imshow('test',rgb)
            cv2.waitKey(1)

    return total_reward, episode_length

def rollout_eval_llm(args, feedback_state, environment_state, env, agent, steps):
    agent.reset()
    obs, feedback, environment_description, info = env.episode_reset(feedback_state, environment_state, train=True)
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


def get_environment_states(args, combination):
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
    if args.num_objects == 7:
        drawer_state = random.choice(drawer_states)
        slider_state = random.choice(slider_states)
        lightbulb_state = random.choice(lightbulb_states)
        led_state = random.choice(led_states)
        block_objects = ['red_block', 'pink_block', 'blue_block']
        block_states = ['slider_left', 'slider_right', 'table']
        environment_state = {}
        for block in block_objects:
            block_state = random.choice(block_states)
            block_states.remove(block_state)
            environment_state[block] = block_state
            if block in combination:
                if 'slider' in block_state:
                    slider_state = 'left' if 'right' in block_state else 'right'
        
        environment_state['drawer'] = drawer_state
        environment_state['lightbulb'] = lightbulb_state
        environment_state['led'] = led_state
        environment_state['slider'] = slider_state
    
    return environment_state
        
       

def eval(args, base_env, agent):
    num_steps = base_env.trial_steps * base_env.num_trials
    # Get all possible environment states
    train_combinations, eval_combinations = base_env.get_object_combinations(args.num_objects)
    if args.eval:
        combinations = eval_combinations
    else:
        combinations = train_combinations
    total_reward = 0
    total_steps = 0
    stats = {}
    for combination in combinations:
        feedback_reward = 0
        feedback_steps = 0
        for reset in range(args.num_resets):
            #base_env.base_env.seed(args.seed + reset)
            print("################New Episode#################")
            random.shuffle(combination)
            state = get_environment_states(args, combination)
            if args.eval_llm:
                reward, episode_length = rollout_eval_llm(args, combination, state, base_env, agent, num_steps)
            else:
                reward, episode_length= rollout_eval(args, combination, state, base_env, agent, num_steps, args.feedback)
            feedback_reward += reward
            feedback_steps += episode_length 
            total_reward += reward
            total_steps += episode_length

        print(f"Feedback state: {combination}, Reward: {feedback_reward}, Steps: {feedback_steps / (len(combination) * args.num_resets)}")
        feedback_state_str = ' '.join(combination)
        stats[feedback_state_str] = {'total_reward': feedback_reward, 'average_reward': feedback_reward / (len(combination) * args.num_resets), \
                                     'episode_length': feedback_steps / (len(combination) * args.num_resets)}
    
    stats['overall_reward'] = {'total_reward': total_reward, 'average_reward': total_reward / (len(combinations) * len(combination) * args.num_resets), \
                               'episode_length': total_steps / (len(combinations) * len(combination) * args.num_resets)}

    return stats 