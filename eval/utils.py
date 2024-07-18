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

def get_environment_states(args, env):
    possible_feedback = []
    environment_states = []

    drawer_states = ['open', 'closed']
    slider_states = ['left', 'right']
    lightbulb_states = [0, 1]
    led_states = [0, 1]
    red_block_states = ['table']

    train_combinations, eval_combinations = env.get_object_combinations()

    if args.num_objs == 3:
        for drawer_state in drawer_states:
            for slider_state in slider_states:
                for lightbulb_state in lightbulb_states:
                    environment_states.append({'drawer': drawer_state, 'slider': slider_state, 'lightbulb': lightbulb_state, 'led': 0,
                                                'red_block': 'table', 'pink_block': 'slider_right', 'blue_block': 'slider_left'})
    elif args.num_objs == 5:
        for drawer_state in drawer_states:
            for slider_state in slider_states:
                for lightbulb_state in lightbulb_states:
                    for led_state in led_states:
                        for red_block_state in red_block_states:
                            environment_states.append({'drawer': drawer_state, 'slider': slider_state, 'lightbulb': lightbulb_state, 'led': led_state,
                                                'red_block': red_block_state, 'pink_block': 'slider_right', 'blue_block': 'slider_left'})
    elif args.num_objects == 7:
        raise NotImplementedError('7 objects not supported yet')
    
    if args.eval_performance:
        combinations = eval_combinations
        assert args.num_objs != 3, "Evaluation for 3 objects not supported"
    else:
        combinations = train_combinations
        
    return combinations, environment_states
       

def eval(args, base_env, model):
    num_steps = base_env.trial_steps * base_env.num_trials
    # Get all possible environment states
    combinations, environment_states = get_environment_states(args, base_env)
    total_reward = 0
    total_steps = 0
    stats = {}
    for combination in combinations:
        feedback_reward = 0
        feedback_steps = 0
        for state in environment_states:
            for reset in range(args.num_resets):
                #base_env.base_env.seed(args.seed + reset)
                print("################New Episode#################")
                random.shuffle(combination)
                reward, episode_length= rollout_eval(args, combination, state, base_env, model, num_steps, args.feedback)
                feedback_reward += reward
                feedback_steps += episode_length 
                total_reward += reward
                total_steps += episode_length

        print(f"Feedback state: {combination}, Reward: {feedback_reward}, Steps: {feedback_steps / (len(environment_states) * args.num_resets)}")
        feedback_state_str = ' '.join(combination)
        stats[feedback_state_str] = {'total_reward': feedback_reward, 'average_reward': feedback_reward / (len(environment_states) * args.num_resets), \
                                     'episode_length': feedback_steps / (len(environment_states) * args.num_resets)}
    
    stats['overall_reward'] = {'total_reward': total_reward, 'average_reward': total_reward / (len(combinations) * len(environment_states) * args.num_resets), \
                               'episode_length': total_steps / (len(combinations) * len(environment_states) * args.num_resets)}

    return stats 