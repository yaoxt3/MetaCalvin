import cv2
import itertools

def rollout_eval(args, feedback_state, environment_state, env, model, steps):
    model.reset()
    obs, feedback, _ = env.episode_reset(feedback_state, environment_state, train=True)
    train_feedback = feedback['legacy_feedback']
    print(train_feedback)
    current_trial = 0
    total_reward = 0
    for i in range(steps):
        action = model.step(obs, train_feedback)
        obs, reward, feedback, trial, trial_done, _ = env.step(action)
        
        if feedback is not None:
            train_feedback = feedback['train_feedback']

        if trial_done:
            print('trial done')
            current_trial += 1
            total_reward += reward
            if current_trial >= env.num_trials:
                break
            obs, feedback, _ = env.trial_reset()
            train_feedback = feedback['legacy_feedback']
            print(train_feedback)
            


        if args.render:
            rgb = env.render(mode="rgb_array")[:,:,::-1]
            cv2.imshow('test',rgb)
            cv2.waitKey(1)
    
    return total_reward

def get_environment_states(args):
    possible_feedback = []
    environment_states = []
    if args.feedback == 'legacy':
        if args.num_objs == 3:
            objects = ['drawer', 'slider', 'lightbulb']
            # compute all permutations of objects without replacement
            possible_feedback.extend(list(itertools.permutations(objects, 3)))
            possible_feedback = [list(feedback) for feedback in possible_feedback]
            
            drawer_states = ['open', 'closed']
            slider_states = ['left', 'right']
            lightbulb_states = [0, 1]
            for drawer_state in drawer_states:
                for slider_state in slider_states:
                    for lightbulb_state in lightbulb_states:
                        environment_states.append({'drawer': drawer_state, 'slider': slider_state, 'lightbulb': lightbulb_state, 'led': 0,
                                                   'red_block': 'table', 'pink_block': 'slider_right', 'blue_block': 'slider_left'})
        else:
            raise NotImplementedError('Only 3 objects supported for legacy feedback so far')
    else:
        raise NotImplementedError('Only legacy feedback supported so far')
    return possible_feedback, environment_states
       

def eval(args, base_env, model):
    num_steps = base_env.trial_steps * base_env.num_trials
    # Get all possible environment states
    possible_feedback, environment_states = get_environment_states(args)
    total_reward = 0
    stats = {}
    for feedback_state in possible_feedback:
        feedback_reward = 0
        for state in environment_states:
            for reset in range(args.num_resets):
                #base_env.base_env.seed(args.seed + reset)
                print("################New Episode#################")
                reward = rollout_eval(args, feedback_state, state, base_env, model, num_steps)
                feedback_reward += reward
                total_reward += reward

        feedback_state_str = ' '.join(feedback_state)
        stats[feedback_state_str] = {'total_reward': feedback_reward, 'average_reward': feedback_reward / (len(environment_states) * args.num_resets)}
    
    stats['overall_reward'] = {'total_reward': total_reward, 'average_reward': total_reward / (len(possible_feedback) * len(environment_states) * args.num_resets)}

    return stats 