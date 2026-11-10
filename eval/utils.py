import cv2
import itertools
import numpy as np
from pathlib import Path
import random
import time

from eval.recording import (
    action_summary,
    append_jsonl,
    failure_category,
    latency_summary,
    template_descriptor,
    utc_now,
)


def _video_path(args, label, index, reset):
    if not args.record_video:
        return None

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or Path(args.model_checkpoint).stem
    safe_run_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
    safe_label = "_".join(str(item) for item in label)
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_label)
    return video_dir / f"{safe_run_name}__{index:03d}_{safe_label}__reset_{reset:02d}.mp4"


def _open_video_writer(obs, video_path, fps):
    frame = obs["rgb_obs"]["rgb_static"]
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 video writer for {video_path}")
    return writer


def _write_video_frame(writer, obs, status):
    frame = obs["rgb_obs"]["rgb_static"].copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(
        frame,
        status[:70],
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    writer.write(frame)


def _target_tasks(meta_task):
    if isinstance(meta_task, (list, tuple, set)):
        return [str(task) for task in meta_task]
    return [str(meta_task)] if meta_task is not None else []


def _completed_tasks(env):
    if hasattr(env, "task_success_dict"):
        return sorted(
            task for task, completed in env.task_success_dict.items() if completed
        )
    if hasattr(env, "instruction_success_dict"):
        return sorted(str(task) for task in env.instruction_success_dict)
    if hasattr(env, "subtask_success_dict"):
        return sorted(
            task for task, completed in env.subtask_success_dict.items() if completed
        )
    return []


def _task_execution_result(
    reward,
    completed_tasks,
    target_tasks,
    trial_steps,
    trial_step_limit,
):
    completed_target_count = len(set(completed_tasks) & set(target_tasks))
    if reward > 0:
        return "full_meta_task_success"
    if completed_target_count > 0:
        return "partial_meta_task_success"
    if completed_tasks:
        return "only_nonrewarding_tasks_completed"
    if trial_steps >= trial_step_limit:
        return "timeout_no_task_completed"
    return "no_task_completed"


def rollout_eval(
    args,
    feedback_state,
    environment_state,
    env,
    model,
    steps,
    feedback_type='train_feedback',
    meta_task=None,
    video_path=None,
    record_context=None,
):
    model.reset()
    obs, feedback, _, _ = env.episode_reset(feedback_state, environment_state, train=False if args.eval else True, meta_task=meta_task)
    train_feedback = feedback[feedback_type]
    print(train_feedback)
    record_context = dict(record_context or {})
    record_context.setdefault("feedback_state", list(feedback_state))
    record_context.setdefault("meta_task", meta_task)
    record_context.setdefault("initial_environment_state", dict(environment_state))
    if hasattr(env, "sampled_tasks"):
        record_context.setdefault("candidate_tasks", list(env.sampled_tasks))
    elif hasattr(env, "sampled_subtasks"):
        record_context.setdefault("candidate_tasks", list(env.sampled_subtasks))
    target_tasks = _target_tasks(meta_task)
    episode_started_at_utc = utc_now()
    episode_start_time = time.perf_counter()
    initial_robot_obs = np.asarray(obs.get("robot_obs", []), dtype=np.float32).tolist()
    current_trial = 0
    total_reward = 0
    episode_length = 0
    first_reward_trial = None
    trial_start_step = 0
    trial_start_time = time.perf_counter()
    trial_input_feedback = train_feedback
    trial_template = template_descriptor(env, feedback_type, "initial")
    trial_actions = []
    trial_latencies = []
    trial_step_rewards = []
    trial_feedback_events = [
        {
            "step_in_trial": 0,
            "kind": "trial_start",
            "feedback": train_feedback,
            **trial_template,
        }
    ]
    episode_actions = []
    episode_latencies = []
    trial_records = []
    episode_completed_tasks = []
    previous_completed_tasks = set()
    trial_completed_task_events = []
    writer = _open_video_writer(obs, video_path, args.video_fps) if video_path is not None else None
    try:
        if writer is not None:
            _write_video_frame(writer, obs, f"trial 1 | {train_feedback}")

        for i in range(steps):
            episode_length += 1
            inference_start = time.perf_counter()
            action = model.step(obs, train_feedback)
            inference_latency = time.perf_counter() - inference_start
            action_values = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
            trial_actions.append(action_values)
            episode_actions.append(action_values)
            trial_latencies.append(inference_latency)
            episode_latencies.append(inference_latency)
            obs, reward, feedback, trial, trial_done, step_info = env.step(action)
            trial_step_rewards.append(float(reward))
            current_completed_tasks = set(_completed_tasks(env))
            newly_completed_tasks = sorted(current_completed_tasks - previous_completed_tasks)
            for completed_task in newly_completed_tasks:
                trial_completed_task_events.append(
                    {
                        "step_in_trial": episode_length - trial_start_step,
                        "task": completed_task,
                        "is_target_subtask": completed_task in target_tasks,
                    }
                )
            previous_completed_tasks = current_completed_tasks

            if writer is not None:
                _write_video_frame(writer, obs, f"trial {current_trial + 1} step {i + 1} reward {reward}")

            if feedback is not None:
                train_feedback = feedback[feedback_type]
                print(train_feedback)
                trial_feedback_events.append(
                    {
                        "step_in_trial": episode_length - trial_start_step,
                        "kind": "intra_trial",
                        "feedback": train_feedback,
                    }
                )

            if trial_done:
                print('trial done')
                trial_number = current_trial + 1
                completed_tasks = _completed_tasks(env)
                for completed_task in completed_tasks:
                    if completed_task not in episode_completed_tasks:
                        episode_completed_tasks.append(completed_task)
                target_subtasks_completed = [
                    task for task in target_tasks if task in completed_tasks
                ]
                full_meta_task_success = bool(
                    target_tasks and len(target_subtasks_completed) == len(target_tasks)
                )
                trial_step_count = episode_length - trial_start_step
                trial_record = {
                    "record_type": "trial",
                    "run_name": args.run_name,
                    "protocol_label": args.protocol_label,
                    "evaluation_seed": args.seed,
                    "evaluation_split": "legacy_eval" if args.eval else "legacy_train",
                    "feedback_type": feedback_type,
                    "checkpoint": str(Path(args.model_checkpoint).resolve()),
                    **record_context,
                    "episode_started_at_utc": episode_started_at_utc,
                    "trial_index": trial_number,
                    "trial_steps": trial_step_count,
                    "trial_wall_seconds": time.perf_counter() - trial_start_time,
                    "input_feedback": trial_input_feedback,
                    "input_feedback_template": trial_template,
                    "feedback_events": trial_feedback_events,
                    "reward": float(reward),
                    "success": bool(reward > 0),
                    "completed_tasks": completed_tasks,
                    "completed_task_events": trial_completed_task_events,
                    "target_tasks": target_tasks,
                    "target_subtasks_completed": target_subtasks_completed,
                    "target_subtask_count": len(target_subtasks_completed),
                    "full_meta_task_success": full_meta_task_success,
                    "partial_meta_task_success": bool(
                        target_subtasks_completed and not full_meta_task_success
                    ),
                    "task_execution_result": _task_execution_result(
                        reward,
                        completed_tasks,
                        target_tasks,
                        trial_step_count,
                        env.trial_steps,
                    ),
                    "cumulative_completed_tasks": list(episode_completed_tasks),
                    "failure_category": failure_category(reward, completed_tasks),
                    "actions": trial_actions,
                    "step_rewards": trial_step_rewards,
                    "action_summary": action_summary(trial_actions),
                    "inference_latency": latency_summary(trial_latencies),
                    "step_info": step_info,
                }
                append_jsonl(args.trial_log_path, trial_record)
                trial_records.append(trial_record)
                current_trial += 1
                total_reward += reward
                if reward > 0 and first_reward_trial is None:
                    first_reward_trial = trial_number
                if current_trial >= env.num_trials:
                    break
                obs, feedback, _, _ = env.trial_reset()
                train_feedback = feedback[feedback_type]
                print(train_feedback)
                template_stage = "reward" if reward > 0 else "failure"
                trial_start_step = episode_length
                trial_start_time = time.perf_counter()
                trial_input_feedback = train_feedback
                trial_template = template_descriptor(env, feedback_type, template_stage)
                trial_actions = []
                trial_latencies = []
                trial_step_rewards = []
                trial_feedback_events = [
                    {
                        "step_in_trial": 0,
                        "kind": "trial_start",
                        "feedback": train_feedback,
                        **trial_template,
                    }
                ]
                previous_completed_tasks = set()
                trial_completed_task_events = []
                if writer is not None:
                    _write_video_frame(writer, obs, f"trial {current_trial + 1} | {train_feedback}")

            if args.render:
                rgb = env.render(mode="rgb_array")[:,:,::-1]
                cv2.imshow('test',rgb)
                cv2.waitKey(1)
    finally:
        if writer is not None:
            writer.release()
            print(f"Video saved: {video_path.resolve()}")

    episode_record = {
        "record_type": "episode",
        "run_name": args.run_name,
        "protocol_label": args.protocol_label,
        "evaluation_seed": args.seed,
        "evaluation_split": "legacy_eval" if args.eval else "legacy_train",
        "feedback_type": feedback_type,
        "checkpoint": str(Path(args.model_checkpoint).resolve()),
        **record_context,
        "episode_started_at_utc": episode_started_at_utc,
        "episode_completed_at_utc": utc_now(),
        "episode_wall_seconds": time.perf_counter() - episode_start_time,
        "initial_robot_obs": initial_robot_obs,
        "reward": float(total_reward),
        "reward_found": bool(total_reward > 0),
        "first_reward_trial": first_reward_trial,
        "target_tasks": target_tasks,
        "cumulative_completed_tasks": list(episode_completed_tasks),
        "target_subtasks_ever_completed": [
            task for task in target_tasks if task in episode_completed_tasks
        ],
        "full_meta_task_success_trials": sum(
            record["full_meta_task_success"] for record in trial_records
        ),
        "partial_meta_task_success_trials": sum(
            record["partial_meta_task_success"] for record in trial_records
        ),
        "episode_steps": episode_length,
        "trials_completed": current_trial,
        "trial_summaries": [
            {
                "trial_index": record["trial_index"],
                "trial_steps": record["trial_steps"],
                "reward": record["reward"],
                "success": record["success"],
                "completed_tasks": record["completed_tasks"],
                "target_subtasks_completed": record["target_subtasks_completed"],
                "target_subtask_count": record["target_subtask_count"],
                "full_meta_task_success": record["full_meta_task_success"],
                "partial_meta_task_success": record["partial_meta_task_success"],
                "task_execution_result": record["task_execution_result"],
                "failure_category": record["failure_category"],
            }
            for record in trial_records
        ],
        "action_summary": action_summary(episode_actions),
        "inference_latency": latency_summary(episode_latencies),
        "video_path": str(video_path.resolve()) if video_path is not None else None,
    }
    append_jsonl(args.episode_log_path, episode_record)

    return total_reward, episode_length

def rollout_eval_llm(args, feedback_state, environment_state, env, agent, steps, meta_task=None, video_path=None):
    agent.reset()
    obs, feedback, environment_description, _ = env.episode_reset(feedback_state, environment_state, train=False if args.eval else True, meta_task=meta_task)
    env_feedback = feedback['lm_feedback']
    total_reward = 0
    current_trial = 0
    episode_length = 0
    instruction_step = True
    writer = _open_video_writer(obs, video_path, args.video_fps) if video_path is not None else None
    try:
        if writer is not None:
            _write_video_frame(writer, obs, "trial 1 | reasoning agent")

        for i in range(steps):
            episode_length += 1
            action, instruction, _ = agent.step(obs, env_feedback, use_reasoning=True, env_description=environment_description,
                                                      instruction_step=instruction_step)

            obs, reward, feedback_new, trial, trial_done, step_info = env.step(action, instruction)
            if writer is not None:
                _write_video_frame(writer, obs, f"trial {current_trial + 1} step {i + 1} reward {reward}")

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
                if writer is not None:
                    _write_video_frame(writer, obs, f"trial {current_trial + 1} | reasoning agent")

            if args.render:
                rgb = env.render(mode="rgb_array")[:,:,::-1]
                cv2.imshow('test',rgb)
                cv2.waitKey(1)
    finally:
        if writer is not None:
            writer.release()
            print(f"Video saved: {video_path.resolve()}")
    
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
    
    return environment_state, meta_task, list(sampled_tasks), idx

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
    if args.max_combinations is not None:
        tasks = tasks[:args.max_combinations]

    total_reward = 0
    total_steps = 0
    total_episode_reward_found = 0
    stats = {}
    for task_index, task in enumerate(tasks):
        task_reward = 0
        task_steps = 0
        task_episode_reward_found = 0
        for reset in range(args.num_resets):
            episode_index = task_index * args.num_resets + reset
            expected_episode_count = len(tasks) * args.num_resets
            print(
                "################New Combined Episode "
                f"{episode_index + 1}/{expected_episode_count}#################"
            )
            environment_state, feedback_state = get_environment_states_for_task(args, task, base_env)
            print(environment_state)
            video_path = _video_path(args, task, task_index, reset)
            record_context = {
                "episode_index": episode_index,
                "expected_episode_count": expected_episode_count,
                "task_pair_index": task_index,
                "reset_index": reset,
                "simulator_seed": None,
                "feedback_state": list(feedback_state),
                "meta_task": list(task),
                "canonical_meta_task_pair": sorted(task),
                "initial_environment_state": dict(environment_state),
            }
            if args.eval_llm:
                reward, episode_length = rollout_eval_llm(args, feedback_state, environment_state, base_env, agent, num_steps, task, video_path)
            else:
                reward, episode_length = rollout_eval(
                    args,
                    feedback_state,
                    environment_state,
                    base_env,
                    agent,
                    num_steps,
                    args.feedback,
                    task,
                    video_path,
                    record_context,
                )
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
    if args.max_combinations is not None:
        combinations = combinations[:args.max_combinations]
    total_reward = 0
    total_steps = 0
    total_episode_reward_found = 0
    stats = {}
    for combination_index, combination in enumerate(combinations):
        feedback_reward = 0
        feedback_steps = 0
        feedback_episode_reward_found = 0
        exploration_limits = {i: 0 for i in range(args.num_trials - 1)}
        for reset in range(args.num_resets):
            #base_env.base_env.seed(args.seed + reset)
            episode_index = combination_index * args.num_resets + reset
            expected_episode_count = len(combinations) * args.num_resets
            print(
                "################New Episode "
                f"{episode_index + 1}/{expected_episode_count}#################"
            )
            random.shuffle(combination)
            print(exploration_limits)
            state, meta_task, candidate_tasks, meta_task_index = get_environment_states(
                args,
                base_env,
                combination,
                exploration_limits,
            )
            print(exploration_limits)
            print(state)
            feedback_state = list(combination)
            video_path = _video_path(args, feedback_state, combination_index, reset)
            record_context = {
                "episode_index": episode_index,
                "expected_episode_count": expected_episode_count,
                "combination_index": combination_index,
                "reset_index": reset,
                "simulator_seed": None,
                "feedback_state": feedback_state,
                "candidate_tasks": candidate_tasks,
                "meta_task": meta_task,
                "meta_task_index": meta_task_index,
                "initial_environment_state": dict(state),
                "exploration_limits_after_selection": dict(exploration_limits),
            }
            if args.eval_llm:
                reward, episode_length = rollout_eval_llm(args, feedback_state, state, base_env, agent, num_steps, meta_task, video_path)
            else:
                reward, episode_length = rollout_eval(
                    args,
                    feedback_state,
                    state,
                    base_env,
                    agent,
                    num_steps,
                    args.feedback,
                    meta_task,
                    video_path,
                    record_context,
                )
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
