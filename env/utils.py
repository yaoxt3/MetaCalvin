from protoenv import ProtoEnv
from multi_env import MultiEnv

def get_env(args, base_env, task_oracle):
    if args.multi_env:
        return MultiEnv(base_env, task_oracle, num_trials=args.num_trials, trial_steps=int(args.rollout_steps/args.num_trials), num_instructions=args.num_instructions,
                    provide_feedback=args.provide_feedback, reset_robot_pos=args.reset_robot_pos, num_objs=args.num_objs, provide_train_feedback=True)
    else:
        return ProtoEnv(base_env, task_oracle, num_trials=args.num_trials, trial_steps=int(args.rollout_steps/args.num_trials),
                    provide_feedback=args.provide_feedback, reset_robot_pos=args.reset_robot_pos, num_objs=args.num_objs, provide_train_feedback=True)