#!/bin/bash

#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=0-18:00:00
#SBATCH --output=/dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/slurm_out/output/%j.out

cd /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/data_gen/

source activate calvin_test
python -u create_data.py --use_reasoning --provide_feedback --rollout_steps 1600 --num_trials 4  --data_episodes 300  \
--save_dir /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/training_explore \
--multi_env --report_interval 2 --one_trial_limit 100  --two_trial_limit 100 --zero_trial_limit 100 --reset_robot_pos