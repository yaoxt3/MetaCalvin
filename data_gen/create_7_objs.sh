#!/bin/bash

#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=0-30:00:00
#SBATCH --output=/dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/slurm_out/output/%j.out

source activate calvin_test
cd /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/data_gen/

python -u create_data.py --use_reasoning --provide_feedback --rollout_steps 1200 --num_trials 5  --data_episodes 1120 \
 --save_dir /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/training_7_objs_pos_4 --report_interval 2 --one_trial_limit 15 \
 --two_trial_limit 30 --zero_trial_limit 5 --num_objs 7 --seed 77 --reset_robot_pos