#!/bin/bash

#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=0-16:00:00
#SBATCH --output=/dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/slurm_out/output/%x.%j.out

cd /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/

source activate calvin_test

python evaluate.py --num_objs 7 --trial_steps 300 --eval_llm --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/non_seq_results.csv \
--num_resets 15 --reasoning_model google/gemma-2-2b-it --tokenizer google/gemma-2-2b-it --num_trials 5 \
--model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth \
--run_name gemma-2-2b-it_7_objs --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/results_llm.csv --seed 1 --eval

python evaluate.py --num_objs 7 --trial_steps 300 --eval_llm --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/non_seq_results.csv \
--num_resets 15 --reasoning_model google/gemma-2-9b-it --tokenizer google/gemma-2-9b-it --num_trials 5 \
--model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth \
--run_name gemma-2-9b-it_7_objs --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/results_llm.csv --seed 1 --eval

python evaluate.py --num_objs 7 --trial_steps 300 --eval_llm --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/non_seq_results.csv \
--num_resets 15 --reasoning_model google/gemma-2-27b-it --tokenizer google/gemma-2-27b-it --num_trials 5 \
--model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth \
--run_name gemma-2-27b-it_7_objs --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/results_llm.csv --seed 1 --eval