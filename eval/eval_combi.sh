#!/bin/bash

#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --output=/dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/slurm_out/output/%x.%j.out

cd /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/

source activate calvin_test


python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0001/4_objs_explore_non_seq_36_con/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_36_mpt_dolly_3b_3000.pth3000_iter.pth \
    --decoder_type lstm \
    --window_size 36 \
    --trial_steps 500 \
    --num_resets 25 \
    --run_name NonSeq36_3000_eval \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/combi_results.csv \
    --eval \
    --combi_explore



python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0001/4_objs_explore_non_seq_72_con/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_72_mpt_dolly_3b_8000.pth8000_iter.pth \
    --decoder_type lstm \
    --window_size 72 \
    --trial_steps 500 \
    --num_resets 25 \
    --run_name NonSeq72_8000_eval \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/combi_results.csv \
    --eval \
    --combi_explore


python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/4_objs_explore_non_seq_pretrained/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4000.pth4000_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --trial_steps 500 \
    --num_resets 25 \
    --run_name NonSeqPretrained_4000_eval \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/combi_results.csv \
    --eval \
    --combi_explore

python -u evaluate.py \
        --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/4_objs_explore_non_seq_alt/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_14000.pth14000_iter.pth \
        --decoder_type lstm \
        --window_size 12 \
        --trial_steps 500 \
        --num_resets 10 \
        --run_name NonSeqAlt12_14000_eval \
        --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/combi_results.csv \
        --eval \
        --combi_explore \
        --feedback alt_feedback


python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/4_objs_explore_non_seq_alt/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_14000.pth14000_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --trial_steps 500 \
    --num_resets 25 \
    --run_name NonSeqAlt12_14000_eval \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/combi_results.csv \
    --eval \
    --combi_explore \
    --feedback alt_feedback
