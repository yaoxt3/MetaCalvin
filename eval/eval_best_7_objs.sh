#!/bin/bash

#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --output=/dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/slurm_out/output/%x.%j.out

cd /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/

source activate calvin_test

python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/7_objs_non_seq_pos_4/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_13000.pth13000_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --num_objs 7 \
    --num_trials 5 \
    --num_resets 15 \
    --run_name NonSeqScratch_7_Objs_13000 \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/results_7_objs.csv \
    --eval \
    --seed 1 \

python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/7_objs_non_seq_pos_4/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_13000.pth13000_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --num_objs 7 \
    --num_trials 5 \
    --num_resets 15 \
    --run_name NonSeqScratch_7_Objs_13000_Alt \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/results_7_objs.csv \
    --eval \
    --seed 1 \
    --feedback alt_feedback

python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0001/7_objs_32_seq/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_32_mpt_dolly_3b_lstm_seq_768_seq_9000.pth9000_iter.pth \
    --decoder_type lstm_seq \
    --window_size 1 \
    --num_objs 7 \
    --num_trials 5 \
    --num_resets 15 \
    --run_name SeqScratch_7_Objs_9000 \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/results_7_objs.csv \
    --eval \
    --seed 1

python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/7_objs_non_seq_pretrained/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_12000.pth12000_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --num_objs 7 \
    --num_trials 5 \
    --num_resets 15 \
    --run_name NonSeqScratchPretrained_7_Objs_12000 \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/results_7_objs.csv \
    --eval \
    --seed 1

python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/7_objs_non_seq_alt/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_9000.pth9000_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --num_objs 7 \
    --num_trials 5 \
    --num_resets 15 \
    --run_name AltFeedbackNonSeqScratch_7_Objs_9000 \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/results_7_objs.csv \
    --eval \
    --feedback alt_feedback \
    --seed 1