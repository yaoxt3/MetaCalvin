#!/bin/bash

#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=0-28:00:00
#SBATCH --output=/dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/slurm_out/output/%x.%j.out

cd /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/

source activate calvin_test


python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0001/5_objs_12_nonseq/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_9000.pth9000_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --num_objs 5 \
    --num_trials 4 \
    --num_resets 30 \
    --run_name NonSeqScratch_5_Objs_9000 \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/eval_best_5_objs.csv \
    --eval \
    --seed 1 \

python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0001/5_objs_12_nonseq/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_9000.pth9000_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --num_objs 5 \
    --num_trials 4 \
    --num_resets 30 \
    --run_name NonSeqScratch_5_Objs_9000_Alt \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/eval_best_5_objs.csv \
    --eval \
    --seed 1 \
    --feedback alt_feedback

python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0001/5_objs_12_nonseq_alt/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7500.pth7500_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --num_objs 5 \
    --num_trials 4 \
    --num_resets 30 \
    --run_name AltFeedback_NonSeqScratch_5_Objs_7500 \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/eval_best_5_objs.csv \
    --eval \
    --feedback alt_feedback \
    --seed 1

python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0001/5_objs_12_nonseq_pretrained/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7000.pth7000_iter.pth \
    --decoder_type lstm \
    --window_size 12 \
    --num_objs 5 \
    --num_resets 30 \
    --run_name NonSeqPretrained_5_Objs_7000 \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/eval_best_5_objs.csv \
    --eval \
    --seed 1

python -u evaluate.py \
    --model_checkpoint /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0001/5_objs_32_seq/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_32_mpt_dolly_3b_lstm_seq_768_seq_5000.pth5000_iter.pth \
    --decoder_type lstm_seq \
    --window_size 1 \
    --num_objs 5 \
    --num_trials 4 \
    --num_resets 30 \
    --run_name SeqScratch_5_Objs_5000 \
    --path_to_results /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/eval/results/eval_best_5_objs.csv \
    --eval \
    --seed 1