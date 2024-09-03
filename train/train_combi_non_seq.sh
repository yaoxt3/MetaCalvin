#!/bin/bash

#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:2
#SBATCH --time=0-24:00:00
#SBATCH --output=/dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/slurm_out/output/%x.%j.out

cd /dss/dsshome1/0F/ge26wuh2/workspace/RoboTest/train/

source activate calvin_test

torchrun --nnodes=1 --nproc_per_node=2 --master_port=6039 train_meta_calvin.py  --calvin_dataset /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/training_explore  \
--run_name /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/4_objs_explore_non_seq --rgb_pad 10 --gripper_pad 4 --use_gripper --traj_cons --fusion_mode post \
--window_size 12 --cross_attn_every_n_layers 1 --gradient_accumulation_steps 1 --batch_size_calvin 8 --precision fp32 \
--num_epochs 3 \
--decoder_type lstm --checkpoint_path_foundation /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint/open_flamingo_checkpoint.pt \
--warmup_steps 1000 --learning_rate 1e-5 --save_every_iter 1000