# Installation Guide and Examples for Running Data Creation, Training and Evaluation
This guide is mainly for the current datasets stored on the LRZ account and the already downloaded foundation models.
Branch cleanup is most up to date and should work the best!

## Installation
To create the venv meta-calvin run:
```
conda env create -f environment.yml
```

The required models are already downloaded on the LRZ account in the huggingface hub and the RoboCheckpoint folder\
**/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint**

If you want to install all required models for training from scratch do the following:

### 1)
Download the released [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) models:

|# params|Language model|Vision encoder|Xattn interval*|COCO 4-shot CIDEr|VQAv2 4-shot Accuracy|Avg Len|Weights|
|------------|--------------|--------------|----------|-----------|-------|-----|----|
|3B| anas-awadalla/mpt-1b-redpajama-200b-dolly | openai CLIP ViT-L/14 | 1 | 82.7 | 45.7 | 4.09 | [Link](https://huggingface.co/openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct)|

### 2)
Download the RoboFlamingo [checkpoint](https://huggingface.co/roboflamingo/RoboFlamingo/blob/main/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth)

All other required Hugginface models should install automatically if they are not already in the local Huggingface Hub.
I case of manual installation they can be found here:\
Gemma-2-27b [huggingface](https://huggingface.co/google/gemma-2-27b-it)\
CLIP Vit-L/14 [huggingface](https://huggingface.co/openai/clip-vit-large-patch14)\
MPT [huggingface](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b-dolly)\


To instantiate the base RoboFlamingo model:
```
from models.factory import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="PATH/TO/LLM/DIR",
    tokenizer_path="PATH/TO/LLM/DIR",
    cross_attn_every_n_layers=1,
    decoder_type='lstm',
)
```

## Data Creation
Example script for dataset generation of 7 objs Explore dataset. Assumes the CALVIN repo to be present in Folder and the current paths of the LRZ account
```
python -u create_data.py
    --checkpoint_path 
    --use_reasoning 
    --provide_feedback 
    --rollout_steps 1200 
    --num_trials 5  
    --data_episodes 1120
    --save_dir /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/training_7_example_data # Path to store dataset
    --report_interval 2 
    --one_trial_limit 15
    --two_trial_limit 30 
    --zero_trial_limit 5 
    --num_objs 7 
    --seed 77 
    --reset_robot_pos
```
The actual batch scripts used to create the datasets can be found in the data gen folder

## Training
Scripts assume the current paths to the dataset on the LRZ account!

To train the model on e.g. the 7 objs Explore dataset on a single GPU run:

```
torchrun --nnodes=1 --nproc_per_node=1 --master_port=6043 train_meta_calvin.py  
    --calvin_dataset /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/training_7_objs_pos_4
    --run_name /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/7_objs_non_seq_example 
    --rgb_pad 10 
    --gripper_pad 4 
    --use_gripper 
    --traj_cons 
    --fusion_mode post
    --window_size 12 
    --cross_attn_every_n_layers 1 
    --gradient_accumulation_steps 1 
    --batch_size_calvin 12 
    --precision fp32
    --num_epochs 3
    --decoder_type lstm 
    --checkpoint_path_foundation /dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint/open_flamingo_checkpoint.pt
    --warmup_steps 1000 
    --learning_rate 1e-5 
    --save_every_iter 1000
    --dist
```

## Evaluation
Example script of evaluating a trained model on the 7 objs Explore dataset:
```
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
```
Batch scripts for all our evaluation results can be found under eval


#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### OpenAI CLIP
Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)

#### OpenFlamingo
Original: [https://github.com/mlfoundations/open_flamingo](https://github.com/mlfoundations/open_flamingo)
License: [MIT](https://github.com/mlfoundations/open_flamingo/blob/main/LICENSE)

#### RoboFlamingo
Original: [https://github.com/RoboFlamingo/RoboFlamingo](https://github.com/RoboFlamingo/RoboFlamingo?tab=readme-ov-file)
License:  [MIT](https://github.com/RoboFlamingo/RoboFlamingo/blob/main/LICENSE)