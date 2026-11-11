# Installation Guide and Examples for Running Data Creation, Training and Evaluation
This repostory is the implementation of **MetaCALVIN: Feedback-Conditioned Behavioral Distillation for Long-Horizon Manipulation**, which was accepted by IEEE Robotics and Automation Letters (RA-L) Nov. 2026.

The required dataset and checkpoints are avaliable on [MetaCALVIN-dataset](https://huggingface.co/datasets/ele-xiaotong13/metaCalvin_dataset) and [checkpoints](https://huggingface.co/datasets/ele-xiaotong13/meta-checkpoints)

---

## Installation
To create the venv meta-calvin run:
```
conda env create -f environment.yml
```

If you want to install all required models for training from scratch do the following:

### 1)
Clone the [CALVIN](https://github.com/mees/calvin) repo and download the debug dataset into the datasets folder

### 2)
Download the released [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) models:

|# params|Language model|Vision encoder|Xattn interval*|COCO 4-shot CIDEr|VQAv2 4-shot Accuracy|Avg Len|Weights|
|------------|--------------|--------------|----------|-----------|-------|-----|----|
|3B| anas-awadalla/mpt-1b-redpajama-200b-dolly | openai CLIP ViT-L/14 | 1 | 82.7 | 45.7 | 4.09 | [Link](https://huggingface.co/openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct)|

### 3)
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
---

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
    --save_dir /your_path/training_7_example_data # Path to store dataset
    --report_interval 2 
    --one_trial_limit 15
    --two_trial_limit 30 
    --zero_trial_limit 5 
    --num_objs 7 
    --seed 77 
    --reset_robot_pos
```
The actual batch scripts used to create the datasets can be found in the data gen folder

---

## Training
Scripts assume the current paths to the dataset on the specific path!

To train the model on e.g. the 7 objs Explore dataset on a single GPU run:

See ./train/launch/README.md

---

## More training protocol

See ./protocol/README.md

---

## Evaluation
Example script of evaluating a trained model on the 7 objs Explore dataset: 

See ./eval/launch/run_E0-E7_C0-C7_resource_rerun.sh

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

---

