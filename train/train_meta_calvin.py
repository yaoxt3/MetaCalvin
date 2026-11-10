""" Main training script, adapted from RobotFlamingo """
import sys
sys.path.append("../calvin/calvin_models/")
sys.path.append("../calvin/calvin_env/")
sys.path.append("../calvin/calvin_env/tacto")
sys.path.append("../calvin/calvin_models/calvin_agent/")
sys.path.append("../open_flamingo/")
sys.path.append("../models/")
sys.path.append("../")



import argparse
import copy
import glob
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import wandb
from huggingface_hub import hf_hub_download

from torch.nn.parallel import DistributedDataParallel as DDP

from data.data import get_data
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from train.utils import (
    get_checkpoint,
    get_ckpt_name,
    get_ckpt_name_pattern,
    periodic_checkpoint_steps,
    preserve_epoch_checkpoint,
    train_one_epoch,
    train_one_epoch_diff,
)
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from factory import create_model_and_transforms, mpt_dict


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def none_or_str(value):
    if value == 'None':
        return None
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repo_dir, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name):
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def write_run_manifest(args, dataset, val_dataset, requested_learning_rate):
    if args.rank != 0:
        return

    run_dir = Path(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(__file__).resolve().parents[3]
    git_status = _git_output(repo_dir, "status", "--short")
    git_diff = _git_output(repo_dir, "diff", "--binary") or ""
    robottest_dir = Path(__file__).resolve().parents[1]
    code_paths = [
        Path(__file__).resolve(),
        robottest_dir / "train" / "utils.py",
        robottest_dir / "data" / "data.py",
        robottest_dir / "data" / "sampler.py",
        robottest_dir / "data" / "utils.py",
        robottest_dir / "data" / "rotation_conversions.py",
        robottest_dir / "models" / "factory.py",
    ]

    manifest_paths = {
        "train": Path(args.calvin_train_manifest).resolve(),
        "validation": Path(args.calvin_val_manifest).resolve(),
    }
    data_manifests = {}
    for split, path in manifest_paths.items():
        content = json.loads(path.read_text(encoding="utf-8"))
        frozen_copy = run_dir / f"frozen_{split}_manifest.json"
        if not frozen_copy.exists():
            shutil.copyfile(path, frozen_copy)
        group_count = content.get(
            "num_object_combinations", content.get("num_meta_task_pairs")
        )
        if group_count is None:
            raise ValueError(f"Manifest has no recognized group count: {path}")
        data_manifests[split] = {
            "path": str(path),
            "frozen_copy": str(frozen_copy),
            "sha256": sha256_file(path),
            "protocol": content["protocol"],
            "split_seed": content["split_seed"],
            "grouping_unit": content.get("grouping_unit"),
            "num_groups": group_count,
            "num_episodes": content["num_episodes"],
            "num_windows_ws12": content["num_windows_ws12"],
            "source_index_sha256": content["source_index_sha256"],
        }

    batches_per_epoch = dataset.dataloader.num_batches
    total_training_batches = batches_per_epoch * args.num_epochs
    optimizer_steps_per_epoch = (
        batches_per_epoch + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps
    try:
        environment_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        environment_freeze = ""
    environment_path = run_dir / "environment_freeze.txt"
    if not environment_path.exists():
        environment_path.write_text(environment_freeze, encoding="utf-8")

    run_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "protocol": args.protocol_name,
        "checkpoint_selection": {
            "split": "validation",
            "metric": "mean_behavior_cloning_loss",
            "mode": "min",
            "frequency": (
                "epoch_end" if args.validate_every_iter == -1
                else f"every_{args.validate_every_iter}_batches_and_epoch_end"
            ),
            "final_test_used_for_selection": False,
        },
        "checkpoint_schedule": {
            "unit": "completed_global_batch_one_indexed",
            "base_every_batches": args.save_every_iter,
            "dense_start_batch": args.dense_save_start,
            "dense_end_batch": args.dense_save_end,
            "dense_every_batches": args.dense_save_every_iter,
            "planned_periodic_checkpoints": periodic_checkpoint_steps(
                args, total_training_batches
            ),
            "contains_optimizer_and_scheduler_state": True,
            "retain_each_epoch_end": True,
            "epoch_end_filename_pattern": "epoch_{epoch_number:02d}_checkpoint.pth",
            "latest_and_epoch_file_share_hard_link_storage": True,
        },
        "data_manifests": data_manifests,
        "dataset_path": str(Path(args.calvin_dataset).resolve()),
        "foundation_checkpoint": {
            "path": str(Path(args.checkpoint_path_foundation).resolve()),
            "sha256": sha256_file(args.checkpoint_path_foundation),
        },
        "training_budget": {
            "epochs": args.num_epochs,
            "batches_per_epoch": batches_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": optimizer_steps_per_epoch * args.num_epochs,
            "batch_size_per_gpu": args.batch_size_calvin,
            "world_size": args.world_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.batch_size_calvin
                * args.world_size
                * args.gradient_accumulation_steps
            ),
            "train_windows_used_per_epoch": dataset.dataloader.num_samples,
            "validation_windows": val_dataset.dataloader.num_samples,
            "sampling_mode": "episodic" if args.seq_training else "non_episodic",
        },
        "optimization": {
            "optimizer": "AdamW",
            "requested_learning_rate": requested_learning_rate,
            "effective_optimizer_learning_rate": args.learning_rate,
            "learning_rate_scaling_rule": "requested_lr * batch_size_per_gpu / 6",
            "lr_scheduler": args.lr_scheduler,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "loss": "Huber(continuous_actions) + 0.01 * BCE(gripper)",
        },
        "reproducibility": {
            "training_seed": args.seed,
            "command": [sys.executable, *sys.argv],
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "package_versions": {
                name: _package_version(name)
                for name in (
                    "numpy",
                    "transformers",
                    "tokenizers",
                    "open-clip-torch",
                    "timm",
                    "wandb",
                )
            },
            "environment_freeze": {
                "path": str(environment_path),
                "sha256": sha256_file(environment_path),
            },
            "git_commit": _git_output(repo_dir, "rev-parse", "HEAD"),
            "git_status": git_status,
            "git_diff_sha256": hashlib.sha256(git_diff.encode("utf-8")).hexdigest(),
            "source_file_sha256": {
                str(path.relative_to(repo_dir)): sha256_file(path)
                for path in code_paths
            },
        },
        "arguments": vars(args),
    }
    output = run_dir / "run_manifest.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        immutable_checks = {
            "experiment_id": (existing.get("experiment_id"), run_manifest["experiment_id"]),
            "protocol": (existing.get("protocol"), run_manifest["protocol"]),
            "training_seed": (
                existing.get("reproducibility", {}).get("training_seed"),
                run_manifest["reproducibility"]["training_seed"],
            ),
            "effective_batch_size": (
                existing.get("training_budget", {}).get("effective_batch_size"),
                run_manifest["training_budget"]["effective_batch_size"],
            ),
            "train_manifest_sha256": (
                existing.get("data_manifests", {}).get("train", {}).get("sha256"),
                run_manifest["data_manifests"]["train"]["sha256"],
            ),
            "validation_manifest_sha256": (
                existing.get("data_manifests", {}).get("validation", {}).get("sha256"),
                run_manifest["data_manifests"]["validation"]["sha256"],
            ),
            "foundation_sha256": (
                existing.get("foundation_checkpoint", {}).get("sha256"),
                run_manifest["foundation_checkpoint"]["sha256"],
            ),
        }
        mismatches = {
            key: values for key, values in immutable_checks.items() if values[0] != values[1]
        }
        if mismatches:
            raise ValueError(
                f"Refusing to resume with a changed frozen protocol: {mismatches}"
            )
        with (run_dir / "resume_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "resume_from_checkpoint": args.resume_from_checkpoint,
                        "command": [sys.executable, *sys.argv],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        print(f"Verified frozen run manifest for resume: {output}")
        return
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"Wrote rebuttal run manifest: {output}")

@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="anas-awadalla/mpt-1b-redpajama-200b-dolly", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default="f0a13e41fcee2217cd701219ffa1eaef7fe955ea",
        help="Pinned Hugging Face revision for the MPT tokenizer/model and remote code",
    )
    parser.add_argument(
        "--checkpoint_path_foundation", 
        type=str, 
        default='/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/RoboCheckpoint/open_flamingo_checkpoint.pt',
        help="Path to pretrained foundation model. Can either be from RoboFlamingo or OpenFlamingo")
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="MetaCalvin",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--stop_after_epochs",
        type=int,
        default=-1,
        help=(
            "Optionally stop cleanly after this many completed epochs while "
            "keeping num_epochs as the frozen total training budget. Use -1 "
            "to run through num_epochs."
        ),
    )
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size_calvin", type=int, default=12)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--resume_batch_offset",
        type=int,
        default=-1,
        help=(
            "Batch offset within an incomplete epoch. Use -1 to infer it from "
            "completed_global_step in the checkpoint."
        ),
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # 1e-4
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument(
        "--calvin_dataset",
        type=str,
        help="path to calvin_dataset",
    )
    parser.add_argument(
        "--calvin_val_dataset",
        type=none_or_str,
        default=None,
        help="path to validation dataset"
    )
    parser.add_argument(
        "--calvin_train_manifest",
        type=none_or_str,
        default=None,
        help="P0 manifest selecting training episodes from calvin_dataset",
    )
    parser.add_argument(
        "--calvin_val_manifest",
        type=none_or_str,
        default=None,
        help="P0 manifest selecting validation episodes from calvin_dataset",
    )
    parser.add_argument("--experiment_id", type=str, default="unregistered")
    parser.add_argument(
        "--protocol_name",
        type=str,
        default="legacy",
        help="Frozen protocol identifier recorded in run_manifest.json",
    )
    parser.add_argument("--validate_every_iter", type=int, default=100)
    parser.add_argument("--loss_multiplier_calvin", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    # hot fix for torch.distributed.launch
    # parser.add_argument("--local-rank", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_calvin", type=int, default=100)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Stop an epoch after this many batches; intended for smoke tests.",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=-1,
        help="Limit validation batches; intended only for smoke tests.",
    )
    parser.add_argument(
        "--skip_checkpoint_saves",
        action="store_true",
        help="Do not write checkpoints; intended only for smoke tests.",
    )
    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument(
        "--freeze_embed",
        default=False,
        action="store_true",
        help="freeze the parameters of embedding layer",
    )
    parser.add_argument(
        "--use_gripper",
        default=False,
        action="store_true",
        help="whether to use gripper image as input",
    )
    parser.add_argument(
        "--use_state",
        default=False,
        action="store_true",
        help="whether to use low-dim state as input",
    )
    parser.add_argument(
        "--action_tokens",
        default=False,
        action="store_true",
        help="use additional action tokens if dataset includes them"
    )
    parser.add_argument(
        "--fusion_mode",
        default="pre",
        type=str,
        help="pre or post to fusion multi vision info",
    )
    parser.add_argument("--hist_window", type=int, default=1)  # input history window size for the model
    # history window size when evaluating, for FC head equals to hist_window, for LSTM head means refresh frequency
    parser.add_argument("--eval_hist_size", type=int, default=-1)
    parser.add_argument(
        "--sep_resampler",
        default=False,
        action="store_true",
        help="whether use separate resamplers for third party and gripper camera",
    )
    parser.add_argument("--train_params", type=int, default=-1)
    parser.add_argument('--rgb_pad', type=int, default=-1)
    parser.add_argument('--gripper_pad', type=int, default=-1)
    parser.add_argument('--n_timesteps', type=int, default=150, help="diffusion time steps")
    parser.add_argument(
        "--predict_epsilon",
        default=False,
        action="store_true",
        help="whether diffusion model should predict epsilon",
    )
    parser.add_argument('--head_type', type=str, default="lstm")
    parser.add_argument(
        "--from_scratch",
        default=False,
        action="store_true",
        help="whether to train the model from scratch",
    )
    parser.add_argument("--n_obs_steps", default=6, type=int)
    parser.add_argument("--diff_horizon", default=32, type=int)
    parser.add_argument(
        "--last_action",
        default=False,
        action="store_true",
        help="whether using last action as input",
    )
    parser.add_argument(
        "--use_hist",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--traj_cons",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--sep_lm_head",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--clip_state",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--unfreeze_vit",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--text_aug",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--residual",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--tcp_rel",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--dif_ws",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--partial_data",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--freeze_sampler",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--fwd_pred",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--fwd_pred_hand",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--no_pretrain",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--no_image_patch",
        default=False,
        action="store_true"
    )
    # Co-Train settings
    parser.add_argument(
        "--cotrain",
        default=False,
        action="store_true"
    )
    parser.add_argument("--batch_size_vl", type=int, default=20)
    parser.add_argument("--vl_task_weights", type=float, default=0.005)

    parser.add_argument("--global_latent", type=int, default=1)
    parser.add_argument("--save_every_iter", type=int, default=-1)
    parser.add_argument("--dense_save_start", type=int, default=-1)
    parser.add_argument("--dense_save_end", type=int, default=-1)
    parser.add_argument("--dense_save_every_iter", type=int, default=-1)
    # For GPT decoder
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--decoder_type", type=str, default='lstm')
    
    parser.add_argument("--min_window_size", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=24)
    parser.add_argument("--llm_name", type=str, default='mpt_dolly_3b')
    parser.add_argument("--pooling", type=str, default='max')
    parser.add_argument("--multi_step_action", type=int, default=1, help="multiple step action prediction")

    parser.add_argument("--seq_training", default=False, action="store_true", help="Load batches from sequences instead of random windows")
    parser.add_argument("--feedback_type", type=str, default="train_feedback", 
                        help="Type of feedback to use for training. train_feedback for constant feedback, alt_feedback for multiple different feedback types ")
    parser.add_argument("--dist", default=True, action="store_false", help="wether to use distributed training")

    args = parser.parse_args()

    if args.save_every_iter == 0 or args.save_every_iter < -1:
        raise ValueError("save_every_iter must be -1 (disabled) or a positive integer")
    dense_values = (
        args.dense_save_start,
        args.dense_save_end,
        args.dense_save_every_iter,
    )
    dense_disabled = dense_values == (-1, -1, -1)
    if not dense_disabled and (
        any(value <= 0 for value in dense_values)
        or args.dense_save_start > args.dense_save_end
    ):
        raise ValueError(
            "dense checkpoint schedule requires positive start/end/interval "
            "with start <= end, or all three values set to -1"
        )
    
    assert args.feedback_type in ["train_feedback", "alt_feedback"]
    if (args.calvin_train_manifest is None) != (args.calvin_val_manifest is None):
        raise ValueError(
            "calvin_train_manifest and calvin_val_manifest must be provided together"
        )
    if args.calvin_train_manifest is not None and args.protocol_name == "legacy":
        raise ValueError("A non-legacy --protocol_name is required with P0 manifests")

    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion":
            args.eval_hist_size = args.n_obs_steps
        else:
            args.n_obs_steps = 0
    if args.tcp_rel:
        args.clip_state = True
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.stop_after_epochs != -1 and not (
        1 <= args.stop_after_epochs <= args.num_epochs
    ):
        raise ValueError(
            "stop_after_epochs must be -1 or between 1 and num_epochs; "
            f"got {args.stop_after_epochs} with num_epochs={args.num_epochs}"
        )

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)

    random_seed(args.seed)
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_gripper=args.use_gripper,
        use_state=args.use_state,
        use_hist=args.use_hist,
        fusion_mode=args.fusion_mode,
        use_local_files=args.offline,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
        window_size=args.eval_hist_size,
        freeze_embed=args.freeze_embed,
        train_params=args.train_params,
        sep_resampler=args.sep_resampler,
        last_action=args.last_action,
        use_diff=(args.head_type == "diffusion"), # Diff still have bugs of loaded data mismatch
        n_timesteps=args.n_timesteps,
        diff_horizon=args.diff_horizon,
        predict_epsilon=args.predict_epsilon,
        sep_lm_head=args.sep_lm_head,
        unfreeze_vit=args.unfreeze_vit,
        multi_step_action=args.multi_step_action,
        llm_name=args.llm_name,
        pooling=args.pooling,
        residual=args.residual,
        tcp_rel=args.tcp_rel,
        decoder_type=args.decoder_type,
        hidden_size=args.hidden_size,
        freeze_sampler=args.freeze_sampler,
        fwd_pred=args.fwd_pred,
        fwd_pred_hand=args.fwd_pred_hand,
        no_image_patch=args.no_image_patch,
        global_latent=args.global_latent,
        action_tokens=args.action_tokens,
        hf_revision=args.hf_revision,
    )
    # Load the policy model and handle training details
    checkpoint_path = args.checkpoint_path_foundation
    # check if a finetuning checkpoint exists for this run
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        latest_checkpoint = os.path.join(args.run_name, "latest_checkpoint.pth")
        if os.path.exists(latest_checkpoint):
            args.resume_from_checkpoint = latest_checkpoint
            print(f"Found resumable checkpoint {latest_checkpoint}.")
        else:
            ckpt_name = get_ckpt_name_pattern(args)
            checkpoint_list = glob.glob(f"{args.run_name}/{ckpt_name}")
            print(ckpt_name)
            checkpoint_list = [_ for _ in checkpoint_list if "__sep" not in _ and 'iter' not in _ and 'weights' not in _]
            if len(checkpoint_list) == 0:
                print(f"Found no checkpoints for run {args.run_name}.")
            else:
                args.resume_from_checkpoint = sorted(
                    checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )[-1]
                print(
                    f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
                )
    
    if not args.debug and not args.no_pretrain:
        if args.resume_from_checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True)
        else:
            checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu', mmap=True)
            checkpoint_path = args.resume_from_checkpoint
        
        if 'open_flamingo' in checkpoint_path:
            for key in list(checkpoint.keys()):
                # remove module from key name
                new_key = key.replace('module.', '')
                checkpoint[new_key] = checkpoint.pop(key)
                # We need to replace the embeddings if we include more tokens
                if args.action_tokens:
                    if new_key == 'lang_encoder.transformer.wte.weight':
                        del checkpoint[new_key]
            model.load_state_dict(checkpoint, strict=False)
        else:
            for key in list(checkpoint['model_state_dict'].keys()):
                #remove module from key name
                checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
        
            model.load_state_dict(checkpoint['model_state_dict'], False)

        if args.residual:
            model.lang_encoder.clone_parameters()


        print(f"Loaded model from {checkpoint_path}")

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()

    model = model.to(device_id)

    if args.dist:
        ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    else:
        ddp_model = model

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": [p for p in params_with_wd if p.requires_grad], "weight_decay": args.weight_decay},
            {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0},
        ]
    requested_learning_rate = args.learning_rate
    args.learning_rate = args.learning_rate * args.batch_size_calvin / 6 # adaptive lr
    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), eps=1e-8, lr=args.learning_rate)

    #Load dataset and LR scheduling
    if args.seq_training:
        dataset = get_data(
            args,
            args.calvin_dataset,
            image_processor,
            tokenizer,
            'seq',
            manifest_path=args.calvin_train_manifest,
            validation=False,
        )
        if args.calvin_val_manifest is not None or args.calvin_val_dataset is not None:
            val_dataset = get_data(
                args,
                args.calvin_val_dataset or args.calvin_dataset,
                image_processor,
                tokenizer,
                'seq',
                manifest_path=args.calvin_val_manifest,
                validation=True,
            )
    else:
        dataset = get_data(
            args,
            args.calvin_dataset,
            image_processor,
            tokenizer,
            'random',
            manifest_path=args.calvin_train_manifest,
            validation=False,
        )
        if args.calvin_val_manifest is not None or args.calvin_val_dataset is not None:
            val_dataset = get_data(
                args,
                args.calvin_val_dataset or args.calvin_dataset,
                image_processor,
                tokenizer,
                'random',
                manifest_path=args.calvin_val_manifest,
                validation=True,
            )
    
    if args.head_type == "diffusion" and (not args.debug):
        normalizer = model.diffusion_model.normalizer
        actions_idx = [i for i in range(0,10000)]
        all_actions = np.vstack([dataset.dataset.__getitem__(actions_idx, True)["actions"]])
        normalizer.fit(all_actions, last_n_dims=1, mode='limits')

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    optimizer_steps_per_epoch = (
        dataset.dataloader.num_batches + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps
    total_training_steps = optimizer_steps_per_epoch * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.calvin_train_manifest is not None:
        # Only rank 0 owns run metadata. Multiple DDP ranks writing the same
        # temporary manifest and environment-freeze files can race during
        # startup, especially when a new run directory is created.
        if args.rank == 0:
            write_run_manifest(
                args,
                dataset,
                val_dataset,
                requested_learning_rate=requested_learning_rate,
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    
    if args.resume_from_checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
    resume_from_epoch = 0
    resume_batch_offset = 0
    if args.resume_from_checkpoint is not None:
        resume_from_epoch = int(checkpoint.get("epoch", -1))
        if checkpoint.get("epoch_complete", False):
            resume_from_epoch += 1
            if args.resume_batch_offset not in (-1, 0):
                raise ValueError(
                    "An epoch-complete checkpoint must resume with batch offset 0."
                )
        else:
            completed_global_step = int(
                checkpoint.get(
                    "completed_global_step",
                    int(checkpoint.get("global_step", -1)) + 1,
                )
            )
            inferred_offset = completed_global_step - (
                resume_from_epoch * dataset.dataloader.num_batches
            )
            if not 0 <= inferred_offset < dataset.dataloader.num_batches:
                raise ValueError(
                    "Checkpoint implies an invalid batch offset: "
                    f"completed_global_step={completed_global_step}, "
                    f"epoch={resume_from_epoch}, "
                    f"batches_per_epoch={dataset.dataloader.num_batches}."
                )
            if args.resume_batch_offset == -1:
                resume_batch_offset = inferred_offset
            elif args.resume_batch_offset != inferred_offset:
                raise ValueError(
                    "Requested resume batch offset does not match checkpoint: "
                    f"requested={args.resume_batch_offset}, "
                    f"checkpoint={inferred_offset}."
                )
            else:
                resume_batch_offset = args.resume_batch_offset
        resume_from_epoch = max(0, resume_from_epoch)
        print(
            f"Resuming from epoch index {resume_from_epoch}, "
            f"batch offset {resume_batch_offset}."
        )
    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    ddp_model.train()
    best_val_loss = (
        checkpoint.get("validation_loss")
        if args.resume_from_checkpoint is not None
        else None
    )
    execution_end_epoch = (
        args.num_epochs
        if args.stop_after_epochs == -1
        else min(args.num_epochs, args.stop_after_epochs)
    )
    if resume_from_epoch >= execution_end_epoch:
        print(
            "Requested epoch boundary already reached: "
            f"resume epoch {resume_from_epoch}, execution end {execution_end_epoch}."
        )
    for epoch in range(resume_from_epoch, execution_end_epoch):
        dataset.set_epoch(epoch)
        calvin_loader = dataset.dataloader

        if args.head_type == "diffusion":
            train_one_epoch_diff(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb)
        elif args.fusion_mode == 'two_way':
            raise NotImplementedError
        else:
            best_val_loss = train_one_epoch(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                calvin_val_loader=(
                    val_dataset.dataloader
                    if args.calvin_val_manifest is not None or args.calvin_val_dataset is not None
                    else None
                ),
                device_id=device_id,
                wandb=wandb,
                best_val_loss=best_val_loss,
                resume_batch_offset=(
                    resume_batch_offset if epoch == resume_from_epoch else 0
                ),
            )
        if args.rank == 0 and not args.skip_checkpoint_saves:
            from train.utils import save_checkpoint
            save_checkpoint(
                args,
                ddp_model,
                optimizer,
                lr_scheduler,
                epoch,
                (epoch + 1) * dataset.dataloader.num_batches - 1,
                checkpoint_label="latest",
                validation_loss=best_val_loss,
                epoch_complete=True,
            )
            preserve_epoch_checkpoint(args, epoch)

        # Rank 0 may spend several minutes serializing a multi-GB optimizer
        # checkpoint. Keep all DDP ranks alive until that atomic save and its
        # epoch hard link are complete, especially when stopping at a planned
        # epoch boundary.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    if args.rank == 0 and not args.skip_checkpoint_saves:
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        ckpt_name = get_ckpt_name(args,)
        torch.save(get_checkpoint(ddp_model), f"{args.run_name}/{ckpt_name}")
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/{ckpt_name}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
