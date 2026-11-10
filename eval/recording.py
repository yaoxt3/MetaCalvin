import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def append_jsonl(path, record):
    if path is None:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": utc_now(),
        **jsonable(record),
    }
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def write_json_atomic(path, payload):
    if path is None:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(payload), output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, output_path)


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while True:
            chunk = input_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def template_descriptor(env, feedback_type, stage):
    combined_environment = hasattr(env, "end_trial_template")
    if feedback_type == "train_feedback":
        if combined_environment:
            template_ids = {
                "initial": "fixed:combined:first_trial",
                "reward": "fixed:combined:trial_end",
                "failure": "fixed:combined:trial_end",
            }
        else:
            template_ids = {
                "initial": "fixed:first_trial",
                "reward": "fixed:reward_last_trial",
                "failure": "fixed:no_reward_last_trial",
            }
        return {
            "feedback_family": "fixed",
            "template_stage": stage,
            "template_id": template_ids[stage],
            "template_text": None,
        }

    if combined_environment:
        attribute_names = {
            "initial": "first_trial_template",
            "reward": "end_trial_template",
            "failure": "end_trial_template",
        }
    else:
        attribute_names = {
            "initial": "first_trial_template",
            "reward": "reward_template",
            "failure": "further_trial_template",
        }
    template_text = str(getattr(env, attribute_names[stage], ""))
    template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest()[:16]
    return {
        "feedback_family": "diverse",
        "template_stage": stage,
        "template_id": f"diverse:{stage}:{template_hash}",
        "template_text": template_text,
    }


def latency_summary(latencies_seconds):
    if not latencies_seconds:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "mean_seconds": None,
            "p50_seconds": None,
            "p95_seconds": None,
            "max_seconds": None,
        }

    values = np.asarray(latencies_seconds, dtype=np.float64)
    return {
        "count": int(values.size),
        "total_seconds": float(values.sum()),
        "mean_seconds": float(values.mean()),
        "p50_seconds": float(np.percentile(values, 50)),
        "p95_seconds": float(np.percentile(values, 95)),
        "max_seconds": float(values.max()),
    }


def action_summary(actions):
    if not actions:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "maximum": None,
            "gripper_open_steps": 0,
            "gripper_closed_steps": 0,
        }

    values = np.asarray(actions, dtype=np.float32)
    gripper = values[:, -1]
    return {
        "count": int(values.shape[0]),
        "mean": values.mean(axis=0).tolist(),
        "minimum": values.min(axis=0).tolist(),
        "maximum": values.max(axis=0).tolist(),
        "gripper_open_steps": int(np.count_nonzero(gripper > 0)),
        "gripper_closed_steps": int(np.count_nonzero(gripper <= 0)),
    }


def failure_category(reward, completed_tasks):
    if reward > 0:
        return None
    if completed_tasks:
        return "completed_only_nonrewarding_tasks"
    return "no_task_completed"
