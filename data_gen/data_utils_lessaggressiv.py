"""Minimal-filter Combined-Explore collection with exact strict-data matching.

This module intentionally lives beside, rather than modifies, the legacy
collector.  A trajectory is eligible when at least one subtask is completed.
Every saved trajectory is clipped to the length of one reference trajectory
from the frozen strict manifests.  Pair labels, episode counts, frame counts,
and ws=12 window counts therefore match the strict dataset exactly.
"""

import gc
import hashlib
import json
import os
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from data_utils import create_episode_dict


TASK_ORDER = (
    "move_slider_left",
    "move_slider_right",
    "open_drawer",
    "close_drawer",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
)
STATE_FILENAME = "collection_state_lessaggressiv.json"
METADATA_FILENAME = "episode_metadata_lessaggressiv.jsonl"
AUDIT_FILENAME = "collection_audit_lessaggressiv.json"


def _format_duration(seconds):
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return "%dh %02dm %02ds" % (hours, minutes, secs)
    return "%dm %02ds" % (minutes, secs)


def _canonical_pair(pair):
    return tuple(sorted(pair, key=TASK_ORDER.index))


def _task_object(task):
    if "slider" in task:
        return "slider"
    if "drawer" in task:
        return "drawer"
    if "lightbulb" in task:
        return "lightbulb"
    if "led" in task:
        return "led"
    raise ValueError("Unsupported Combined subtask: %s" % task)


def combined_env_state_lessaggressiv(meta_task, sampling_stats, sampling_limits):
    """Create a valid four-object state and a balanced exploration ordering."""
    pair = _canonical_pair(meta_task)
    meta_objects = [_task_object(task) for task in pair]
    if len(set(meta_objects)) != 2:
        raise ValueError("Combined task pair must use two distinct objects: %s" % (pair,))

    environment_state = {
        "slider": random.choice(("left", "right")),
        "drawer": random.choice(("closed", "open")),
        "lightbulb": random.choice((0, 1)),
        "led": random.choice((0, 1)),
        "red_block": "table",
        "pink_block": "slider_right",
        "blue_block": "slider_left",
    }
    for task in pair:
        if task == "move_slider_left":
            environment_state["slider"] = "right"
        elif task == "move_slider_right":
            environment_state["slider"] = "left"
        elif task == "open_drawer":
            environment_state["drawer"] = "closed"
        elif task == "close_drawer":
            environment_state["drawer"] = "open"
        elif task == "turn_on_lightbulb":
            environment_state["lightbulb"] = 0
        elif task == "turn_off_lightbulb":
            environment_state["lightbulb"] = 1
        elif task == "turn_on_led":
            environment_state["led"] = 0
        elif task == "turn_off_led":
            environment_state["led"] = 1

    available_depths = [
        depth
        for depth in ("0", "1", "2")
        if sampling_stats[depth] < sampling_limits[depth]
    ]
    if not available_depths:
        raise RuntimeError("All exploration-depth sampling limits are exhausted")
    depth = random.choice(available_depths)
    sampling_stats[depth] += 1
    if depth == "0":
        positions = [0, 1]
        random.shuffle(positions)
    elif depth == "1":
        first = random.choice((0, 1, 2))
        second = random.choice((0, 1)) if first == 2 else 2
        positions = [first, second]
    else:
        positions = [random.choice((0, 1)), 3]
    if random.choice((False, True)):
        positions.reverse()

    feedback_state = [None] * 4
    for obj, position in zip(meta_objects, positions):
        feedback_state[position] = obj
    remaining_objects = [
        obj
        for obj in ("slider", "drawer", "lightbulb", "led")
        if obj not in meta_objects
    ]
    random.shuffle(remaining_objects)
    for index, value in enumerate(feedback_state):
        if value is None:
            feedback_state[index] = remaining_objects.pop()
    return environment_state, feedback_state, int(depth)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def _atomic_numpy(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(str(temporary), str(path))


def load_reference_targets(train_manifest_path, validation_manifest_path):
    """Return one collection target for every strict reference episode."""
    manifests = {}
    targets = []
    pair_splits = {}
    for split, path in (
        ("train", Path(train_manifest_path).resolve()),
        ("validation", Path(validation_manifest_path).resolve()),
    ):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("split") != split:
            raise ValueError("Invalid %s reference manifest: %s" % (split, path))
        manifests[split] = {
            "path": str(path),
            "sha256": _sha256(path),
            "protocol": manifest["protocol"],
            "split_seed": manifest["split_seed"],
            "num_episodes": int(manifest["num_episodes"]),
            "num_frames": int(manifest["num_frames"]),
            "num_windows_ws12": int(manifest["num_windows_ws12"]),
        }
        for record in manifest["episodes"]:
            pair = _canonical_pair(record["meta_task_pair"])
            previous_split = pair_splits.setdefault(pair, split)
            if previous_split != split:
                raise ValueError("Task pair appears in both splits: %s" % (pair,))
            length = int(record["end"]) - int(record["start"]) + 1
            if length <= 12:
                raise ValueError("Reference episode is too short: %s" % record)
            targets.append(
                {
                    "split": split,
                    "meta_task_pair": list(pair),
                    "target_length": length,
                    "reference_episode_index": int(record["episode_index"]),
                    "reference_start": int(record["start"]),
                    "reference_end": int(record["end"]),
                }
            )

    # Interleave all train/validation task-pair groups so a staged half-dataset
    # is representative instead of containing only the first few pairs.  Within
    # a pair, shorter targets come first to expose infeasible cutoffs early.
    grouped_targets = {}
    for target in targets:
        group = (target["split"], tuple(target["meta_task_pair"]))
        grouped_targets.setdefault(group, []).append(target)
    for group_targets in grouped_targets.values():
        group_targets.sort(
            key=lambda item: (
                item["target_length"], item["reference_episode_index"]
            )
        )
    group_order = sorted(
        grouped_targets,
        key=lambda group: (
            0 if group[0] == "train" else 1,
            tuple(TASK_ORDER.index(task) for task in group[1]),
        ),
    )
    targets = []
    round_index = 0
    while True:
        added = False
        for group in group_order:
            group_targets = grouped_targets[group]
            if round_index < len(group_targets):
                targets.append(group_targets[round_index])
                added = True
        if not added:
            break
        round_index += 1
    for split in ("train", "validation"):
        selected = [item for item in targets if item["split"] == split]
        frames = sum(item["target_length"] for item in selected)
        windows = sum(item["target_length"] - 12 for item in selected)
        if len(selected) != manifests[split]["num_episodes"]:
            raise ValueError("Reference episode count does not match its manifest")
        if frames != manifests[split]["num_frames"]:
            raise ValueError("Reference frame count does not match its manifest")
        if windows != manifests[split]["num_windows_ws12"]:
            raise ValueError("Reference ws=12 count does not match its manifest")
    return targets, manifests


def rollout_minimal(env, agent, steps, meta_task, feedback_state, environment_state,
                    use_reasoning=False, render=False, annotation=None):
    """Roll out all trials and report the first completed-subtask frame."""
    episode_data = []
    robot = deepcopy(env.base_env.robot)
    obs, feedback, environment_description, _ = env.episode_reset(
        feedback_state, environment_state, list(meta_task), train=True
    )
    env_feedback = feedback["lm_feedback"]
    current_trial = 0
    instruction_step = True
    first_completion_frames = None
    completed_subtasks = 0
    full_meta_successes = 0
    trials_before_first_full_reward = 0
    saw_full_reward = False

    for frame_index in range(steps):
        before_completed = int(env.completed_instructions)
        action, instruction, output = agent.step(
            obs,
            env_feedback,
            use_reasoning=use_reasoning,
            env_description=deepcopy(environment_description),
            annotation=annotation,
            instruction_step=instruction_step,
        )
        action_save = action.copy()
        abs_action = robot.relative_to_absolute(action_save.copy())
        if not instruction_step:
            feedback["lm_feedback"] = None
            output = None
            if current_trial != 0:
                environment_description = None
        episode_data.append(
            create_episode_dict(
                obs,
                action_save,
                abs_action,
                feedback,
                instruction,
                output,
                environment_description,
                current_trial,
                frame_index,
            )
        )

        obs, reward, feedback_new, _, trial_done, _ = env.step(action, instruction)
        newly_completed = max(0, int(env.completed_instructions) - before_completed)
        if newly_completed:
            completed_subtasks += newly_completed
            if first_completion_frames is None:
                # The saved frame contains the action that completed the subtask.
                first_completion_frames = len(episode_data)

        if feedback_new is None:
            instruction_step = False
            env_feedback = None
        else:
            instruction_step = True
            feedback = feedback_new
            env_feedback = feedback["lm_feedback"]

        if trial_done:
            current_trial += 1
            if reward:
                full_meta_successes += 1
                saw_full_reward = True
            elif not saw_full_reward:
                trials_before_first_full_reward += 1

            if current_trial >= env.num_trials:
                break
            obs, feedback, environment_description, _ = env.trial_reset(instruction)
            env_feedback = feedback["lm_feedback"]
            instruction_step = True
            agent.policy_model.reset()

        if render:
            rgb = env.render(mode="rgb_array")[:, :, ::-1]
            cv2.imshow("lessaggressiv collection", rgb)
            cv2.waitKey(1)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "episode_data": episode_data,
        "rollout_length": len(episode_data),
        "first_completion_frames": first_completion_frames,
        "completed_subtasks": completed_subtasks,
        "full_meta_successes": full_meta_successes,
        "trials_before_first_full_reward": trials_before_first_full_reward,
    }


def _save_episode(episode_data, save_dir, start):
    for offset, frame in enumerate(episode_data):
        path = save_dir / ("episode_%06d.npz" % (start + offset))
        serializable_frame = dict(frame)
        actions = serializable_frame.get("actions")
        if isinstance(actions, (tuple, list)):
            # CALVIN absolute actions contain position (3), orientation (3),
            # and gripper (1). NumPy >=1.24 requires the ragged container to
            # be made explicit; dtype/object and shape (3,) match legacy data.
            object_actions = np.empty(len(actions), dtype=object)
            object_actions[:] = actions
            serializable_frame["actions"] = object_actions

        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **serializable_frame)
        os.replace(str(temporary), str(path))


def _write_dataset_files(save_dir, state):
    records = state["records"]
    ranges = np.asarray(
        [[record["start"], record["end"]] for record in records], dtype=np.int64
    )
    _atomic_numpy(save_dir / "ep_start_end_ids.npy", ranges)

    pair_counts = Counter("_".join(record["meta_task_pair"]) for record in records)
    np.savez(
        str(save_dir / "meta_task_stats.npz"),
        **{key: int(value) for key, value in pair_counts.items()}
    )
    exploration_counts = Counter(
        str(min(2, record["trials_before_first_full_reward"])) for record in records
    )
    np.savez(
        str(save_dir / "exploration_stats.npz"),
        **{key: int(exploration_counts.get(key, 0)) for key in ("0", "1", "2")}
    )
    metadata_path = save_dir / METADATA_FILENAME
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(str(temporary), str(metadata_path))


def _initial_state(save_dir, targets, reference_manifests, args):
    existing = list(save_dir.iterdir())
    if existing:
        raise RuntimeError(
            "Refusing to start in non-empty directory without %s: %s"
            % (STATE_FILENAME, save_dir)
        )
    return {
        "schema_version": 1,
        "filtering": "minimal_at_least_one_completed_subtask",
        "matching": "pair_and_reference_episode_length",
        "complete": False,
        "next_target_index": 0,
        "next_frame_index": 0,
        "attempts": 0,
        "rejections": {},
        "records": [],
        "targets": targets,
        "reference_manifests": reference_manifests,
        "collection_seed": int(args.seed),
        "rollout_steps": int(args.rollout_steps),
        "num_trials": int(args.num_trials),
    }


def _validate_resume_state(state, targets, reference_manifests, args):
    immutable = {
        "targets": targets,
        "reference_manifests": reference_manifests,
        "collection_seed": int(args.seed),
        "rollout_steps": int(args.rollout_steps),
        "num_trials": int(args.num_trials),
    }
    for key, expected in immutable.items():
        if state.get(key) != expected:
            raise RuntimeError("Resume mismatch for %s" % key)


def collect_data_lessaggressiv(args, agent, env):
    """Collect the single Combined Minimal dataset requested for F1."""
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    targets, reference_manifests = load_reference_targets(
        args.reference_train_manifest, args.reference_validation_manifest
    )
    state_path = save_dir / STATE_FILENAME
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _validate_resume_state(state, targets, reference_manifests, args)
    else:
        state = _initial_state(save_dir, targets, reference_manifests, args)
        _atomic_json(state_path, state)

    if state["complete"]:
        print("Collection is already complete: %s" % save_dir)
        return

    stage_limit = int(args.stop_after_episodes)
    if stage_limit and state["next_target_index"] >= stage_limit:
        print(
            "Staged collection already reached %d/%d episodes; rerun with a "
            "larger --stop_after_episodes value or 0 to finish."
            % (state["next_target_index"], len(targets))
        )
        return

    # This controls environment ordering only; acceptance never depends on
    # reward discovery depth in the Minimal condition.
    sampling_stats = {"0": 0, "1": 0, "2": 0}
    sampling_limits = {"0": 10 ** 9, "1": 10 ** 9, "2": 10 ** 9}
    saved_at_start = int(state["next_target_index"])
    started = time.time()

    while state["next_target_index"] < len(targets):
        if args.max_attempts and state["attempts"] >= args.max_attempts:
            raise RuntimeError("Reached --max_attempts before matching all targets")
        target = targets[state["next_target_index"]]
        pair = tuple(target["meta_task_pair"])
        target_length = int(target["target_length"])
        attempt_seed = (int(args.seed) + int(state["attempts"])) % (2 ** 32)
        random.seed(attempt_seed)
        np.random.seed(attempt_seed)
        torch.manual_seed(attempt_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(attempt_seed)
        environment_state, feedback_state, intended_depth = (
            combined_env_state_lessaggressiv(pair, sampling_stats, sampling_limits)
        )
        agent.reset()
        result = rollout_minimal(
            env,
            agent,
            args.rollout_steps,
            pair,
            feedback_state,
            environment_state,
            use_reasoning=args.use_reasoning,
            render=args.render,
        )
        state["attempts"] += 1

        reason = None
        first_completion = result["first_completion_frames"]
        if first_completion is None:
            reason = "no_subtask_completed"
        elif first_completion > target_length:
            reason = "first_completion_after_matched_cutoff"
        elif result["rollout_length"] < target_length:
            reason = "rollout_shorter_than_reference_episode"

        if reason is not None:
            state["rejections"][reason] = state["rejections"].get(reason, 0) + 1
            _atomic_json(state_path, state)
            print(
                "Rejected attempt %d for %s (%s, target=%d, rollout=%d)"
                % (
                    state["attempts"],
                    pair,
                    reason,
                    target_length,
                    result["rollout_length"],
                )
            )
            continue

        clipped = result["episode_data"][:target_length]
        start = int(state["next_frame_index"])
        end = start + target_length - 1
        _save_episode(clipped, save_dir, start)
        record = {
            "episode_index": len(state["records"]),
            "start": start,
            "end": end,
            "meta_task_pair": list(pair),
            "split": target["split"],
            "filtering": "minimal_at_least_one_completed_subtask",
            "completed_subtasks_in_full_rollout": result["completed_subtasks"],
            "first_completion_frames": int(first_completion),
            "full_meta_successes": result["full_meta_successes"],
            "trials_before_first_full_reward": result[
                "trials_before_first_full_reward"
            ],
            "rollout_length": result["rollout_length"],
            "saved_length": target_length,
            "reference_episode_index": target["reference_episode_index"],
            "reference_start": target["reference_start"],
            "reference_end": target["reference_end"],
            "attempt_seed": attempt_seed,
            "intended_exploration_depth": intended_depth,
        }
        state["records"].append(record)
        state["next_frame_index"] = end + 1
        state["next_target_index"] += 1
        _write_dataset_files(save_dir, state)
        _atomic_json(state_path, state)

        saved = state["next_target_index"]
        if saved % args.report_interval == 0 or saved == len(targets):
            elapsed = time.time() - started
            saved_this_run = saved - saved_at_start
            seconds_per_episode = elapsed / saved_this_run
            remaining_seconds = seconds_per_episode * (len(targets) - saved)
            estimated_finish = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(time.time() + remaining_seconds),
            )
            print(
                "Saved %d/%d matched episodes (%d frames); attempts=%d; "
                "avg=%s/episode; ETA=%s; estimated_finish=%s"
                % (
                    saved,
                    len(targets),
                    state["next_frame_index"],
                    state["attempts"],
                    _format_duration(seconds_per_episode),
                    _format_duration(remaining_seconds),
                    estimated_finish,
                )
            )
        if stage_limit and saved >= stage_limit:
            print(
                "Paused staged collection at %d/%d accepted episodes. State and "
                "dataset indices are saved; rerun to resume at episode %d."
                % (saved, len(targets), saved + 1)
            )
            return

    state["complete"] = True
    state["completed_at_unix"] = time.time()
    _write_dataset_files(save_dir, state)
    _atomic_json(state_path, state)

    split_stats = {}
    for split in ("train", "validation"):
        records = [record for record in state["records"] if record["split"] == split]
        split_stats[split] = {
            "num_episodes": len(records),
            "num_frames": sum(record["saved_length"] for record in records),
            "num_windows_ws12": sum(record["saved_length"] - 12 for record in records),
        }
        reference = reference_manifests[split]
        for key in ("num_episodes", "num_frames", "num_windows_ws12"):
            if split_stats[split][key] != reference[key]:
                raise RuntimeError("Final %s %s mismatch" % (split, key))
    audit = {
        "schema_version": 1,
        "complete": True,
        "filtering": state["filtering"],
        "matching": state["matching"],
        "reference_manifests": reference_manifests,
        "splits": split_stats,
        "attempts": state["attempts"],
        "accepted": len(state["records"]),
        "rejections": state["rejections"],
        "source_index_sha256": _sha256(save_dir / "ep_start_end_ids.npy"),
    }
    _atomic_json(save_dir / AUDIT_FILENAME, audit)
    print("Exact matched Minimal collection complete: %s" % save_dir)
