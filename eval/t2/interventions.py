"""Deterministic textual history interventions for T2."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


INTERVENTIONS = ("full", "masked", "shuffled", "counterbalanced")
FEEDBACK_TYPES = ("train_feedback", "alt_feedback")
DEFAULT_DONOR_SEED = 42

TASK_FEEDBACK_TEXT = {
    "move_slider_left": "move the door to the left",
    "move_slider_right": "move the door to the right",
    "open_drawer": "pull the handle to open the drawer",
    "close_drawer": "push the handle to close the drawer",
    "turn_on_lightbulb": "turn on the lightbulb",
    "turn_off_lightbulb": "turn off the lightbulb",
    "turn_on_led": "turn on the led",
    "turn_off_led": "turn off the led",
    "lift_red_block_table": "lift the red block from the table",
    "lift_pink_block_table": "lift the pink block from the table",
    "lift_blue_block_table": "lift the blue block from the table",
    "lift_red_block_slider": "lift the red block from the slider",
    "lift_pink_block_slider": "lift the pink block from the slider",
    "lift_blue_block_slider": "lift the blue block from the slider",
}


def counterbalanced_order(objects: Sequence[str], episode_index: int) -> list[str]:
    values = list(objects)
    if len(values) < 2:
        raise ValueError("Counterbalancing requires at least two objects")
    offset = 1 + (int(episode_index) % (len(values) - 1))
    return values[offset:] + values[:offset]


def render_objects(objects: Sequence[str]) -> str:
    return ", ".join(str(item).replace("_", " ") for item in objects)


def replace_episode_object_order(
    text: str, original_objects: Sequence[str], new_objects: Sequence[str]
) -> str:
    """Replace a displayed full candidate list in either feedback family.

    Reward-only diverse feedback does not display the candidate list, so it is
    intentionally left unchanged.
    """
    original = render_objects(original_objects)
    if original not in text:
        return text
    return text.replace(original, render_objects(new_objects), 1)


def masked_previous_outcomes(
    text: str,
    trial_index: int,
    episode: Mapping[str, Any],
    feedback_type: str,
) -> str:
    if trial_index <= 1:
        return text
    rendered = render_objects(episode["feedback_state"])
    if feedback_type == "alt_feedback":
        # Reuse this episode's frozen held-out initial template. This removes
        # both reward/no-reward identity and explored-object history without
        # borrowing a template or numeric parameter from another experiment.
        template = str(episode["feedback_templates"]["first_trial_template"])
        return template.format(rendered)
    return (
        f"Trial {trial_index - 1} completed. "
        "Completed rewarding tasks last trial: information withheld. "
        "Completed unrewarding tasks last trial: information withheld. "
        f"Episode objects: {rendered}"
    )


def load_full_trials(path: Path) -> Dict[tuple[str, int], Dict[str, Any]]:
    lookup: Dict[tuple[str, int], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["episode_id"]), int(record["trial_index"]))
            if key in lookup:
                raise ValueError(f"Duplicate Full trial record at line {line_number}: {key}")
            if "input_feedback" not in record:
                raise ValueError(f"Full trial record at line {line_number} has no input_feedback")
            lookup[key] = dict(record)
    if not lookup:
        raise ValueError(f"Full trial log contains no records: {path}")
    return lookup


def load_full_feedback(path: Path) -> Dict[tuple[str, int], str]:
    return {
        key: str(record["input_feedback"])
        for key, record in load_full_trials(path).items()
    }


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _structure_signature(episode: Mapping[str, Any]) -> tuple[tuple[str, ...], int]:
    objects = tuple(sorted(str(item) for item in episode["feedback_state"]))
    candidate_count = len(episode["candidate_tasks"])
    if len(objects) != candidate_count or len(set(objects)) != len(objects):
        raise ValueError(
            f"Episode {episode['episode_id']} has an invalid candidate structure"
        )
    return objects, candidate_count


def build_donor_map(
    episodes: Sequence[Mapping[str, Any]],
    full_feedback: Mapping[tuple[str, int], str],
    *,
    donor_seed: int = DEFAULT_DONOR_SEED,
) -> Dict[str, str]:
    """Predeclare a seeded donor derangement without inspecting outcomes.

    Donors must have the same object/task-slot structure. Their concrete task
    labels are translated into the target episode's feasible candidate labels
    before the history is presented to the policy.
    """
    groups: Dict[tuple[tuple[str, ...], int], list[str]] = defaultdict(list)
    available = {episode_id for episode_id, _ in full_feedback}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        if episode_id in available:
            groups[_structure_signature(episode)].append(episode_id)

    mapping: Dict[str, str] = {}
    for episode_ids in groups.values():
        if len(episode_ids) < 2:
            continue
        episode_ids.sort(key=lambda value: _stable_key(donor_seed, value))
        for index, episode_id in enumerate(episode_ids):
            mapping[episode_id] = episode_ids[(index + 1) % len(episode_ids)]
    missing = sorted(available - set(mapping))
    if missing:
        raise ValueError(
            "Shuffled history requires at least two Full episodes in every used "
            f"object-combination stratum; missing donors for {missing[:5]}"
        )
    return mapping


def _aligned_tasks_by_object(episode: Mapping[str, Any]) -> Dict[str, str]:
    objects = [str(value) for value in episode["feedback_state"]]
    tasks = [str(value) for value in episode["candidate_tasks"]]
    if len(objects) != len(tasks):
        raise ValueError(
            f"Episode {episode['episode_id']} has unaligned objects and tasks"
        )
    return dict(zip(objects, tasks))


def build_history_translation(
    donor: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    donor_seed: int = DEFAULT_DONOR_SEED,
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Map donor history to target candidates through a seeded derangement."""
    if _structure_signature(donor) != _structure_signature(target):
        raise ValueError(
            f"Donor {donor['episode_id']} does not match target "
            f"{target['episode_id']} structure"
        )
    donor_objects = sorted(str(value) for value in donor["feedback_state"])
    target_objects = sorted(str(value) for value in target["feedback_state"])
    offset = 1 + (
        int(_stable_key(donor_seed, str(target["episode_id"])), 16)
        % (len(target_objects) - 1)
    )
    rotated_targets = target_objects[offset:] + target_objects[:offset]
    object_map = dict(zip(donor_objects, rotated_targets))
    donor_tasks = _aligned_tasks_by_object(donor)
    target_tasks = _aligned_tasks_by_object(target)
    task_map = {
        donor_tasks[donor_object]: target_tasks[target_object]
        for donor_object, target_object in object_map.items()
    }
    return object_map, task_map


def _replace_simultaneously(text: str, replacements: Mapping[str, str]) -> str:
    placeholders: Dict[str, str] = {}
    result = text
    for index, source in enumerate(sorted(replacements, key=len, reverse=True)):
        if not source:
            continue
        placeholder = f"__T2_HISTORY_REPLACEMENT_{index}__"
        if source in result:
            result = result.replace(source, placeholder)
            placeholders[placeholder] = replacements[source]
    for placeholder, replacement in placeholders.items():
        result = result.replace(placeholder, replacement)
    return result


def translate_donor_feedback(
    text: str,
    donor: Mapping[str, Any],
    target: Mapping[str, Any],
    object_map: Mapping[str, str],
    task_map: Mapping[str, str],
    feedback_type: str,
) -> str:
    """Keep donor outcomes but express all identities in target vocabulary."""
    candidate_placeholder = "__T2_TARGET_CANDIDATE_LIST__"
    result = text.replace(
        render_objects(donor["feedback_state"]), candidate_placeholder, 1
    )
    if feedback_type == "train_feedback":
        donor_surfaces = TASK_FEEDBACK_TEXT
        target_surfaces = TASK_FEEDBACK_TEXT
    else:
        donor_surfaces = donor["sampled_task_instructions"]
        target_surfaces = target["sampled_task_instructions"]
    task_replacements = {
        str(donor_surfaces[donor_task]): str(target_surfaces[target_task])
        for donor_task, target_task in task_map.items()
    }
    object_replacements = {
        str(source).replace("_", " "): str(destination).replace("_", " ")
        for source, destination in object_map.items()
    }
    # Apply task phrases and bare object identities atomically. Otherwise an
    # object name inside an already translated target task could be remapped a
    # second time.
    result = _replace_simultaneously(
        result, {**object_replacements, **task_replacements}
    )
    return result.replace(candidate_placeholder, render_objects(target["feedback_state"]))


class FeedbackIntervention:
    def __init__(
        self,
        name: str,
        episodes: Sequence[Mapping[str, Any]],
        full_trials_path: Path | None = None,
        feedback_type: str = "train_feedback",
        donor_seed: int = DEFAULT_DONOR_SEED,
    ) -> None:
        if name not in INTERVENTIONS:
            raise ValueError(f"Unknown T2 intervention: {name!r}")
        if feedback_type not in FEEDBACK_TYPES:
            raise ValueError(f"Unknown feedback type: {feedback_type!r}")
        self.name = name
        self.feedback_type = feedback_type
        self.donor_seed = int(donor_seed)
        self.episodes = {
            str(episode["episode_id"]): dict(episode) for episode in episodes
        }
        self.full_trials: Dict[tuple[str, int], Dict[str, Any]] = {}
        self.full_feedback: Dict[tuple[str, int], str] = {}
        self.donor_map: Dict[str, str] = {}
        self.donor_object_maps: Dict[str, Dict[str, str]] = {}
        self.donor_task_maps: Dict[str, Dict[str, str]] = {}
        if full_trials_path is not None:
            if not full_trials_path.is_file():
                raise ValueError(f"Full trial reference does not exist: {full_trials_path}")
            self.full_trials = load_full_trials(full_trials_path)
            self.full_feedback = {
                key: str(record["input_feedback"])
                for key, record in self.full_trials.items()
            }
        if name == "shuffled":
            if not self.full_trials:
                raise ValueError("Shuffled intervention requires an existing Full trials.jsonl")
            self.donor_map = build_donor_map(
                episodes, self.full_feedback, donor_seed=self.donor_seed
            )
            for target_id, donor_id in self.donor_map.items():
                object_map, task_map = build_history_translation(
                    self.episodes[donor_id],
                    self.episodes[target_id],
                    donor_seed=self.donor_seed,
                )
                self.donor_object_maps[target_id] = object_map
                self.donor_task_maps[target_id] = task_map

    def full_reference(
        self, episode_id: str, trial_index: int
    ) -> Mapping[str, Any] | None:
        return self.full_trials.get((str(episode_id), int(trial_index)))

    def apply(
        self,
        raw_feedback: Any,
        *,
        episode: Mapping[str, Any],
        trial_index: int,
        stage: str,
    ) -> str:
        text = str(raw_feedback)
        if self.name == "full":
            return text
        if self.name == "masked":
            if stage != "trial_start":
                return text
            return masked_previous_outcomes(
                text, trial_index, episode, self.feedback_type
            )
        if self.name == "shuffled":
            if stage != "trial_start" or trial_index <= 1:
                return text
            donor = self.donor_map[str(episode["episode_id"])]
            try:
                donor_feedback = self.full_feedback[(donor, trial_index)]
            except KeyError as error:
                raise ValueError(
                    f"Full donor {donor} has no trial {trial_index} feedback"
                ) from error
            target_id = str(episode["episode_id"])
            return translate_donor_feedback(
                donor_feedback,
                self.episodes[donor],
                episode,
                self.donor_object_maps[target_id],
                self.donor_task_maps[target_id],
                self.feedback_type,
            )
        if self.name == "counterbalanced":
            order = counterbalanced_order(
                episode["feedback_state"], int(episode["episode_index"])
            )
            return replace_episode_object_order(
                text, episode["feedback_state"], order
            )
        raise AssertionError(self.name)

    def input_candidate_order(self, episode: Mapping[str, Any]) -> list[str]:
        candidates = list(episode["candidate_tasks"])
        if self.name != "counterbalanced":
            return candidates
        return counterbalanced_order(candidates, int(episode["episode_index"]))

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feedback_type": self.feedback_type,
            "scope": (
                "previous-trial boundary feedback only"
                if self.name in {"masked", "shuffled"}
                else (
                    "candidate-order text at trial boundaries"
                    if self.name == "counterbalanced"
                    else "trial-boundary feedback unchanged"
                )
            ),
            "causal_role": {
                "full": "unmodified within-T2 paired reference",
                "masked": "remove prior success/failure outcome information",
                "shuffled": "replace prior outcomes with a matched donor episode",
                "counterbalanced": "test fixed original-order versus presented-order behavior",
            }[self.name],
            "counterbalance_rule": (
                "cyclically rotate feedback_state and aligned candidate_tasks by "
                "1 + episode_index mod (candidate_count - 1)"
                if self.name == "counterbalanced"
                else None
            ),
            "masked_rule": (
                "replace later trial-boundary feedback with the episode's frozen "
                "initial diverse template and unchanged candidate order"
                if self.name == "masked" and self.feedback_type == "alt_feedback"
                else (
                    "replace rewarding and unrewarding history fields with "
                    "information-withheld markers"
                    if self.name == "masked"
                    else None
                )
            ),
            "donor_map": dict(sorted(self.donor_map.items())),
            "donor_seed": self.donor_seed if self.name == "shuffled" else None,
            "donor_pairing_rule": (
                "seeded cyclic derangement within identical sorted feedback_state "
                "and candidate-count strata; independent of Full outcomes"
                if self.name == "shuffled"
                else None
            ),
            "donor_object_maps": dict(sorted(self.donor_object_maps.items())),
            "donor_task_maps": dict(sorted(self.donor_task_maps.items())),
            "paired_full_reference_available": bool(self.full_trials),
        }
