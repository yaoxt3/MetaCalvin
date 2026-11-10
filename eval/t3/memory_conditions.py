"""Leakage-safe textual-memory conditions for T3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


TEXT_MEMORY_MODES = ("local", "cumulative")

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


def render_objects(objects: Sequence[str]) -> str:
    return ", ".join(str(value).replace("_", " ") for value in objects)


def _render_items(values: Sequence[str]) -> str:
    return ", ".join(TASK_FEEDBACK_TEXT.get(value, value.replace("_", " ")) for value in values) or "none"


@dataclass(frozen=True)
class TrialFacts:
    trial_index: int
    rewarding_tasks: tuple[str, ...]
    unrewarding_tasks: tuple[str, ...]
    completed_tasks: tuple[str, ...]
    reward: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "rewarding_tasks": list(self.rewarding_tasks),
            "unrewarding_tasks": list(self.unrewarding_tasks),
            "completed_tasks": list(self.completed_tasks),
            "reward": self.reward,
        }


class TextMemoryLedger:
    """Build text only from observable trial outcomes, never the hidden target."""

    def __init__(self, candidate_tasks: Sequence[str], episode_objects: Sequence[str]):
        self.candidate_tasks = tuple(str(value) for value in candidate_tasks)
        self.episode_objects = tuple(str(value) for value in episode_objects)
        if not self.candidate_tasks:
            raise ValueError("T3 requires at least one candidate task")
        if len(set(self.candidate_tasks)) != len(self.candidate_tasks):
            raise ValueError("T3 candidate tasks must be unique")
        self.trials: list[TrialFacts] = []
        self.known_rewarding: set[str] = set()
        self.known_unrewarding: set[str] = set()

    def ordered(self, values: Sequence[str] | set[str]) -> list[str]:
        available = set(values)
        return [task for task in self.candidate_tasks if task in available]

    def snapshot(self) -> Dict[str, Any]:
        known_rewarding = self.ordered(self.known_rewarding)
        known_unrewarding = self.ordered(self.known_unrewarding)
        tested = self.known_rewarding | self.known_unrewarding
        return {
            "known_rewarding_tasks": known_rewarding,
            "known_unrewarding_tasks": known_unrewarding,
            "untested_tasks": [
                task for task in self.candidate_tasks if task not in tested
            ],
            "last_trial": self.trials[-1].to_dict() if self.trials else None,
            "trial_count_observed": len(self.trials),
        }

    def render(self, mode: str, trial_index: int, raw_initial_feedback: Any) -> str:
        if mode not in TEXT_MEMORY_MODES:
            raise ValueError(f"Unknown T3 text-memory mode: {mode!r}")
        if trial_index == 1:
            if self.trials:
                raise ValueError("Trial 1 feedback requested after outcomes were recorded")
            return str(raw_initial_feedback)
        if len(self.trials) != trial_index - 1:
            raise ValueError(
                f"T3 ledger has {len(self.trials)} trials before trial {trial_index}"
            )
        last = self.trials[-1]
        if mode == "cumulative":
            cumulative = self.snapshot()
            known_rewarding = _render_items(cumulative["known_rewarding_tasks"])
            known_unrewarding = _render_items(cumulative["known_unrewarding_tasks"])
            untested = _render_items(cumulative["untested_tasks"])
        else:
            known_rewarding = "information not provided"
            known_unrewarding = "information not provided"
            untested = "information not provided"
        return (
            f"Trial {trial_index - 1} completed. "
            f"Completed rewarding tasks last trial: {_render_items(last.rewarding_tasks)}. "
            f"Completed unrewarding tasks last trial: {_render_items(last.unrewarding_tasks)}. "
            f"Known rewarding tasks so far: {known_rewarding}. "
            f"Known unrewarding tasks so far: {known_unrewarding}. "
            f"Untested tasks: {untested}. "
            f"Episode objects: {render_objects(self.episode_objects)}"
        )

    def observe(self, trial_index: int, completed_tasks: Sequence[str], reward: float) -> TrialFacts:
        if trial_index != len(self.trials) + 1:
            raise ValueError(f"Unexpected T3 trial outcome index: {trial_index}")
        completed = tuple(self.ordered(completed_tasks))
        unknown = sorted(set(completed_tasks) - set(self.candidate_tasks))
        if unknown:
            raise ValueError(f"Observed tasks outside the frozen candidate set: {unknown}")
        if reward > 0:
            if len(completed) != 1:
                raise ValueError(
                    "A rewarding 7-Object trial must expose exactly one completed task; "
                    f"observed {list(completed)}"
                )
            rewarding = completed
            unrewarding: tuple[str, ...] = ()
            self.known_rewarding.update(completed)
            self.known_unrewarding.difference_update(completed)
        else:
            rewarding = ()
            unrewarding = completed
            self.known_unrewarding.update(completed)
        facts = TrialFacts(
            trial_index=trial_index,
            rewarding_tasks=rewarding,
            unrewarding_tasks=unrewarding,
            completed_tasks=completed,
            reward=float(reward),
        )
        self.trials.append(facts)
        return facts

    def metadata(self, mode: str) -> Dict[str, Any]:
        if mode not in TEXT_MEMORY_MODES:
            raise ValueError(mode)
        return {
            "text_memory": mode,
            "renderer": "t3_matched_structured_fixed_v1",
            "trial_1": "raw frozen-manifest feedback",
            "later_trials": (
                "last-trial outcomes plus cumulative positive/negative/untested facts"
                if mode == "cumulative"
                else "last-trial outcomes; cumulative fields explicitly withheld"
            ),
            "leakage_rule": (
                "derive text only from completed_tasks and observed reward; "
                "the renderer never receives meta_task"
            ),
            "immediate_feedback": "preserved unchanged within trial",
        }
