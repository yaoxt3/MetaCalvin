"""High-level planners used by the T1 comparison.

This module deliberately has no CALVIN, PyBullet, or torch dependency so the
planner semantics can be unit-tested without a simulator or GPU.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def normalize_instruction(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def instruction_to_task(
    instruction: str,
    candidates: Sequence[str],
    clear_instructions: Mapping[str, str],
) -> Optional[str]:
    """Map an LLM instruction to exactly one feasible canonical task."""

    normalized = normalize_instruction(instruction)
    if not normalized:
        return None

    exact_matches = [
        task
        for task in candidates
        if normalized
        in {normalize_instruction(task), normalize_instruction(clear_instructions[task])}
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    # Gemma sometimes wraps a concise instruction in a sentence. Only accept a
    # unique contained clear instruction; ambiguous matches remain invalid.
    contained_matches = [
        task
        for task in candidates
        if normalize_instruction(clear_instructions[task]) in normalized
    ]
    if len(contained_matches) == 1:
        return contained_matches[0]

    words = set(normalized.split())
    semantic_matches = []
    for task in candidates:
        parts = task.split("_")
        matches = False
        if task.startswith("move_slider_"):
            direction = parts[-1]
            matches = direction in words and bool({"slider", "door"} & words)
        elif task in {"open_drawer", "close_drawer"}:
            matches = parts[0] in words and "drawer" in words
        elif task.startswith("turn_on_") or task.startswith("turn_off_"):
            state = parts[1]
            obj = parts[-1]
            matches = state in words and obj in words
        elif task.startswith("lift_"):
            color = parts[1]
            object_words = {"block", "cube"}
            action_words = {"lift", "lifting", "pick", "pickup", "grasp", "grab"}
            matches = color in words and bool(object_words & words) and bool(action_words & words)
        if matches:
            semantic_matches.append(task)
    if len(semantic_matches) == 1:
        return semantic_matches[0]
    return None


@dataclass
class PlannerDecision:
    selected_task: Optional[str]
    instruction: str
    valid: bool = True
    raw_output: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BasePlanner:
    name = "base"

    def __init__(self, clear_instructions: Mapping[str, str]):
        self.clear_instructions = dict(clear_instructions)
        self.candidates: List[str] = []
        self.meta_task: Optional[str] = None

    def reset(
        self,
        candidates: Sequence[str],
        *,
        meta_task: Optional[str] = None,
        environment_description: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> None:
        self.candidates = list(candidates)
        if not self.candidates:
            raise ValueError("T1 planners require at least one candidate task")
        unknown = set(self.candidates) - set(self.clear_instructions)
        if unknown:
            raise ValueError(f"Missing clear instructions for tasks: {sorted(unknown)}")
        self.meta_task = meta_task

    def _decision(self, task: str, **metadata: Any) -> PlannerDecision:
        return PlannerDecision(
            selected_task=task,
            instruction=self.clear_instructions[task],
            metadata=metadata,
        )

    def begin_trial(self, feedback: Any, history: Sequence[Mapping[str, Any]]) -> PlannerDecision:
        raise NotImplementedError

    def observe_trial(self, outcome: Mapping[str, Any]) -> None:
        pass


class RandomPlanner(BasePlanner):
    """Uniform independent selection on every trial, ignoring outcome history."""

    name = "random"

    def __init__(self, clear_instructions: Mapping[str, str], seed: int):
        super().__init__(clear_instructions)
        self.rng = random.Random(seed)
        self.positive_task: Optional[str] = None

    def reset(self, candidates: Sequence[str], **kwargs: Any) -> None:
        super().reset(candidates, **kwargs)
        self.observed_negative: List[str] = []

    def begin_trial(self, feedback: Any, history: Sequence[Mapping[str, Any]]) -> PlannerDecision:
        task = self.rng.choice(self.candidates)
        return self._decision(
            task,
            phase="uniform_independent",
            known_negative_before=task in self.observed_negative,
        )

    def observe_trial(self, outcome: Mapping[str, Any]) -> None:
        selected = outcome.get("selected_task")
        if (
            outcome.get("reward", 0) <= 0
            and outcome.get("selected_task_completed")
            and selected in self.candidates
            and selected not in self.observed_negative
        ):
            # Retained only to report repetition. It never affects selection.
            self.observed_negative.append(str(selected))


class NoRepeatPlanner(BasePlanner):
    """Enumerate in manifest order, then exploit a discovered positive task."""

    name = "no_repeat"

    def reset(self, candidates: Sequence[str], **kwargs: Any) -> None:
        super().reset(candidates, **kwargs)
        self.confirmed_negative: List[str] = []
        self.attempt_counts = {task: 0 for task in self.candidates}
        self.positive_task: Optional[str] = None

    def begin_trial(self, feedback: Any, history: Sequence[Mapping[str, Any]]) -> PlannerDecision:
        if self.positive_task is not None:
            return self._decision(self.positive_task, phase="exploit")
        feasible = [task for task in self.candidates if task not in self.confirmed_negative]
        if feasible:
            task = min(
                feasible,
                key=lambda item: (self.attempt_counts[item], self.candidates.index(item)),
            )
            phase = "enumerate"
        else:
            task = self.candidates[0]
            phase = "exhausted_fallback"
        self.attempt_counts[task] += 1
        return self._decision(
            task,
            phase=phase,
            confirmed_negative_before=list(self.confirmed_negative),
            known_negative_before=task in self.confirmed_negative,
        )

    def observe_trial(self, outcome: Mapping[str, Any]) -> None:
        selected = outcome.get("selected_task")
        if outcome.get("reward", 0) > 0 and selected in self.candidates:
            self.positive_task = str(selected)
        elif (
            outcome.get("selected_task_completed")
            and selected in self.candidates
            and selected not in self.confirmed_negative
        ):
            self.confirmed_negative.append(str(selected))


class RuleBasedPlanner(BasePlanner):
    """Eliminate completed negative tasks and exploit completed positives."""

    name = "rule_based"

    def reset(self, candidates: Sequence[str], **kwargs: Any) -> None:
        super().reset(candidates, **kwargs)
        self.positive: List[str] = []
        self.negative: List[str] = []
        self.attempt_counts = {task: 0 for task in self.candidates}

    def begin_trial(self, feedback: Any, history: Sequence[Mapping[str, Any]]) -> PlannerDecision:
        if self.positive:
            task = self.positive[0]
            phase = "exploit"
        else:
            unknown = [task for task in self.candidates if task not in self.negative]
            if unknown:
                task = min(unknown, key=lambda item: (self.attempt_counts[item], self.candidates.index(item)))
                phase = "eliminate"
            else:
                task = min(self.candidates, key=lambda item: (self.attempt_counts[item], self.candidates.index(item)))
                phase = "exhausted_fallback"
        known_negative_before = task in self.negative
        self.attempt_counts[task] += 1
        return self._decision(
            task,
            phase=phase,
            positive_before=list(self.positive),
            negative_before=list(self.negative),
            known_negative_before=known_negative_before,
        )

    def observe_trial(self, outcome: Mapping[str, Any]) -> None:
        completed = [task for task in outcome.get("completed_tasks", []) if task in self.candidates]
        if outcome.get("reward", 0) > 0:
            selected = outcome.get("selected_task")
            positive = selected if selected in completed else next(iter(completed), None)
            if positive is not None and positive not in self.positive:
                self.positive.append(positive)
            return
        for task in completed:
            if task not in self.negative:
                self.negative.append(task)


class OraclePlanner(BasePlanner):
    name = "oracle"

    def begin_trial(self, feedback: Any, history: Sequence[Mapping[str, Any]]) -> PlannerDecision:
        if self.meta_task not in self.candidates:
            raise ValueError(f"Oracle meta-task {self.meta_task!r} is not feasible")
        return self._decision(str(self.meta_task), phase="oracle")


class GemmaPlanner(BasePlanner):
    name = "gemma"

    def __init__(self, clear_instructions: Mapping[str, str], reasoning_llm: Any):
        super().__init__(clear_instructions)
        self.reasoning_llm = reasoning_llm
        self.messages: List[Mapping[str, str]] = []
        self.call_count = 0

    def reset(
        self,
        candidates: Sequence[str],
        *,
        meta_task: Optional[str] = None,
        environment_description: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> None:
        # meta_task is accepted for a common interface but never added to the
        # Gemma prompt.
        super().reset(candidates, meta_task=None, environment_description=environment_description)
        self.messages = copy.deepcopy(list(environment_description or []))
        self.call_count = 0

    def begin_trial(self, feedback: Any, history: Sequence[Mapping[str, Any]]) -> PlannerDecision:
        if feedback is not None:
            if isinstance(feedback, Mapping):
                self.messages.append(dict(feedback))
            else:
                self.messages.append({"role": "user", "content": str(feedback)})
        instruction, raw_output = self.reasoning_llm.get_instruction(self.messages)
        self.call_count += 1
        if isinstance(raw_output, Mapping):
            self.messages.append(dict(raw_output))
        else:
            self.messages.append({"role": "assistant", "content": str(raw_output)})
        selected = instruction_to_task(instruction, self.candidates, self.clear_instructions)
        return PlannerDecision(
            selected_task=selected,
            instruction=str(instruction).strip(),
            valid=selected is not None,
            raw_output=raw_output,
            metadata={"phase": "llm", "llm_call_index": self.call_count},
        )


def build_planner(
    name: str,
    clear_instructions: Mapping[str, str],
    *,
    seed: int,
    reasoning_llm: Any = None,
) -> BasePlanner:
    if name == "random":
        return RandomPlanner(clear_instructions, seed)
    if name == "no_repeat":
        return NoRepeatPlanner(clear_instructions)
    if name == "rule_based":
        return RuleBasedPlanner(clear_instructions)
    if name == "oracle":
        return OraclePlanner(clear_instructions)
    if name == "gemma":
        if reasoning_llm is None:
            raise ValueError("Gemma planner requires a reasoning LLM")
        return GemmaPlanner(clear_instructions, reasoning_llm)
    raise ValueError(f"Unknown T1 planner: {name}")
