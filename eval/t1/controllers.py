"""Common controller adapters for direct policies and planner/executor systems."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from eval.t1.planners import BasePlanner, PlannerDecision


def _synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass


@dataclass
class TimedAction:
    action: Any
    latency_seconds: float


class DirectController:
    """A student or untuned executor consuming ambiguous feedback directly."""

    def __init__(self, policy: Any, name: str):
        self.policy = policy
        self.name = name
        self.goal: Optional[str] = None

    def reset(self, episode: Mapping[str, Any], environment_description: Any = None) -> None:
        self.policy.reset()
        self.goal = None

    def begin_trial(self, feedback: Any, history: Sequence[Mapping[str, Any]]) -> PlannerDecision:
        self.goal = str(feedback)
        return PlannerDecision(
            selected_task=None,
            instruction=self.goal,
            valid=True,
            metadata={"phase": "direct_ambiguous_feedback", "planner_applicable": False},
        )

    def act(self, obs: Mapping[str, Any]) -> TimedAction:
        _synchronize_cuda()
        started = time.perf_counter()
        action = self.policy.step(obs, self.goal)
        _synchronize_cuda()
        return TimedAction(action=action, latency_seconds=time.perf_counter() - started)

    def observe_trial(self, outcome: Mapping[str, Any]) -> None:
        pass

    @property
    def llm_calls(self) -> int:
        return 0


class PlannerExecutorController:
    """A high-level planner controlling one shared clear-instruction executor."""

    def __init__(self, policy: Any, planner: BasePlanner):
        self.policy = policy
        self.planner = planner
        self.name = planner.name
        self.decision: Optional[PlannerDecision] = None

    def reset(self, episode: Mapping[str, Any], environment_description: Any = None) -> None:
        self.policy.reset()
        self.planner.reset(
            episode["candidate_tasks"],
            meta_task=episode.get("meta_task"),
            environment_description=environment_description,
        )
        self.decision = None

    def begin_trial(self, feedback: Any, history: Sequence[Mapping[str, Any]]) -> PlannerDecision:
        _synchronize_cuda()
        started = time.perf_counter()
        decision = self.planner.begin_trial(feedback, history)
        _synchronize_cuda()
        decision.metadata["planner_latency_seconds"] = time.perf_counter() - started
        self.decision = decision
        return decision

    def act(self, obs: Mapping[str, Any]) -> TimedAction:
        if self.decision is None:
            raise RuntimeError("begin_trial must be called before act")
        _synchronize_cuda()
        started = time.perf_counter()
        action = self.policy.step(obs, self.decision.instruction)
        _synchronize_cuda()
        return TimedAction(action=action, latency_seconds=time.perf_counter() - started)

    def observe_trial(self, outcome: Mapping[str, Any]) -> None:
        self.planner.observe_trial(outcome)

    @property
    def llm_calls(self) -> int:
        return int(getattr(self.planner, "call_count", 0))


def action_to_list(action: Any) -> list:
    return np.asarray(action, dtype=np.float32).reshape(-1).tolist()

