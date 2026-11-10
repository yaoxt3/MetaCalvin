"""Audited trial-boundary state control for T3."""

from __future__ import annotations

from typing import Any, Dict


RECURRENT_STATE_MODES = ("reset", "persistent")
QUEUE_NAMES = ("img_queue", "gripper_queue", "state_queue", "mask_queue", "text_queue")


def _heads(policy: Any) -> list[tuple[str, Any]]:
    model = policy.model
    result = [("lang_encoder.lm_head", model.lang_encoder.lm_head)]
    if bool(getattr(model, "sep_lm_head", False)):
        result.append(("lm_head", model.lm_head))
    return result


def recurrent_snapshot(policy: Any) -> Dict[str, Any]:
    heads = {}
    for name, head in _heads(policy):
        history = getattr(head, "history_memory", [])
        heads[name] = {
            "hidden_present": getattr(head, "hidden_state", None) is not None,
            "history_length": len(history),
            "rnn_out_present": getattr(head, "rnn_out", None) is not None,
        }
    return {
        "heads": heads,
        "queue_lengths": {
            name: len(getattr(policy, name))
            for name in QUEUE_NAMES
            if hasattr(policy, name)
        },
        "feature_cache_present": getattr(policy, "feature_cache", None) is not None,
        "dt_feature_cache_length": len(getattr(policy, "dt_feat_cache", [])),
    }


def _clear_input_caches(policy: Any) -> None:
    for name in QUEUE_NAMES:
        queue = getattr(policy, name, None)
        if queue is not None:
            queue.clear()
    policy.feature_cache = None
    policy.dt_feat_cache = []


def _clear_recurrent_state(policy: Any) -> None:
    for _, head in _heads(policy):
        head.hidden_state = None
        head.history_memory = []
        if hasattr(head, "rnn_out"):
            head.rnn_out = None


def prepare_trial_boundary(policy: Any, mode: str, trial_index: int) -> Dict[str, Any]:
    if mode not in RECURRENT_STATE_MODES:
        raise ValueError(f"Unknown T3 recurrent-state mode: {mode!r}")
    if trial_index <= 1:
        raise ValueError("Trial-boundary control is only valid after trial 1")
    before = recurrent_snapshot(policy)
    # Both conditions drop short sensory/text queues. This prevents the input
    # wrapper from becoming a third cross-trial memory channel.
    _clear_input_caches(policy)
    if mode == "reset":
        _clear_recurrent_state(policy)
    after = recurrent_snapshot(policy)
    if any(after["queue_lengths"].values()):
        raise RuntimeError("T3 failed to clear trial-boundary input queues")
    if mode == "reset" and any(
        value["hidden_present"] or value["history_length"]
        for value in after["heads"].values()
    ):
        raise RuntimeError("T3 failed to reset recurrent state")
    return {
        "trial_index": trial_index,
        "state_mode": mode,
        "input_queues_cleared_in_both_modes": True,
        "recurrent_reset_applied": mode == "reset",
        "before": before,
        "after": after,
    }


def metadata(mode: str) -> Dict[str, Any]:
    if mode not in RECURRENT_STATE_MODES:
        raise ValueError(mode)
    return {
        "recurrent_state": mode,
        "episode_start": "full ModelWrapper.reset in every condition",
        "trial_boundary_common": "clear wrapper sensory/text queues and feature caches",
        "trial_boundary_intervention": (
            "clear decoder hidden_state, history_memory, and rnn_out"
            if mode == "reset"
            else "preserve decoder hidden_state, history_memory, and rnn_out"
        ),
    }
