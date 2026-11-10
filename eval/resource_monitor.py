#!/usr/bin/env python3
"""Sample CPU, host-memory, and NVIDIA GPU resources for one process tree.

The monitor deliberately uses only the Python standard library and nvidia-smi so
that it can run in the same environments as both legacy and T1 evaluation.
Raw samples and an aggregate summary are written independently from task metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
MIB = 1024 * 1024
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
LOGICAL_CPU_COUNT = os.cpu_count() or 1
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"mean": None, "p50": None, "p95": None, "peak": None}
    return {
        "mean": statistics.fmean(present),
        "p50": percentile(present, 0.50),
        "p95": percentile(present, 0.95),
        "peak": max(present),
    }


def parse_proc_stat(pid: int) -> Optional[Dict[str, int]]:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    close_paren = text.rfind(")")
    if close_paren < 0:
        return None
    fields = text[close_paren + 2 :].split()
    if len(fields) < 22:
        return None
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "start_ticks": int(fields[19]),
        "rss_bytes": max(0, int(fields[21])) * PAGE_SIZE,
    }


def process_tree(root_pid: int) -> Tuple[Optional[Dict[str, int]], List[Dict[str, int]]]:
    records: Dict[int, Dict[str, int]] = {}
    children: Dict[int, List[int]] = defaultdict(list)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        record = parse_proc_stat(int(entry.name))
        if record is None:
            continue
        records[record["pid"]] = record
        children[record["ppid"]].append(record["pid"])
    root = records.get(root_pid)
    if root is None:
        return None, []
    descendants: List[Dict[str, int]] = []
    pending = [root_pid]
    visited = set()
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        record = records.get(pid)
        if record is not None:
            descendants.append(record)
            pending.extend(children.get(pid, []))
    return root, descendants


def total_cpu_ticks() -> int:
    first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    fields = first_line.split()
    if not fields or fields[0] != "cpu":
        raise RuntimeError("Unexpected /proc/stat CPU line")
    return sum(int(value) for value in fields[1:])


def run_nvidia_smi(query: str) -> Tuple[List[List[str]], Optional[str]]:
    command = [
        "nvidia-smi",
        f"--query-{query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return [], f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return [], f"nvidia-smi exit {completed.returncode}: {detail}"
    return [row for row in csv.reader(completed.stdout.splitlines()) if row], None


def optional_float(value: str) -> Optional[float]:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def gpu_sample(target_pids: set[int]) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    gpu_rows, gpu_error = run_nvidia_smi(
        "gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    )
    if gpu_error is not None:
        return [], [gpu_error]

    process_rows, process_error = run_nvidia_smi(
        "compute-apps=pid,gpu_uuid,used_gpu_memory"
    )
    if process_error is not None:
        errors.append(process_error)

    target_memory_mib: Dict[str, float] = defaultdict(float)
    for row in process_rows:
        if len(row) < 3:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue
        used_mib = optional_float(row[2])
        if pid in target_pids and used_mib is not None:
            target_memory_mib[row[1].strip()] += used_mib

    records: List[Dict[str, Any]] = []
    for row in gpu_rows:
        if len(row) < 8:
            continue
        uuid = row[1].strip()
        records.append(
            {
                "index": int(row[0].strip()),
                "uuid": uuid,
                "name": row[2].strip(),
                "gpu_utilization_percent": optional_float(row[3]),
                "memory_controller_utilization_percent": optional_float(row[4]),
                "device_memory_used_bytes": (
                    int(optional_float(row[5]) * MIB)
                    if optional_float(row[5]) is not None
                    else None
                ),
                "device_memory_total_bytes": (
                    int(optional_float(row[6]) * MIB)
                    if optional_float(row[6]) is not None
                    else None
                ),
                "power_draw_watts": optional_float(row[7]),
                "target_process_memory_bytes": int(target_memory_mib[uuid] * MIB),
            }
        )
    return records, errors


def handle_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def summarize(
    *,
    args: argparse.Namespace,
    started_at_utc: str,
    started_monotonic: float,
    completed_at_utc: str,
    completed_monotonic: float,
    samples: Sequence[Dict[str, Any]],
    errors: Sequence[str],
    termination_reason: str,
) -> Dict[str, Any]:
    cpu_percent = [sample.get("process_cpu_percent") for sample in samples]
    cpu_capacity = [sample.get("host_capacity_percent") for sample in samples]
    rss_values = [sample.get("process_tree_rss_bytes") for sample in samples]
    process_counts = [sample.get("process_count") for sample in samples]
    gpu_records = [gpu for sample in samples for gpu in sample.get("gpus", [])]
    observed_gpu_indices = sorted({gpu["index"] for gpu in gpu_records})
    first_cpu_seconds = samples[0]["process_tree_cpu_seconds"] if samples else None
    last_cpu_seconds = samples[-1]["process_tree_cpu_seconds"] if samples else None

    return {
        "schema_version": SCHEMA_VERSION,
        "label": args.label,
        "root_pid": args.pid,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "duration_seconds": max(0.0, completed_monotonic - started_monotonic),
        "sample_interval_seconds": args.interval,
        "sample_count": len(samples),
        "termination_reason": termination_reason,
        "cpu": {
            "logical_cpu_count": LOGICAL_CPU_COUNT,
            "process_cpu_percent_one_core_basis": stats(cpu_percent),
            "host_capacity_percent": stats(cpu_capacity),
            "process_tree_cpu_time_seconds_observed": (
                max(0.0, last_cpu_seconds - first_cpu_seconds)
                if first_cpu_seconds is not None and last_cpu_seconds is not None
                else None
            ),
            "process_tree_rss_bytes": stats(rss_values),
            "process_count": stats(process_counts),
        },
        "gpu": {
            "observed_gpu_indices": observed_gpu_indices,
            "target_process_memory_bytes": stats(
                gpu.get("target_process_memory_bytes") for gpu in gpu_records
            ),
            "device_memory_used_bytes": stats(
                gpu.get("device_memory_used_bytes") for gpu in gpu_records
            ),
            "gpu_utilization_percent": stats(
                gpu.get("gpu_utilization_percent") for gpu in gpu_records
            ),
            "memory_controller_utilization_percent": stats(
                gpu.get("memory_controller_utilization_percent") for gpu in gpu_records
            ),
            "power_draw_watts": stats(gpu.get("power_draw_watts") for gpu in gpu_records),
        },
        "sampling_errors": dict(Counter(errors)),
        "notes": {
            "process_cpu_percent_one_core_basis": (
                "100% equals one fully occupied logical CPU and values may exceed 100%."
            ),
            "host_capacity_percent": (
                "Target process-tree CPU time divided by total host CPU time; range 0-100%."
            ),
            "gpu_utilization_percent": (
                "Device-wide nvidia-smi utilization. The launchers serialize compute jobs with a global GPU lock."
            ),
            "target_process_memory_bytes": (
                "nvidia-smi compute-process memory summed over the monitored process tree."
            ),
        },
    }


def monitor(args: argparse.Namespace) -> int:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    started_at_utc = utc_now()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + args.startup_timeout
    initial_root: Optional[Dict[str, int]] = None
    while time.monotonic() <= deadline and not STOP_REQUESTED:
        initial_root = parse_proc_stat(args.pid)
        if initial_root is not None:
            break
        time.sleep(0.05)

    samples: List[Dict[str, Any]] = []
    errors: List[str] = []
    termination_reason = "root_process_not_found"
    previous_monotonic: Optional[float] = None
    previous_process_ticks: Optional[int] = None
    previous_total_ticks: Optional[int] = None

    with args.output_jsonl.open("w", encoding="utf-8", buffering=1) as raw_output:
        while initial_root is not None and not STOP_REQUESTED:
            sampled_monotonic = time.monotonic()
            root, tree = process_tree(args.pid)
            if root is None or root["start_ticks"] != initial_root["start_ticks"]:
                termination_reason = "root_process_exited"
                break

            process_ticks = sum(record["cpu_ticks"] for record in tree)
            system_ticks = total_cpu_ticks()
            elapsed = (
                sampled_monotonic - previous_monotonic
                if previous_monotonic is not None
                else None
            )
            process_delta = (
                process_ticks - previous_process_ticks
                if previous_process_ticks is not None
                else None
            )
            total_delta = (
                system_ticks - previous_total_ticks
                if previous_total_ticks is not None
                else None
            )
            process_cpu_percent = (
                process_delta / CLOCK_TICKS / elapsed * 100.0
                if process_delta is not None and elapsed and elapsed > 0
                else None
            )
            host_capacity_percent = (
                process_delta / total_delta * 100.0
                if process_delta is not None and total_delta and total_delta > 0
                else None
            )

            gpus, gpu_errors = gpu_sample({record["pid"] for record in tree})
            errors.extend(gpu_errors)
            sample = {
                "schema_version": SCHEMA_VERSION,
                "sampled_at_utc": utc_now(),
                "elapsed_seconds": sampled_monotonic - started_monotonic,
                "root_pid": args.pid,
                "process_count": len(tree),
                "process_tree_cpu_seconds": process_ticks / CLOCK_TICKS,
                "process_cpu_percent": process_cpu_percent,
                "host_capacity_percent": host_capacity_percent,
                "process_tree_rss_bytes": sum(record["rss_bytes"] for record in tree),
                "gpus": gpus,
                "sampling_errors": gpu_errors,
            }
            raw_output.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
            samples.append(sample)
            previous_monotonic = sampled_monotonic
            previous_process_ticks = process_ticks
            previous_total_ticks = system_ticks
            termination_reason = "stop_signal" if STOP_REQUESTED else "monitoring"

            sleep_until = sampled_monotonic + args.interval
            while not STOP_REQUESTED:
                remaining = sleep_until - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.1, remaining))

    completed_monotonic = time.monotonic()
    if STOP_REQUESTED:
        termination_reason = "stop_signal"
    summary = summarize(
        args=args,
        started_at_utc=started_at_utc,
        started_monotonic=started_monotonic,
        completed_at_utc=utc_now(),
        completed_monotonic=completed_monotonic,
        samples=samples,
        errors=errors,
        termination_reason=termination_reason,
    )
    atomic_json(args.summary_json, summary)
    return 0 if samples else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="root process PID")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=2.0)
    parser.add_argument("--label", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.pid <= 0:
        raise SystemExit("--pid must be positive")
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.startup_timeout < 0:
        raise SystemExit("--startup-timeout must be non-negative")
    return monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
