#!/usr/bin/env python3
"""Run staged gateway stress test and collect resource samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


STAGE_START_RE = re.compile(r"start stage camera_count=(\d+)")
STAGE_DONE_RE = re.compile(r"stage cameras=(\d+)\b")
NPU_LINE_RE = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s+\|\s+([0-9a-fA-F:.]+)\s+\|\s+(\d+)\s+(\d+)\s*/\s*(\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--stress-script", default="stress_multi_docker_5fps.py")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--camera-ids", required=True)
    parser.add_argument("--host", default="http://192.168.100.64")
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--stage-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--start-cameras", type=int, default=1)
    parser.add_argument("--max-cameras", type=int, default=15)
    parser.add_argument("--request-workers", type=int, default=256)
    parser.add_argument("--max-in-flight", type=int, default=2000)
    parser.add_argument("--video-path", default="video/test1.mp4")
    parser.add_argument("--npu-ids", default="0,1")
    parser.add_argument("--save-records", action="store_true")
    parser.add_argument("--disable-face-recognition", action="store_true")
    parser.add_argument("--disable-behavior-detection", action="store_true")
    parser.add_argument("--disable-spatial-positioning", action="store_true")
    parser.add_argument("--disable-target-tracking", action="store_true")
    return parser.parse_args()


def read_proc_stat() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + values[4]
    total = sum(values)
    return total, idle


def read_meminfo() -> dict[str, int]:
    data: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        data[key] = int(raw.strip().split()[0])
    return data


def cpu_percent(prev: tuple[int, int] | None, cur: tuple[int, int]) -> float | None:
    if prev is None:
        return None
    total_delta = cur[0] - prev[0]
    idle_delta = cur[1] - prev[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))


def parse_npu_smi(text: str, npu_ids: set[int]) -> dict[int, dict[str, int]]:
    devices: dict[int, dict[str, int]] = {}
    for line in text.splitlines():
        match = NPU_LINE_RE.match(line)
        if not match:
            continue
        npu_id = int(match.group(1))
        chip_id = int(match.group(2))
        if npu_id not in npu_ids or chip_id != npu_id:
            continue
        devices[npu_id] = {
            "aicore_percent": int(match.group(4)),
            "memory_used_mb": int(match.group(5)),
            "memory_total_mb": int(match.group(6)),
        }
    return devices


def sample_npu(npu_ids: set[int]) -> dict[int, dict[str, int]]:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    except Exception:
        return {}
    if result.returncode != 0:
        return {}
    return parse_npu_smi(result.stdout, npu_ids)


class Monitor:
    def __init__(self, sample_path: Path, interval: float, npu_ids: set[int]):
        self.sample_path = sample_path
        self.interval = interval
        self.npu_ids = npu_ids
        self.current_stage: int | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.prev_cpu: tuple[int, int] | None = None

    def set_stage(self, stage: int | None) -> None:
        with self.lock:
            self.current_stage = stage

    def get_stage(self) -> int | None:
        with self.lock:
            return self.current_stage

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.interval * 2.0))

    def run(self) -> None:
        self.sample_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp",
            "elapsed_sec",
            "stage_camera_count",
            "cpu_percent",
            "mem_used_mb",
            "mem_total_mb",
            "mem_percent",
            "npu0_aicore_percent",
            "npu0_mem_used_mb",
            "npu0_mem_total_mb",
            "npu1_aicore_percent",
            "npu1_mem_used_mb",
            "npu1_mem_total_mb",
        ]
        started_at = time.time()
        with self.sample_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            while not self.stop_event.is_set():
                now = time.time()
                cur_cpu = read_proc_stat()
                cpu = cpu_percent(self.prev_cpu, cur_cpu)
                self.prev_cpu = cur_cpu
                mem = read_meminfo()
                mem_total = mem.get("MemTotal", 0) / 1024.0
                mem_available = mem.get("MemAvailable", 0) / 1024.0
                mem_used = max(0.0, mem_total - mem_available)
                npu = sample_npu(self.npu_ids)
                row: dict[str, Any] = {
                    "timestamp": datetime.fromtimestamp(now).isoformat(),
                    "elapsed_sec": round(now - started_at, 3),
                    "stage_camera_count": self.get_stage() or "",
                    "cpu_percent": "" if cpu is None else round(cpu, 2),
                    "mem_used_mb": round(mem_used, 1),
                    "mem_total_mb": round(mem_total, 1),
                    "mem_percent": round(mem_used * 100.0 / mem_total, 2) if mem_total else "",
                    "npu0_aicore_percent": "",
                    "npu0_mem_used_mb": "",
                    "npu0_mem_total_mb": "",
                    "npu1_aicore_percent": "",
                    "npu1_mem_used_mb": "",
                    "npu1_mem_total_mb": "",
                }
                for npu_id in self.npu_ids:
                    data = npu.get(npu_id)
                    if not data:
                        continue
                    prefix = f"npu{npu_id}"
                    row[f"{prefix}_aicore_percent"] = data["aicore_percent"]
                    row[f"{prefix}_mem_used_mb"] = data["memory_used_mb"]
                    row[f"{prefix}_mem_total_mb"] = data["memory_total_mb"]
                writer.writerow(row)
                handle.flush()
                self.stop_event.wait(self.interval)


def build_stress_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python_bin,
        "-u",
        args.stress_script,
        "--docker-count",
        "1",
        "--host",
        args.host,
        "--start-port",
        str(args.port),
        "--camera-ids",
        args.camera_ids,
        "--target-fps",
        str(args.target_fps),
        "--start-cameras",
        str(args.start_cameras),
        "--max-cameras",
        str(args.max_cameras),
        "--step-cameras",
        "1",
        "--stage-seconds",
        str(args.stage_seconds),
        "--warmup-seconds",
        str(args.warmup_seconds),
        "--timeout",
        str(args.timeout),
        "--request-workers",
        str(args.request_workers),
        "--max-in-flight",
        str(args.max_in_flight),
        "--video-path",
        args.video_path,
        "--output-dir",
        args.output_dir,
        "--run-name",
        args.run_name,
    ]
    if args.save_records:
        cmd.append("--save-records")
    if args.disable_face_recognition:
        cmd.append("--disable-face-recognition")
    if args.disable_behavior_detection:
        cmd.append("--disable-behavior-detection")
    if args.disable_spatial_positioning:
        cmd.append("--disable-spatial-positioning")
    if args.disable_target_tracking:
        cmd.append("--disable-target-tracking")
    return cmd


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        try:
            values.append(float(raw))
        except ValueError:
            pass
    return values


def avg(rows: list[dict[str, str]], key: str) -> float | None:
    values = numeric_values(rows, key)
    return statistics.mean(values) if values else None


def max_value(rows: list[dict[str, str]], key: str) -> float | None:
    values = numeric_values(rows, key)
    return max(values) if values else None


def aggregate_resources(sample_path: Path) -> dict[str, Any]:
    with sample_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        raw_stage = row.get("stage_camera_count") or ""
        if not raw_stage:
            continue
        grouped.setdefault(int(raw_stage), []).append(row)

    stages: dict[str, Any] = {}
    for stage, stage_rows in sorted(grouped.items()):
        stages[str(stage)] = {
            "sample_count": len(stage_rows),
            "cpu_avg_percent": avg(stage_rows, "cpu_percent"),
            "cpu_max_percent": max_value(stage_rows, "cpu_percent"),
            "mem_avg_mb": avg(stage_rows, "mem_used_mb"),
            "mem_max_mb": max_value(stage_rows, "mem_used_mb"),
            "mem_avg_percent": avg(stage_rows, "mem_percent"),
            "npu0_aicore_avg_percent": avg(stage_rows, "npu0_aicore_percent"),
            "npu0_aicore_max_percent": max_value(stage_rows, "npu0_aicore_percent"),
            "npu0_mem_avg_mb": avg(stage_rows, "npu0_mem_used_mb"),
            "npu0_mem_max_mb": max_value(stage_rows, "npu0_mem_used_mb"),
            "npu1_aicore_avg_percent": avg(stage_rows, "npu1_aicore_percent"),
            "npu1_aicore_max_percent": max_value(stage_rows, "npu1_aicore_percent"),
            "npu1_mem_avg_mb": avg(stage_rows, "npu1_mem_used_mb"),
            "npu1_mem_max_mb": max_value(stage_rows, "npu1_mem_used_mb"),
        }
    return {"stages": stages, "total_samples": len(rows)}


def main() -> int:
    args = parse_args()
    if not args.run_name:
        args.run_name = datetime.now().strftime("gateway_5fps_mon_%Y%m%d_%H%M%S")

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / "resource_samples.csv"
    stdout_path = run_dir / "stress_stdout.log"
    monitor_summary_path = run_dir / "monitor_summary.json"
    command_path = run_dir / "command.json"

    npu_ids = {int(item.strip()) for item in args.npu_ids.split(",") if item.strip()}
    cmd = build_stress_command(args)
    command_path.write_text(json.dumps(cmd, indent=2, ensure_ascii=False), encoding="utf-8")

    monitor = Monitor(samples_path, args.sample_interval, npu_ids)
    monitor.start()
    process: subprocess.Popen[str] | None = None
    return_code = 1
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle:
            process = subprocess.Popen(
                cmd,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert process.stdout is not None
            for line in process.stdout:
                stdout_handle.write(line)
                stdout_handle.flush()
                print(line, end="", flush=True)
                stage_start = STAGE_START_RE.search(line)
                if stage_start:
                    monitor.set_stage(int(stage_start.group(1)))
                stage_done = STAGE_DONE_RE.search(line)
                if stage_done:
                    monitor.set_stage(None)
            return_code = process.wait()
    except KeyboardInterrupt:
        if process and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        monitor.stop()

    resource_summary = aggregate_resources(samples_path)
    stress_summary_path = run_dir / "summary.json"
    stress_summary: dict[str, Any] | None = None
    if stress_summary_path.exists():
        stress_summary = json.loads(stress_summary_path.read_text(encoding="utf-8"))

    combined = {
        "return_code": return_code,
        "run_dir": str(run_dir),
        "samples_path": str(samples_path),
        "stress_stdout_path": str(stdout_path),
        "stress_summary_path": str(stress_summary_path),
        "resource_summary": resource_summary,
        "stress_summary": stress_summary,
    }
    monitor_summary_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"monitor_summary={monitor_summary_path}")
    return return_code


if __name__ == "__main__":
    sys.exit(main())
