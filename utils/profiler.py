import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


PROFILE_ENABLED = _env_bool("ENABLE_PROFILING", False)
PROFILE_LOG_EVERY = max(1, _env_int("PROFILE_LOG_EVERY", 100))


class RequestProfiler:
    def __init__(self):
        self.enabled = PROFILE_ENABLED
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self._starts = {}

    def start(self, name):
        if self.enabled:
            self._starts[name] = time.perf_counter()

    def stop(self, name):
        if not self.enabled:
            return
        started_at = self._starts.pop(name, None)
        if started_at is None:
            return
        elapsed = time.perf_counter() - started_at
        self.times[name] += elapsed
        self.counts[name] += 1

    @contextmanager
    def section(self, name):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def merge(self, other):
        if not self.enabled or other is None:
            return
        for name, elapsed in other.times.items():
            self.times[name] += elapsed
        for name, count in other.counts.items():
            self.counts[name] += count


class ProfilingStats:
    def __init__(self, name, log_every=None):
        self.name = name
        self.log_every = PROFILE_LOG_EVERY if log_every is None else max(1, int(log_every))
        self._lock = threading.Lock()
        self._requests = 0
        self._total_times = defaultdict(float)
        self._total_counts = defaultdict(int)

    def record(self, profiler):
        if not PROFILE_ENABLED or profiler is None:
            return

        with self._lock:
            self._requests += 1
            for name, elapsed in profiler.times.items():
                self._total_times[name] += elapsed
            for name, count in profiler.counts.items():
                self._total_counts[name] += count

            if self._requests % self.log_every == 0:
                self._print_report_locked()

    def _print_report_locked(self):
        rows = []
        for name, total_seconds in self._total_times.items():
            count = max(1, self._total_counts.get(name, 1))
            avg_ms = total_seconds * 1000.0 / count
            per_request_ms = total_seconds * 1000.0 / max(1, self._requests)
            rows.append((name, avg_ms, per_request_ms, count))

        rows.sort(key=lambda item: item[2], reverse=True)
        parts = [
            f"{name}:avg={avg_ms:.1f}ms,req_avg={per_request_ms:.1f}ms,n={count}"
            for name, avg_ms, per_request_ms, count in rows
        ]
        print(f"[Profiler:{self.name}] requests={self._requests} " + " | ".join(parts))
