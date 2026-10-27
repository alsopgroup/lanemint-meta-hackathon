"""In-memory abuse guard for the live Muse endpoint.

Per-IP sliding windows, a global daily call/spend ceiling, and a
concurrency slot. State is per-process: it resets on redeploy and is
per-replica, which is fine for a single Railway replica demo service
but is not a distributed rate limiter.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


class LiveCallGuard:
    """Thread-safe guard consulted before every live Muse call."""

    def __init__(self, max_concurrent: int = 2):
        self._lock = threading.Lock()
        self._minute: dict[str, deque[float]] = defaultdict(deque)
        self._hour: dict[str, deque[float]] = defaultdict(deque)
        self._slots = threading.Semaphore(max_concurrent)
        self._day = self._today()
        self._daily_calls = 0
        self._daily_spend_usd = 0.0
        self._reserved_usd = 0.0

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def reset(self) -> None:
        """Test helper: clear all windows, counters, and slots."""
        with self._lock:
            self._minute.clear()
            self._hour.clear()
            self._day = self._today()
            self._daily_calls = 0
            self._daily_spend_usd = 0.0
            self._reserved_usd = 0.0
        # Drain and refill slots so a leaked acquire cannot wedge tests.
        while self._slots.acquire(blocking=False):
            pass
        for _ in range(2):
            self._slots.release()

    @staticmethod
    def _prune(window: deque[float], now: float, span: float) -> None:
        while window and now - window[0] >= span:
            window.popleft()

    def _roll_day_locked(self) -> None:
        if self._day != self._today():
            self._day = self._today()
            self._daily_calls = 0
            self._daily_spend_usd = 0.0
            self._reserved_usd = 0.0

    def acquire(self, client_ip: str, reserve_usd: float = 0.0) -> tuple[bool, int, str]:
        """Record an attempted live call, or explain why it is denied.

        Returns (allowed, retry_after_seconds, reason). Denied calls are
        not recorded in the windows. ``reserve_usd`` is the worst-case
        cost of the call being admitted: it is held against the daily
        budget until settle() converts it to actual spend, so calls
        that fail after being billed still count against the cap.
        """
        now = time.monotonic()
        per_minute = _env_int("LIVE_RATE_PER_MINUTE", 5)
        per_hour = _env_int("LIVE_RATE_PER_HOUR", 30)
        daily_max = _env_int("LIVE_DAILY_MAX_CALLS", 300)
        daily_budget = _env_float("LIVE_DAILY_BUDGET_USD", 5.0)

        with self._lock:
            self._roll_day_locked()
            minute = self._minute[client_ip]
            hour = self._hour[client_ip]
            self._prune(minute, now, 60)
            self._prune(hour, now, 3600)
            # 0 is a hard block (no live calls), not "unlimited".
            if len(minute) >= per_minute:
                retry = max(1, int(60 - (now - minute[0])) + 1) if minute else 60
                return False, retry, "rate limit: too many requests this minute"
            if len(hour) >= per_hour:
                retry = max(1, int(3600 - (now - hour[0])) + 1) if hour else 3600
                return False, retry, "rate limit: too many requests this hour"
            if self._daily_calls >= daily_max:
                return False, 3600, "daily live-call cap reached"
            if self._daily_spend_usd >= daily_budget:
                return False, 3600, "daily live-call budget reached"
            if reserve_usd > 0 and (
                self._daily_spend_usd + self._reserved_usd + reserve_usd > daily_budget
            ):
                return False, 3600, "daily live-call budget reached"
            if not self._slots.acquire(blocking=False):
                return False, 5, "too many live calls in progress"
            minute.append(now)
            hour.append(now)
            self._daily_calls += 1
            self._reserved_usd += max(0.0, reserve_usd)
            return True, 0, ""

    def release(self) -> None:
        self._slots.release()

    def settle(self, reserve_usd: float, actual_usd: float) -> None:
        """Convert a reservation into actual spend after the call ends.

        Callers pass actual=reserve for failed calls: the upstream may
        have billed the attempt even when its response was unusable,
        so failures settle conservatively at the reserved worst case.
        """
        with self._lock:
            self._roll_day_locked()
            self._reserved_usd = max(0.0, self._reserved_usd - max(0.0, reserve_usd))
            self._daily_spend_usd += max(0.0, actual_usd)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "day": self._day,
                "daily_calls": self._daily_calls,
                "daily_spend_usd": round(self._daily_spend_usd, 4),
                "reserved_usd": round(self._reserved_usd, 4),
            }


# Shared process-wide guard.
guard = LiveCallGuard()
