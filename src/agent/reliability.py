"""LLM/도구 호출의 재시도, 오류 분류, 폴백 추적 공통 계층."""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.35
    max_delay_seconds: float = 2.0
    jitter_ratio: float = 0.15


@dataclass
class AttemptEvent:
    operation: str
    attempt: int
    ok: bool
    retryable: bool = False
    error_type: str | None = None
    error: str | None = None
    delay_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RetryExhaustedError(RuntimeError):
    def __init__(self, operation: str, cause: Exception, events: list[AttemptEvent]):
        super().__init__(f"{operation} failed after {len(events)} attempt(s): {cause}")
        self.operation = operation
        self.cause = cause
        self.events = events


def is_transient_error(exc: Exception) -> bool:
    """HTTP/SDK/네트워크 예외 중 재시도 가치가 있는 오류만 분류한다."""
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    return any(token in name for token in (
        "timeout", "connection", "ratelimit", "internalserver", "serviceunavailable",
    ))


def call_with_retry(
    operation: str,
    fn: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    retry_if: Callable[[Exception], bool] = is_transient_error,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, list[AttemptEvent]]:
    """지수 백오프로 호출하고 모든 시도를 구조화 로그로 반환한다."""
    policy = policy or RetryPolicy()
    events: list[AttemptEvent] = []
    last: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            value = fn()
            events.append(AttemptEvent(operation, attempt, True))
            return value, events
        except Exception as exc:  # 호출 경계에서 오류를 정규화한다.
            last = exc
            retryable = bool(retry_if(exc))
            should_retry = retryable and attempt < policy.max_attempts
            delay = 0.0
            if should_retry:
                delay = min(policy.max_delay_seconds,
                            policy.base_delay_seconds * (2 ** (attempt - 1)))
                if policy.jitter_ratio:
                    delay *= 1 + random.uniform(-policy.jitter_ratio, policy.jitter_ratio)
                delay = max(0.0, delay)
            events.append(AttemptEvent(
                operation, attempt, False, retryable,
                type(exc).__name__, str(exc)[:500], round(delay, 3),
            ))
            if not should_retry:
                break
            sleep(delay)
    assert last is not None
    raise RetryExhaustedError(operation, last, events) from last
