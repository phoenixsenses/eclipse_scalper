from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

LatencyMode = Literal["fixed", "normal", "empirical"]


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x:  # NaN
            return float(default)
        return x
    except Exception:
        return float(default)


def _clip_non_negative(x: float) -> float:
    return max(0.0, float(x))


def _u01(seed: int, key: str) -> float:
    msg = f"{int(seed)}|{str(key)}".encode("utf-8", errors="ignore")
    h = hashlib.sha256(msg).digest()
    n = int.from_bytes(h[:8], byteorder="big", signed=False)
    return n / float(2**64 - 1)


@dataclass(frozen=True)
class StageLatency:
    send_ms: float = 0.0
    exchange_recv_ms: float = 0.0
    book_effective_ms: float = 0.0
    ack_ms: float = 0.0
    fill_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return float(self.send_ms + self.exchange_recv_ms + self.book_effective_ms + self.ack_ms + self.fill_ms)


@dataclass(frozen=True)
class LatencyProfile:
    enabled: bool
    mode: LatencyMode
    send_ms: float
    exchange_recv_ms: float
    book_effective_ms: float
    ack_ms: float
    fill_ms: float
    jitter_ms: float
    empirical_ms: Optional[List[float]] = None
    empirical_weights: Optional[List[float]] = None


def parse_latency_profile(params: Dict[str, Any]) -> LatencyProfile:
    enabled = str(params.get("latency_enabled", "")).strip().lower() in {"1", "true", "yes", "on"}
    mode_raw = str(params.get("latency_profile", "fixed")).strip().lower()
    mode: LatencyMode = "fixed"
    if mode_raw in {"normal", "empirical"}:
        mode = mode_raw  # type: ignore[assignment]

    # Backward-compatible mapping from existing keys:
    # decision_to_ack ~= send + exchange_recv + ack
    decision_to_ack_ms = _clip_non_negative(_to_float(params.get("latency_decision_to_ack_ms"), 0.0))
    queue_entry_ms = _clip_non_negative(_to_float(params.get("latency_queue_entry_ms"), 0.0))
    feed_lag_ms = _clip_non_negative(_to_float(params.get("latency_feed_lag_ms"), 0.0))

    send_ms = _clip_non_negative(_to_float(params.get("latency_send_ms"), decision_to_ack_ms * 0.30))
    exchange_recv_ms = _clip_non_negative(_to_float(params.get("latency_exchange_recv_ms"), decision_to_ack_ms * 0.40))
    ack_ms = _clip_non_negative(_to_float(params.get("latency_ack_ms"), max(0.0, decision_to_ack_ms - send_ms - exchange_recv_ms)))
    book_effective_ms = _clip_non_negative(_to_float(params.get("latency_book_effective_ms"), queue_entry_ms))
    fill_ms = _clip_non_negative(_to_float(params.get("latency_fill_ms"), feed_lag_ms))
    jitter_ms = _clip_non_negative(_to_float(params.get("latency_jitter_ms"), _to_float(params.get("latency_decision_to_ack_jitter_ms"), 0.0)))

    empirical_ms: Optional[List[float]] = None
    empirical_weights: Optional[List[float]] = None
    if mode == "empirical":
        buckets = params.get("latency_empirical_ms")
        weights = params.get("latency_empirical_weights")
        if isinstance(buckets, (list, tuple)) and buckets:
            empirical_ms = [_clip_non_negative(_to_float(x, 0.0)) for x in buckets]
        if isinstance(weights, (list, tuple)) and weights:
            empirical_weights = [_clip_non_negative(_to_float(x, 0.0)) for x in weights]

    return LatencyProfile(
        enabled=bool(enabled),
        mode=mode,
        send_ms=float(send_ms),
        exchange_recv_ms=float(exchange_recv_ms),
        book_effective_ms=float(book_effective_ms),
        ack_ms=float(ack_ms),
        fill_ms=float(fill_ms),
        jitter_ms=float(jitter_ms),
        empirical_ms=empirical_ms,
        empirical_weights=empirical_weights,
    )


def _draw_stage(mean_ms: float, jitter_ms: float, *, seed: int, key: str, mode: LatencyMode) -> float:
    m = _clip_non_negative(mean_ms)
    if mode == "fixed" or jitter_ms <= 0.0:
        return m
    u = _u01(seed, key)
    # bounded symmetric draw, deterministic
    return _clip_non_negative(m + ((2.0 * u - 1.0) * _clip_non_negative(jitter_ms)))


def _draw_empirical(*, seed: int, key: str, buckets: List[float], weights: Optional[List[float]]) -> float:
    if not buckets:
        return 0.0
    if not weights or len(weights) != len(buckets):
        weights = [1.0] * len(buckets)
    total = sum(max(0.0, float(w)) for w in weights)
    if total <= 0.0:
        return float(buckets[0])
    u = _u01(seed, key)
    cum = 0.0
    for b, w in zip(buckets, weights):
        cum += max(0.0, float(w)) / total
        if u <= cum:
            return _clip_non_negative(float(b))
    return _clip_non_negative(float(buckets[-1]))


def sample_stage_latency(profile: LatencyProfile, *, seed: int, event_id: str) -> StageLatency:
    if not profile.enabled:
        return StageLatency()
    if profile.mode == "empirical" and profile.empirical_ms:
        tot = _draw_empirical(
            seed=seed,
            key=f"{event_id}|lat_emp_total",
            buckets=list(profile.empirical_ms),
            weights=profile.empirical_weights,
        )
        # Allocate total over stages by configured proportions.
        base_sum = max(1e-9, profile.send_ms + profile.exchange_recv_ms + profile.book_effective_ms + profile.ack_ms + profile.fill_ms)
        s = tot * (profile.send_ms / base_sum)
        ex = tot * (profile.exchange_recv_ms / base_sum)
        be = tot * (profile.book_effective_ms / base_sum)
        a = tot * (profile.ack_ms / base_sum)
        f = tot * (profile.fill_ms / base_sum)
        return StageLatency(send_ms=s, exchange_recv_ms=ex, book_effective_ms=be, ack_ms=a, fill_ms=f)
    return StageLatency(
        send_ms=_draw_stage(profile.send_ms, profile.jitter_ms, seed=seed, key=f"{event_id}|lat_send", mode=profile.mode),
        exchange_recv_ms=_draw_stage(
            profile.exchange_recv_ms,
            profile.jitter_ms,
            seed=seed,
            key=f"{event_id}|lat_exchange_recv",
            mode=profile.mode,
        ),
        book_effective_ms=_draw_stage(
            profile.book_effective_ms,
            profile.jitter_ms,
            seed=seed,
            key=f"{event_id}|lat_book_effective",
            mode=profile.mode,
        ),
        ack_ms=_draw_stage(profile.ack_ms, profile.jitter_ms, seed=seed, key=f"{event_id}|lat_ack", mode=profile.mode),
        fill_ms=_draw_stage(profile.fill_ms, profile.jitter_ms, seed=seed, key=f"{event_id}|lat_fill", mode=profile.mode),
    )


def latency_bars(total_ms: float, bucket_sec: float) -> int:
    bsec = max(1e-9, float(bucket_sec))
    return int(max(0, round(float(total_ms) / (1000.0 * bsec))))


def build_latency_timeline(*, decision_ts_ms: int, stage: StageLatency) -> Dict[str, int]:
    d = int(max(0, int(decision_ts_ms)))
    send_ts = d + int(round(stage.send_ms))
    exch_ts = send_ts + int(round(stage.exchange_recv_ms))
    book_ts = exch_ts + int(round(stage.book_effective_ms))
    ack_ts = book_ts + int(round(stage.ack_ms))
    fill_ts = ack_ts + int(round(stage.fill_ms))
    return {
        "decision_ts": d,
        "send_ts": send_ts,
        "exchange_recv_ts": exch_ts,
        "book_effective_ts": book_ts,
        "ack_ts": ack_ts,
        "fill_ts": fill_ts,
    }


def stage_to_legacy_components(stage: StageLatency) -> Dict[str, float]:
    # Legacy compatibility fields expected by existing tooling.
    decision_to_ack_ms = float(stage.send_ms + stage.exchange_recv_ms + stage.ack_ms)
    return {
        "decision_to_ack_ms": decision_to_ack_ms,
        "queue_entry_ms": float(stage.book_effective_ms),
        "feed_lag_ms": float(stage.fill_ms),
        "total_ms": float(stage.total_ms),
    }

