from __future__ import annotations

import csv
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger("alpha_gate")


@dataclass
class AlphaGateDecision:
    blocked: bool
    reason: str
    details: Dict[str, Any]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        out = float(v)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def load_alpha_metrics(path: str | Path) -> tuple[Optional[Dict[str, Any]], Optional[float]]:
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        payload = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        mtime = float(p.stat().st_mtime)
        return payload, mtime
    except Exception:
        return None, None


def load_stability_metrics(path: str | Path) -> tuple[Optional[Dict[str, Any]], Optional[float]]:
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None, None
        payload = rows[0]
        mtime = float(p.stat().st_mtime)
        return payload, mtime
    except Exception:
        return None, None


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_alpha_", dir=str(dst.parent))
    os.close(fd)
    try:
        with src.open("rb") as rf, open(tmp_name, "wb") as wf:
            wf.write(rf.read())
        os.replace(tmp_name, str(dst))
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _atomic_write_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_alpha_", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        os.replace(tmp_name, str(dst))
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _fallback_last_good_json(
    latest_reason: str,
    now_ts: float,
    max_staleness_sec: float,
    fallback_path: Path,
) -> tuple[Optional[Dict[str, Any]], Optional[float], bool, str]:
    fb_metrics, fb_mtime = load_alpha_metrics(fallback_path)
    if not isinstance(fb_metrics, dict):
        return None, None, False, latest_reason
    if fb_mtime is not None and (float(now_ts) - float(fb_mtime)) > float(max_staleness_sec):
        return None, None, False, latest_reason
    _log.info("latest metrics %s, using fallback last_good", latest_reason)
    return fb_metrics, fb_mtime, True, latest_reason


def _fallback_last_good_stability(
    latest_reason: str,
    now_ts: float,
    max_staleness_sec: float,
    fallback_path: Path,
) -> tuple[Optional[Dict[str, Any]], Optional[float], bool, str]:
    fb_metrics, fb_mtime = load_stability_metrics(fallback_path)
    if not isinstance(fb_metrics, dict):
        return None, None, False, latest_reason
    if fb_mtime is not None and (float(now_ts) - float(fb_mtime)) > float(max_staleness_sec):
        return None, None, False, latest_reason
    _log.info("latest stability %s, using fallback last_good", latest_reason)
    return fb_metrics, fb_mtime, True, latest_reason


def _update_last_good_if_ok(
    latest_metrics_path: Optional[Path],
    latest_stability_path: Optional[Path] = None,
    latest_stability_up_path: Optional[Path] = None,
    latest_stability_down_path: Optional[Path] = None,
) -> None:
    if not any(
        p is not None and p.exists()
        for p in (latest_metrics_path, latest_stability_path, latest_stability_up_path, latest_stability_down_path)
    ):
        return
    if latest_metrics_path is not None and latest_metrics_path.exists():
        latest_dir = latest_metrics_path.parent
    elif latest_stability_path is not None and latest_stability_path.exists():
        latest_dir = latest_stability_path.parent
    elif latest_stability_up_path is not None and latest_stability_up_path.exists():
        latest_dir = latest_stability_up_path.parent
    else:
        latest_dir = latest_stability_down_path.parent  # type: ignore[union-attr]

    last_good_dir = Path("runs/last_good")
    if latest_metrics_path is not None and latest_metrics_path.exists():
        _atomic_copy(latest_metrics_path, last_good_dir / "metrics.json")
    if latest_stability_path is not None and latest_stability_path.exists():
        _atomic_copy(latest_stability_path, last_good_dir / "stability.csv")
    if latest_stability_up_path is not None and latest_stability_up_path.exists():
        _atomic_copy(latest_stability_up_path, last_good_dir / "stability_up.csv")
    if latest_stability_down_path is not None and latest_stability_down_path.exists():
        _atomic_copy(latest_stability_down_path, last_good_dir / "stability_down.csv")
    cfg = latest_dir / "config.json"
    if cfg.exists():
        _atomic_copy(cfg, last_good_dir / "config.json")
    _atomic_write_text(last_good_dir / "ts_utc.txt", str(int(time.time())))
    _log.info("updated last_good metrics snapshot")


def evaluate_alpha_gate(
    metrics: Optional[Dict[str, Any]],
    now_ts: float,
    *,
    metrics_mtime_ts: Optional[float] = None,
    max_staleness_sec: float = 300.0,
    min_pnl_net_per_fill: float = 0.0,
    min_fill_rate: float = 0.2,
    max_fee_dominates_frac: float = 0.5,
    max_spread_cost_per_fill: Optional[float] = None,
) -> AlphaGateDecision:
    if not isinstance(metrics, dict):
        return AlphaGateDecision(True, "alpha_metrics_missing", {})
    if metrics_mtime_ts is not None and (float(now_ts) - float(metrics_mtime_ts)) > float(max_staleness_sec):
        return AlphaGateDecision(True, "alpha_metrics_stale", {"age_sec": float(now_ts) - float(metrics_mtime_ts)})

    fills_count = int(_safe_float(metrics.get("fills_count"), 0.0))
    decisions_count = int(_safe_float(metrics.get("decisions_count"), 0.0))
    pnl_net_sum = _safe_float(metrics.get("pnl_net_sum", metrics.get("pnl_sum", 0.0)), 0.0)
    decision_to_fill_rate = _safe_float(metrics.get("decision_to_fill_rate"), 0.0)
    if decisions_count > 0 and decision_to_fill_rate <= 0:
        decision_to_fill_rate = float(fills_count) / float(decisions_count)
    fee_dominates_count = int(_safe_float(metrics.get("fee_dominates_count"), 0.0))
    spread_cost_est_sum = _safe_float(metrics.get("spread_cost_est_sum"), 0.0)

    pnl_net_per_fill = float(pnl_net_sum / fills_count) if fills_count > 0 else float("-inf")
    fee_dominates_frac = float(fee_dominates_count / fills_count) if fills_count > 0 else 1.0
    spread_cost_per_fill = float(spread_cost_est_sum / fills_count) if fills_count > 0 else float("inf")
    details = {
        "fills_count": fills_count,
        "decisions_count": decisions_count,
        "pnl_net_sum": pnl_net_sum,
        "pnl_net_per_fill": pnl_net_per_fill,
        "decision_to_fill_rate": decision_to_fill_rate,
        "fee_dominates_frac": fee_dominates_frac,
        "spread_cost_per_fill": spread_cost_per_fill,
    }
    if pnl_net_per_fill < float(min_pnl_net_per_fill):
        return AlphaGateDecision(True, "alpha_negative_edge", details)
    if decision_to_fill_rate < float(min_fill_rate):
        return AlphaGateDecision(True, "alpha_fill_rate_low", details)
    if fee_dominates_frac > float(max_fee_dominates_frac):
        return AlphaGateDecision(True, "alpha_fee_dominates", details)
    if max_spread_cost_per_fill is not None and spread_cost_per_fill > float(max_spread_cost_per_fill):
        return AlphaGateDecision(True, "alpha_spread_too_high", details)
    return AlphaGateDecision(False, "alpha_ok", details)


def _evaluate_stability_gate(
    row: Optional[Dict[str, Any]],
    *,
    min_pos_frac: float,
    min_st_score: float,
    max_worst: Optional[float],
    unstable_reason: str,
    score_low_reason: str,
    tail_reason: str,
) -> AlphaGateDecision:
    if not isinstance(row, dict):
        return AlphaGateDecision(True, "alpha_stability_missing", {})
    pos_frac = _safe_float(row.get("pos_slices_frac"), 0.0)
    st_score = _safe_float(row.get("stability_score"), 0.0)
    worst = _safe_float(row.get("worst_pnl_net_per_fill"), 0.0)
    slices_count = int(_safe_float(row.get("slices_count"), 0.0))
    details = {
        "pos_slices_frac": pos_frac,
        "stability_score": st_score,
        "worst_pnl_net_per_fill": worst,
        "slices_count": slices_count,
    }
    if slices_count <= 0:
        return AlphaGateDecision(True, "alpha_stability_missing", details)
    if pos_frac < float(min_pos_frac):
        return AlphaGateDecision(True, unstable_reason, details)
    if st_score < float(min_st_score):
        return AlphaGateDecision(True, score_low_reason, details)
    if max_worst is not None and worst < -abs(float(max_worst)):
        return AlphaGateDecision(True, tail_reason, details)
    return AlphaGateDecision(False, "alpha_stability_ok", details)


def evaluate_alpha_gate_from_env(now_ts: Optional[float] = None) -> AlphaGateDecision:
    now = float(time.time() if now_ts is None else now_ts)
    mode = str(os.getenv("ALPHA_GATE_MODE", "metrics") or "metrics").strip().lower()
    if mode not in ("metrics", "stability", "both", "both_regime"):
        mode = "metrics"

    metrics_path = Path(os.getenv("ALPHA_GATE_METRICS_PATH", "").strip() or "runs/latest/metrics.json")
    stability_path = Path(os.getenv("ALPHA_GATE_STABILITY_PATH", "").strip() or "runs/latest/stability.csv")
    stability_up_path = Path(os.getenv("ALPHA_GATE_STABILITY_UP_PATH", "").strip() or "runs/latest/stability_up.csv")
    stability_down_path = Path(os.getenv("ALPHA_GATE_STABILITY_DOWN_PATH", "").strip() or "runs/latest/stability_down.csv")

    max_staleness = _env_float("ALPHA_GATE_MAX_STALENESS_SEC", 300.0)
    max_spread_cfg = os.getenv("ALPHA_GATE_MAX_SPREAD_COST_PER_FILL", "").strip()
    max_spread_val = None if max_spread_cfg == "" else _safe_float(max_spread_cfg, 0.0)

    metrics, mtime = load_alpha_metrics(metrics_path)
    fallback_used_metrics = False
    fallback_reason_metrics = ""
    if mode in ("metrics", "both", "both_regime"):
        if not isinstance(metrics, dict):
            metrics, mtime, fallback_used_metrics, fallback_reason_metrics = _fallback_last_good_json(
                "missing", now, max_staleness, Path("runs/last_good/metrics.json")
            )
        elif mtime is not None and (float(now) - float(mtime)) > float(max_staleness):
            metrics, mtime, fallback_used_metrics, fallback_reason_metrics = _fallback_last_good_json(
                "stale", now, max_staleness, Path("runs/last_good/metrics.json")
            )

    dec_metrics = evaluate_alpha_gate(
        metrics,
        now,
        metrics_mtime_ts=mtime,
        max_staleness_sec=max_staleness,
        min_pnl_net_per_fill=_env_float("ALPHA_GATE_MIN_PNL_NET_PER_FILL", 0.0),
        min_fill_rate=_env_float("ALPHA_GATE_MIN_FILL_RATE", 0.2),
        max_fee_dominates_frac=_env_float("ALPHA_GATE_MAX_FEE_DOMINATES_FRAC", 0.5),
        max_spread_cost_per_fill=max_spread_val,
    )

    fallback_used_stability = False
    fallback_reason_stability = ""
    st_row, st_mtime = load_stability_metrics(stability_path)
    if mode in ("stability", "both", "both_regime"):
        if not isinstance(st_row, dict):
            st_row, st_mtime, fallback_used_stability, fallback_reason_stability = _fallback_last_good_stability(
                "missing", now, max_staleness, Path("runs/last_good/stability.csv")
            )
        elif st_mtime is not None and (float(now) - float(st_mtime)) > float(max_staleness):
            st_row, st_mtime, fallback_used_stability, fallback_reason_stability = _fallback_last_good_stability(
                "stale", now, max_staleness, Path("runs/last_good/stability.csv")
            )

    max_worst_cfg = os.getenv("ALPHA_GATE_MAX_WORST_PNL_NET_PER_FILL", "").strip()
    max_worst = None if max_worst_cfg == "" else abs(_safe_float(max_worst_cfg, 0.0))
    stability_dec = _evaluate_stability_gate(
        st_row,
        min_pos_frac=_env_float("ALPHA_GATE_MIN_POS_SLICES_FRAC", 0.5),
        min_st_score=_env_float("ALPHA_GATE_MIN_STABILITY_SCORE", 0.0),
        max_worst=max_worst,
        unstable_reason="alpha_unstable",
        score_low_reason="alpha_stability_score_low",
        tail_reason="alpha_tail_risk",
    )

    up_fallback_used = False
    up_fallback_reason = ""
    up_row, up_mtime = load_stability_metrics(stability_up_path)
    if mode == "both_regime":
        if not isinstance(up_row, dict):
            up_row, up_mtime, up_fallback_used, up_fallback_reason = _fallback_last_good_stability(
                "missing", now, max_staleness, Path("runs/last_good/stability_up.csv")
            )
        elif up_mtime is not None and (float(now) - float(up_mtime)) > float(max_staleness):
            up_row, up_mtime, up_fallback_used, up_fallback_reason = _fallback_last_good_stability(
                "stale", now, max_staleness, Path("runs/last_good/stability_up.csv")
            )

    down_fallback_used = False
    down_fallback_reason = ""
    down_row, down_mtime = load_stability_metrics(stability_down_path)
    if mode == "both_regime":
        if not isinstance(down_row, dict):
            down_row, down_mtime, down_fallback_used, down_fallback_reason = _fallback_last_good_stability(
                "missing", now, max_staleness, Path("runs/last_good/stability_down.csv")
            )
        elif down_mtime is not None and (float(now) - float(down_mtime)) > float(max_staleness):
            down_row, down_mtime, down_fallback_used, down_fallback_reason = _fallback_last_good_stability(
                "stale", now, max_staleness, Path("runs/last_good/stability_down.csv")
            )

    up_dec = _evaluate_stability_gate(
        up_row,
        min_pos_frac=_env_float("ALPHA_GATE_MIN_POS_SLICES_FRAC_UP", _env_float("ALPHA_GATE_MIN_POS_SLICES_FRAC", 0.5)),
        min_st_score=_env_float("ALPHA_GATE_MIN_STABILITY_SCORE_UP", _env_float("ALPHA_GATE_MIN_STABILITY_SCORE", 0.0)),
        max_worst=max_worst,
        unstable_reason="alpha_regime_up_unstable",
        score_low_reason="alpha_regime_up_score_low",
        tail_reason="alpha_regime_up_tail_risk",
    )
    down_dec = _evaluate_stability_gate(
        down_row,
        min_pos_frac=_env_float("ALPHA_GATE_MIN_POS_SLICES_FRAC_DOWN", _env_float("ALPHA_GATE_MIN_POS_SLICES_FRAC", 0.5)),
        min_st_score=_env_float("ALPHA_GATE_MIN_STABILITY_SCORE_DOWN", _env_float("ALPHA_GATE_MIN_STABILITY_SCORE", 0.0)),
        max_worst=max_worst,
        unstable_reason="alpha_regime_down_unstable",
        score_low_reason="alpha_regime_down_score_low",
        tail_reason="alpha_regime_down_tail_risk",
    )

    if mode == "metrics":
        dec = dec_metrics
    elif mode == "stability":
        dec = AlphaGateDecision(stability_dec.blocked, stability_dec.reason, dict(stability_dec.details))
    elif mode == "both":
        if dec_metrics.blocked:
            dec = dec_metrics
        elif stability_dec.blocked:
            dec = AlphaGateDecision(True, stability_dec.reason, dict(stability_dec.details))
        else:
            merged = dict(dec_metrics.details or {})
            merged.update(dict(stability_dec.details or {}))
            dec = AlphaGateDecision(False, "alpha_ok", merged)
    else:  # both_regime
        if dec_metrics.blocked:
            dec = dec_metrics
        elif stability_dec.blocked:
            dec = AlphaGateDecision(True, stability_dec.reason, dict(stability_dec.details))
        elif up_dec.blocked:
            dec = AlphaGateDecision(True, up_dec.reason, dict(up_dec.details))
        elif down_dec.blocked:
            dec = AlphaGateDecision(True, down_dec.reason, dict(down_dec.details))
        else:
            merged = dict(dec_metrics.details or {})
            merged.update({"stability_all": dict(stability_dec.details or {})})
            merged.update({"stability_up": dict(up_dec.details or {})})
            merged.update({"stability_down": dict(down_dec.details or {})})
            dec = AlphaGateDecision(False, "alpha_ok", merged)

    if isinstance(dec.details, dict):
        dec.details["alpha_gate_mode"] = mode
        dec.details["metrics_fallback_used"] = bool(fallback_used_metrics)
        if fallback_reason_metrics:
            dec.details["metrics_fallback_reason"] = str(fallback_reason_metrics)
        dec.details["metrics_path_used"] = "runs/last_good/metrics.json" if fallback_used_metrics else str(metrics_path)
        dec.details["stability_fallback_used"] = bool(fallback_used_stability)
        if fallback_reason_stability:
            dec.details["stability_fallback_reason"] = str(fallback_reason_stability)
        dec.details["stability_path_used"] = "runs/last_good/stability.csv" if fallback_used_stability else str(stability_path)
        dec.details["regime_up_fallback_used"] = bool(up_fallback_used)
        if up_fallback_reason:
            dec.details["regime_up_fallback_reason"] = str(up_fallback_reason)
        dec.details["regime_up_path_used"] = "runs/last_good/stability_up.csv" if up_fallback_used else str(stability_up_path)
        dec.details["regime_down_fallback_used"] = bool(down_fallback_used)
        if down_fallback_reason:
            dec.details["regime_down_fallback_reason"] = str(down_fallback_reason)
        dec.details["regime_down_path_used"] = "runs/last_good/stability_down.csv" if down_fallback_used else str(stability_down_path)

    update_allowed = False
    if mode == "metrics":
        update_allowed = (not dec.blocked) and (not fallback_used_metrics)
    elif mode == "stability":
        update_allowed = (not dec.blocked) and (not fallback_used_stability)
    elif mode == "both":
        update_allowed = (not dec.blocked) and (not fallback_used_metrics) and (not fallback_used_stability)
    else:
        update_allowed = (
            (not dec.blocked)
            and (not fallback_used_metrics)
            and (not fallback_used_stability)
            and (not up_fallback_used)
            and (not down_fallback_used)
        )

    if update_allowed:
        _update_last_good_if_ok(
            metrics_path if mode in ("metrics", "both", "both_regime") else None,
            stability_path if mode in ("stability", "both", "both_regime") else None,
            stability_up_path if mode == "both_regime" else None,
            stability_down_path if mode == "both_regime" else None,
        )
    return dec
