from __future__ import annotations

import argparse
import json
import re
import random
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Dict, List, Tuple

from config.costs import DEFAULT_MAKER_FEE_BPS
from tools.run_summary import build_run_summary
from tools.validate_passive_pocket_forward import validate_pocket_forward


def _parse_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in _parse_list(raw)]


def _parse_seed_list(raw: str) -> str:
    return ",".join(_parse_list(raw))


def _pocket_id(
    cand: Dict[str, Any],
    *,
    rule: str,
    side: str,
    horizon_override: int = 0,
) -> str:
    horizon = int(horizon_override) if int(horizon_override) > 0 else int(cand.get("horizon_sec", 0) or 0)
    return (
        f"{str(cand.get('symbol', ''))} rule={str(rule)} side={str(side)} "
        f"h={horizon} imb>={float(cand.get('min_imbalance', 0.0)):.2f} "
        f"int>={float(cand.get('min_trade_intensity', 0.0)):.0f} "
        f"spr<={float(cand.get('max_spread', 0.0)):.6f}"
    )


_FIELD_ALIASES: Dict[str, List[str]] = {
    "symbol": ["symbol"],
    "horizon_sec": ["horizon_sec", "horizon", "h", "horizonsec"],
    "min_imbalance": ["min_imbalance", "min_imb", "imb>=", "imb", "minimbalance"],
    "min_trade_intensity": ["min_trade_intensity", "min_int", "int>=", "trade_intensity", "mintradeintensity"],
    "max_spread": ["max_spread", "spr<=", "spr", "spread", "maxspread"],
    "pass": ["pass", "PASS"],
}


def _canon(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _find_col_idx(headers: List[str], aliases: List[str]) -> int:
    canon_headers = [_canon(h) for h in headers]
    wanted = {_canon(a) for a in aliases}
    for i, h in enumerate(canon_headers):
        if h in wanted:
            return i
    return -1


def _is_sep_row(parts: List[str]) -> bool:
    if not parts:
        return True
    for p in parts:
        t = p.strip()
        if not t:
            continue
        if not re.fullmatch(r"[-:]+", t):
            return False
    return True


def _to_float(x: str) -> float:
    s = str(x or "").strip().replace("%", "").replace("+", "")
    return float(s)


def _pass_yes(raw: str) -> bool:
    return _canon(raw) in {"yes", "true", "1", "y"}


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in candidates:
        k = (
            str(c.get("symbol", "")),
            int(c.get("horizon_sec", 0)),
            float(c.get("min_imbalance", 0.0)),
            float(c.get("min_trade_intensity", 0.0)),
            float(c.get("max_spread", 0.0)),
        )
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return uniq


def _parse_candidates_from_md(path: Path, debug: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "path": str(path),
        "total_rows_seen": 0,
        "table_rows_seen": 0,
        "rows_with_pass_yes": 0,
        "candidates_parsed": 0,
        "candidates_unique": 0,
        "rows_skipped_missing_fields": 0,
        "skipped_examples": [],
        "parsed_examples": [],
    }
    if not path.exists():
        stats["error"] = "missing_file"
        return [], stats

    lines = path.read_text(encoding="utf-8").splitlines()
    stats["total_rows_seen"] = len(lines)
    headers: List[str] = []
    idx_map: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []

    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        parts = [p.strip() for p in s.strip("|").split("|")]
        if not parts:
            continue
        stats["table_rows_seen"] += 1
        if _is_sep_row(parts):
            continue

        # header detection / re-header handling
        parts_canon = [_canon(p) for p in parts]
        if "symbol" in parts_canon and ("horizonsec" in parts_canon or "horizon" in parts_canon or "h" in parts_canon):
            headers = parts
            idx_map = {k: _find_col_idx(headers, v) for k, v in _FIELD_ALIASES.items()}
            continue

        if not headers:
            # fallback: expected positional schema from classic sweep output
            # symbol|horizon|min_imbalance|min_trade_intensity|max_spread|...|pass
            if len(parts) >= 6:
                headers = ["symbol", "horizon_sec", "min_imbalance", "min_trade_intensity", "max_spread"] + [f"col{i}" for i in range(5, len(parts) - 1)] + ["pass"]
                idx_map = {k: _find_col_idx(headers, v) for k, v in _FIELD_ALIASES.items()}
            else:
                stats["rows_skipped_missing_fields"] += 1
                if len(stats["skipped_examples"]) < 5:
                    stats["skipped_examples"].append({"row": parts, "reason": "no_header_and_too_few_cols"})
                continue

        needed = ["symbol", "horizon_sec", "min_imbalance", "min_trade_intensity", "max_spread", "pass"]
        missing = [k for k in needed if idx_map.get(k, -1) < 0]
        if missing:
            stats["rows_skipped_missing_fields"] += 1
            if len(stats["skipped_examples"]) < 5:
                stats["skipped_examples"].append({"row": parts, "reason": f"missing_cols={missing}"})
            continue

        try:
            pass_raw = parts[idx_map["pass"]]
            if _pass_yes(pass_raw):
                stats["rows_with_pass_yes"] += 1
            else:
                continue
            cand = {
                "symbol": str(parts[idx_map["symbol"]]).strip(),
                "horizon_sec": int(float(_to_float(parts[idx_map["horizon_sec"]]))),
                "min_imbalance": float(_to_float(parts[idx_map["min_imbalance"]])),
                "min_trade_intensity": float(_to_float(parts[idx_map["min_trade_intensity"]])),
                "max_spread": float(_to_float(parts[idx_map["max_spread"]])),
            }
            out.append(cand)
            if len(stats["parsed_examples"]) < 5:
                stats["parsed_examples"].append(cand)
        except Exception as exc:
            stats["rows_skipped_missing_fields"] += 1
            if len(stats["skipped_examples"]) < 5:
                stats["skipped_examples"].append({"row": parts, "reason": f"parse_error={exc}"})

    stats["candidates_parsed"] = len(out)
    uniq = _dedupe_candidates(out)
    stats["candidates_unique"] = len(uniq)
    if debug:
        print(
            "[parse] "
            f"path={path} total_rows_seen={stats['total_rows_seen']} table_rows_seen={stats['table_rows_seen']} "
            f"rows_with_pass_yes={stats['rows_with_pass_yes']} candidates_parsed={stats['candidates_parsed']} "
            f"candidates_unique={stats['candidates_unique']} rows_skipped_missing_fields={stats['rows_skipped_missing_fields']}"
        )
        for ex in stats["parsed_examples"]:
            print(f"[parse] parsed_example={ex}")
        for ex in stats["skipped_examples"]:
            print(f"[parse] skipped_example={ex}")
    return uniq, stats


def _combo_pass(row: Dict[str, Any], min_n: int) -> bool:
    return bool(
        int(row.get("filled_n", 0) or 0) >= int(min_n)
        and float(row.get("filled_avg_net", 0.0)) >= 0.000005
        and float(row.get("filled_p90_net", 0.0)) >= 0.00005
    )


def _split_seed_stability(per_combo: List[Dict[str, Any]]) -> float:
    by_split: Dict[int, List[float]] = {}
    for r in per_combo:
        sid = int(r.get("split", 0) or 0)
        by_split.setdefault(sid, []).append(float(r.get("filled_avg_net", 0.0)))
    if not by_split:
        return 0.0
    stds = []
    for vals in by_split.values():
        stds.append(pstdev(vals) if len(vals) > 1 else 0.0)
    return sum(stds) / len(stds)


def _aggregate_eval(res: Dict[str, Any], min_n: int) -> Dict[str, Any]:
    per_combo = list(res.get("per_combo", []))
    if not per_combo:
        return {
            "pass_rate": 0.0,
            "median_filled_avg_net": 0.0,
            "worst_split_avg_net": 0.0,
            "worst_split_p90_net": 0.0,
            "stability_std": 0.0,
            "median_net_per_attempt": 0.0,
            "attempt_fill_rate_median": 0.0,
            "attempts_per_min_median": 0.0,
            "rows": 0,
            "gate_reject_ratio": 0.0,
            "fill_rate_after_gate": 0.0,
            "avg_adverse_bps_on_fills": 0.0,
            "avg_fee_bps": 0.0,
            "avg_scratch_bps_on_fills": 0.0,
            "avg_raw_return_bps_on_fills": 0.0,
            "avg_net_return_bps_on_fills": 0.0,
            "reject_vol_quantile_reject": 0,
            "reject_spread_too_wide": 0,
            "reject_imbalance_too_low": 0,
            "reject_intensity_too_low": 0,
            "reject_other_gate": 0,
            "per_combo": [],
        }
    flags = [_combo_pass(r, min_n=min_n) for r in per_combo]
    pass_rate = sum(1 for f in flags if f) / len(flags)
    med_avg = float(median(float(r.get("filled_avg_net", 0.0)) for r in per_combo))
    by_split: Dict[int, List[Dict[str, Any]]] = {}
    for r in per_combo:
        sid = int(r.get("split", 0) or 0)
        by_split.setdefault(sid, []).append(r)
    split_avg = []
    split_p90 = []
    for grp in by_split.values():
        split_avg.append(sum(float(x.get("filled_avg_net", 0.0)) for x in grp) / len(grp))
        split_p90.append(sum(float(x.get("filled_p90_net", 0.0)) for x in grp) / len(grp))
    total_events = sum(int(r.get("n_events_total", 0) or 0) for r in per_combo)
    total_gate_reject = sum(int(r.get("n_rejected_attempt_gate", 0) or 0) for r in per_combo)
    total_attempts_after_gate = sum(int(r.get("n_attempts_after_gate", r.get("val_attempts_after_gate", r.get("val_attempts", 0))) or 0) for r in per_combo)
    total_filled = sum(int(r.get("n_filled", r.get("filled_n", 0)) or 0) for r in per_combo)
    rej_vol = sum(int(r.get("reject_vol_quantile_reject", 0) or 0) for r in per_combo)
    rej_spread = sum(int(r.get("reject_spread_too_wide", 0) or 0) for r in per_combo)
    rej_imb = sum(int(r.get("reject_imbalance_too_low", 0) or 0) for r in per_combo)
    rej_int = sum(int(r.get("reject_intensity_too_low", 0) or 0) for r in per_combo)
    rej_other = sum(int(r.get("reject_other_gate", 0) or 0) for r in per_combo)
    filled_weights = [max(0, int(r.get("n_filled", r.get("filled_n", 0)) or 0)) for r in per_combo]
    wsum = sum(filled_weights)

    def _wavg(key: str) -> float:
        if wsum <= 0:
            return 0.0
        return sum(float(r.get(key, 0.0) or 0.0) * w for r, w in zip(per_combo, filled_weights)) / wsum

    return {
        "pass_rate": pass_rate,
        "median_filled_avg_net": med_avg,
        "worst_split_avg_net": min(split_avg) if split_avg else 0.0,
        "worst_split_p90_net": min(split_p90) if split_p90 else 0.0,
        "stability_std": _split_seed_stability(per_combo),
        "median_net_per_attempt": float(median(float(r.get("net_per_attempt", 0.0)) for r in per_combo)),
        "attempt_fill_rate_median": float(median(float(r.get("attempt_fill_rate", 0.0)) for r in per_combo)),
        "attempts_per_min_median": float(median(float(r.get("attempts_per_min", 0.0)) for r in per_combo)),
        "median_effective_min_n": int(median(int(r.get("effective_min_n", 0) or 0) for r in per_combo)),
        "rows": len(per_combo),
        "gate_reject_ratio": (float(total_gate_reject) / float(total_events)) if total_events > 0 else 0.0,
        "fill_rate_after_gate": (float(total_filled) / float(total_attempts_after_gate)) if total_attempts_after_gate > 0 else 0.0,
        "avg_adverse_bps_on_fills": _wavg("avg_adverse_bps_on_fills"),
        "avg_fee_bps": _wavg("avg_fee_bps"),
        "avg_scratch_bps_on_fills": _wavg("avg_scratch_bps_on_fills"),
        "avg_raw_return_bps_on_fills": _wavg("avg_raw_return_bps_on_fills"),
        "avg_net_return_bps_on_fills": _wavg("avg_net_return_bps_on_fills"),
        "reject_vol_quantile_reject": int(rej_vol),
        "reject_spread_too_wide": int(rej_spread),
        "reject_imbalance_too_low": int(rej_imb),
        "reject_intensity_too_low": int(rej_int),
        "reject_other_gate": int(rej_other),
        "per_combo": per_combo,
    }


def _fee_score(agg: Dict[str, Any]) -> float:
    pass_rate = float(agg.get("pass_rate", 0.0))
    med_bps = float(agg.get("median_filled_avg_net", 0.0)) * 10000.0
    worst_bps = max(0.0, float(agg.get("worst_split_avg_net", 0.0)) * 10000.0)
    return pass_rate * med_bps * worst_bps


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_ratio(num: Any, den: Any) -> float | None:
    n = _safe_float(num, default=0.0)
    d = _safe_float(den, default=0.0)
    if d <= 0.0:
        return None
    return n / d


def _failure_reason_top(agg: Dict[str, Any]) -> str:
    gate_reject_ratio = _safe_float(agg.get("gate_reject_ratio", 0.0), default=0.0)
    fill_rate_after_gate = _safe_float(agg.get("fill_rate_after_gate", 0.0), default=0.0)
    avg_net_bps = _safe_float(agg.get("avg_net_return_bps_on_fills", 0.0), default=0.0)
    avg_adv_bps = abs(_safe_float(agg.get("avg_adverse_bps_on_fills", 0.0), default=0.0))
    avg_fee_bps = abs(_safe_float(agg.get("avg_fee_bps", 0.0), default=0.0))
    if gate_reject_ratio > 0.5:
        return "gate_reject"
    if fill_rate_after_gate < 0.1:
        return "no_fills"
    if avg_net_bps < 0.0:
        if avg_adv_bps >= avg_fee_bps:
            return "adverse_dominates"
        return "fees_dominate"
    return "mixed"


def _npa_decomposition_from_eval(eval_row: Dict[str, Any]) -> Dict[str, Any]:
    fill_after = _safe_float(eval_row.get("fill_rate_after_gate", 0.0), 0.0)
    gross_npa = (_safe_float(eval_row.get("avg_raw_return_bps_on_fills", 0.0), 0.0) / 10000.0) * fill_after
    fee_cost_npa = (_safe_float(eval_row.get("avg_fee_bps", 0.0), 0.0) / 10000.0) * fill_after
    adverse_cost_npa = (_safe_float(eval_row.get("avg_adverse_bps_on_fills", 0.0), 0.0) / 10000.0) * fill_after
    scratch_cost_npa = (_safe_float(eval_row.get("avg_scratch_bps_on_fills", 0.0), 0.0) / 10000.0) * fill_after
    net_npa = gross_npa - fee_cost_npa - adverse_cost_npa - scratch_cost_npa
    observed_npa = _safe_float(eval_row.get("median_net_per_attempt", 0.0), 0.0)
    residual = observed_npa - net_npa
    return {
        "gross_edge_npa": float(gross_npa),
        "fee_cost_npa": float(fee_cost_npa),
        "adverse_cost_npa": float(adverse_cost_npa),
        "scratch_cost_npa": float(scratch_cost_npa),
        "net_npa": float(net_npa),
        "observed_net_npa": float(observed_npa),
        "residual_npa": float(residual),
    }


def _bootstrap_mean_ci(values: List[float], *, samples: int, seed: int) -> Tuple[float, float, float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return 0.0, 0.0, 1.0
    if len(clean) == 1:
        mu = float(clean[0])
        p = 0.0 if mu > 0.0 else 1.0
        return mu, mu, p
    rng = random.Random(int(seed))
    n = len(clean)
    means: List[float] = []
    for _ in range(max(10, int(samples))):
        s = 0.0
        for _ in range(n):
            s += clean[rng.randrange(0, n)]
        means.append(s / float(n))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    nonpos = sum(1 for m in means if m <= 0.0)
    p_one = nonpos / float(len(means))
    return float(lo), float(hi), float(max(0.0, min(1.0, p_one)))


def _bh_adjust(items: List[Tuple[int, float]]) -> Dict[int, float]:
    if not items:
        return {}
    ordered = sorted(((idx, float(max(0.0, min(1.0, p)))) for idx, p in items), key=lambda x: x[1])
    m = len(ordered)
    raw = [0.0] * m
    for i, (_, p) in enumerate(ordered, start=1):
        raw[i - 1] = min(1.0, p * m / float(i))
    adj = [0.0] * m
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, raw[i])
        adj[i] = running
    return {ordered[i][0]: float(adj[i]) for i in range(m)}


def _bonferroni_adjust(items: List[Tuple[int, float]]) -> Dict[int, float]:
    if not items:
        return {}
    m = max(1, len(items))
    out: Dict[int, float] = {}
    for idx, p in items:
        pv = float(max(0.0, min(1.0, p)))
        out[int(idx)] = float(min(1.0, pv * float(m)))
    return out


def _summarize_liquidation_scoring_impact(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    impacted = [
        r
        for r in rows
        if str(r.get("rule", "")).startswith("micro_edge_v")
    ]
    if not impacted:
        return {"available": False, "count": 0}
    avg_delta_score = sum(float(r.get("delta_score_raw_core", 0.0) or 0.0) for r in impacted) / float(len(impacted))
    avg_delta_npa = sum(float(r.get("delta_npa_core", 0.0) or 0.0) for r in impacted) / float(len(impacted))
    avg_delta_pass = sum(float(r.get("delta_pass_rate_core", 0.0) or 0.0) for r in impacted) / float(len(impacted))
    positive = sum(1 for r in impacted if float(r.get("delta_score_raw_core", 0.0) or 0.0) > 0.0)
    improved = max(impacted, key=lambda r: float(r.get("delta_score_raw_core", 0.0) or 0.0))
    degraded = min(impacted, key=lambda r: float(r.get("delta_score_raw_core", 0.0) or 0.0))
    return {
        "available": True,
        "count": int(len(impacted)),
        "positive_delta_score_count": int(positive),
        "avg_delta_score_raw_core": float(avg_delta_score),
        "avg_delta_npa_core": float(avg_delta_npa),
        "avg_delta_pass_rate_core": float(avg_delta_pass),
        "top_improved": {
            "symbol": improved.get("symbol"),
            "rule": improved.get("rule"),
            "delta_score_raw_core": float(improved.get("delta_score_raw_core", 0.0) or 0.0),
            "delta_npa_core": float(improved.get("delta_npa_core", 0.0) or 0.0),
            "delta_pass_rate_core": float(improved.get("delta_pass_rate_core", 0.0) or 0.0),
        },
        "top_degraded": {
            "symbol": degraded.get("symbol"),
            "rule": degraded.get("rule"),
            "delta_score_raw_core": float(degraded.get("delta_score_raw_core", 0.0) or 0.0),
            "delta_npa_core": float(degraded.get("delta_npa_core", 0.0) or 0.0),
            "delta_pass_rate_core": float(degraded.get("delta_pass_rate_core", 0.0) or 0.0),
        },
    }


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank passive pockets by forward robustness.")
    p.add_argument("--candidates-md", required=True)
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--lookback-min", type=int, default=1440)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--horizon-sec", type=int, default=0, help="Optional override for candidate horizon_sec. 0 keeps per-candidate horizon.")
    p.add_argument("--rule", default="intensity_spike_imbalance_cont")
    p.add_argument("--rules", default="", help="Optional comma list to compare rules on identical pockets.")
    p.add_argument("--side", default="auto")
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--min-n-frac", type=float, default=0.0)
    p.add_argument("--maker-fee-bps-grid", default=f"{float(DEFAULT_MAKER_FEE_BPS)}")
    p.add_argument("--passive-adverse-mult-grid", default="0.8,1.0,1.2")
    p.add_argument("--v2-min-score", type=float, default=0.0)
    p.add_argument("--v2-min-persistence", type=float, default=0.0)
    p.add_argument("--v2-min-confidence", type=float, default=0.0)
    p.add_argument("--passive-profile-in", default="state/passive_realistic_profiles.json")
    p.add_argument("--passive-max-wait-buckets", type=int, default=0, help="Optional extra passive wait buckets before considering the order unfilled.")
    p.add_argument("--out-md", default="reports/PASSIVE_POCKET_RANKING.md")
    p.add_argument("--out-json", default="reports/PASSIVE_POCKET_RANKING.json")
    p.add_argument("--debug-parse", action="store_true")
    p.add_argument("--only-pocket", default="", help="Filter pockets by exact substring match against pocket identifier.")
    p.add_argument("--only-pocket-regex", default="", help="Filter pockets by regex (re.search) against pocket identifier.")
    p.add_argument("--min-attempt-fill-rate", type=float, default=0.10,
                   help="Skip pockets whose core-eval attempt_fill_rate_median is below this threshold.")
    p.add_argument("--max-insufficient-fill-rate", type=float, default=0.50,
                   help="Skip pockets whose core-eval insufficient_fill_rate exceeds this threshold.")
    p.add_argument("--min-intensity-strong", type=float, default=0.0, help="Pre-attempt gate: skip when trade_intensity < this.")
    p.add_argument("--min-imbalance-strong", type=float, default=0.0, help="Pre-attempt gate: skip when |imbalance| < this.")
    p.add_argument("--max-spread-tight", type=float, default=0.0, help="Pre-attempt gate: skip when spread > this.")
    p.add_argument("--max-volatility-extreme", type=float, default=None, help="Pre-attempt gate: skip when volatility proxy > this. For anti_adverse_v2, this overrides default 0.0020.")
    p.add_argument("--vol-quantile-reject", type=float, default=0.01, help="For anti_adverse_v3: reject top X fraction of volatility via train-slice quantile (e.g. 0.01).")
    p.add_argument("--scratch-bps", type=float, default=0.0, help="Optional post-fill adverse move threshold (bps) for early scratch exit; 0 disables.")
    p.add_argument("--scratch-window-sec", type=int, default=0, help="Optional scratch window in seconds; 0 disables.")
    p.add_argument("--scratch-taker-fee-bps", type=float, default=0.0, help="Extra one-way taker fee bps when scratch triggers.")
    p.add_argument("--scratch-slippage-bps", type=float, default=0.0, help="Extra one-way slippage bps when scratch triggers.")
    p.add_argument("--diagnostic-breakdown", action="store_true", help="Print top-pocket cost/decomposition diagnostics and bps->fraction checks.")
    p.add_argument("--emit-fee-cliff-summary", action="store_true", help="Emit sidecar fee-cliff/decomposition summary JSON next to --out-json.")
    p.add_argument("--bootstrap-ci", action="store_true", help="Compute bootstrap confidence intervals/p-values on core net_per_attempt.")
    p.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap sample count for --bootstrap-ci.")
    p.add_argument("--bootstrap-seed", type=int, default=42, help="Deterministic RNG seed for --bootstrap-ci.")
    p.add_argument("--bh-correction", action="store_true", help="Apply Benjamini-Hochberg correction to bootstrap p-values.")
    p.add_argument("--mtc-method", default="none", choices=["none", "bh", "bonferroni"], help="Multiple testing correction method for bootstrap p-values.")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for bootstrap/BH gating.")
    p.add_argument("--pass-threshold", type=float, default=0.5, help="Minimum pass_rate required for robust_core/stress flags.")
    p.add_argument("--research-mode", action="store_true", help="Set a softer robustness gate for exploration (default pass-threshold=0.33 unless overridden).")
    p.add_argument(
        "--regime",
        default="none",
        choices=["none", "up", "down"],
        help=(
            "Regime filter: 'up' keeps only signals where rolling 1h log-return >= 0; "
            "'down' keeps only DOWN-regime signals; 'none' disables filter (default)."
        ),
    )
    p.add_argument(
        "--mitigation-profile",
        default="baseline",
        choices=["baseline", "anti_adverse_v1", "anti_adverse_v2", "anti_adverse_v3", "anti_adverse_v4", "anti_adverse_v5", "anti_adverse_v6", "event_block_v1", "event_block_book_proxy_v1", "event_block_volatility_v1", "event_block_eth_v1", "event_block_eth_micro_v1", "event_block_eth_micro_imb05_v1", "event_block_eth_micro_imb085_v1"],
        help=(
            "Signal filter profile to reduce adverse selection. "
            "'baseline' = no change. "
            "'anti_adverse_v1' = require stronger imbalance (×1.25) and tighter spread (×0.75), "
            "trading fill-rate for lower adverse selection. "
            "'anti_adverse_v2' = light-touch fixed volatility-extreme guard. "
            "'anti_adverse_v3' = quantile-based volatility-extreme guard. "
            "'anti_adverse_v4' = anti_adverse_v3 + conservative scratch/escape defaults. "
            "'anti_adverse_v5' = anti_adverse_v4 + extra passive wait for event-driven fills. "
            "'anti_adverse_v6' = anti_adverse_v5 + taker fallback after passive miss. "
            "'event_block_v1' = block negative event lanes book_proxy_pressure and volatility_burst. "
            "'event_block_book_proxy_v1' = block only book_proxy_pressure. "
            "'event_block_volatility_v1' = block only volatility_burst. "
            "'event_block_eth_v1' = same block rule, but only for ETH symbol candidates. "
            "'event_block_eth_micro_v1' = same block rule, but only for ETH + micro_edge_v3_passive_alpha candidates. "
            "'event_block_eth_micro_imb05_v1' = same block rule, but only for ETH + micro_edge_v3_passive_alpha + min_imbalance>=0.5 candidates. "
            "'event_block_eth_micro_imb085_v1' = same block rule, but only for ETH + micro_edge_v3_passive_alpha + min_imbalance>=0.85 candidates."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _args()
    if bool(args.research_mode) and float(args.pass_threshold) == 0.5:
        args.pass_threshold = 0.33
    if bool(args.research_mode):
        print(f"RESEARCH MODE enabled: pass_threshold={float(args.pass_threshold):.2f}")
    md_paths = [Path(x) for x in _parse_list(args.candidates_md)]
    candidates: List[Dict[str, Any]] = []
    parse_stats_all: List[Dict[str, Any]] = []
    for p in md_paths:
        parsed, st = _parse_candidates_from_md(p, debug=bool(args.debug_parse))
        parse_stats_all.append(st)
        candidates.extend(parsed)
    candidates = _dedupe_candidates(candidates)
    total_rows_seen = sum(int(s.get("total_rows_seen", 0)) for s in parse_stats_all)
    table_rows_seen = sum(int(s.get("table_rows_seen", 0)) for s in parse_stats_all)
    rows_with_pass_yes = sum(int(s.get("rows_with_pass_yes", 0)) for s in parse_stats_all)
    candidates_parsed = sum(int(s.get("candidates_parsed", 0)) for s in parse_stats_all)
    rows_skipped_missing = sum(int(s.get("rows_skipped_missing_fields", 0)) for s in parse_stats_all)
    print(
        "candidate_parse "
        f"total_rows_seen={total_rows_seen} table_rows_seen={table_rows_seen} "
        f"rows_with_pass_yes={rows_with_pass_yes} candidates_parsed={candidates_parsed} "
        f"candidates_unique={len(candidates)} rows_skipped_missing_fields={rows_skipped_missing}"
    )
    if bool(args.debug_parse):
        shown = 0
        for st in parse_stats_all:
            for ex in st.get("skipped_examples", []):
                print(f"[parse] skipped_row={ex}")
                shown += 1
                if shown >= 8:
                    break
            if shown >= 8:
                break
    if not candidates:
        print(
            "ERROR no candidates parsed from candidates-md. "
            "Check markdown headers and PASS values or use --debug-parse for diagnostics."
        )
        return 2
    fee_grid = _parse_float_list(args.maker_fee_bps_grid)
    adverse_grid = _parse_float_list(args.passive_adverse_mult_grid)
    rules = _parse_list(args.rules) if str(args.rules).strip() else [str(args.rule)]
    seed_str = _parse_seed_list(args.seeds)

    # Optional pocket filtering prior to any validation work.
    only_sub = str(args.only_pocket or "").strip()
    only_rx_raw = str(args.only_pocket_regex or "").strip()
    only_rx = None
    if only_rx_raw:
        try:
            only_rx = re.compile(only_rx_raw)
        except re.error as exc:
            print(f"ERROR invalid --only-pocket-regex: {exc}")
            return 2
    if only_sub or only_rx is not None:
        filtered_candidates: List[Dict[str, Any]] = []
        for c in candidates:
            ids = [
                _pocket_id(
                    c,
                    rule=r,
                    side=str(args.side),
                    horizon_override=int(args.horizon_sec),
                )
                for r in rules
            ]
            matched = False
            for pid in ids:
                if only_sub and only_sub in pid:
                    matched = True
                    break
                if only_rx is not None and only_rx.search(pid):
                    matched = True
                    break
            if matched:
                filtered_candidates.append(c)
        candidates = filtered_candidates
        print(
            f"pocket_filter candidates_after={len(candidates)} "
            f"only_pocket={only_sub or '-'} only_pocket_regex={only_rx_raw or '-'}"
        )
        if not candidates:
            print(
                "ERROR pocket filter matched 0 candidates. "
                "Adjust --only-pocket/--only-pocket-regex."
            )
            return 2

    mitigation_profile = str(args.mitigation_profile)
    if mitigation_profile != "baseline":
        print(f"MITIGATION PROFILE: {mitigation_profile}")
    eff_scratch_bps = float(args.scratch_bps)
    eff_scratch_window_sec = int(args.scratch_window_sec)
    eff_scratch_taker_fee_bps = float(args.scratch_taker_fee_bps)
    eff_scratch_slippage_bps = float(args.scratch_slippage_bps)
    eff_passive_max_wait_buckets = int(args.passive_max_wait_buckets)
    if mitigation_profile in {"anti_adverse_v4", "anti_adverse_v5", "anti_adverse_v6"}:
        if eff_scratch_bps <= 0.0:
            eff_scratch_bps = 4.0
        if eff_scratch_window_sec <= 0:
            eff_scratch_window_sec = 10
        if eff_scratch_taker_fee_bps <= 0.0:
            eff_scratch_taker_fee_bps = 1.0
        if eff_scratch_slippage_bps <= 0.0:
            eff_scratch_slippage_bps = 0.5
    if mitigation_profile in {"anti_adverse_v5", "anti_adverse_v6"} and eff_passive_max_wait_buckets <= 0:
        eff_passive_max_wait_buckets = 2

    def _mitigation_overrides(c: Dict[str, Any], rule_name: str) -> Dict[str, float]:
        """Return extra kwargs for validate_pocket_forward based on mitigation profile."""
        if mitigation_profile == "anti_adverse_v1":
            # Require stronger imbalance confirmation + tighter spread
            # to trade fill-rate for lower adverse selection exposure.
            return {
                "min_imbalance_strong": float(c["min_imbalance"]) * 1.25,
                "max_spread_tight": float(c["max_spread"]) * 0.75,
            }
        if mitigation_profile == "anti_adverse_v2":
            # Light-touch: do not tighten spread/intensity beyond candidate.
            # Only suppress volatility extremes likely tied to adverse bursts.
            return {
                "max_volatility_extreme": float(args.max_volatility_extreme) if args.max_volatility_extreme is not None else 0.0020,
            }
        if mitigation_profile == "anti_adverse_v3":
            # Adaptive light-touch: reject only top-X volatility events, threshold per split from train slice.
            return {
                "vol_quantile_reject": max(0.0, min(0.50, float(args.vol_quantile_reject))),
            }
        if mitigation_profile == "anti_adverse_v4":
            # anti_adverse_v3 + conservative post-fill scratch.
            return {
                "vol_quantile_reject": max(0.0, min(0.50, float(args.vol_quantile_reject))),
                "scratch_bps": (float(args.scratch_bps) if float(args.scratch_bps) > 0.0 else 4.0),
                "scratch_window_sec": (int(args.scratch_window_sec) if int(args.scratch_window_sec) > 0 else 10),
                "scratch_taker_fee_bps": (float(args.scratch_taker_fee_bps) if float(args.scratch_taker_fee_bps) > 0.0 else 1.0),
                "scratch_slippage_bps": (float(args.scratch_slippage_bps) if float(args.scratch_slippage_bps) > 0.0 else 0.5),
            }
        if mitigation_profile == "anti_adverse_v5":
            return {
                "vol_quantile_reject": max(0.0, min(0.50, float(args.vol_quantile_reject))),
                "scratch_bps": (float(args.scratch_bps) if float(args.scratch_bps) > 0.0 else 4.0),
                "scratch_window_sec": (int(args.scratch_window_sec) if int(args.scratch_window_sec) > 0 else 10),
                "scratch_taker_fee_bps": (float(args.scratch_taker_fee_bps) if float(args.scratch_taker_fee_bps) > 0.0 else 1.0),
                "scratch_slippage_bps": (float(args.scratch_slippage_bps) if float(args.scratch_slippage_bps) > 0.0 else 0.5),
                "passive_max_wait_buckets": (int(args.passive_max_wait_buckets) if int(args.passive_max_wait_buckets) > 0 else 2),
            }
        if mitigation_profile == "anti_adverse_v6":
            return {
                "vol_quantile_reject": max(0.0, min(0.50, float(args.vol_quantile_reject))),
                "scratch_bps": (float(args.scratch_bps) if float(args.scratch_bps) > 0.0 else 4.0),
                "scratch_window_sec": (int(args.scratch_window_sec) if int(args.scratch_window_sec) > 0 else 10),
                "scratch_taker_fee_bps": (float(args.scratch_taker_fee_bps) if float(args.scratch_taker_fee_bps) > 0.0 else 1.0),
                "scratch_slippage_bps": (float(args.scratch_slippage_bps) if float(args.scratch_slippage_bps) > 0.0 else 0.5),
                "passive_max_wait_buckets": (int(args.passive_max_wait_buckets) if int(args.passive_max_wait_buckets) > 0 else 2),
                "exec_model": "passive_then_taker",
            }
        if mitigation_profile == "event_block_v1":
            return {
                "event_block_lanes": "book_proxy_pressure,volatility_burst",
            }
        if mitigation_profile == "event_block_book_proxy_v1":
            return {
                "event_block_lanes": "book_proxy_pressure",
            }
        if mitigation_profile == "event_block_volatility_v1":
            return {
                "event_block_lanes": "volatility_burst",
            }
        if mitigation_profile == "event_block_eth_v1":
            if str(c.get("symbol", "")).upper() != "ETHUSDT":
                return {}
            return {
                "event_block_lanes": "book_proxy_pressure,volatility_burst",
            }
        if mitigation_profile == "event_block_eth_micro_v1":
            if str(c.get("symbol", "")).upper() != "ETHUSDT":
                return {}
            if str(rule_name) != "micro_edge_v3_passive_alpha":
                return {}
            return {
                "event_block_lanes": "book_proxy_pressure,volatility_burst",
            }
        if mitigation_profile == "event_block_eth_micro_imb05_v1":
            if str(c.get("symbol", "")).upper() != "ETHUSDT":
                return {}
            if str(rule_name) != "micro_edge_v3_passive_alpha":
                return {}
            if float(c.get("min_imbalance", 0)) < 0.5:
                return {}
            return {
                "event_block_lanes": "book_proxy_pressure,volatility_burst",
            }
        if mitigation_profile == "event_block_eth_micro_imb085_v1":
            if str(c.get("symbol", "")).upper() != "ETHUSDT":
                return {}
            if str(rule_name) != "micro_edge_v3_passive_alpha":
                return {}
            if float(c.get("min_imbalance", 0)) < 0.85:
                return {}
            return {
                "event_block_lanes": "book_proxy_pressure,volatility_burst",
            }
        return {}  # baseline: honour the args values directly

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        for rule_name in rules:
            mit_overrides = _mitigation_overrides(c, str(rule_name))

            def _evaluate_grid(profile_overrides: Dict[str, float]) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for fee in fee_grid:
                    for adv in adverse_grid:
                        key = f"fee={fee:.3f}|adv={adv:.3f}"
                        res = validate_pocket_forward(
                            db=str(args.db),
                            symbol=str(c["symbol"]),
                            lookback_min=int(args.lookback_min),
                            bucket_sec=int(args.bucket_sec),
                            horizon_sec=(int(args.horizon_sec) if int(args.horizon_sec) > 0 else int(c["horizon_sec"])),
                            rule=str(rule_name),
                            side=str(args.side),
                            min_imbalance=float(c["min_imbalance"]),
                            min_trade_intensity=float(c["min_trade_intensity"]),
                            max_spread=float(c["max_spread"]),
                            splits=int(args.splits),
                            seeds=seed_str,
                            min_n=int(args.min_n),
                            min_n_frac=float(args.min_n_frac),
                            maker_fee_bps=float(fee),
                            passive_profile_in=str(args.passive_profile_in),
                            passive_max_wait_buckets=int(profile_overrides.get("passive_max_wait_buckets", eff_passive_max_wait_buckets)),
                            passive_adverse_mult=float(adv),
                            v2_min_score=float(args.v2_min_score),
                            v2_min_persistence=float(args.v2_min_persistence),
                            v2_min_confidence=float(args.v2_min_confidence),
                            min_intensity_strong=float(profile_overrides.get("min_intensity_strong", args.min_intensity_strong)),
                            min_imbalance_strong=float(profile_overrides.get("min_imbalance_strong", args.min_imbalance_strong)),
                            max_spread_tight=float(profile_overrides.get("max_spread_tight", args.max_spread_tight)),
                            max_volatility_extreme=float(profile_overrides.get("max_volatility_extreme", args.max_volatility_extreme if args.max_volatility_extreme is not None else 0.0)),
                            vol_quantile_reject=float(profile_overrides.get("vol_quantile_reject", 0.0)),
                            event_allow_lanes=str(profile_overrides.get("event_allow_lanes", "")),
                            event_block_lanes=str(profile_overrides.get("event_block_lanes", "")),
                            scratch_bps=float(profile_overrides.get("scratch_bps", args.scratch_bps)),
                            scratch_window_sec=int(profile_overrides.get("scratch_window_sec", args.scratch_window_sec)),
                            scratch_taker_fee_bps=float(profile_overrides.get("scratch_taker_fee_bps", args.scratch_taker_fee_bps)),
                            scratch_slippage_bps=float(profile_overrides.get("scratch_slippage_bps", args.scratch_slippage_bps)),
                            exec_model=str(profile_overrides.get("exec_model", "passive_realistic")),
                            regime_filter=str(args.regime).upper() if str(args.regime).lower() not in ("none", "") else "",
                        )
                        agg = _aggregate_eval(res, min_n=int(args.min_n))
                        attr = res.get("failure_attribution_median") or {}
                        agg["rows_total"] = int(res.get("rows_total", 0))
                        agg["pass_count"] = int(res.get("pass_count", 0))
                        agg["pass_rate_raw"] = float(res.get("pass_rate", 0.0))
                        agg["insufficient_fill_rate"] = float(res.get("insufficient_fill_rate", 0.0))
                        agg["event_filter"] = dict(res.get("event_filter") or {})
                        # Prefer explicit validator median attribution when present.
                        for k in [
                            "n_events_total",
                            "n_rejected_attempt_gate",
                            "n_attempts_after_gate",
                            "n_filled",
                            "n_unfilled",
                            "avg_fill_prob",
                            "avg_adverse_bps_on_fills",
                            "avg_fee_bps",
                            "avg_scratch_bps_on_fills",
                            "avg_raw_return_bps_on_fills",
                            "avg_net_return_bps_on_fills",
                            "net_return_bps_p10",
                            "net_return_bps_p50",
                            "net_return_bps_p90",
                            "reject_vol_quantile_reject",
                            "reject_spread_too_wide",
                            "reject_imbalance_too_low",
                            "reject_intensity_too_low",
                            "reject_other_gate",
                        ]:
                            if k in attr and attr.get(k) is not None:
                                agg[k] = attr.get(k)
                        # Recompute derived ratios from enriched fields.
                        agg["gate_reject_ratio"] = _safe_ratio(
                            agg.get("n_rejected_attempt_gate"),
                            agg.get("n_events_total"),
                        )
                        agg["fill_rate_after_gate"] = _safe_ratio(
                            agg.get("n_filled"),
                            agg.get("n_attempts_after_gate"),
                        )
                        out[key] = agg
                return out

            pocket_evals: Dict[str, Any] = _evaluate_grid(mit_overrides)
            baseline_evals: Dict[str, Any] = _evaluate_grid({}) if mitigation_profile != "baseline" else pocket_evals

            def get_eval(fee: float, adv: float) -> Dict[str, Any]:
                return pocket_evals.get(f"fee={fee:.3f}|adv={adv:.3f}", {})

            fee_min = min(fee_grid)
            fee_max = max(fee_grid)
            adv_max = max(adverse_grid)
            adv_one = min(adverse_grid, key=lambda x: abs(x - 1.0))
            fee_one = min(fee_grid, key=lambda x: abs(x - 1.0))

            core_eval = get_eval(fee_one, adv_one)
            stress_eval = get_eval(fee_one, adv_max)
            base_core_eval = baseline_evals.get(f"fee={fee_one:.3f}|adv={adv_one:.3f}", {})
            base_stress_eval = baseline_evals.get(f"fee={fee_one:.3f}|adv={adv_max:.3f}", {})
            failure_reason_top = _failure_reason_top(core_eval)
            core_decomp = _npa_decomposition_from_eval(core_eval)
            stress_decomp = _npa_decomposition_from_eval(stress_eval)

            # NPA-based scoring (net_per_attempt; capacity-honest metric)
            core_npa = float(core_eval.get("median_net_per_attempt", 0.0))
            stress_npa = float(stress_eval.get("median_net_per_attempt", 0.0))
            stab_bps = float(core_eval.get("stability_std", 0.0)) * 10000.0
            npa_bps = core_npa * 10000.0
            insufficient_fill_rate = float(core_eval.get("insufficient_fill_rate", 0.0))
            capacity_penalty = 1.0 + 0.5 * insufficient_fill_rate
            base_score = max(0.0, npa_bps) / (1.0 + max(0.0, stab_bps))

            # Robustness: edge must be positive at both core and stress conditions
            pass_rate_core = float(core_eval.get("pass_rate", 0.0))
            pass_rate_stress = float(stress_eval.get("pass_rate", 0.0))
            robust_core = (pass_rate_core >= float(args.pass_threshold)) and (core_npa > 0.0)
            robust_stress = (pass_rate_stress >= float(args.pass_threshold)) and (stress_npa > 0.0)
            final_score = (base_score / capacity_penalty) if (robust_core and robust_stress) else 0.0

            # Raw score fields — always populated regardless of robustness gate
            score_raw_core = float(get_eval(fee_min, adv_one).get("median_filled_avg_net", 0.0))
            score_raw_stress = float(get_eval(fee_max, adv_max).get("median_filled_avg_net", 0.0))
            score_raw_min = min(
                (float(e.get("median_filled_avg_net", 0.0)) for e in pocket_evals.values()),
                default=0.0,
            )
            base_score_raw_core = float(baseline_evals.get(f"fee={fee_min:.3f}|adv={adv_one:.3f}", {}).get("median_filled_avg_net", 0.0))
            base_pass_rate_core = float(base_core_eval.get("pass_rate", 0.0))
            base_pass_rate_stress = float(base_stress_eval.get("pass_rate", 0.0))
            base_npa_core = float(base_core_eval.get("median_net_per_attempt", 0.0))
            base_npa_stress = float(base_stress_eval.get("median_net_per_attempt", 0.0))

            # Capacity filter: skip pockets that cannot generate meaningful flow
            core_afr = float(core_eval.get("attempt_fill_rate_median", 0.0))
            core_eff_min = int(core_eval.get("median_effective_min_n", 0) or 0)
            if core_afr < float(args.min_attempt_fill_rate):
                print(
                    f"[cap_filter] skip symbol={c['symbol']} rule={rule_name} h={c['horizon_sec']} "
                    f"attempt_fill_rate={core_afr:.4f} < threshold={args.min_attempt_fill_rate:.4f} "
                    f"effective_min_n_median={core_eff_min}"
                )
                continue
            if insufficient_fill_rate > float(args.max_insufficient_fill_rate):
                cur_frac = float(args.min_n_frac)
                suggest_frac = max(0.00001, cur_frac * 0.5)
                suggest_splits = max(2, int(args.splits) - 1)
                suggest_min_n = max(10, int(args.min_n * 0.8))
                print(
                    f"[cap_filter] skip symbol={c['symbol']} rule={rule_name} h={c['horizon_sec']} "
                    f"insufficient_fill_rate={insufficient_fill_rate:.4f} > threshold={args.max_insufficient_fill_rate:.4f} "
                    f"effective_min_n_median={core_eff_min}. "
                    f"Hint: current min_n_frac={cur_frac:.6f}, min_n={int(args.min_n)}, splits={int(args.splits)}. "
                    f"Try lower --min-n-frac (e.g. {suggest_frac:.6f}), lower --min-n (e.g. {suggest_min_n}), or fewer --splits (e.g. {suggest_splits})."
                )
                continue

            scored.append(
                {
                    **c,
                    "rule": str(rule_name),
                    "score": float(final_score),
                    "stability_std_bps": float(stab_bps),
                    "insufficient_fill_rate": float(insufficient_fill_rate),
                    "robust_core": bool(robust_core),
                    "robust_stress": bool(robust_stress),
                    "score_raw_core": score_raw_core,
                    "score_raw_stress": score_raw_stress,
                    "score_raw_min": score_raw_min,
                    "net_per_attempt": core_npa,
                    "npa_core": core_npa,
                    "npa_stress": stress_npa,
                    "pass_rate_core": pass_rate_core,
                    "pass_rate_stress": pass_rate_stress,
                    "baseline_score_raw_core": base_score_raw_core,
                    "baseline_npa_core": base_npa_core,
                    "baseline_npa_stress": base_npa_stress,
                    "baseline_pass_rate_core": base_pass_rate_core,
                    "baseline_pass_rate_stress": base_pass_rate_stress,
                    "delta_score_raw_core": score_raw_core - base_score_raw_core,
                    "delta_npa_core": core_npa - base_npa_core,
                    "delta_npa_stress": stress_npa - base_npa_stress,
                    "delta_pass_rate_core": pass_rate_core - base_pass_rate_core,
                    "delta_pass_rate_stress": pass_rate_stress - base_pass_rate_stress,
                    "event_allow_lanes": list((core_eval.get("event_filter") or {}).get("allow_lanes", [])),
                    "event_block_lanes": list((core_eval.get("event_filter") or {}).get("block_lanes", [])),
                    "event_filter_kept_ratio": float((core_eval.get("event_filter") or {}).get("kept_ratio", 1.0) or 0.0),
                    "failure_reason_top": failure_reason_top,
                    "n_events_total": core_eval.get("n_events_total"),
                    "n_rejected_attempt_gate": core_eval.get("n_rejected_attempt_gate"),
                    "n_attempts_after_gate": core_eval.get("n_attempts_after_gate"),
                    "n_filled": core_eval.get("n_filled"),
                    "n_unfilled": core_eval.get("n_unfilled"),
                    "avg_fill_prob": core_eval.get("avg_fill_prob"),
                    "avg_adverse_bps_on_fills": core_eval.get("avg_adverse_bps_on_fills"),
                    "avg_fee_bps": core_eval.get("avg_fee_bps"),
                    "avg_scratch_bps_on_fills": core_eval.get("avg_scratch_bps_on_fills"),
                    "avg_raw_return_bps_on_fills": core_eval.get("avg_raw_return_bps_on_fills"),
                    "avg_net_return_bps_on_fills": core_eval.get("avg_net_return_bps_on_fills"),
                    "net_return_bps_p10": core_eval.get("net_return_bps_p10"),
                    "net_return_bps_p50": core_eval.get("net_return_bps_p50"),
                    "net_return_bps_p90": core_eval.get("net_return_bps_p90"),
                    "reject_breakdown": {
                        "vol_quantile_reject": int(_safe_float(core_eval.get("reject_vol_quantile_reject", 0), 0.0)),
                        "spread_too_wide": int(_safe_float(core_eval.get("reject_spread_too_wide", 0), 0.0)),
                        "imbalance_too_low": int(_safe_float(core_eval.get("reject_imbalance_too_low", 0), 0.0)),
                        "intensity_too_low": int(_safe_float(core_eval.get("reject_intensity_too_low", 0), 0.0)),
                        "other_gate": int(_safe_float(core_eval.get("reject_other_gate", 0), 0.0)),
                    },
                    "effective_trade_count_core": {
                        "n_events_total": int(_safe_float(core_eval.get("n_events_total", 0), 0.0)),
                        "n_attempts_after_gate": int(_safe_float(core_eval.get("n_attempts_after_gate", 0), 0.0)),
                        "n_filled": int(_safe_float(core_eval.get("n_filled", 0), 0.0)),
                    },
                    "effective_trade_count_stress": {
                        "n_events_total": int(_safe_float(stress_eval.get("n_events_total", 0), 0.0)),
                        "n_attempts_after_gate": int(_safe_float(stress_eval.get("n_attempts_after_gate", 0), 0.0)),
                        "n_filled": int(_safe_float(stress_eval.get("n_filled", 0), 0.0)),
                    },
                    "decomposition_core": core_decomp,
                    "decomposition_stress": stress_decomp,
                    "gate_reject_ratio": core_eval.get("gate_reject_ratio"),
                    "fill_rate_after_gate": core_eval.get("fill_rate_after_gate"),
                    "attempt_fill_rate": core_afr,
                    "attempts_per_min": float(core_eval.get("attempts_per_min_median", 0.0)),
                    "per_combo": list(core_eval.get("per_combo", [])),
                    "best_fee_survive": max(
                        [f for f in fee_grid if float(get_eval(f, adv_one).get("pass_rate", 0.0)) > 0.0] or [0.0]
                    ),
                    "evals": pocket_evals,
                }
            )

    # Primary sort: score (gated); secondary: score_raw_core so zero-gated pockets remain comparable
    scored.sort(
        key=lambda r: (
            float(r.get("score", 0.0)),
            float(r.get("pass_rate_stress", 0.0)),
            float(r.get("npa_stress", 0.0)),
            float(r.get("score_raw_core", 0.0)),
        ),
        reverse=True,
    )

    if bool(args.bootstrap_ci):
        pvals: List[Tuple[int, float]] = []
        for i, row in enumerate(scored):
            per_combo = list(row.get("per_combo", []))
            npa_vals = [float(x.get("net_per_attempt", 0.0) or 0.0) for x in per_combo]
            lo, hi, p_one = _bootstrap_mean_ci(
                npa_vals,
                samples=int(args.bootstrap_samples),
                seed=int(args.bootstrap_seed) + i,
            )
            row["bootstrap_ci_low"] = float(lo)
            row["bootstrap_ci_high"] = float(hi)
            row["bootstrap_p_value"] = float(p_one)
            pvals.append((i, p_one))
        mtc = str(args.mtc_method).strip().lower()
        if bool(args.bh_correction) and mtc == "none":
            mtc = "bh"
        if mtc == "bh":
            qvals = _bh_adjust(pvals)
        elif mtc == "bonferroni":
            qvals = _bonferroni_adjust(pvals)
        else:
            qvals = {}
        for i, row in enumerate(scored):
            qv = float(qvals.get(i, row.get("bootstrap_p_value", 1.0)))
            row["bootstrap_q_value"] = qv
            sig = qv <= float(args.alpha)
            row["multiple_testing_method"] = str(mtc)
            row["significant"] = bool(sig)
            if not bool(sig):
                row["robust_core"] = False
                row["robust_stress"] = False
                row["score"] = 0.0

    top10 = scored[:10]
    print("top 10 pockets overall")
    for i, r in enumerate(top10, start=1):
        print(
            f"{i:2d}. {r['symbol']} rule={r['rule']} h={r['horizon_sec']} imb>={r['min_imbalance']:.2f} "
            f"int>={r['min_trade_intensity']:.0f} spr<={r['max_spread']:.6f} "
            f"score={r['score']:.6e} npa={r.get('net_per_attempt', 0.0):.6e} "
            f"pass_core={r.get('pass_rate_core', 0.0):.2%} pass_stress={r.get('pass_rate_stress', 0.0):.2%} "
            f"afr={r.get('attempt_fill_rate', 0.0):.2%} "
            f"score_raw_core={r.get('score_raw_core', 0.0):.6e} "
            f"failure_reason_top={r.get('failure_reason_top', 'mixed')}"
        )
    for sym in sorted(set(r["symbol"] for r in scored)):
        print(f"top 5 {sym}")
        cnt = 0
        for r in scored:
            if r["symbol"] != sym:
                continue
            print(
                f" - rule={r['rule']} h={r['horizon_sec']} imb>={r['min_imbalance']:.2f} int>={r['min_trade_intensity']:.0f} "
                f"spr<={r['max_spread']:.6f} score={r['score']:.6e} npa={r.get('net_per_attempt', 0.0):.6e} "
                f"pass_core={r.get('pass_rate_core', 0.0):.2%} pass_stress={r.get('pass_rate_stress', 0.0):.2%} "
                f"score_raw_core={r.get('score_raw_core', 0.0):.6e} "
                f"failure_reason_top={r.get('failure_reason_top', 'mixed')}"
            )
            cnt += 1
            if cnt >= 5:
                break
    survive = sum(1 for r in scored if float(r["evals"].get("fee=1.000|adv=1.000", {}).get("pass_rate", 0.0)) >= 0.5)
    liq_impact = _summarize_liquidation_scoring_impact(scored)
    print(f"pockets survive fee>=1.0 with pass_rate>=0.5: {survive}")
    if bool(liq_impact.get("available")):
        print(
            "liquidation_scoring_impact "
            f"count={int(liq_impact.get('count', 0))} "
            f"positive_delta_score_count={int(liq_impact.get('positive_delta_score_count', 0))} "
            f"avg_delta_score_raw_core={float(liq_impact.get('avg_delta_score_raw_core', 0.0)):+.6e} "
            f"avg_delta_npa_core={float(liq_impact.get('avg_delta_npa_core', 0.0)):+.6e} "
            f"avg_delta_pass_rate_core={float(liq_impact.get('avg_delta_pass_rate_core', 0.0)):+.2%}"
        )
    if bool(args.diagnostic_breakdown) and scored:
        top = scored[0]
        maker_bps = float(top.get("avg_fee_bps", 0.0) or 0.0)
        maker_frac = maker_bps / 10000.0
        scratch_bps = float(top.get("avg_scratch_bps_on_fills", 0.0) or 0.0)
        scratch_frac = scratch_bps / 10000.0
        print("[diagnostic_breakdown] top pocket:", f"{top.get('symbol')} rule={top.get('rule')} h={top.get('horizon_sec')}")
        print(
            "[diagnostic_breakdown] bps_to_fraction",
            f"maker_fee_bps={maker_bps:.6f} -> {maker_frac:.8f}",
            f"scratch_bps={scratch_bps:.6f} -> {scratch_frac:.8f}",
            f"scratch_taker_fee_bps={float(eff_scratch_taker_fee_bps):.6f} -> {float(eff_scratch_taker_fee_bps)/10000.0:.8f}",
            f"scratch_slippage_bps={float(eff_scratch_slippage_bps):.6f} -> {float(eff_scratch_slippage_bps)/10000.0:.8f}",
        )
        if abs(maker_bps) > 0.0 and abs(maker_frac / maker_bps - 0.0001) > 1e-12:
            print("WARNING fee conversion sanity check failed for maker fee bps.")
        if abs(scratch_bps) > 0.0 and abs(scratch_frac / scratch_bps - 0.0001) > 1e-12:
            print("WARNING fee conversion sanity check failed for scratch bps.")

    out_json = Path(str(args.out_json))
    payload = {
        "count": len(scored),
        "mitigation_profile": mitigation_profile,
        "gate_config": {
            "min_intensity_strong": float(args.min_intensity_strong),
            "min_imbalance_strong": float(args.min_imbalance_strong),
            "max_spread_tight": float(args.max_spread_tight),
            "max_volatility_extreme": None if args.max_volatility_extreme is None else float(args.max_volatility_extreme),
            "vol_quantile_reject": float(args.vol_quantile_reject),
            "scratch_bps": float(eff_scratch_bps),
            "scratch_window_sec": int(eff_scratch_window_sec),
            "scratch_taker_fee_bps": float(eff_scratch_taker_fee_bps),
            "scratch_slippage_bps": float(eff_scratch_slippage_bps),
            "passive_max_wait_buckets": int(eff_passive_max_wait_buckets),
            "horizon_sec_override": int(args.horizon_sec),
            "event_allow_lanes": [],
            "event_block_lanes": (
                ["book_proxy_pressure"]
                if mitigation_profile == "event_block_book_proxy_v1"
                else (
                    ["volatility_burst"]
                    if mitigation_profile == "event_block_volatility_v1"
                    else (
                        ["book_proxy_pressure", "volatility_burst"]
                        if mitigation_profile in {"event_block_v1", "event_block_eth_v1", "event_block_eth_micro_v1", "event_block_eth_micro_imb05_v1", "event_block_eth_micro_imb085_v1"}
                        else []
                    )
                )
            ),
            "event_profile_lane_scope": (
                ["book_proxy_pressure"]
                if mitigation_profile == "event_block_book_proxy_v1"
                else (
                    ["volatility_burst"]
                    if mitigation_profile == "event_block_volatility_v1"
                    else (
                        ["book_proxy_pressure", "volatility_burst"]
                        if mitigation_profile in {"event_block_v1", "event_block_eth_v1", "event_block_eth_micro_v1", "event_block_eth_micro_imb05_v1", "event_block_eth_micro_imb085_v1"}
                        else []
                    )
                )
            ),
            "event_profile_symbol_scope": ("ETHUSDT" if mitigation_profile in {"event_block_eth_v1", "event_block_eth_micro_v1", "event_block_eth_micro_imb05_v1", "event_block_eth_micro_imb085_v1"} else None),
            "event_profile_rule_scope": ("micro_edge_v3_passive_alpha" if mitigation_profile in {"event_block_eth_micro_v1", "event_block_eth_micro_imb05_v1", "event_block_eth_micro_imb085_v1"} else None),
        },
        "statistical": {
            "bootstrap_ci": bool(args.bootstrap_ci),
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.bootstrap_seed),
            "alpha": float(args.alpha),
            "multiple_testing_method": (
                ("bh" if bool(args.bh_correction) and str(args.mtc_method).strip().lower() == "none" else str(args.mtc_method).strip().lower())
                if bool(args.bootstrap_ci)
                else "none"
            ),
            "splits": int(args.splits),
        },
        "decomposition": [
            {
                "pocket": _pocket_id(
                    r,
                    rule=str(r.get("rule", "")),
                    side=str(args.side),
                    horizon_override=(int(args.horizon_sec) if int(args.horizon_sec) > 0 else 0),
                ),
                "n_samples": int(_safe_float(r.get("n_events_total", 0), 0.0)),
                "score_raw_core": float(r.get("score_raw_core", 0.0)),
                "gross_edge_npa": float((r.get("decomposition_core") or {}).get("gross_edge_npa", 0.0)),
                "fee_cost_npa": float((r.get("decomposition_core") or {}).get("fee_cost_npa", 0.0)),
                "adverse_cost_npa": float((r.get("decomposition_core") or {}).get("adverse_cost_npa", 0.0)),
                "scratch_cost_npa": float((r.get("decomposition_core") or {}).get("scratch_cost_npa", 0.0)),
                "net_npa": float((r.get("decomposition_core") or {}).get("net_npa", 0.0)),
                "residual_npa": float((r.get("decomposition_core") or {}).get("residual_npa", 0.0)),
                "pass_core": float(r.get("pass_rate_core", 0.0)),
                "pass_stress": float(r.get("pass_rate_stress", 0.0)),
                "reject_breakdown": dict(r.get("reject_breakdown", {})),
                "effective_trade_count_core": dict(r.get("effective_trade_count_core", {})),
                "effective_trade_count_stress": dict(r.get("effective_trade_count_stress", {})),
            }
            for r in scored
        ],
        "liquidation_scoring_impact": liq_impact,
        "ranking": scored,
    }
    payload["run_summary"] = build_run_summary(
        run_type="rank_passive_pockets_forward",
        inputs={"candidates_md": str(args.candidates_md), "db": str(args.db), "rules": rules, "min_n_frac": float(args.min_n_frac)},
        metrics={"count": len(scored), "candidate_count": len(candidates), "survive_fee1_passrate_ge_0_5": int(survive)},
        artifacts={"json": str(out_json), "md": str(args.out_md)},
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md = Path(str(args.out_md))
    lines = [
        "# PASSIVE_POCKET_RANKING",
        "",
        f"candidates={len(candidates)} ranked={len(scored)}",
        (
            f"statistical bootstrap_ci={bool(args.bootstrap_ci)} bootstrap_samples={int(args.bootstrap_samples)} "
            f"alpha={float(args.alpha):.4f} mtc_method="
            f"{('bh' if bool(args.bh_correction) and str(args.mtc_method).strip().lower() == 'none' else str(args.mtc_method).strip().lower()) if bool(args.bootstrap_ci) else 'none'} "
            f"splits={int(args.splits)} (recommended=5 for 60-day retest)"
        ),
        f"candidate_parse total_rows_seen={total_rows_seen} table_rows_seen={table_rows_seen} rows_with_pass_yes={rows_with_pass_yes} candidates_parsed={candidates_parsed} candidates_unique={len(candidates)} rows_skipped_missing_fields={rows_skipped_missing}",
        f"fee_grid={fee_grid} adverse_mult_grid={adverse_grid}",
        f"pass_threshold={float(args.pass_threshold):.3f}",
        (
            f"liquidation_scoring_impact available={bool(liq_impact.get('available'))} "
            f"count={int(liq_impact.get('count', 0))} "
            f"positive_delta_score_count={int(liq_impact.get('positive_delta_score_count', 0))} "
            f"avg_delta_score_raw_core={float(liq_impact.get('avg_delta_score_raw_core', 0.0)):+.6e} "
            f"avg_delta_npa_core={float(liq_impact.get('avg_delta_npa_core', 0.0)):+.6e} "
            f"avg_delta_pass_rate_core={float(liq_impact.get('avg_delta_pass_rate_core', 0.0)):+.2%}"
        ),
        (
            f"mitigation_profile={mitigation_profile} gate_config "
            f"min_intensity_strong={float(args.min_intensity_strong):.6f} "
            f"min_imbalance_strong={float(args.min_imbalance_strong):.6f} "
            f"max_spread_tight={float(args.max_spread_tight):.6f} "
            f"max_volatility_extreme={args.max_volatility_extreme} "
            f"vol_quantile_reject={float(args.vol_quantile_reject):.6f} "
            f"scratch_bps={float(eff_scratch_bps):.4f} scratch_window_sec={int(eff_scratch_window_sec)} "
            f"scratch_taker_fee_bps={float(eff_scratch_taker_fee_bps):.4f} scratch_slippage_bps={float(eff_scratch_slippage_bps):.4f} "
            f"passive_max_wait_buckets={int(eff_passive_max_wait_buckets)} "
            f"horizon_sec_override={int(args.horizon_sec)}"
        ),
        "",
        "| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, r in enumerate(scored, start=1):
        lines.append(
            f"| {idx} | {r['symbol']} | {r['rule']} | {r['horizon_sec']} | {r['min_imbalance']:.2f} | {r['min_trade_intensity']:.0f} | {r['max_spread']:.6f} | "
            f"{r['score']:.6e} | {r['robust_core']} | {r['robust_stress']} | "
            f"{r.get('pass_rate_core', 0.0):.2%} | {r.get('pass_rate_stress', 0.0):.2%} | {r.get('npa_core', 0.0):+.6e} | {r.get('npa_stress', 0.0):+.6e} | "
            f"{r.get('attempt_fill_rate', 0.0):.2%} | {r.get('attempts_per_min', 0.0):.2f} | "
            f"{r.get('score_raw_core', 0.0):+.8f} | {r.get('score_raw_stress', 0.0):+.8f} | {r.get('score_raw_min', 0.0):+.8f} | "
            f"{r.get('baseline_pass_rate_core', 0.0):.2%} | {r.get('baseline_pass_rate_stress', 0.0):.2%} | "
            f"{r.get('baseline_npa_core', 0.0):+.6e} | {r.get('baseline_score_raw_core', 0.0):+.8f} | "
            f"{r.get('delta_pass_rate_core', 0.0):+.2%} | {r.get('delta_npa_core', 0.0):+.6e} | {r.get('delta_score_raw_core', 0.0):+.8f} | "
            f"{r.get('failure_reason_top', 'mixed')} | "
            f"{(r.get('gate_reject_ratio') if r.get('gate_reject_ratio') is not None else 0.0):.2%} | "
            f"{(r.get('fill_rate_after_gate') if r.get('fill_rate_after_gate') is not None else 0.0):.2%} | "
            f"{_safe_float(r.get('avg_fee_bps'), 0.0):.3f} | {_safe_float(r.get('avg_adverse_bps_on_fills'), 0.0):.3f} | "
            f"{_safe_float(r.get('avg_raw_return_bps_on_fills'), 0.0):+.3f} | {_safe_float(r.get('avg_net_return_bps_on_fills'), 0.0):+.3f} | "
            f"{r['stability_std_bps']:.3f} | {r['best_fee_survive']:.2f} | {float(r.get('insufficient_fill_rate', 0.0)):.2%} |"
        )
    lines += [
        "",
        f"survive_fee1_passrate_ge_0.5={survive}",
        "",
        "## Decomposition",
        "",
        "| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, r in enumerate(scored, start=1):
        dc = r.get("decomposition_core") or {}
        ec = r.get("effective_trade_count_core") or {}
        lines.append(
            f"| {idx} | {r.get('symbol')} | {r.get('rule')} | {int(_safe_float(r.get('horizon_sec', 0), 0.0))} | "
            f"{_safe_float(dc.get('gross_edge_npa', 0.0), 0.0):+.6e} | "
            f"{_safe_float(dc.get('fee_cost_npa', 0.0), 0.0):+.6e} | "
            f"{_safe_float(dc.get('adverse_cost_npa', 0.0), 0.0):+.6e} | "
            f"{_safe_float(dc.get('scratch_cost_npa', 0.0), 0.0):+.6e} | "
            f"{_safe_float(dc.get('net_npa', 0.0), 0.0):+.6e} | "
            f"{_safe_float(dc.get('observed_net_npa', 0.0), 0.0):+.6e} | "
            f"{_safe_float(dc.get('residual_npa', 0.0), 0.0):+.6e} | "
            f"{(_safe_float(r.get('gate_reject_ratio', 0.0), 0.0)):.2%} | "
            f"{int(_safe_float(ec.get('n_events_total', 0), 0.0))} | "
            f"{int(_safe_float(ec.get('n_attempts_after_gate', 0), 0.0))} | "
            f"{int(_safe_float(ec.get('n_filled', 0), 0.0))} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    if bool(args.emit_fee_cliff_summary):
        fee_out = out_json.with_name(f"{out_json.stem}.fee_cliff.json")
        fee_rows: List[Dict[str, Any]] = []
        for r in scored:
            fee_rows.append(
                {
                    "pocket": _pocket_id(
                        r,
                        rule=str(r.get("rule", "")),
                        side=str(args.side),
                        horizon_override=(int(args.horizon_sec) if int(args.horizon_sec) > 0 else 0),
                    ),
                    "symbol": r.get("symbol"),
                    "rule": r.get("rule"),
                    "horizon_sec": int(_safe_float(r.get("horizon_sec", 0), 0.0)),
                    "score_raw_core": float(r.get("score_raw_core", 0.0)),
                    "npa_core": float(r.get("npa_core", 0.0)),
                    "pass_rate_core": float(r.get("pass_rate_core", 0.0)),
                    "decomposition_core": dict(r.get("decomposition_core") or {}),
                }
            )
        fee_out.write_text(
            json.dumps(
                {
                    "maker_fee_bps_grid": fee_grid,
                    "passive_adverse_mult_grid": adverse_grid,
                    "rows": fee_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {fee_out}")
    if str(args.regime).lower() not in ("none", ""):
        min_attempts = min(
            (
                int(row.get("val_attempts", 999))
                for s in scored
                for row in s.get("per_combo", [])
                if row.get("val_attempts") is not None
            ),
            default=999,
        )
        if min_attempts < 50:
            print(
                f"WARN regime_filter={args.regime}: minimum val_attempts per fold = {min_attempts} < 50. "
                "Results may be noisy. Consider collecting more data or relaxing the pocket filter."
            )
        else:
            print(f"regime_filter={args.regime}: minimum val_attempts per fold = {min_attempts} (OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
