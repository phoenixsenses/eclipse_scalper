from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from data.labels.forward_return import direction_label


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_record(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True, sort_keys=True)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
        f.flush()


def parse_jsonl_lines(lines: Iterable[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in lines:
        s = str(raw or "").strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return parse_jsonl_lines(path.read_text(encoding="utf-8", errors="ignore").splitlines())


def extract_best_rule_delta(record: Dict[str, Any]) -> Optional[float]:
    return extract_best_rule_delta_min_n(record, min_rule_n=0)


def extract_best_rule_delta_min_n(record: Dict[str, Any], min_rule_n: int = 0) -> Optional[float]:
    rules = record.get("naive_rules")
    if not isinstance(rules, dict) or not rules:
        return None
    best: Optional[float] = None
    for _, rv in rules.items():
        if not isinstance(rv, dict):
            continue
        try:
            n = int(rv.get("n", 0) or 0)
        except Exception:
            n = 0
        if n < int(max(0, min_rule_n)):
            continue
        d = rv.get("delta_vs_baseline")
        if d is None:
            continue
        try:
            x = float(d)
        except Exception:
            continue
        if best is None or x > best:
            best = x
    return best


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if q <= 0:
        return vals[0]
    if q >= 1:
        return vals[-1]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def build_bucket_features(
    trades: Sequence[Dict[str, float | int]],
    marks: Sequence[Dict[str, float | int]],
    bucket_sec: int,
    vol_window: int = 12,
) -> List[Dict[str, Optional[float]]]:
    bms = max(1, int(bucket_sec)) * 1000
    mark_buckets: Dict[int, float] = {}
    for m in marks:
        ts = int(m.get("ts_ms", 0) or 0)
        px = m.get("mark_price")
        if px is None:
            continue
        try:
            pxf = float(px)
        except Exception:
            continue
        b = (ts // bms) * bms
        mark_buckets[b] = pxf

    agg: Dict[int, Dict[str, float]] = {}
    for t in trades:
        ts = int(t.get("ts_ms", 0) or 0)
        b = (ts // bms) * bms
        row = agg.get(b)
        if row is None:
            row = {
                "trade_count": 0.0,
                "sum_abs_qty": 0.0,
                "signed_qty": 0.0,
                "sum_qty": 0.0,
                "sum_abs_dev_ratio": 0.0,
                "spread_n": 0.0,
            }
            agg[b] = row
        try:
            q = float(t.get("quantity", 0.0) or 0.0)
            p = float(t.get("price", 0.0) or 0.0)
            is_bm = int(t.get("is_buyer_maker", 0) or 0)
        except Exception:
            continue
        signed = -q if is_bm == 1 else q
        row["trade_count"] += 1.0
        row["sum_abs_qty"] += abs(q)
        row["signed_qty"] += signed
        row["sum_qty"] += q
        mp = mark_buckets.get(b)
        if mp is not None and mp > 0:
            row["sum_abs_dev_ratio"] += abs(p - mp) / mp
            row["spread_n"] += 1.0

    all_buckets = sorted(set(mark_buckets.keys()) | set(agg.keys()))
    out: List[Dict[str, Optional[float]]] = []
    prev_mid: Optional[float] = None
    logrets: List[float] = []
    last_mark: Optional[float] = None
    for b in all_buckets:
        if b in mark_buckets:
            last_mark = mark_buckets[b]
        mid = last_mark
        a = agg.get(b, {})
        trade_count = float(a.get("trade_count", 0.0))
        sum_abs_qty = float(a.get("sum_abs_qty", 0.0))
        signed_qty = float(a.get("signed_qty", 0.0))
        spread_proxy = None
        if float(a.get("spread_n", 0.0)) > 0:
            spread_proxy = float(a.get("sum_abs_dev_ratio", 0.0)) / float(a.get("spread_n", 1.0))
        imbalance = None
        if sum_abs_qty > 0:
            imbalance = signed_qty / sum_abs_qty
        trade_intensity = trade_count * (60.0 / max(1.0, float(bucket_sec)))
        volume = float(a.get("sum_qty", 0.0))
        ret_1 = None
        if mid is not None and prev_mid is not None and mid > 0 and prev_mid > 0:
            ret_1 = math.log(mid / prev_mid)
            logrets.append(ret_1)
        if mid is not None and mid > 0:
            prev_mid = mid
        micro_vol = None
        if len(logrets) >= 2:
            w = logrets[-max(2, int(vol_window)) :]
            mu = sum(w) / len(w)
            var = sum((x - mu) ** 2 for x in w) / (len(w) - 1)
            micro_vol = math.sqrt(max(0.0, var))
        liq_proxy = None
        if spread_proxy is not None:
            liq_proxy = volume / (1.0 + spread_proxy)
        out.append(
            {
                "ts_ms": float(b),
                "mid": mid,
                "spread": spread_proxy,
                "imbalance": imbalance,
                "trade_intensity": trade_intensity,
                "volume": volume,
                "ret_1": ret_1,
                "micro_volatility": micro_vol,
                "liquidity_proxy": liq_proxy,
            }
        )
    return out


def evaluate_naive_rules(
    rows: Sequence[Dict[str, Optional[float]]],
    labels: Sequence[Optional[int]],
    baseline_hit_rate: Optional[float],
) -> Dict[str, Dict[str, Optional[float] | int]]:
    def _hit(pred: List[int], act: List[int]) -> Optional[float]:
        if not pred or not act or len(pred) != len(act):
            return None
        return sum(1 for p, a in zip(pred, act) if p == a) / len(pred)

    imbs = [float(r["imbalance"]) for r in rows if r.get("imbalance") is not None]
    ints = [float(r["trade_intensity"]) for r in rows if r.get("trade_intensity") is not None]
    sprs = [float(r["spread"]) for r in rows if r.get("spread") is not None]
    q10_imb = quantile(imbs, 0.10)
    q90_imb = quantile(imbs, 0.90)
    q90_int = quantile(ints, 0.90)
    q90_spr = quantile(sprs, 0.90)
    rules: Dict[str, Dict[str, Optional[float] | int]] = {}

    def _build(name: str, pred: List[int], act: List[int]) -> None:
        hit = _hit(pred, act)
        rules[name] = {
            "n": len(pred),
            "hit_rate": hit,
            "baseline": baseline_hit_rate,
            "delta_vs_baseline": (None if hit is None or baseline_hit_rate is None else hit - baseline_hit_rate),
        }

    pred: List[int] = []
    act: List[int] = []
    if q90_imb is not None:
        for i, r in enumerate(rows):
            lbl = labels[i] if i < len(labels) else None
            imb = r.get("imbalance")
            if lbl in (None, 0) or imb is None:
                continue
            if float(imb) > q90_imb:
                pred.append(1)
                act.append(int(lbl))
    if pred:
        _build("imbalance_gt_q90_up", pred, act)

    pred, act = [], []
    if q10_imb is not None:
        for i, r in enumerate(rows):
            lbl = labels[i] if i < len(labels) else None
            imb = r.get("imbalance")
            if lbl in (None, 0) or imb is None:
                continue
            if float(imb) < q10_imb:
                pred.append(-1)
                act.append(int(lbl))
    if pred:
        _build("imbalance_lt_q10_down", pred, act)

    pred, act = [], []
    if q90_int is not None:
        for i, r in enumerate(rows):
            lbl = labels[i] if i < len(labels) else None
            ti = r.get("trade_intensity")
            imb = r.get("imbalance")
            if lbl in (None, 0) or ti is None or imb is None:
                continue
            if float(ti) > q90_int and abs(float(imb)) > 0.10:
                side = rule_predicted_side("intensity_spike_imbalance_cont", r, default_side="LONG")
                if side is None:
                    continue
                pred.append(1 if side == "LONG" else -1)
                act.append(int(lbl))
    if pred:
        _build("intensity_spike_imbalance_cont", pred, act)

    pred, act = [], []
    if q90_spr is not None:
        for i, r in enumerate(rows):
            lbl = labels[i] if i < len(labels) else None
            sp = r.get("spread")
            r1 = r.get("ret_1")
            if lbl in (None, 0) or sp is None or r1 is None:
                continue
            if float(sp) > q90_spr and float(r1) != 0:
                side = rule_predicted_side("spread_spike_reversal", r, default_side="LONG")
                if side is None:
                    continue
                pred.append(1 if side == "LONG" else -1)
                act.append(int(lbl))
    if pred:
        _build("spread_spike_reversal", pred, act)

    pred, act = [], []
    thr = compute_rule_thresholds(rows)
    for i, r in enumerate(rows):
        lbl = labels[i] if i < len(labels) else None
        if lbl in (None, 0):
            continue
        if rule_fires("micro_edge_v2_passive_alpha", r, thr):
            side = rule_predicted_side("micro_edge_v2_passive_alpha", r, default_side="LONG")
            if side is None:
                continue
            pred.append(1 if side == "LONG" else -1)
            act.append(int(lbl))
    if pred:
        _build("micro_edge_v2_passive_alpha", pred, act)

    pred, act = [], []
    for i, r in enumerate(rows):
        lbl = labels[i] if i < len(labels) else None
        if lbl in (None, 0):
            continue
        if rule_fires("micro_edge_v3_passive_alpha", r, thr):
            side = rule_predicted_side("micro_edge_v3_passive_alpha", r, default_side="LONG")
            if side is None:
                continue
            pred.append(1 if side == "LONG" else -1)
            act.append(int(lbl))
    if pred:
        _build("micro_edge_v3_passive_alpha", pred, act)

    return rules


def filter_rules_min_n(
    rules: Dict[str, Dict[str, Optional[float] | int]],
    min_rule_n: int,
) -> Dict[str, Dict[str, Optional[float] | int]]:
    out: Dict[str, Dict[str, Optional[float] | int]] = {}
    thr = int(max(0, min_rule_n))
    for name, rv in (rules or {}).items():
        try:
            n = int((rv or {}).get("n", 0) or 0)
        except Exception:
            n = 0
        if n >= thr:
            out[name] = rv
    return out


def compute_rule_thresholds(rows: Sequence[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    imbs = [float(r["imbalance"]) for r in rows if r.get("imbalance") is not None]
    ints = [float(r["trade_intensity"]) for r in rows if r.get("trade_intensity") is not None]
    sprs = [float(r["spread"]) for r in rows if r.get("spread") is not None]
    v2_scores = [abs(float(r["v2_score"])) for r in rows if r.get("v2_score") is not None]
    v2_persist = [abs(float(r["v2_imbalance_persist"])) for r in rows if r.get("v2_imbalance_persist") is not None]
    v3_scores = [abs(float(r["v3_score"])) for r in rows if r.get("v3_score") is not None]
    v3_persist = [abs(float(r["v2_imbalance_persist"])) for r in rows if r.get("v2_imbalance_persist") is not None]
    v3_shrink = [float(r["v3_spread_shrink_ratio"]) for r in rows if r.get("v3_spread_shrink_ratio") is not None]
    v3_int_slope = [float(r["v3_intensity_slope"]) for r in rows if r.get("v3_intensity_slope") is not None]
    return {
        "imb_q10": quantile(imbs, 0.10),
        "imb_q90": quantile(imbs, 0.90),
        "int_q90": quantile(ints, 0.90),
        "spr_q90": quantile(sprs, 0.90),
        "v2_score_q70": quantile(v2_scores, 0.70),
        "v2_score_q80": quantile(v2_scores, 0.80),
        "v2_persist_q60": quantile(v2_persist, 0.60),
        "v3_score_q80": quantile(v3_scores, 0.80),
        "v3_persist_q60": quantile(v3_persist, 0.60),
        "v3_shrink_q60": quantile(v3_shrink, 0.60),
        "v3_int_slope_q60": quantile(v3_int_slope, 0.60),
    }


def infer_rule_side(rule_name: str) -> str:
    rn = str(rule_name or "").lower()
    if "down" in rn or "reversal" in rn:
        return "SHORT"
    return "LONG"


def rule_predicted_side(
    rule_name: str,
    row: Dict[str, Optional[float]],
    default_side: str = "LONG",
) -> Optional[str]:
    rn = str(rule_name or "")
    if rn == "intensity_spike_imbalance_cont":
        imb = row.get("imbalance")
        if imb is None:
            return None
        if float(imb) == 0.0:
            return None
        return "LONG" if float(imb) > 0 else "SHORT"
    if rn == "spread_spike_reversal":
        r1 = row.get("ret_1")
        if r1 is None or float(r1) == 0.0:
            return None
        return "SHORT" if float(r1) > 0 else "LONG"
    if rn == "micro_edge_v2_passive_alpha":
        ss = row.get("v2_side_signal")
        if ss is None or float(ss) == 0.0:
            return None
        return "LONG" if float(ss) > 0 else "SHORT"
    if rn == "micro_edge_v3_passive_alpha":
        ss = row.get("v3_side_signal")
        if ss is None or float(ss) == 0.0:
            return None
        return "LONG" if float(ss) > 0 else "SHORT"
    return str(default_side).upper()


def rule_fires(
    rule_name: str,
    row: Dict[str, Optional[float]],
    thresholds: Dict[str, Optional[float]],
) -> bool:
    rn = str(rule_name or "")
    imb = row.get("imbalance")
    ti = row.get("trade_intensity")
    sp = row.get("spread")
    r1 = row.get("ret_1")
    if rn == "imbalance_gt_q90_up":
        q = thresholds.get("imb_q90")
        return imb is not None and q is not None and float(imb) > float(q)
    if rn == "imbalance_lt_q10_down":
        q = thresholds.get("imb_q10")
        return imb is not None and q is not None and float(imb) < float(q)
    if rn == "intensity_spike_imbalance_cont":
        q = thresholds.get("int_q90")
        return ti is not None and imb is not None and q is not None and float(ti) > float(q) and abs(float(imb)) > 0.10
    if rn == "spread_spike_reversal":
        q = thresholds.get("spr_q90")
        return sp is not None and r1 is not None and q is not None and float(sp) > float(q) and float(r1) != 0.0
    if rn == "micro_edge_v2_passive_alpha":
        score = row.get("v2_score")
        persist = row.get("v2_imbalance_persist")
        conf = row.get("v2_confidence")
        if score is None or persist is None:
            return False
        min_score = thresholds.get("v2_min_score")
        if min_score is None:
            min_score = thresholds.get("v2_score_q80")
        min_persist = thresholds.get("v2_min_persistence")
        if min_persist is None:
            min_persist = thresholds.get("v2_persist_q60")
        min_conf = thresholds.get("v2_min_confidence")
        if min_conf is None:
            min_conf = 0.50
        return (
            abs(float(score)) >= float(min_score or 0.0)
            and abs(float(persist)) >= float(min_persist or 0.0)
            and float(conf or 0.0) >= float(min_conf)
        )
    if rn == "micro_edge_v3_passive_alpha":
        score = row.get("v3_score")
        persist = row.get("v2_imbalance_persist")
        conf = row.get("v3_confidence")
        shrink = row.get("v3_spread_shrink_ratio")
        int_slope = row.get("v3_intensity_slope")
        if score is None or persist is None or conf is None or shrink is None or int_slope is None:
            return False
        min_score = thresholds.get("v3_min_score")
        if min_score is None:
            min_score = thresholds.get("v3_score_q80")
        min_persist = thresholds.get("v3_min_persistence")
        if min_persist is None:
            min_persist = thresholds.get("v3_persist_q60")
        min_conf = thresholds.get("v3_min_confidence")
        if min_conf is None:
            min_conf = 0.55
        min_shrink = thresholds.get("v3_min_spread_shrink")
        if min_shrink is None:
            min_shrink = thresholds.get("v3_shrink_q60")
        min_int_slope = thresholds.get("v3_min_intensity_slope")
        if min_int_slope is None:
            min_int_slope = thresholds.get("v3_int_slope_q60")
        return (
            abs(float(score)) >= float(min_score or 0.0)
            and abs(float(persist)) >= float(min_persist or 0.0)
            and float(conf or 0.0) >= float(min_conf)
            and float(shrink or 0.0) >= float(min_shrink or 0.0)
            and float(int_slope or 0.0) >= float(min_int_slope or 0.0)
        )
    return False


def signal_aligned_forward_returns(
    mids: Sequence[Optional[float]],
    horizon_steps: int,
) -> List[Optional[float]]:
    h = max(1, int(horizon_steps))
    n = len(mids)
    out: List[Optional[float]] = [None] * n
    # rule fires at t, trade enters at t+1, exits at t+1+h
    for t in range(n):
        e = t + 1
        x = e + h
        if x >= n:
            out[t] = None
            continue
        m_e = mids[e]
        m_x = mids[x]
        if m_e is None or m_x is None:
            out[t] = None
            continue
        entry = float(m_e)
        exitp = float(m_x)
        if entry <= 0.0 or exitp <= 0.0:
            out[t] = None
            continue
        out[t] = (exitp / entry) - 1.0
    return out


def signal_aligned_labels(
    mids: Sequence[Optional[float]],
    horizon_steps: int,
    threshold: float,
) -> tuple[List[Optional[float]], List[Optional[int]]]:
    rets = signal_aligned_forward_returns(mids, horizon_steps=horizon_steps)
    labels: List[Optional[int]] = []
    for r in rets:
        if r is None:
            labels.append(None)
        else:
            labels.append(direction_label(float(r), float(threshold)))
    return rets, labels


def label_value_to_text(lbl: Optional[int]) -> str:
    if lbl is None:
        return "none"
    if int(lbl) > 0:
        return "up"
    if int(lbl) < 0:
        return "down"
    return "flat"
