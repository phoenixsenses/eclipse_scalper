from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IN_JSON = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.json"
FALLBACK_JSON = ROOT / "reports" / "research" / "s34" / "S34_CROSS_ASSET_ABSORPTION_POOL.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V3_ENERGY_LATENT_MODEL.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V3_ENERGY_LATENT_MODEL.md"

ENERGY_FEATURES = ("running_notional", "running_accel")
STATIC_FEATURES = ("bid_depth_usd", "total_top_depth_usd", "book_imbalance", "spread_bps")
MIN_SPLIT_N = 40


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [finite(r.get("net_bps")) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate_pct": None,
            "t3r_bps": 0.0,
            "max_loss_bps": None,
            "tail_lt_100": 0,
            "tail_lt_200": 0,
        }
    ordered = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "sum_bps": round(sum(vals), 1),
        "mean_bps": round(sum(vals) / len(vals), 1),
        "median_bps": round(median(vals), 1),
        "win_rate_pct": round(100.0 * sum(1 for v in vals if v > 0) / len(vals), 1),
        "t3r_bps": round(sum(ordered[3:]) if len(ordered) > 3 else sum(vals), 1),
        "max_loss_bps": round(min(vals), 1),
        "tail_lt_100": sum(1 for v in vals if v < -100.0),
        "tail_lt_200": sum(1 for v in vals if v < -200.0),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted((v, i) for i, v in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for _, idx in indexed[i:j]:
            out[idx] = rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return pearson(ranks(xs), ranks(ys))


def percentile(vals: list[float], q: float) -> float | None:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return None
    idx = int(round((len(xs) - 1) * q))
    return xs[max(0, min(len(xs) - 1, idx))]


def split_rows(rows: list[dict[str, Any]], payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    hold_months = set(payload.get("split", {}).get("holdout_months", []))
    cal = [r for r in rows if str(r.get("month")) not in hold_months]
    hold = [r for r in rows if str(r.get("month")) in hold_months]
    return cal, hold, hold_months


def cal_route_stats(cal_rows: list[dict[str, Any]], features: tuple[str, ...]) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for route in sorted({str(r["route_id"]) for r in cal_rows}):
        rrows = [r for r in cal_rows if str(r["route_id"]) == route]
        out[route] = {}
        for feature in features:
            vals = [finite(r.get(feature)) for r in rrows]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                continue
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
            if sd > 0:
                out[route][feature] = (mu, sd)
    return out


def annotate_z(rows: list[dict[str, Any]], stats: dict[str, dict[str, tuple[float, float]]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        route_stats = stats.get(str(row["route_id"]), {})
        energy_parts = []
        static_parts = []
        for feature in ENERGY_FEATURES + STATIC_FEATURES:
            value = finite(row.get(feature))
            pair = route_stats.get(feature)
            if value is None or not pair:
                continue
            mu, sd = pair
            z = (value - mu) / sd
            r[f"{feature}_cal_route_z"] = z
            if feature in ENERGY_FEATURES:
                energy_parts.append(z)
            else:
                static_parts.append(z)
        if energy_parts:
            r["energy_z"] = sum(energy_parts) / len(energy_parts)
        if static_parts:
            r["static_structure_z"] = sum(static_parts) / len(static_parts)
        out.append(r)
    return out


def feature_corr(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    xs = []
    ys = []
    for row in rows:
        x = finite(row.get(feature))
        y = finite(row.get("net_bps"))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return {
        "feature": feature,
        "n": len(xs),
        "pearson": None if len(xs) < 3 else round(float(pearson(xs, ys) or 0.0), 4),
        "spearman": None if len(xs) < 3 else round(float(spearman(xs, ys) or 0.0), 4),
    }


def gate_by_cut(
    rows: list[dict[str, Any]],
    feature: str,
    cut: float,
    *,
    high_label: str = "high",
    low_label: str = "low",
) -> dict[str, Any]:
    high = [r for r in rows if (v := finite(r.get(feature))) is not None and v >= cut]
    low = [r for r in rows if (v := finite(r.get(feature))) is not None and v < cut]
    return {
        "feature": feature,
        "cut": round(cut, 4),
        high_label: metrics(high),
        low_label: metrics(low),
        "delta_high_low_sum_bps": round(float(metrics(high)["sum_bps"] or 0.0) - float(metrics(low)["sum_bps"] or 0.0), 1),
        "delta_high_low_t3r_bps": round(float(metrics(high)["t3r_bps"] or 0.0) - float(metrics(low)["t3r_bps"] or 0.0), 1),
    }


def confluence(rows: list[dict[str, Any]], feature: str, cut: float) -> list[dict[str, Any]]:
    base = [r for r in rows if (v := finite(r.get(feature))) is not None and v >= cut]
    specs = [
        ("energy_high", lambda r: True),
        ("energy_high+sync", lambda r: str(r.get("sync_gate")) == "sync"),
        ("energy_high+idio", lambda r: str(r.get("sync_gate")) == "idio"),
        ("energy_high+mixed", lambda r: str(r.get("absorption_gate")) == "mixed"),
        ("energy_high+not_absorbed", lambda r: str(r.get("absorption_gate")) != "absorbed"),
        ("energy_high+sync+mixed", lambda r: str(r.get("sync_gate")) == "sync" and str(r.get("absorption_gate")) == "mixed"),
        ("energy_high+sync+not_absorbed", lambda r: str(r.get("sync_gate")) == "sync" and str(r.get("absorption_gate")) != "absorbed"),
    ]
    out = []
    for name, fn in specs:
        sub = [r for r in base if fn(r)]
        out.append({"confluence": name, "summary": metrics(sub)})
    out.sort(key=lambda r: (float(r["summary"]["t3r_bps"] or -1e18), float(r["summary"]["sum_bps"] or -1e18)), reverse=True)
    return out


def by_symbol(rows: list[dict[str, Any]], feature: str, cut: float) -> list[dict[str, Any]]:
    out = []
    for symbol in sorted({str(r["symbol"]) for r in rows}):
        srows = [r for r in rows if str(r["symbol"]) == symbol]
        out.append(
            {
                "symbol": symbol,
                "all": metrics(srows),
                "high": metrics([r for r in srows if (v := finite(r.get(feature))) is not None and v >= cut]),
                "low": metrics([r for r in srows if (v := finite(r.get(feature))) is not None and v < cut]),
            }
        )
    return out


def by_route(rows: list[dict[str, Any]], feature: str, cut: float) -> list[dict[str, Any]]:
    out = []
    for route in sorted({str(r["route_id"]) for r in rows}):
        rrows = [r for r in rows if str(r["route_id"]) == route]
        high = [r for r in rrows if (v := finite(r.get(feature))) is not None and v >= cut]
        low = [r for r in rrows if (v := finite(r.get(feature))) is not None and v < cut]
        out.append(
            {
                "route_id": route,
                "all": metrics(rrows),
                "high": metrics(high),
                "low": metrics(low),
                "delta_t3r_bps": round(float(metrics(high)["t3r_bps"] or 0.0) - float(metrics(low)["t3r_bps"] or 0.0), 1),
            }
        )
    out.sort(key=lambda r: (r["delta_t3r_bps"], r["high"]["n"]), reverse=True)
    return out


def build_report(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    rows = list(payload["rows"])
    cal, hold, _ = split_rows(rows, payload)
    stats = cal_route_stats(cal, ENERGY_FEATURES + STATIC_FEATURES)
    zrows = annotate_z(rows, stats)
    zcal, zhold, _ = split_rows(zrows, payload)

    features = [
        "running_notional_cal_route_z",
        "running_accel_cal_route_z",
        "energy_z",
        "bid_depth_usd_cal_route_z",
        "total_top_depth_usd_cal_route_z",
        "book_imbalance_cal_route_z",
        "spread_bps_cal_route_z",
        "static_structure_z",
    ]
    cuts = {}
    for feature in ("running_notional_cal_route_z", "running_accel_cal_route_z", "energy_z", "static_structure_z"):
        vals = [finite(r.get(feature)) for r in zcal]
        vals = [v for v in vals if v is not None]
        if vals:
            cuts[feature] = {
                "median": percentile(vals, 0.5),
                "q75": percentile(vals, 0.75),
                "q90": percentile(vals, 0.9),
            }
    energy_cut = float(cuts["energy_z"]["q75"]) if "energy_z" in cuts and cuts["energy_z"]["q75"] is not None else 0.0

    gate_tests = []
    for feature, cutset in cuts.items():
        for name, cut in cutset.items():
            if cut is None:
                continue
            gate_tests.append(
                {
                    "feature": feature,
                    "cut_name": name,
                    "cut": round(float(cut), 4),
                    "cal": gate_by_cut(zcal, feature, float(cut)),
                    "hold": gate_by_cut(zhold, feature, float(cut)),
                    "all": gate_by_cut(zrows, feature, float(cut)),
                }
            )
    gate_tests.sort(
        key=lambda r: (
            float(r["hold"]["high"]["t3r_bps"] or -1e18),
            float(r["hold"]["high"]["sum_bps"] or -1e18),
            float(r["hold"]["high"]["n"] or 0),
        ),
        reverse=True,
    )

    return {
        "generated_at_utc": utc_now(),
        "source": str(source),
        "discipline": {
            "zscore_stats": "Calibration route mean/std only; holdout never used to define z-score.",
            "min_split_n": MIN_SPLIT_N,
        },
        "split": payload.get("split", {}),
        "coverage": {
            "rows": len(rows),
            "cal_rows": len(zcal),
            "hold_rows": len(zhold),
            "z_energy_rows": sum(1 for r in zrows if finite(r.get("energy_z")) is not None),
        },
        "correlations": {
            "cal": [feature_corr(zcal, f) for f in features],
            "hold": [feature_corr(zhold, f) for f in features],
            "all": [feature_corr(zrows, f) for f in features],
        },
        "cuts": cuts,
        "gate_tests": gate_tests,
        "energy_cut_q75": round(energy_cut, 4),
        "holdout_energy_confluence": confluence(zhold, "energy_z", energy_cut),
        "cal_energy_confluence": confluence(zcal, "energy_z", energy_cut),
        "by_symbol_holdout": by_symbol(zhold, "energy_z", energy_cut),
        "by_route_holdout": by_route(zhold, "energy_z", energy_cut),
        "rows": zrows,
    }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s['n']} sum={s['sum_bps']} med={s['median_bps']} "
        f"T3R={s['t3r_bps']} max_loss={s['max_loss_bps']} tail<-100={s['tail_lt_100']}"
    )


def corr_fmt(c: Any) -> str:
    return "NA" if c is None else f"{float(c):.3f}"


def render(report: dict[str, Any]) -> str:
    cal_corr = {r["feature"]: r for r in report["correlations"]["cal"]}
    hold_corr = {r["feature"]: r for r in report["correlations"]["hold"]}
    all_corr = {r["feature"]: r for r in report["correlations"]["all"]}
    lines = [
        "# S34 v3 Energy Latent Model",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live/paper/executor changes.",
        "",
        "## Discipline",
        "",
        f"- Z-score rule: {report['discipline']['zscore_stats']}",
        f"- Min split N for candidate promotion: `{report['discipline']['min_split_n']}`.",
        "",
        "## Coverage",
        "",
    ]
    for key, value in report["coverage"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Correlation: Route-Normalized Features vs Net Bps",
        "",
        "| Feature | Cal N | Cal Pearson | Cal Spearman | Hold N | Hold Pearson | Hold Spearman | All Pearson |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature, c in cal_corr.items():
        h = hold_corr.get(feature, {})
        a = all_corr.get(feature, {})
        lines.append(
            f"| `{feature}` | {c['n']} | {corr_fmt(c['pearson'])} | {corr_fmt(c['spearman'])} | "
            f"{h.get('n', 0)} | {corr_fmt(h.get('pearson'))} | {corr_fmt(h.get('spearman'))} | {corr_fmt(a.get('pearson'))} |"
        )
    lines += [
        "",
        "## Gate Tests By Calibration Cuts",
        "",
        "| Rank | Feature | Cut | Calibration High | Holdout High | Holdout Low | Hold dT3R |",
        "| ---: | --- | ---: | --- | --- | --- | ---: |",
    ]
    for idx, row in enumerate(report["gate_tests"], start=1):
        lines.append(
            f"| {idx} | `{row['feature']}:{row['cut_name']}` | {row['cut']} | "
            f"{fmt(row['cal']['high'])} | {fmt(row['hold']['high'])} | {fmt(row['hold']['low'])} | "
            f"{row['hold']['delta_high_low_t3r_bps']} |"
        )
    lines += [
        "",
        "## Holdout Energy Confluence",
        "",
        f"Energy high cut: `energy_z >= {report['energy_cut_q75']}` from calibration q75.",
        "",
        "| Rank | Confluence | Holdout summary |",
        "| ---: | --- | --- |",
    ]
    for idx, row in enumerate(report["holdout_energy_confluence"], start=1):
        lines.append(f"| {idx} | `{row['confluence']}` | {fmt(row['summary'])} |")
    lines += [
        "",
        "## Holdout By Symbol",
        "",
        "| Symbol | All | Energy high | Energy low |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["by_symbol_holdout"]:
        lines.append(f"| `{row['symbol']}` | {fmt(row['all'])} | {fmt(row['high'])} | {fmt(row['low'])} |")
    lines += [
        "",
        "## Best Holdout Route-Level Energy Deltas",
        "",
        "| Rank | Route | All | Energy high | Energy low | dT3R |",
        "| ---: | --- | --- | --- | --- | ---: |",
    ]
    for idx, row in enumerate(report["by_route_holdout"][:15], start=1):
        lines.append(
            f"| {idx} | `{row['route_id']}` | {fmt(row['all'])} | {fmt(row['high'])} | {fmt(row['low'])} | {row['delta_t3r_bps']} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "- Energy is only useful if calibration-cut high energy improves holdout without collapsing N.",
        "- If high-energy confluence is still tail-heavy, it is a sizing/context variable, not a standalone entry rule.",
        "- N<40 per split remains a hypothesis even when the row looks strong.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    source = IN_JSON if IN_JSON.exists() else FALLBACK_JSON
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = build_report(payload, source)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
