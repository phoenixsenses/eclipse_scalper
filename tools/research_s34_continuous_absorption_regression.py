from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


IN_JSON = Path("reports/research/s34/S34_CROSS_ASSET_ABSORPTION_POOL.json")
OUT_JSON = Path("reports/research/s34/S34_CONTINUOUS_ABSORPTION_REGRESSION.json")
OUT_MD = Path("reports/research/s34/S34_CONTINUOUS_ABSORPTION_REGRESSION.md")

TARGET = "net_bps"
FEATURES = [
    "book_imbalance",
    "bid_depth_usd",
    "ask_depth_usd",
    "total_top_depth_usd",
    "spread_bps",
    "vdepth_bps",
    "running_notional",
    "running_liq_count",
    "running_accel",
]


def clean_float(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def summarize(rows):
    vals = [clean_float(r.get(TARGET)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "sum": 0.0, "mean": 0.0, "median": 0.0, "win_rate": 0.0, "t3r": 0.0, "max_loss": 0.0, "tail_lt_100": 0}
    ordered = sorted(vals, reverse=True)
    t3r = sum(ordered[3:]) if len(ordered) > 3 else sum(vals)
    return {
        "n": len(vals),
        "sum": round(sum(vals), 1),
        "mean": round(sum(vals) / len(vals), 1),
        "median": round(median(vals), 1),
        "win_rate": round(100.0 * sum(1 for v in vals if v > 0) / len(vals), 1),
        "t3r": round(t3r, 1),
        "max_loss": round(min(vals), 1),
        "tail_lt_100": sum(1 for v in vals if v < -100),
    }


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(values):
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


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    return pearson(ranks(xs), ranks(ys))


def regression_slope(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def feature_pairs(rows, feature):
    xs = []
    ys = []
    for row in rows:
        x = clean_float(row.get(feature))
        y = clean_float(row.get(TARGET))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def route_zscore_rows(rows):
    by_route = defaultdict(list)
    for row in rows:
        by_route[row["route_id"]].append(row)

    stats = {}
    for route, rrows in by_route.items():
        route_stats = {}
        for feature in FEATURES:
            vals = [clean_float(r.get(feature)) for r in rrows]
            vals = [v for v in vals if v is not None]
            if len(vals) < 3:
                continue
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
            if sd > 0:
                route_stats[feature] = (mu, sd)
        stats[route] = route_stats

    out = []
    for row in rows:
        zrow = dict(row)
        for feature in FEATURES:
            pair = stats.get(row["route_id"], {}).get(feature)
            value = clean_float(row.get(feature))
            if pair and value is not None:
                mu, sd = pair
                zrow[f"{feature}_route_z"] = (value - mu) / sd
        out.append(zrow)
    return out


def corr_table(rows, features):
    table = []
    for feature in features:
        xs, ys = feature_pairs(rows, feature)
        row = {
            "feature": feature,
            "n": len(xs),
            "pearson": None,
            "spearman": None,
            "slope_bps_per_unit": None,
        }
        if len(xs) >= 3:
            row["pearson"] = pearson(xs, ys)
            row["spearman"] = spearman(xs, ys)
            row["slope_bps_per_unit"] = regression_slope(xs, ys)
        table.append(row)
    return table


def quantile_edges(values):
    vals = sorted(values)
    if not vals:
        return []
    def q(p):
        idx = int(round((len(vals) - 1) * p))
        return vals[max(0, min(len(vals) - 1, idx))]

    return [q(0.25), q(0.5), q(0.75)]


def quartile_screen(rows, feature):
    vals = [clean_float(r.get(feature)) for r in rows]
    vals = [v for v in vals if v is not None]
    if len(vals) < 12:
        return None
    q1, q2, q3 = quantile_edges(vals)
    low = [r for r in rows if (clean_float(r.get(feature)) is not None and clean_float(r.get(feature)) <= q1)]
    high = [r for r in rows if (clean_float(r.get(feature)) is not None and clean_float(r.get(feature)) >= q3)]
    return {
        "feature": feature,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "low": summarize(low),
        "high": summarize(high),
        "delta_high_low_t3r": round(summarize(high)["t3r"] - summarize(low)["t3r"], 1),
        "delta_high_low_sum": round(summarize(high)["sum"] - summarize(low)["sum"], 1),
    }


def split_rows(rows, split):
    holdout_months = set(split.get("holdout_months", []))
    cal = [r for r in rows if r.get("month") not in holdout_months]
    hold = [r for r in rows if r.get("month") in holdout_months]
    return cal, hold


def fmt_corr(value):
    if value is None:
        return "NA"
    return f"{value:.3f}"


def fmt_summary(s):
    return (
        f"N={s['n']} sum={s['sum']:.1f} med={s['median']:.1f} "
        f"T3R={s['t3r']:.1f} max_loss={s['max_loss']:.1f} tail<-100={s['tail_lt_100']}"
    )


def main():
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    rows = data["rows"]
    cal, hold = split_rows(rows, data.get("split", {}))
    zrows = route_zscore_rows(rows)
    zcal, zhold = split_rows(zrows, data.get("split", {}))

    route_z_features = [f"{feature}_route_z" for feature in FEATURES]
    raw_corr = {
        "all": corr_table(rows, FEATURES),
        "calibration": corr_table(cal, FEATURES),
        "holdout": corr_table(hold, FEATURES),
    }
    route_z_corr = {
        "all": corr_table(zrows, route_z_features),
        "calibration": corr_table(zcal, route_z_features),
        "holdout": corr_table(zhold, route_z_features),
    }
    quartiles = [q for q in (quartile_screen(rows, f) for f in FEATURES) if q]
    zquartiles = [q for q in (quartile_screen(zrows, f) for f in route_z_features) if q]

    by_symbol = {}
    for symbol in sorted({r["symbol"] for r in rows}):
        srows = [r for r in rows if r["symbol"] == symbol]
        by_symbol[symbol] = {
            "summary": summarize(srows),
            "raw_corr": corr_table(srows, ["book_imbalance", "bid_depth_usd", "spread_bps", "vdepth_bps"]),
        }

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(IN_JSON),
        "n": len(rows),
        "split": data.get("split", {}),
        "overall": summarize(rows),
        "calibration": summarize(cal),
        "holdout": summarize(hold),
        "raw_corr": raw_corr,
        "route_z_corr": route_z_corr,
        "quartiles": quartiles,
        "route_z_quartiles": zquartiles,
        "by_symbol": by_symbol,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# S34 Continuous Absorption Regression",
        "",
        f"Generated: `{output['generated_at_utc']}`",
        "",
        "Research-only. Uses cross-asset pooled rows; no live/paper/executor changes.",
        "",
        "## Sample",
        "",
        f"- Overall: {fmt_summary(output['overall'])}",
        f"- Calibration: {fmt_summary(output['calibration'])}",
        f"- Holdout: {fmt_summary(output['holdout'])}",
        "",
        "## Raw Feature Correlation vs Net Bps",
        "",
        "| Feature | N | Pearson all | Spearman all | Pearson cal | Pearson hold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    cal_by_feature = {r["feature"]: r for r in raw_corr["calibration"]}
    hold_by_feature = {r["feature"]: r for r in raw_corr["holdout"]}
    for row in raw_corr["all"]:
        lines.append(
            f"| `{row['feature']}` | {row['n']} | {fmt_corr(row['pearson'])} | "
            f"{fmt_corr(row['spearman'])} | {fmt_corr(cal_by_feature[row['feature']]['pearson'])} | "
            f"{fmt_corr(hold_by_feature[row['feature']]['pearson'])} |"
        )

    lines += [
        "",
        "## Route-Normalized Feature Correlation",
        "",
        "Each feature is z-scored inside its own `route_id` before correlation. This tests whether a route has unusually high/low absorption relative to its own baseline.",
        "",
        "| Feature | N | Pearson all | Spearman all | Pearson cal | Pearson hold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    zcal_by_feature = {r["feature"]: r for r in route_z_corr["calibration"]}
    zhold_by_feature = {r["feature"]: r for r in route_z_corr["holdout"]}
    for row in route_z_corr["all"]:
        lines.append(
            f"| `{row['feature']}` | {row['n']} | {fmt_corr(row['pearson'])} | "
            f"{fmt_corr(row['spearman'])} | {fmt_corr(zcal_by_feature[row['feature']]['pearson'])} | "
            f"{fmt_corr(zhold_by_feature[row['feature']]['pearson'])} |"
        )

    lines += [
        "",
        "## Quartile Extremes",
        "",
        "| Feature | Low quartile | High quartile | Delta high-low T3R |",
        "| --- | --- | --- | ---: |",
    ]
    for row in sorted(quartiles, key=lambda r: r["delta_high_low_t3r"], reverse=True):
        lines.append(
            f"| `{row['feature']}` | {fmt_summary(row['low'])} | {fmt_summary(row['high'])} | {row['delta_high_low_t3r']:.1f} |"
        )

    lines += [
        "",
        "## Route-Normalized Quartile Extremes",
        "",
        "| Feature | Low route-z quartile | High route-z quartile | Delta high-low T3R |",
        "| --- | --- | --- | ---: |",
    ]
    for row in sorted(zquartiles, key=lambda r: r["delta_high_low_t3r"], reverse=True):
        lines.append(
            f"| `{row['feature']}` | {fmt_summary(row['low'])} | {fmt_summary(row['high'])} | {row['delta_high_low_t3r']:.1f} |"
        )

    lines += [
        "",
        "## By Symbol Snapshot",
        "",
        "| Symbol | Summary | book_imbalance r | bid_depth r | spread r | vdepth r |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for symbol, payload in by_symbol.items():
        corr = {r["feature"]: r for r in payload["raw_corr"]}
        lines.append(
            f"| `{symbol}` | {fmt_summary(payload['summary'])} | "
            f"{fmt_corr(corr['book_imbalance']['pearson'])} | "
            f"{fmt_corr(corr['bid_depth_usd']['pearson'])} | "
            f"{fmt_corr(corr['spread_bps']['pearson'])} | "
            f"{fmt_corr(corr['vdepth_bps']['pearson'])} |"
        )

    lines += [
        "",
        "## Read",
        "",
        "- Treat this as diagnostics, not model selection. The sample is pooled across route definitions.",
        "- A sign flip between raw and route-normalized features means the pooled binary gate is probably mixing route identity with absorption.",
        "- A feature is interesting only if the sign is directionally stable in calibration and holdout.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:80]))


if __name__ == "__main__":
    main()
