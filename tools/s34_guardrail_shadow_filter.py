from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Iterable


DEFAULT_DB = Path("data/s34_intelligence.db")
DEFAULT_MD = Path("reports/research/s34/S34_GUARDRAIL_SHADOW_FILTER.md")
DEFAULT_JSON = Path("reports/research/s34/S34_GUARDRAIL_SHADOW_FILTER.json")


@dataclass
class TradeRow:
    trade_id: str
    signal_id: str
    rule_name: str
    signal_ts_utc: str
    exit_reason: str
    net_bps: float
    guardrail_level: str
    guardrail_headline: str


def _connect(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_closed_trades(conn: sqlite3.Connection) -> list[TradeRow]:
    rows = conn.execute(
        """
        SELECT
            o.trade_id,
            o.signal_id,
            o.rule_name,
            s.signal_ts_utc,
            o.exit_reason,
            COALESCE(o.net_bps, 0.0) AS net_bps,
            COALESCE(g.level, 'missing') AS guardrail_level,
            COALESCE(g.headline, 'missing guardrail') AS guardrail_headline
        FROM s34_outcomes o
        JOIN s34_signals s ON s.signal_id = o.signal_id
        LEFT JOIN s34_model_guardrails g ON g.signal_id = o.signal_id
        ORDER BY o.exit_ts_ms ASC, o.trade_id ASC
        """
    ).fetchall()
    return [
        TradeRow(
            trade_id=str(r["trade_id"]),
            signal_id=str(r["signal_id"]),
            rule_name=str(r["rule_name"]),
            signal_ts_utc=str(r["signal_ts_utc"]),
            exit_reason=str(r["exit_reason"]),
            net_bps=float(r["net_bps"]),
            guardrail_level=str(r["guardrail_level"]),
            guardrail_headline=str(r["guardrail_headline"]),
        )
        for r in rows
    ]


def _pct(value: float) -> float:
    return round(value * 100.0, 2)


def _metrics(trades: Iterable[TradeRow]) -> dict[str, object]:
    rows = list(trades)
    nets = [t.net_bps for t in rows]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    return {
        "n": len(rows),
        "cum_net_bps": round(sum(nets), 2) if nets else 0.0,
        "mean_net_bps": round(mean(nets), 2) if nets else 0.0,
        "median_net_bps": round(median(nets), 2) if nets else 0.0,
        "win_rate_pct": _pct(len(wins) / len(rows)) if rows else 0.0,
        "loss_rate_pct": _pct(len(losses) / len(rows)) if rows else 0.0,
        "avg_win_bps": round(mean(wins), 2) if wins else 0.0,
        "avg_loss_bps": round(mean(losses), 2) if losses else 0.0,
    }


def _scenario_keep(levels_to_skip: set[str]):
    def keep(row: TradeRow) -> bool:
        return row.guardrail_level not in levels_to_skip

    return keep


def _scenario_only(levels_to_keep: set[str]):
    def keep(row: TradeRow) -> bool:
        return row.guardrail_level in levels_to_keep

    return keep


SCENARIOS = {
    "baseline_all_closed": ("Keep every closed trade", lambda row: True),
    "skip_warning": ("Skip guardrail level WARNING", _scenario_keep({"warning"})),
    "skip_warning_caution": (
        "Skip guardrail levels WARNING and CAUTION",
        _scenario_keep({"warning", "caution"}),
    ),
    "only_ok": ("Take only guardrail level OK", _scenario_only({"ok"})),
}


def _evaluate_scenarios(trades: list[TradeRow]) -> dict[str, dict[str, object]]:
    baseline = _metrics(trades)
    result: dict[str, dict[str, object]] = {}
    for scenario, (description, predicate) in SCENARIOS.items():
        kept = [t for t in trades if predicate(t)]
        skipped = [t for t in trades if not predicate(t)]
        m = _metrics(kept)
        m.update(
            {
                "description": description,
                "kept_n": len(kept),
                "skipped_n": len(skipped),
                "kept_ratio_pct": _pct(len(kept) / len(trades)) if trades else 0.0,
                "delta_cum_vs_baseline_bps": round(
                    float(m["cum_net_bps"]) - float(baseline["cum_net_bps"]), 2
                ),
                "skipped_cum_net_bps": round(sum(t.net_bps for t in skipped), 2),
            }
        )
        result[scenario] = m
    return result


def _evaluate_by_rule(trades: list[TradeRow]) -> dict[str, dict[str, dict[str, object]]]:
    rules = sorted({t.rule_name for t in trades})
    return {rule: _evaluate_scenarios([t for t in trades if t.rule_name == rule]) for rule in rules}


def _level_breakdown(trades: list[TradeRow]) -> dict[str, dict[str, object]]:
    levels = sorted({t.guardrail_level for t in trades})
    return {level: _metrics([t for t in trades if t.guardrail_level == level]) for level in levels}


def _skipped_examples(trades: list[TradeRow], level: str = "warning", limit: int = 10) -> dict[str, list[dict[str, object]]]:
    skipped = [t for t in trades if t.guardrail_level == level]
    winners = sorted([t for t in skipped if t.net_bps > 0], key=lambda t: t.net_bps, reverse=True)[:limit]
    losers = sorted([t for t in skipped if t.net_bps <= 0], key=lambda t: t.net_bps)[:limit]

    def pack(row: TradeRow) -> dict[str, object]:
        return {
            "trade_id": row.trade_id,
            "rule_name": row.rule_name,
            "signal_ts_utc": row.signal_ts_utc,
            "exit_reason": row.exit_reason,
            "net_bps": round(row.net_bps, 2),
            "guardrail_headline": row.guardrail_headline,
        }

    return {
        "skipped_warning_winners": [pack(t) for t in winners],
        "skipped_warning_losers": [pack(t) for t in losers],
    }


def _table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _write_report(md_path: Path, payload: dict[str, object]) -> None:
    scenarios = payload["overall_scenarios"]
    levels = payload["level_breakdown"]
    by_rule = payload["by_rule"]
    examples = payload["examples"]

    scenario_rows = []
    for name, m in scenarios.items():
        scenario_rows.append(
            [
                name,
                m["kept_n"],
                m["skipped_n"],
                m["kept_ratio_pct"],
                m["cum_net_bps"],
                m["delta_cum_vs_baseline_bps"],
                m["median_net_bps"],
                m["win_rate_pct"],
            ]
        )

    level_rows = []
    for level, m in levels.items():
        level_rows.append(
            [
                level,
                m["n"],
                m["cum_net_bps"],
                m["mean_net_bps"],
                m["median_net_bps"],
                m["win_rate_pct"],
                m["loss_rate_pct"],
            ]
        )

    rule_rows = []
    for rule, scenarios_for_rule in by_rule.items():
        base = scenarios_for_rule["baseline_all_closed"]
        skip_warning = scenarios_for_rule["skip_warning"]
        only_ok = scenarios_for_rule["only_ok"]
        rule_rows.append(
            [
                rule,
                base["n"],
                base["cum_net_bps"],
                base["median_net_bps"],
                skip_warning["kept_n"],
                skip_warning["cum_net_bps"],
                skip_warning["delta_cum_vs_baseline_bps"],
                only_ok["kept_n"],
                only_ok["cum_net_bps"],
            ]
        )

    winner_rows = [
        [
            x["trade_id"],
            x["rule_name"],
            x["exit_reason"],
            x["net_bps"],
            x["guardrail_headline"],
        ]
        for x in examples["skipped_warning_winners"]
    ]
    loser_rows = [
        [
            x["trade_id"],
            x["rule_name"],
            x["exit_reason"],
            x["net_bps"],
            x["guardrail_headline"],
        ]
        for x in examples["skipped_warning_losers"]
    ]

    lines = [
        "# S34 Guardrail Shadow Filter",
        "",
        f"Generated at: `{payload['generated_at_utc']}`",
        "",
        "Scope: closed S34 paper trades in `data/s34_intelligence.db`. This is a paper-only counterfactual. It does not change the runner, config, or live rules.",
        "",
        "## Overall Scenarios",
        "",
        _table(
            [
                "Scenario",
                "Kept N",
                "Skipped N",
                "Kept %",
                "Cum Net",
                "Delta vs Base",
                "Median",
                "WR %",
            ],
            scenario_rows,
        ),
        "",
        "## Guardrail Level Breakdown",
        "",
        _table(["Level", "N", "Cum Net", "Mean", "Median", "WR %", "Loss %"], level_rows),
        "",
        "## Rule-Level Shadow Result",
        "",
        _table(
            [
                "Rule",
                "Base N",
                "Base Cum",
                "Base Median",
                "Skip Warning N",
                "Skip Warning Cum",
                "Delta",
                "Only OK N",
                "Only OK Cum",
            ],
            rule_rows,
        ),
        "",
        "## Warning Trades That Would Have Been Skipped",
        "",
        "### Largest Skipped Winners",
        "",
        _table(["Trade", "Rule", "Exit", "Net", "Guardrail"], winner_rows) if winner_rows else "None.",
        "",
        "### Largest Skipped Losers",
        "",
        _table(["Trade", "Rule", "Exit", "Net", "Guardrail"], loser_rows) if loser_rows else "None.",
        "",
        "## Read",
        "",
        "If skipping warnings improves cumulative net while discarding many winners, the guardrail is useful but too blunt. If `only_ok` is strong but low-N, it is a candidate for a separate pre-registered validation gate, not an immediate production filter.",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(db_path: Path) -> dict[str, object]:
    with _connect(db_path) as conn:
        trades = _load_closed_trades(conn)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "closed_trade_n": len(trades),
        "overall_scenarios": _evaluate_scenarios(trades),
        "level_breakdown": _level_breakdown(trades),
        "by_rule": _evaluate_by_rule(trades),
        "examples": _skipped_examples(trades),
        "trades": [asdict(t) for t in trades],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="S34 guardrail shadow-filter counterfactual report.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    payload = build_payload(args.db)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(args.out_md, payload)

    scenarios = payload["overall_scenarios"]
    print(
        json.dumps(
            {
                "closed_trade_n": payload["closed_trade_n"],
                "baseline_cum_net_bps": scenarios["baseline_all_closed"]["cum_net_bps"],
                "skip_warning_cum_net_bps": scenarios["skip_warning"]["cum_net_bps"],
                "skip_warning_delta_bps": scenarios["skip_warning"]["delta_cum_vs_baseline_bps"],
                "only_ok_n": scenarios["only_ok"]["kept_n"],
                "only_ok_cum_net_bps": scenarios["only_ok"]["cum_net_bps"],
                "out_md": str(args.out_md),
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
