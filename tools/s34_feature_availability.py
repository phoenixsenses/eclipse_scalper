from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "reports" / "research" / "s34" / "FEATURE_AVAILABILITY_REGISTRY.md"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "FEATURE_AVAILABILITY_REGISTRY.json"


class FeatureClass(str, Enum):
    POINT_IN_TIME = "POINT_IN_TIME"
    RUNNING_CLUSTER = "RUNNING_CLUSTER"
    TERMINAL_CLUSTER = "TERMINAL_CLUSTER"
    FORWARD_OUTCOME = "FORWARD_OUTCOME"


class LookaheadViolation(RuntimeError):
    def __init__(self, feature_name: str, knowable_at_ts: int | None, entry_ts: int, context: str = ""):
        self.feature_name = feature_name
        self.knowable_at_ts = knowable_at_ts
        self.entry_ts = entry_ts
        self.context = context
        suffix = f" context={context}" if context else ""
        super().__init__(
            f"LookaheadViolation feature={feature_name} knowable_at_ts={knowable_at_ts} "
            f"entry_ts={entry_ts}{suffix}"
        )


@dataclass(frozen=True)
class FeatureValue:
    name: str
    value: Any
    knowable_at_ts: int | None
    feature_class: FeatureClass
    source_table: str = ""
    definition: str = ""


@dataclass(frozen=True)
class FeatureRegistryRow:
    feature_name: str
    source_table: str
    definition: str
    feature_class: FeatureClass
    knowable_at_rule: str
    used_at_ts: str
    consumed_by_entry: bool

    @property
    def violation(self) -> bool:
        return self.consumed_by_entry and self.feature_class in {
            FeatureClass.TERMINAL_CLUSTER,
            FeatureClass.FORWARD_OUTCOME,
        }


def assert_feature_available(feature: FeatureValue, entry_ts: int, *, context: str = "") -> None:
    if feature.feature_class in {FeatureClass.TERMINAL_CLUSTER, FeatureClass.FORWARD_OUTCOME}:
        raise LookaheadViolation(feature.name, feature.knowable_at_ts, int(entry_ts), context)
    if feature.knowable_at_ts is None:
        raise LookaheadViolation(feature.name, None, int(entry_ts), context)
    if int(feature.knowable_at_ts) > int(entry_ts):
        raise LookaheadViolation(feature.name, int(feature.knowable_at_ts), int(entry_ts), context)


def assert_feature_set_available(features: Iterable[FeatureValue], entry_ts: int, *, context: str = "") -> None:
    for feature in features:
        assert_feature_available(feature, entry_ts, context=context)


def signal_entry_features(signal: dict[str, Any]) -> list[FeatureValue]:
    """Features currently consumed by the live/replay paper runner entry path.

    These are legal only because the runner signal timestamp is the threshold-cross
    timestamp and the cluster quantities are partial values accumulated only up to
    that timestamp. Terminal feature-factory aggregates must not use these names.
    """
    ts = int(signal.get("ts_ms") or signal.get("threshold_cross_ts_ms") or signal.get("entry_ts_ms") or 0)
    names = [
        ("liq_total_notional", "running_notional at threshold-cross"),
        ("liq_count", "running liquidation count at threshold-cross"),
        ("liq_max_notional", "running max single liquidation notional at threshold-cross"),
        ("cluster_max_single_liq_share", "running max single liquidation share at threshold-cross"),
        ("intensity_per_sec", "running notional/sec over elapsed cluster duration at threshold-cross"),
        ("inter_cluster_gap_sec", "previous accepted signal/liquidation gap known at threshold-cross"),
        ("day_trend_bps", "UTC day-so-far mark return known at threshold-cross"),
        ("day_range_bps", "UTC day-so-far range known at threshold-cross"),
        ("day_buy_liq_notional", "UTC day-so-far BUY liquidation notional known at threshold-cross"),
        ("day_agg_trade_count", "UTC day-so-far agg trade count known at threshold-cross"),
    ]
    out: list[FeatureValue] = []
    for name, definition in names:
        if name in signal and signal.get(name) is not None:
            cls = FeatureClass.RUNNING_CLUSTER if name.startswith(("liq_", "cluster_", "intensity", "inter_")) else FeatureClass.POINT_IN_TIME
            out.append(
                FeatureValue(
                    name=name,
                    value=signal.get(name),
                    knowable_at_ts=ts,
                    feature_class=cls,
                    source_table="runner_signal",
                    definition=definition,
                )
            )
    return out


def registry_rows() -> list[FeatureRegistryRow]:
    rows = [
        FeatureRegistryRow("liq_total_notional", "runner_signal", "running cumulative notional up to threshold-cross", FeatureClass.RUNNING_CLUSTER, "signal.ts_ms", "entry_ts_ms", True),
        FeatureRegistryRow("liq_count", "runner_signal", "running liquidation count up to threshold-cross", FeatureClass.RUNNING_CLUSTER, "signal.ts_ms", "entry_ts_ms", True),
        FeatureRegistryRow("liq_max_notional", "runner_signal", "running max single liquidation up to threshold-cross", FeatureClass.RUNNING_CLUSTER, "signal.ts_ms", "entry_ts_ms", True),
        FeatureRegistryRow("cluster_max_single_liq_share", "runner_signal", "running max single share up to threshold-cross", FeatureClass.RUNNING_CLUSTER, "signal.ts_ms", "entry_ts_ms", True),
        FeatureRegistryRow("intensity_per_sec", "runner_signal", "running notional/sec up to threshold-cross", FeatureClass.RUNNING_CLUSTER, "signal.ts_ms", "entry_ts_ms", True),
        FeatureRegistryRow("day_trend_bps", "mark_prices", "UTC day-so-far return at decision time", FeatureClass.POINT_IN_TIME, "signal.ts_ms", "entry_ts_ms", True),
        FeatureRegistryRow("day_range_bps", "mark_prices", "UTC day-so-far range at decision time", FeatureClass.POINT_IN_TIME, "signal.ts_ms", "entry_ts_ms", True),
        FeatureRegistryRow("cluster_notional", "liq_event_features", "full terminal cluster notional", FeatureClass.TERMINAL_CLUSTER, "cluster_end_ts_ms", "event_ts_ms / entry_ts_ms", True),
        FeatureRegistryRow("cluster_count", "liq_event_features", "full terminal cluster count", FeatureClass.TERMINAL_CLUSTER, "cluster_end_ts_ms", "event_ts_ms / entry_ts_ms", True),
        FeatureRegistryRow("cluster_duration_sec", "liq_event_features", "full terminal cluster duration", FeatureClass.TERMINAL_CLUSTER, "cluster_end_ts_ms", "event_ts_ms / entry_ts_ms", True),
        FeatureRegistryRow("cluster_max_notional", "liq_event_features", "full terminal max single liquidation", FeatureClass.TERMINAL_CLUSTER, "cluster_end_ts_ms", "event_ts_ms / entry_ts_ms", True),
        FeatureRegistryRow("max_single_liq_share", "liq_event_features", "full terminal max single liquidation share", FeatureClass.TERMINAL_CLUSTER, "cluster_end_ts_ms", "event_ts_ms / entry_ts_ms", True),
        FeatureRegistryRow("shape_label", "liq_event_features", "terminal cluster shape label", FeatureClass.TERMINAL_CLUSTER, "cluster_end_ts_ms", "event_ts_ms / entry_ts_ms", True),
        FeatureRegistryRow("entry_price", "liq_event_outcome_labels", "route label entry price", FeatureClass.FORWARD_OUTCOME, "label generation / route simulation", "entry decision", True),
        FeatureRegistryRow("exit_price", "liq_event_outcome_labels", "route label exit price", FeatureClass.FORWARD_OUTCOME, "after entry path", "entry decision", True),
        FeatureRegistryRow("net_bps", "liq_event_outcome_labels", "route label realized net PnL", FeatureClass.FORWARD_OUTCOME, "after exit", "entry decision", True),
        FeatureRegistryRow("mfe_bps", "liq_event_outcome_labels", "post-entry max favorable excursion", FeatureClass.FORWARD_OUTCOME, "after entry path", "entry decision", True),
        FeatureRegistryRow("mae_bps", "liq_event_outcome_labels", "post-entry max adverse excursion", FeatureClass.FORWARD_OUTCOME, "after entry path", "entry decision", True),
    ]
    return rows


def build_registry_payload() -> dict[str, Any]:
    rows = registry_rows()
    violations = [row for row in rows if row.violation]
    return {
        "violation_count": len(violations),
        "rows": [
            {
                "feature_name": row.feature_name,
                "source_table": row.source_table,
                "definition": row.definition,
                "class": row.feature_class.value,
                "knowable_at_rule": row.knowable_at_rule,
                "used_at_ts": row.used_at_ts,
                "consumed_by_entry": row.consumed_by_entry,
                "violation": row.violation,
            }
            for row in rows
        ],
    }


def write_registry(out_md: Path = OUT_MD, out_json: Path = OUT_JSON) -> dict[str, Any]:
    payload = build_registry_payload()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Feature Availability Registry",
        "",
        f"- violation_count: `{payload['violation_count']}`",
        "",
        "| feature_name | source_table | definition | class | knowable_at rule | used_at_ts | VIOLATION? |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['feature_name']}` | `{row['source_table']}` | {row['definition']} | "
            f"`{row['class']}` | `{row['knowable_at_rule']}` | `{row['used_at_ts']}` | "
            f"{'YES' if row['violation'] else 'NO'} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write S34 feature availability registry.")
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()
    payload = write_registry()
    print(f"violation_count={payload['violation_count']}")
    print(f"md={OUT_MD}")
    print(f"json={OUT_JSON}")
    if args.fail_on_violation and int(payload["violation_count"]) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
