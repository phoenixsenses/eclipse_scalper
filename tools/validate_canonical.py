from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from tools.run_summary import build_run_summary


CONFIG_VERSION = "canonical_validate_v1"
DEFAULT_NAN_THRESHOLD = 0.05
TIMESTAMP_CANDIDATES = ["timestamp", "ts", "ts_ms", "time"]
SYMBOL_CANDIDATES = ["symbol"]
PRICE_CANDIDATES = ["mid", "price"]
NUMERIC_VALIDATION_COLS = ["mid", "price", "spread", "volume", "trade_intensity"]
SANITY_POSITIVE_COLS = ["mid", "price"]


@dataclass
class ValidationResult:
    status: str
    violations: List[Dict[str, Any]]
    column_stats: Dict[str, Any]
    invariant_summary: Dict[str, Any]
    notes: List[str]


def _find_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _stable_run_id(canonical_path: str, nan_threshold: float, config_version: str = CONFIG_VERSION) -> str:
    payload = {
        "canonical_path": str(Path(canonical_path)),
        "nan_threshold": float(nan_threshold),
        "config_version": str(config_version),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _as_timestamp(series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    vals = pd.to_numeric(series, errors="coerce")
    mx = float(vals.dropna().max()) if not vals.dropna().empty else 0.0
    unit = "s"
    if mx > 1e14:
        unit = "ns"
    elif mx > 1e11:
        unit = "ms"
    elif mx > 1e9:
        unit = "s"
    return pd.to_datetime(vals, unit=unit, utc=True, errors="coerce")


def _numeric_convertible(series: pd.Series) -> Tuple[bool, float]:
    if is_numeric_dtype(series):
        return True, 0.0
    converted = pd.to_numeric(series, errors="coerce")
    src_non_null = int(series.notna().sum())
    if src_non_null == 0:
        return True, 0.0
    conv_non_null = int(converted.notna().sum())
    bad = max(0, src_non_null - conv_non_null)
    return bad == 0, (bad / float(src_non_null))


def validate_dataframe(df: pd.DataFrame, *, nan_threshold: float = DEFAULT_NAN_THRESHOLD) -> ValidationResult:
    violations: List[Dict[str, Any]] = []
    notes: List[str] = []
    column_stats: Dict[str, Any] = {}
    invariant_summary: Dict[str, Any] = {}

    ts_col = _find_first_present(df, TIMESTAMP_CANDIDATES)
    sym_col = _find_first_present(df, SYMBOL_CANDIDATES)
    px_col = _find_first_present(df, PRICE_CANDIDATES)

    if ts_col is None:
        violations.append({"type": "schema", "code": "missing_timestamp_col", "severity": "critical"})
    if sym_col is None:
        violations.append({"type": "schema", "code": "missing_symbol_col", "severity": "critical"})
    if px_col is None:
        violations.append({"type": "schema", "code": "missing_price_col", "severity": "critical"})
    if df.empty:
        notes.append("empty_dataframe")
        violations.append({"type": "schema", "code": "empty_dataframe", "severity": "critical"})

    for col in NUMERIC_VALIDATION_COLS:
        if col not in df.columns:
            continue
        ok, bad_ratio = _numeric_convertible(df[col])
        column_stats[col] = {
            "dtype": str(df[col].dtype),
            "numeric_convertible": bool(ok),
            "bad_ratio_on_convert": float(bad_ratio),
            "nan_ratio": float(df[col].isna().mean()),
            "inf_count": int(pd.to_numeric(df[col], errors="coerce").isin([float("inf"), float("-inf")]).sum()),
        }
        if not ok:
            violations.append(
                {
                    "type": "dtype",
                    "code": "object_numeric_drift",
                    "column": col,
                    "severity": "critical",
                    "bad_ratio": float(bad_ratio),
                }
            )

    if ts_col is not None:
        ts = _as_timestamp(df[ts_col])
        ts_nan_ratio = float(ts.isna().mean())
        column_stats[ts_col] = {
            "dtype": str(df[ts_col].dtype),
            "timestamp_convertible_nan_ratio": ts_nan_ratio,
        }
        if not ts.dropna().empty:
            invariant_summary["timestamp_min_utc"] = ts.dropna().min().isoformat()
            invariant_summary["timestamp_max_utc"] = ts.dropna().max().isoformat()
        if ts_nan_ratio > float(nan_threshold):
            violations.append(
                {
                    "type": "dtype",
                    "code": "timestamp_convert_fail_ratio",
                    "column": ts_col,
                    "severity": "critical",
                    "nan_ratio": ts_nan_ratio,
                }
            )

        if sym_col is not None:
            sym_raw = df[sym_col]
            blank_symbol_count = int(sym_raw.isna().sum())
            if not sym_raw.empty:
                blank_symbol_count += int((sym_raw.astype(str).str.strip() == "").sum())
            invariant_summary["blank_symbol_count"] = blank_symbol_count
            if blank_symbol_count > 0:
                violations.append(
                    {
                        "type": "schema",
                        "code": "blank_symbol",
                        "severity": "critical",
                        "count": blank_symbol_count,
                    }
                )

            work = pd.DataFrame({"symbol": sym_raw.astype("string").str.strip(), "ts": ts})
            work = work.dropna(subset=["symbol", "ts"])
            work = work[work["symbol"] != ""]
            if not work.empty:
                dup_count = int(work.duplicated(subset=["symbol", "ts"]).sum())
                backward_total = 0
                for _, grp in work.groupby("symbol", sort=False):
                    d = grp["ts"].diff()
                    backward_total += int((d < pd.Timedelta(0)).sum())
                invariant_summary["duplicate_timestamps_per_symbol"] = dup_count
                invariant_summary["backward_time_jumps"] = int(backward_total)
                if dup_count > 0:
                    violations.append(
                        {
                            "type": "time",
                            "code": "duplicate_timestamp_per_symbol",
                            "severity": "critical",
                            "count": dup_count,
                        }
                    )
                if backward_total > 0:
                    violations.append(
                        {
                            "type": "time",
                            "code": "backward_time_jump",
                            "severity": "critical",
                            "count": int(backward_total),
                        }
                    )
            else:
                notes.append("no_non_null_symbol_ts_rows")

    for col in [c for c in NUMERIC_VALIDATION_COLS if c in df.columns]:
        nan_ratio = float(df[col].isna().mean())
        if nan_ratio > float(nan_threshold):
            violations.append(
                {
                    "type": "nan",
                    "code": "nan_ratio_above_threshold",
                    "column": col,
                    "severity": "critical",
                    "nan_ratio": nan_ratio,
                    "threshold": float(nan_threshold),
                }
            )
        numeric = pd.to_numeric(df[col], errors="coerce")
        inf_cnt = int(numeric.isin([float("inf"), float("-inf")]).sum())
        if inf_cnt > 0:
            violations.append(
                {
                    "type": "nan",
                    "code": "inf_values_present",
                    "column": col,
                    "severity": "critical",
                    "count": inf_cnt,
                }
            )

    if "spread" in df.columns:
        spread_neg = int((pd.to_numeric(df["spread"], errors="coerce") < 0).sum())
        invariant_summary["negative_spread_count"] = spread_neg
        if spread_neg > 0:
            violations.append({"type": "sanity", "code": "negative_spread", "severity": "critical", "count": spread_neg})
    if "volume" in df.columns:
        vol_neg = int((pd.to_numeric(df["volume"], errors="coerce") < 0).sum())
        invariant_summary["negative_volume_count"] = vol_neg
        if vol_neg > 0:
            violations.append({"type": "sanity", "code": "negative_volume", "severity": "critical", "count": vol_neg})
    for col in SANITY_POSITIVE_COLS:
        if col not in df.columns:
            continue
        non_pos = int((pd.to_numeric(df[col], errors="coerce") <= 0).sum())
        invariant_summary[f"non_positive_{col}_count"] = non_pos
        if non_pos > 0:
            violations.append(
                {
                    "type": "sanity",
                    "code": "non_positive_price",
                    "column": col,
                    "severity": "critical",
                    "count": non_pos,
                }
            )

    status = "fail" if violations else "pass"
    invariant_summary["rows"] = int(len(df))
    invariant_summary["violations"] = int(len(violations))
    return ValidationResult(
        status=status,
        violations=violations,
        column_stats=column_stats,
        invariant_summary=invariant_summary,
        notes=notes,
    )


def _load_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported_extension:{suffix}")


def _write_reports(report_json: Path, report_md: Path, payload: Dict[str, Any]) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# CANONICAL_VALIDATION",
        "",
        f"status={payload.get('status')}",
        f"run_id={payload.get('run_id')}",
        f"source={payload.get('source')}",
        f"violations={len(payload.get('violations', []))}",
        "",
        "## Violations",
    ]
    for v in payload.get("violations", []):
        lines.append(f"- {v}")
    lines += ["", "## Invariant Summary", f"- {payload.get('invariant_summary', {})}"]
    if isinstance(payload.get("run_summary"), dict):
        lines += ["", "## Run Summary", f"- {payload.get('run_summary', {})}"]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate canonical dataset invariants (deterministic artifacts).")
    p.add_argument("--in", dest="in_path", default="data/canonical/canonical_merged.parquet")
    p.add_argument("--db", default="")
    p.add_argument("--nan-threshold", type=float, default=DEFAULT_NAN_THRESHOLD)
    p.add_argument("--reports-dir", default="reports")
    return p.parse_args()


def main() -> int:
    args = _args()
    in_path = Path(str(args.in_path))
    run_id = _stable_run_id(str(in_path), float(args.nan_threshold))
    reports_dir = Path(str(args.reports_dir))
    out_json = reports_dir / f"validate_canonical_{run_id}.json"
    out_md = reports_dir / f"validate_canonical_{run_id}.md"

    if str(args.db).strip():
        db_path = Path(str(args.db))
        if not db_path.exists():
            payload = {
                "status": "skip",
                "run_id": run_id,
                "source": str(in_path),
                "notes": ["skipped_missing_db"],
                "violations": [],
                "column_stats": {},
                "invariant_summary": {},
                "run_summary": build_run_summary(
                    run_type="validate_canonical",
                    inputs={"source": str(in_path), "nan_threshold": float(args.nan_threshold), "db": str(db_path)},
                    metrics={"status": "skip", "violation_count": 0, "row_count": 0},
                    artifacts={"json": str(out_json), "md": str(out_md)},
                ),
            }
            _write_reports(out_json, out_md, payload)
            print("validate_canonical status=skip reason=skipped_missing_db")
            return 0

    if not in_path.exists():
        payload = {
            "status": "skip",
            "run_id": run_id,
            "source": str(in_path),
            "notes": ["skipped_missing_data"],
            "violations": [],
            "column_stats": {},
            "invariant_summary": {},
            "run_summary": build_run_summary(
                run_type="validate_canonical",
                inputs={"source": str(in_path), "nan_threshold": float(args.nan_threshold), "db": str(args.db or "")},
                metrics={"status": "skip", "violation_count": 0, "row_count": 0},
                artifacts={"json": str(out_json), "md": str(out_md)},
            ),
        }
        _write_reports(out_json, out_md, payload)
        print("validate_canonical status=skip reason=skipped_missing_data")
        return 0

    try:
        df = _load_df(in_path)
    except Exception as exc:
        payload = {
            "status": "fail",
            "run_id": run_id,
            "source": str(in_path),
            "notes": [f"load_error:{type(exc).__name__}:{exc}"],
            "violations": [{"type": "load", "code": "read_failure", "severity": "critical"}],
            "column_stats": {},
            "invariant_summary": {},
            "run_summary": build_run_summary(
                run_type="validate_canonical",
                inputs={"source": str(in_path), "nan_threshold": float(args.nan_threshold), "db": str(args.db or "")},
                metrics={"status": "fail", "violation_count": 1, "row_count": 0},
                artifacts={"json": str(out_json), "md": str(out_md)},
            ),
        }
        _write_reports(out_json, out_md, payload)
        print(f"validate_canonical status=fail reason=read_failure err={exc}")
        return 3

    res = validate_dataframe(df, nan_threshold=float(args.nan_threshold))
    payload = {
        "status": res.status,
        "run_id": run_id,
        "source": str(in_path),
        "config_version": CONFIG_VERSION,
        "violations": res.violations,
        "column_stats": res.column_stats,
        "invariant_summary": res.invariant_summary,
        "notes": res.notes,
        "run_summary": build_run_summary(
            run_type="validate_canonical",
            inputs={"source": str(in_path), "nan_threshold": float(args.nan_threshold), "db": str(args.db or "")},
            metrics={
                "status": res.status,
                "violation_count": len(res.violations),
                "row_count": int(res.invariant_summary.get("rows", 0) or 0),
            },
            artifacts={"json": str(out_json), "md": str(out_md)},
        ),
    }
    _write_reports(out_json, out_md, payload)
    print(f"validate_canonical status={res.status} run_id={run_id} violations={len(res.violations)}")
    return 0 if res.status in {"pass", "skip"} else 3


if __name__ == "__main__":
    raise SystemExit(main())

