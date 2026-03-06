from __future__ import annotations

import argparse
import importlib
import py_compile
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


KEY_MODULES = [
    "config.costs",
    "execution.passive_execution_simulator",
    "tools.rank_passive_pockets_forward",
    "tools.validate_passive_pocket_forward",
    "tools.run_rank_sweep",
    "tools.summarize_rank_attribution",
    "tools.summarize_liq_regime_tag_impact",
    "tools.summarize_liq_tag_signal_behavior",
    "tools.validate_canonical",
    "tools.report_schema_validator",
    "tools.run_summary",
    "tools.report_check",
    "tools.tooling_audit",
    "tools.validate_microstructure_contract",
    "tools.generate_liq_reversal_candidates",
    "tools.run_liq_reversal_e2e",
    "tools.liquidation_regime_tagger",
    "tools.liquidation_regime_alerts",
    "tools.liquidation_alert_state",
    "tools.liquidation_watchlist",
    "tools.spread_stress_alerts",
    "tools.spread_stress_state",
    "tools.return_shock_alerts",
    "tools.return_shock_state",
    "tools.return_shock_watchlist",
    "tools.spread_stress_watchlist",
    "tools.fill_toxicity_state",
    "tools.latency_stress_state",
    "tools.research_event_watchboard",
    "tools.event_watchboard_trend",
    "tools.event_watchboard_snapshot_append",
    "tools.event_watchboard_trend_from_history",
    "tools.run_research_event_watchboard_cycle",
    "tools.research_event_operator_brief",
]

KEY_FILES = [
    "config/costs.py",
    "execution/passive_execution_simulator.py",
    "tools/rank_passive_pockets_forward.py",
    "tools/validate_passive_pocket_forward.py",
    "tools/run_rank_sweep.py",
    "tools/summarize_rank_attribution.py",
    "tools/summarize_liq_regime_tag_impact.py",
    "tools/summarize_liq_tag_signal_behavior.py",
    "tools/validate_canonical.py",
    "tools/report_schema_validator.py",
    "tools/run_summary.py",
    "tools/report_check.py",
    "tools/tooling_audit.py",
    "tools/validate_microstructure_contract.py",
    "tools/generate_liq_reversal_candidates.py",
    "tools/run_liq_reversal_e2e.py",
    "tools/liquidation_regime_tagger.py",
    "tools/liquidation_regime_alerts.py",
    "tools/liquidation_alert_state.py",
    "tools/liquidation_watchlist.py",
    "tools/spread_stress_alerts.py",
    "tools/spread_stress_state.py",
    "tools/return_shock_alerts.py",
    "tools/return_shock_state.py",
    "tools/return_shock_watchlist.py",
    "tools/spread_stress_watchlist.py",
    "tools/fill_toxicity_state.py",
    "tools/latency_stress_state.py",
    "tools/research_event_watchboard.py",
    "tools/event_watchboard_trend.py",
    "tools/event_watchboard_snapshot_append.py",
    "tools/event_watchboard_trend_from_history.py",
    "tools/run_research_event_watchboard_cycle.py",
    "tools/research_event_operator_brief.py",
]


def check_imports(modules: List[str]) -> List[Tuple[str, bool, str]]:
    out: List[Tuple[str, bool, str]] = []
    for m in modules:
        try:
            importlib.import_module(m)
            out.append((m, True, "ok"))
        except Exception as exc:
            out.append((m, False, f"{type(exc).__name__}: {exc}"))
    return out


def check_py_compile(repo_root: Path, files: List[str]) -> List[Tuple[str, bool, str]]:
    out: List[Tuple[str, bool, str]] = []
    for rel in files:
        p = repo_root / rel
        if not p.exists():
            out.append((rel, False, "missing_file"))
            continue
        try:
            py_compile.compile(str(p), doraise=True)
            out.append((rel, True, "ok"))
        except Exception as exc:
            out.append((rel, False, f"{type(exc).__name__}: {exc}"))
    return out


def run_synthetic_checks() -> List[Tuple[str, bool, str]]:
    out: List[Tuple[str, bool, str]] = []
    try:
        from tools.micro_edge_backtest import compute_gross_return, compute_trade_cost

        long_ret = compute_gross_return(100.0, 101.0, "LONG")
        short_ret = compute_gross_return(100.0, 99.0, "SHORT")
        cost = compute_trade_cost(-1.0, 0.0)  # rebate-friendly
        if abs(long_ret - 0.01) < 1e-12 and short_ret > 0.0 and cost < 0.0:
            out.append(("synthetic_math", True, "ok"))
        else:
            out.append(("synthetic_math", False, f"unexpected_values long={long_ret} short={short_ret} cost={cost}"))
    except Exception as exc:
        out.append(("synthetic_math", False, f"{type(exc).__name__}: {exc}"))
    try:
        import pandas as pd
        from tools.validate_canonical import validate_dataframe

        df = pd.DataFrame(
            {
                "timestamp": [1, 2],
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "mid": [100.0, 100.1],
                "spread": [0.1, 0.1],
                "volume": [1.0, 2.0],
            }
        )
        res = validate_dataframe(df, nan_threshold=0.5)
        if res.status == "pass":
            out.append(("synthetic_validate_canonical", True, "ok"))
        else:
            out.append(("synthetic_validate_canonical", False, f"status={res.status}"))
    except Exception as exc:
        out.append(("synthetic_validate_canonical", False, f"{type(exc).__name__}: {exc}"))
    try:
        import json
        import shutil
        import sqlite3
        import uuid
        from pathlib import Path

        from tools.validate_microstructure_contract import analyze_contract
        from tools.report_schema_validator import validate_payload

        tmp = Path("localtests") / f"smoke_micro_contract_{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True, exist_ok=True)
        db = tmp / "micro.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute(
                "CREATE TABLE agg_trades (ts_ms INTEGER, symbol TEXT, price REAL, quantity REAL, notional REAL, is_buyer_maker INTEGER)"
            )
            conn.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL)")
            conn.execute(
                "CREATE TABLE liquidations (ts_ms INTEGER, symbol TEXT, side TEXT, price REAL, quantity REAL, notional REAL)"
            )
            conn.execute("INSERT INTO agg_trades VALUES (1700000000000, 'ETHUSDT', 100.0, 1.0, 100.0, 0)")
            conn.execute("INSERT INTO mark_prices VALUES (1700000000000, 'ETHUSDT', 100.1)")
            conn.execute("INSERT INTO liquidations VALUES (1700000000500, 'ETHUSDT', 'SELL', 99.9, 2.0, 199.8)")
            conn.commit()
        finally:
            conn.close()
        payload = analyze_contract(db, ["ETHUSDT"])
        schema_errors = validate_payload(payload, "validate_microstructure_contract")
        if payload.get("status") == "warn" and not schema_errors:
            out.append(("synthetic_validate_microstructure_contract", True, "ok"))
        else:
            out.append(
                (
                    "synthetic_validate_microstructure_contract",
                    False,
                    f"status={payload.get('status')} schema_errors={json.dumps(schema_errors)}",
                )
            )
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as exc:
        out.append(("synthetic_validate_microstructure_contract", False, f"{type(exc).__name__}: {exc}"))
    return out


def check_db_if_present(db_path: Path) -> Tuple[str, bool, str]:
    if not db_path.exists():
        return ("db_check", True, "skipped_missing_db")
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [r[0] for r in cur.fetchall()]
            if not tables:
                return ("db_check", False, "db_has_no_tables")
            return ("db_check", True, f"ok_tables={len(tables)}")
        finally:
            conn.close()
    except Exception as exc:
        return ("db_check", False, f"{type(exc).__name__}: {exc}")


def run_smoke(repo_root: Path, db_path: Path) -> Dict[str, Any]:
    import_results = check_imports(KEY_MODULES)
    compile_results = check_py_compile(repo_root, KEY_FILES)
    synth_results = run_synthetic_checks()
    db_result = check_db_if_present(db_path)

    checks: List[Tuple[str, bool, str]] = []
    checks.extend(import_results)
    checks.extend(compile_results)
    checks.extend(synth_results)
    checks.append(db_result)

    failed = [c for c in checks if not c[1]]
    return {
        "checks": checks,
        "failed": failed,
        "ok": len(failed) == 0,
    }


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast deterministic smoke checks for Eclipse Scalper.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--db", default="data/microstructure.db")
    return p.parse_args()


def main() -> int:
    args = _args()
    repo_root = Path(str(args.repo_root)).resolve()
    db_path = (repo_root / str(args.db)).resolve() if not Path(str(args.db)).is_absolute() else Path(str(args.db))
    res = run_smoke(repo_root=repo_root, db_path=db_path)
    print("smoke_all")
    for name, ok, msg in res["checks"]:
        status = "PASS" if ok else "FAIL"
        print(f"- {status} {name}: {msg}")
    if res["ok"]:
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
