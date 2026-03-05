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
    "tools.validate_canonical",
]

KEY_FILES = [
    "config/costs.py",
    "execution/passive_execution_simulator.py",
    "tools/rank_passive_pockets_forward.py",
    "tools/validate_passive_pocket_forward.py",
    "tools/run_rank_sweep.py",
    "tools/summarize_rank_attribution.py",
    "tools/validate_canonical.py",
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
