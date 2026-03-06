from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tools.replay_strategy import _parse_iso_utc, _parse_symbols, replay_to_decisions


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_parity_", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", newline="\n", closefd=True) as f:
            f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2))
            f.write("\n")
        Path(tmp_name).replace(path)
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _norm_rows(
    rows: Sequence[Dict[str, Any]],
    symbols: Sequence[str],
    start_ts: float,
    end_ts: float,
    id_field: str,
    ts_field: str,
    symbol_field: str,
) -> List[Dict[str, str]]:
    allowed = {str(s).upper() for s in symbols}
    out: List[Dict[str, str]] = []
    for r in rows:
        try:
            ts_utc = str(r.get(ts_field) or "")
            ts = _parse_iso_utc(ts_utc)
            if ts < start_ts or ts > end_ts:
                continue
            symbol = str(r.get(symbol_field) or "").upper()
            if allowed and symbol not in allowed:
                continue
            decision_id = str(r.get(id_field) or "")
            if not decision_id:
                continue
            out.append({"decision_id": decision_id, "ts_utc": ts_utc, "symbol": symbol, "action": str(r.get("action") or "")})
        except Exception:
            continue
    return out


def _hash_ids(rows: Sequence[Dict[str, str]]) -> str:
    blob = "\n".join(str(r["decision_id"]) for r in rows).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_parity_check(
    paper_decisions: Path,
    db: Path,
    symbols: Sequence[str],
    start: str,
    end: str,
    strategy: str,
    strategy_config: Dict[str, Any],
    paper_id_field: str = "decision_id",
    paper_ts_field: str = "ts_utc",
    paper_symbol_field: str = "symbol",
    include_runtime_ts: bool = False,
) -> Dict[str, Any]:
    paper_rows = _read_jsonl(paper_decisions)
    replay_rows, _ = replay_to_decisions(
        db=db,
        symbols=symbols,
        start_iso=start,
        end_iso=end,
        strategy_name=strategy,
        strategy_config=strategy_config,
    )
    start_ts = _parse_iso_utc(start)
    end_ts = _parse_iso_utc(end)
    norm_paper = _norm_rows(
        rows=paper_rows,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=end_ts,
        id_field=paper_id_field,
        ts_field=paper_ts_field,
        symbol_field=paper_symbol_field,
    )
    norm_replay = _norm_rows(
        rows=replay_rows,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=end_ts,
        id_field="decision_id",
        ts_field="ts_utc",
        symbol_field="symbol",
    )

    paper_ids = [r["decision_id"] for r in norm_paper]
    replay_ids = [r["decision_id"] for r in norm_replay]
    n = min(len(paper_ids), len(replay_ids))
    first_div = None
    for i in range(n):
        if paper_ids[i] != replay_ids[i]:
            first_div = i
            break
    if first_div is None and len(paper_ids) != len(replay_ids):
        first_div = n
    lo = max(0, int(first_div or 0) - 1)
    hi = min(max(len(norm_paper), len(norm_replay)), int(first_div or 0) + 2)
    report: Dict[str, Any] = {
        "match": first_div is None,
        "paper_count": len(norm_paper),
        "replay_count": len(norm_replay),
        "first_divergence_index": first_div,
        "paper_divergence": norm_paper[lo:hi] if first_div is not None else [],
        "replay_divergence": norm_replay[lo:hi] if first_div is not None else [],
        "paper_hash": _hash_ids(norm_paper),
        "replay_hash": _hash_ids(norm_replay),
        "notes": (
            "counts_or_order_mismatch"
            if first_div is not None
            else "parity_ok"
        ),
    }
    if include_runtime_ts:
        report["runtime_ts_utc"] = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare paper decisions JSONL with replay decisions for a time slice.")
    p.add_argument("--paper-decisions", required=True)
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--strategy", default="baseline")
    p.add_argument("--strategy-config", default="{}")
    p.add_argument("--paper-id-field", default="decision_id")
    p.add_argument("--paper-ts-field", default="ts_utc")
    p.add_argument("--paper-symbol-field", default="symbol")
    p.add_argument("--out", required=True)
    p.add_argument("--include-runtime-ts", action="store_true")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        cfg = json.loads(str(args.strategy_config))
        if not isinstance(cfg, dict):
            raise ValueError("strategy-config must be JSON object")
        report = run_parity_check(
            paper_decisions=Path(str(args.paper_decisions)),
            db=Path(str(args.db)),
            symbols=_parse_symbols(args.symbols),
            start=str(args.start),
            end=str(args.end),
            strategy=str(args.strategy),
            strategy_config=cfg,
            paper_id_field=str(args.paper_id_field),
            paper_ts_field=str(args.paper_ts_field),
            paper_symbol_field=str(args.paper_symbol_field),
            include_runtime_ts=bool(args.include_runtime_ts),
        )
        out = Path(str(args.out))
        _write_json(out, report)
        if bool(report.get("match")):
            print(f"parity_check ok paper={report['paper_count']} replay={report['replay_count']} out={out}")
            return 0
        print(
            f"parity_check diverged idx={report.get('first_divergence_index')} "
            f"paper={report['paper_count']} replay={report['replay_count']} out={out}"
        )
        return 1
    except Exception as e:
        print(f"parity_check error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

