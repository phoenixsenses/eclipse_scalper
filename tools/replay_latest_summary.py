#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

try:
    from tools import replay_trade
except Exception:
    from eclipse_scalper.tools import replay_trade  # type: ignore


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _latest_corr_or_symbol(events: list[dict]) -> tuple[str, str]:
    corr = ""
    sym = ""
    for ev in reversed(events):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if not corr:
            corr = str(data.get("correlation_id") or ev.get("correlation_id") or "").strip()
        if not sym:
            sym = str(ev.get("symbol") or data.get("symbol") or data.get("k") or "").strip().upper()
        if corr and sym:
            break
    return corr, sym


def build_summary(telemetry_path: Path, journal_path: Path, *, limit: int = 10) -> str:
    events = _load_jsonl(telemetry_path)
    corr, sym = _latest_corr_or_symbol(events)
    out = replay_trade.replay(journal_path, correlation_id=corr, symbol=("" if corr else sym))
    transitions = out.get("transitions", []) if isinstance(out, dict) else []
    lines = [
        "Execution Replay Latest",
        "=======================",
        f"telemetry: {telemetry_path}",
        f"journal: {journal_path}",
        f"filter correlation_id={corr or 'n/a'} symbol={sym or 'n/a'}",
        f"events={int(out.get('count', 0) or 0)} last_state={str(out.get('last_state') or '')}",
        "",
        "Transitions:",
    ]
    if not transitions:
        lines.append("- none")
    else:
        for tr in transitions[-max(1, int(limit)):]:
            lines.append(
                f"- {tr.get('machine')} {tr.get('entity')} "
                f"{tr.get('from')}->{tr.get('to')} ({tr.get('reason')})"
            )
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build latest execution replay summary artifact")
    ap.add_argument("--telemetry", default="logs/telemetry.jsonl")
    ap.add_argument("--journal", default="logs/execution_journal.jsonl")
    ap.add_argument("--output", default="logs/replay_latest.txt")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)

    summary = build_summary(Path(args.telemetry), Path(args.journal), limit=max(1, int(args.limit)))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    print(summary.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
