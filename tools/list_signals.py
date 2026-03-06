from __future__ import annotations

import argparse
import json

from src.microphys.alpha.library import built_in_signal_specs


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="List built-in microphysics alpha signals.")
    p.add_argument("--json", action="store_true")
    return p


def main() -> int:
    args = _parser().parse_args()
    specs = built_in_signal_specs()
    if args.json:
        print(json.dumps([s.to_dict() for s in specs], ensure_ascii=True, sort_keys=True, indent=2))
        return 0
    print(f"signals={len(specs)}")
    for s in specs:
        tags = ",".join((s.meta.get("tags") if isinstance(s.meta, dict) else []) or [])
        print(
            f"- {s.name} side={s.side} entry={s.entry} horizon={s.horizon_bars} cooldown={s.cooldown_bars} tags={tags}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
