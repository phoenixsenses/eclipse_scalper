from __future__ import annotations

import argparse
from pathlib import Path

from src.microphys.risk.policy import dump_risk_policy, load_risk_policy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write risk policy markdown/json doc.")
    p.add_argument("--risk-policy", default="")
    p.add_argument("--out-md", default="reports/risk_policy.md")
    p.add_argument("--out-json", default="")
    p.add_argument("--starting-equity", type=float, default=10000.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        policy = load_risk_policy(str(args.risk_policy), starting_equity_override=float(args.starting_equity))
        lines = [
            "# Risk Policy",
            "",
        ]
        for k, v in sorted(policy.model_dump().items()):
            lines.append(f"- `{k}`: `{v}`")
        out_md = Path(str(args.out_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        if str(args.out_json).strip():
            dump_risk_policy(policy, Path(str(args.out_json)))
        print(f"write_risk_policy_doc ok out={out_md}")
        return 0
    except Exception as e:
        print(f"write_risk_policy_doc error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

