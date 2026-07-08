"""Mutation raporu ureticisi -> reports/research/s34/AMI_MUTATION_REPORT.md

Run: python -m ami.run_mutation_report
"""
from __future__ import annotations
import sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.mutation_suite import run_all

OUT = ROOT / "reports" / "research" / "s34" / "AMI_MUTATION_REPORT.md"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    base = Path(tempfile.mkdtemp(prefix="ami_mut_"))
    results = run_all(base)
    n_pass = sum(1 for r in results if r["passed"])
    lines = ["# AMI Mutation / Adversarial Test Report", "",
             f"> {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} — "
             f"**{n_pass}/{len(results)} ihlal yakalandi**", "",
             "| # | Injected violation | Expected rejection | Actual | Blocking component | Pass |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            r["name"], r["injected"], r["expected"],
            str(r["actual"]).replace("|", "/")[:120],
            r["blocked_by"], "✅" if r["passed"] else "❌"))
    lines += ["", "Audit kanallari: `data/ami/knowledge.sqlite:audit_log` "
              "(PUT/LINK/AUTHORIZE/BREAKER_TRIP/BINDING_INVALID/EVIDENCE_REJECTED/ARCHIVE_FAILURE), "
              "`research.sqlite:processed_trades(reject_reason)`, `decisions.jsonl`.",
              "", "*Kaynak: `ami/mutation_suite.py` — pytest esdegeri: "
              "`tests/test_ami_mutation_suite.py`*"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    for r in results:
        print(("PASS " if r["passed"] else "FAIL ") + r["name"])
    print(f"\n{n_pass}/{len(results)}  ->  {OUT}")


if __name__ == "__main__":
    main()
