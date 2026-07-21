"""S34 Diversification Meta — SQL'den kombinasyon cikarimi.

Kaynak: reports/research/s34/S34_ALL.db (research_clean, trust'li)
Hedef: conviction-weighted sizing / feature-interaction (iki-sinyal) /
premium sleeve (score>=4/5) sonuclarini risk-ayarli sirala.

risk_adj = total_bps / max(|mdd_bps| or |worst_bps|, 50)
Cikti: reports/research/s34/S34_DIVERS_META_SQL.json + .md
"""
from __future__ import annotations
import json, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "reports" / "research" / "s34" / "S34_ALL.db"
OJ = ROOT / "reports" / "research" / "s34" / "S34_DIVERS_META_SQL.json"
OM = ROOT / "reports" / "research" / "s34" / "S34_DIVERS_META_SQL.md"

BASE = ("SELECT report,key,label,n,per_month,wr_norm,avg_bps,total_bps,worst_bps,"
        "tail_n,mdd_bps,mc_p,trust FROM research_clean "
        "WHERE n>=8 AND trust>=2 AND avg_bps IS NOT NULL AND total_bps IS NOT NULL ")

CATS = {
    "interaction": "AND (key LIKE 'I_%' OR key LIKE '%sync%whale%' OR key LIKE '%shelf%be%' "
                   "OR key LIKE '%sync%shelf%' OR key LIKE '%sync%rv%' OR label LIKE '%both%' "
                   "OR label LIKE '%&%')",
    "premium_sleeve": "AND (key LIKE '%ge4%' OR key LIKE '%ge5%' OR key LIKE '%s4%' OR key LIKE '%s5%' "
                      "OR key LIKE '%score4%' OR key LIKE '%score5%' OR key LIKE '%premium%' "
                      "OR label LIKE '%>=4%' OR label LIKE '%>=5%')",
    "sizing_weighted": "AND (key LIKE '%weight%' OR key LIKE '%sizing%' OR key LIKE '%conviction%' "
                       "OR label LIKE '%weight%' OR label LIKE '%sizing%')",
    "lean_composite": "AND (key LIKE '%lean%' OR key LIKE '%min3%' OR key LIKE '%consensus%')",
    "portfolio": "AND (key LIKE '%portfolio%' OR key LIKE '%P1%' OR key LIKE '%P2%' OR key LIKE '%P3%' "
                 "OR key LIKE '%union%' OR key LIKE '%combo%' OR label LIKE '%portfolio%')",
}


def risk_adj(row) -> float:
    dd = row["mdd_bps"] if row["mdd_bps"] is not None else row["worst_bps"]
    denom = max(abs(dd) if dd is not None else 100.0, 50.0)
    return (row["total_bps"] or 0.0) / denom


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out = {}
    for cat, cond in CATS.items():
        rows = [dict(r) | {"risk_adj": round(risk_adj(r), 2)}
                for r in conn.execute(BASE + cond)]
        # dedupe by (key) keeping best risk_adj
        best = {}
        for r in rows:
            k = r["key"]
            if k not in best or r["risk_adj"] > best[k]["risk_adj"]:
                best[k] = r
        top = sorted(best.values(), key=lambda r: r["risk_adj"], reverse=True)[:15]
        out[cat] = top
        print(f"\n=== {cat} (uniq={len(best)}) ===")
        for r in top[:10]:
            print("  %-42s N=%-4s /ay=%-5s WR=%-5s avg=%-7s TOT=%-8s worst=%-7s mdd=%-8s mc=%-6s RA=%s"
                  % (r["key"][:42], r["n"], r["per_month"], r["wr_norm"], r["avg_bps"],
                     r["total_bps"], r["worst_bps"], r["mdd_bps"], r["mc_p"], r["risk_adj"]))
    OJ.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 Diversification Meta (SQL)", "",
             "> research_clean trust>=2, n>=8. risk_adj = total / max(|mdd or worst|, 50).", ""]
    for cat, top in out.items():
        lines += [f"## {cat}", "",
                  "| key | report | N | /ay | WR | avg | TOT | worst | mdd | mc_p | risk_adj |",
                  "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in top:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                         % (r["key"], r["report"][:28], r["n"], r["per_month"], r["wr_norm"],
                            r["avg_bps"], r["total_bps"], r["worst_bps"], r["mdd_bps"],
                            r["mc_p"], r["risk_adj"]))
        lines.append("")
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON: {OJ}\nMD:   {OM}")


if __name__ == "__main__":
    main()
