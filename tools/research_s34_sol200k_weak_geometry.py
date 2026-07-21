"""
S34 SOL 200K Weak Geometry Shadow Tagging
Shadow-observe only. No runner changes, no live blocking.

Tags SOL_BUY_LIQ_LONG_200K trades as SOL_WEAK_GEOMETRY_SHADOW when:
  1. cluster_notional is between 500K and 1M, OR
  2. max_single_liq_share >= 80%, OR
  3. cluster_liq_count <= 2

Reports tagged vs untagged performance.
Writes tags to s34_shadow_geometry_tags table in s34_intelligence.db
for dashboard visibility.

Critical: shadow observe only. No runner block. No live config change.
"""
from __future__ import annotations
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_MD   = OUT_DIR / "S34_SOL200K_WEAK_GEOMETRY.md"

RULE_NAME  = "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30"
TAG_NAME   = "SOL_WEAK_GEOMETRY_SHADOW"

# Weak geometry criteria (any one triggers)
CAS_LO     = 500_000.0
CAS_HI     = 1_000_000.0
SHARE_THRESH = 80.0
MIN_LIQ_COUNT = 2   # <= this → weak


def _r1(v): return round(float(v), 1) if v is not None and math.isfinite(float(v)) else None
def _r3(v): return round(float(v), 3) if v is not None and math.isfinite(float(v)) else None

def _median(vals):
    c = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not c: return None
    i = (len(c) - 1) / 2
    lo, hi = math.floor(i), math.ceil(i)
    return c[lo] if lo == hi else (c[lo] + c[hi]) / 2

def _top3r(nets):
    s = sorted(nets)
    return sum(s[3:]) if len(s) > 3 else sum(s)

def _stats(nets: list[float]) -> dict:
    if not nets:
        return {"n": 0, "median": None, "cum": None, "top3r": None, "wr": None}
    exits_info = ""
    wr = sum(1 for n in nets if n > 0) / len(nets)
    return {
        "n":      len(nets),
        "median": _r1(_median(nets)),
        "cum":    _r1(sum(nets)),
        "top3r":  _r1(_top3r(nets)),
        "wr":     round(wr, 3),
    }


def is_weak_geometry(signal: dict) -> tuple[bool, list[str]]:
    reasons = []
    cas  = signal.get("liq_total_notional")
    cnt  = signal.get("liq_count")
    mx   = signal.get("liq_max_notional")

    share = (mx / cas * 100) if (cas and mx and cas > 0) else None

    if cas is not None and CAS_LO <= cas < CAS_HI:
        reasons.append(f"cascade_500K_1M ({cas:.0f})")
    if share is not None and share >= SHARE_THRESH:
        reasons.append(f"single_share_gte80 ({share:.1f}%)")
    if cnt is not None and cnt <= MIN_LIQ_COUNT:
        reasons.append(f"liq_count_lte2 ({cnt})")

    return bool(reasons), reasons


def load_sol200k_trades(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT trade_id, trade_json FROM s34_trades "
        "WHERE status='CLOSED' AND rule_name=? AND trade_json IS NOT NULL",
        (RULE_NAME,)
    ).fetchall()
    out = []
    for trade_id, raw in rows:
        try:
            t = json.loads(raw)
            if t.get("net_bps") is not None:
                t["_db_trade_id"] = trade_id
                out.append(t)
        except Exception:
            pass
    return out


def ensure_tag_table(conn_rw: sqlite3.Connection):
    conn_rw.execute("""
        CREATE TABLE IF NOT EXISTS s34_shadow_geometry_tags (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at    TEXT NOT NULL,
            trade_id      TEXT NOT NULL,
            rule_name     TEXT NOT NULL,
            tag           TEXT NOT NULL,
            reasons       TEXT NOT NULL,
            cascade_usd   REAL,
            liq_count     INTEGER,
            single_share  REAL,
            net_bps       REAL,
            exit_reason   TEXT
        )
    """)
    conn_rw.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_sgt_trade ON s34_shadow_geometry_tags(trade_id, tag)"
    )
    conn_rw.commit()


def upsert_tags(conn_rw: sqlite3.Connection, tagged: list[dict]):
    now = datetime.now(timezone.utc).isoformat()
    for row in tagged:
        conn_rw.execute("""
            INSERT INTO s34_shadow_geometry_tags
              (created_at, trade_id, rule_name, tag, reasons,
               cascade_usd, liq_count, single_share, net_bps, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(trade_id, tag) DO UPDATE SET
              reasons     = excluded.reasons,
              cascade_usd = excluded.cascade_usd,
              liq_count   = excluded.liq_count,
              single_share= excluded.single_share,
              net_bps     = excluded.net_bps,
              exit_reason = excluded.exit_reason,
              created_at  = excluded.created_at
        """, (
            now,
            row["trade_id"],
            RULE_NAME,
            TAG_NAME,
            json.dumps(row["reasons"]),
            row["cascade_usd"],
            row["liq_count"],
            row["single_share"],
            row["net_bps"],
            row["exit_reason"],
        ))
    conn_rw.commit()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # Read trades (read-only)
    conn_ro = sqlite3.connect(f"file:{INTEL_DB.as_posix()}?mode=ro", uri=True)
    trades  = load_sol200k_trades(conn_ro)
    conn_ro.close()

    print(f"S34 SOL 200K Weak Geometry Shadow Tagger — {now}")
    print(f"Total SOL_200K closed trades: {len(trades)}\n")

    tagged_rows   = []
    untagged_nets = []
    tagged_nets   = []
    tag_detail    = []   # per-trade detail

    for t in trades:
        signal = t.get("signal") or {}
        weak, reasons = is_weak_geometry(signal)

        cas   = signal.get("liq_total_notional")
        cnt   = signal.get("liq_count")
        mx    = signal.get("liq_max_notional")
        share = (mx / cas * 100) if (cas and mx and cas > 0) else None
        net   = float(t["net_bps"])
        exit_ = t.get("exit_reason")
        tid   = t.get("_db_trade_id") or t.get("trade_id")

        if weak:
            tagged_nets.append(net)
            tagged_rows.append({
                "trade_id":   tid,
                "reasons":    reasons,
                "cascade_usd": cas,
                "liq_count":  cnt,
                "single_share": share,
                "net_bps":    net,
                "exit_reason": exit_,
            })
            tag_detail.append(
                f"  [{tid}] net={net:+.1f}  exit={exit_}  "
                f"cas={int(cas) if cas else '?'}  cnt={cnt}  share={share:.1f}%  "
                f"tags={reasons}"
            )
        else:
            untagged_nets.append(net)

    st_tagged   = _stats(tagged_nets)
    st_untagged = _stats(untagged_nets)
    st_all      = _stats(tagged_nets + untagged_nets)

    print(f"TAGGED   (SOL_WEAK_GEOMETRY_SHADOW): N={st_tagged['n']}")
    print(f"  median={st_tagged['median']}  WR={st_tagged['wr']*100 if st_tagged['wr'] else 0:.0f}%  "
          f"cum={st_tagged['cum']}  top3r={st_tagged['top3r']}")
    print()
    print(f"UNTAGGED (clean geometry):           N={st_untagged['n']}")
    print(f"  median={st_untagged['median']}  WR={st_untagged['wr']*100 if st_untagged['wr'] else 0:.0f}%  "
          f"cum={st_untagged['cum']}  top3r={st_untagged['top3r']}")
    print()
    print(f"ALL SOL 200K:                        N={st_all['n']}")
    print(f"  median={st_all['median']}  WR={st_all['wr']*100 if st_all['wr'] else 0:.0f}%  "
          f"cum={st_all['cum']}  top3r={st_all['top3r']}")
    print()
    print("Tagged trade details:")
    for d in tag_detail:
        print(d)

    # Write tags to DB (read-write)
    conn_rw = sqlite3.connect(str(INTEL_DB))
    ensure_tag_table(conn_rw)
    upsert_tags(conn_rw, tagged_rows)
    total_tags = conn_rw.execute(
        "SELECT COUNT(*) FROM s34_shadow_geometry_tags WHERE tag=?", (TAG_NAME,)
    ).fetchone()[0]
    conn_rw.close()
    print(f"\nDB: {total_tags} tags written to s34_shadow_geometry_tags ({INTEL_DB.name})")

    # MD report
    lines = [
        "# S34 SOL 200K Weak Geometry — Shadow Tag Report",
        "",
        f"Generated: `{now}`",
        "",
        "**Shadow observe only. No runner change. No live blocking.**",
        "",
        "Criteria for `SOL_WEAK_GEOMETRY_SHADOW` tag (any one triggers):",
        "- cluster_notional 500K–1M",
        "- max_single_liq_share ≥ 80%",
        "- cluster_liq_count ≤ 2",
        "",
        "## Performance Split",
        "",
        "| Group | N | Median | Cum | Top3R | WR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Tagged (weak geometry) | {st_tagged['n']} | {st_tagged['median']} | "
        f"{st_tagged['cum']} | {st_tagged['top3r']} | "
        f"{st_tagged['wr']*100:.0f}% |" if st_tagged["n"] else "| Tagged | 0 | — | — | — | — |",
        f"| Untagged (clean) | {st_untagged['n']} | {st_untagged['median']} | "
        f"{st_untagged['cum']} | {st_untagged['top3r']} | "
        f"{st_untagged['wr']*100:.0f}% |" if st_untagged["n"] else "| Untagged | 0 | — | — | — | — |",
        f"| All SOL 200K | {st_all['n']} | {st_all['median']} | "
        f"{st_all['cum']} | {st_all['top3r']} | "
        f"{st_all['wr']*100:.0f}% |" if st_all["n"] else "",
        "",
        "## Tagged Trades (SOL_WEAK_GEOMETRY_SHADOW)",
        "",
        "| Trade ID | Net bps | Exit | Cascade | Liq Count | Single Share | Reasons |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in tagged_rows:
        share_str = f"{row['single_share']:.1f}%" if row["single_share"] is not None else "?"
        cas_str   = f"{int(row['cascade_usd']):,}" if row["cascade_usd"] else "?"
        lines.append(
            f"| {row['trade_id']} | {row['net_bps']:+.1f} | {row['exit_reason']} "
            f"| {cas_str} | {row['liq_count']} | {share_str} "
            f"| {', '.join(row['reasons'])} |"
        )

    lines += [
        "",
        "## DB Tag Table",
        "",
        f"Tags written to `s34_shadow_geometry_tags` in `{INTEL_DB.name}`.",
        "Total rows: " + str(total_tags),
        "",
        "Schema: `trade_id, rule_name, tag, reasons, cascade_usd, liq_count, single_share, net_bps, exit_reason`",
        "",
        "## Interpretation",
        "",
        "- **No block recommended** — N_tagged too small for live filter.",
        "- Monitor: if tagged trades continue underperforming as N grows toward 20+, "
          "consider `min_liq_count >= 3` as an exploratory shadow rule.",
        "- The 500K–1M cascade band and single-dominant spike may indicate "
          "a different market dynamic (concentrated forced sell vs distributed cascade).",
        "- Revisit when total SOL 200K N ≥ 50.",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD: {OUT_MD}")


if __name__ == "__main__":
    main()
