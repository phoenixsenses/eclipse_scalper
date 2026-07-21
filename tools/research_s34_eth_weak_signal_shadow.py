# encoding: utf-8
"""
S34 ETH 500K Weak Signal Shadow Tagging
Shadow-observe only. No runner changes, no live blocking.

Tags ETH_BUY_LIQ_LONG_500K trades with:
  - ETH_WEAK_COUNT_SHADOW  : liq_count <= 7
  - ETH_HIGH_SHARE_SHADOW  : max_single_liq_share >= 80%

Writes to s34_shadow_geometry_tags (same table as SOL_WEAK_GEOMETRY_SHADOW).
Reports tagged vs untagged performance to show OOS evidence over time.

Background: 2 historical SLs not caught by existing SOL geo filter are
both ETH trades. Both have cnt<=7 or share>=80%. Adding as shadow tags
to monitor going forward — minimum N=30 needed before live action.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT     = Path("D:/eclipse_scalper")
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_MD   = OUT_DIR / "S34_ETH500K_WEAK_SIGNAL.md"

RULE_NAME     = "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30"
TAG_CNT_WEAK  = "ETH_WEAK_COUNT_SHADOW"
TAG_HIGH_SHARE = "ETH_HIGH_SHARE_SHADOW"

CNT_THRESHOLD   = 7     # count <= this → weak signal
SHARE_THRESHOLD = 80.0  # share >= this → dominated by single liq


def _stats(nets: list[float]) -> dict:
    if not nets:
        return {"n": 0, "median": "n/a", "wr": None, "cum": "n/a", "sl": 0, "sl_pct": "n/a"}
    s = sorted(nets)
    med = s[len(s) // 2]
    sl_n = sum(1 for v in nets if v < -20)
    return {
        "n": len(nets),
        "median": f"{med:+.1f}",
        "wr": sum(1 for v in nets if v > 0) / len(nets),
        "cum": f"{sum(nets):+.1f}",
        "sl": sl_n,
        "sl_pct": f"{sl_n/len(nets)*100:.0f}%",
    }


def ensure_tag_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS s34_shadow_geometry_tags (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at    TEXT NOT NULL,
            trade_id      TEXT NOT NULL,
            rule_name     TEXT NOT NULL,
            tag           TEXT NOT NULL,
            reasons       TEXT,
            cascade_usd   REAL,
            liq_count     INTEGER,
            single_share  REAL,
            net_bps       REAL,
            exit_reason   TEXT
        )
    """)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_sgt_trade ON "
        "s34_shadow_geometry_tags(trade_id, tag)"
    )
    conn.commit()


def upsert_tags(conn: sqlite3.Connection, tagged: list[dict]):
    now = datetime.now(timezone.utc).isoformat()
    for row in tagged:
        conn.execute(
            """
            INSERT INTO s34_shadow_geometry_tags
              (created_at, trade_id, rule_name, tag, reasons,
               cascade_usd, liq_count, single_share, net_bps, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(trade_id, tag) DO UPDATE SET
              created_at   = excluded.created_at,
              reasons      = excluded.reasons,
              net_bps      = excluded.net_bps,
              exit_reason  = excluded.exit_reason
            """,
            (
                now, row["trade_id"], row["rule_name"], row["tag"], row["reasons"],
                row["cascade_usd"], row["liq_count"], row["single_share"],
                row["net_bps"], row["exit_reason"],
            ),
        )
    conn.commit()


def main():
    intel = sqlite3.connect(str(INTEL_DB))
    ensure_tag_table(intel)

    rows = intel.execute(
        "SELECT trade_id, entry_ts_ms, net_bps, exit_reason, trade_json "
        "FROM s34_trades WHERE rule_name=? AND status='CLOSED' AND net_bps IS NOT NULL "
        "ORDER BY entry_ts_ms",
        (RULE_NAME,),
    ).fetchall()

    tagged_cnt   = []
    tagged_share = []
    tagged_both  = []
    untagged     = []
    tag_rows     = []

    for tid, entry_ms, net_bps, exit_r, tj in rows:
        try:
            t   = json.loads(tj)
            sig = t.get("signal") or {}
            cnt = sig.get("liq_count")
            cas = sig.get("liq_total_notional")
            mx  = sig.get("liq_max_notional")
            share = (mx / cas * 100) if (cas and mx and cas > 0) else None

            is_cnt_weak   = cnt is not None and cnt <= CNT_THRESHOLD
            is_share_high = share is not None and share >= SHARE_THRESHOLD

            reasons = []
            if is_cnt_weak:
                reasons.append(f"cnt={cnt}<={CNT_THRESHOLD}")
            if is_share_high:
                reasons.append(f"share={share:.0f}%>={SHARE_THRESHOLD:.0f}%")

            net = float(net_bps)
            base = {
                "trade_id": tid, "rule_name": RULE_NAME,
                "cascade_usd": cas, "liq_count": cnt, "single_share": share,
                "net_bps": net, "exit_reason": exit_r,
            }

            if is_cnt_weak:
                tag_rows.append({**base, "tag": TAG_CNT_WEAK,
                                  "reasons": ";".join(reasons)})
            if is_share_high:
                tag_rows.append({**base, "tag": TAG_HIGH_SHARE,
                                  "reasons": ";".join(reasons)})

            if is_cnt_weak and is_share_high:
                tagged_both.append(net)
            elif is_cnt_weak:
                tagged_cnt.append(net)
            elif is_share_high:
                tagged_share.append(net)
            else:
                untagged.append(net)

        except Exception:
            pass

    upsert_tags(intel, tag_rows)

    cnt_weak_total  = tagged_cnt + tagged_both
    share_high_total = tagged_share + tagged_both
    any_tagged       = tagged_cnt + tagged_share + tagged_both

    st_cnt   = _stats(cnt_weak_total)
    st_share = _stats(share_high_total)
    st_any   = _stats(any_tagged)
    st_clean = _stats(untagged)
    st_all   = _stats(any_tagged + untagged)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("=" * 70)
    print(f"ETH 500K WEAK SIGNAL SHADOW TAGS   {now}")
    print(f"Rule: {RULE_NAME}")
    print("=" * 70)
    print()
    print(f"Criteria:")
    print(f"  {TAG_CNT_WEAK}   : liq_count <= {CNT_THRESHOLD}")
    print(f"  {TAG_HIGH_SHARE}: share >= {SHARE_THRESHOLD:.0f}%")
    print()

    for label, s in [
        (f"cnt<={CNT_THRESHOLD} (ETH_WEAK_COUNT)",        st_cnt),
        (f"share>={SHARE_THRESHOLD:.0f}% (ETH_HIGH_SHARE)", st_share),
        ("EITHER tag",                                     st_any),
        ("CLEAN (no tag)",                                 st_clean),
        ("ALL ETH 500K",                                   st_all),
    ]:
        if s["n"] == 0:
            print(f"  {label:42} N=0")
            continue
        sl_info = f"  SL={s['sl']} ({s['sl_pct']})"
        print(f"  {label:42} N={s['n']:>3}  WR={s['wr']*100:.0f}%  "
              f"med={s['median']:>7}  cum={s['cum']:>8}{sl_info}")

    print()
    print("-" * 70)
    print("TAGGED TRADE DETAIL")
    print("-" * 70)
    print(f"  {'Date':>5}  {'cnt':>4}  {'Share':>6}  {'Exit':>2}  {'Net':>7}  Tags")
    print("  " + "-" * 52)

    intel_ro = sqlite3.connect(f"file:{INTEL_DB}?mode=ro", uri=True)
    tag_detail = intel_ro.execute(
        "SELECT DISTINCT t.trade_id, t.entry_ts_ms, t.net_bps, t.exit_reason, t.trade_json, "
        "GROUP_CONCAT(s.tag, ' | ') as tags "
        "FROM s34_trades t JOIN s34_shadow_geometry_tags s ON t.trade_id=s.trade_id "
        "WHERE t.rule_name=? AND t.status='CLOSED' "
        "AND s.tag IN (?,?) GROUP BY t.trade_id ORDER BY t.entry_ts_ms",
        (RULE_NAME, TAG_CNT_WEAK, TAG_HIGH_SHARE),
    ).fetchall()
    intel_ro.close()

    for tid, entry_ms, net_bps, exit_r, tj, tags in tag_detail:
        try:
            sig = json.loads(tj).get("signal") or {}
            cnt = sig.get("liq_count")
            cas = sig.get("liq_total_notional")
            mx  = sig.get("liq_max_notional")
            share = (mx / cas * 100) if (cas and mx and cas > 0) else None
            dt  = datetime.fromtimestamp(entry_ms/1000, tz=timezone.utc).strftime("%m/%d")
            flag = "SL" if exit_r and "SL" in exit_r else ("BE" if exit_r and "BE" in exit_r else "TP")
            cnt_s  = str(cnt) if cnt is not None else "?"
            sh_s   = f"{share:.0f}%" if share is not None else "?"
            print(f"  {dt:>5}  {cnt_s:>4}  {sh_s:>6}  {flag:>2}  {float(net_bps):>+7.1f}  {tags}")
        except Exception:
            pass

    tag_cnt_db   = intel.execute(
        "SELECT COUNT(*) FROM s34_shadow_geometry_tags WHERE tag=?", (TAG_CNT_WEAK,)
    ).fetchone()[0]
    tag_share_db = intel.execute(
        "SELECT COUNT(*) FROM s34_shadow_geometry_tags WHERE tag=?", (TAG_HIGH_SHARE,)
    ).fetchone()[0]
    intel.close()

    print()
    print(f"DB tags written: {TAG_CNT_WEAK}={tag_cnt_db}  {TAG_HIGH_SHARE}={tag_share_db}")
    print()
    print("NOTE: Shadow observe only. No runner change. No live blocking.")
    print(f"      Need N>=30 tagged trades with consistent SL rate before any live action.")

    # Write MD report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S34 ETH 500K Weak Signal — Shadow Tag Report",
        f"Generated: `{now}`",
        "",
        "**Shadow observe only. No runner change. No live blocking.**",
        "",
        f"## Criteria",
        "",
        f"| Tag | Condition |",
        "| --- | --- |",
        f"| `{TAG_CNT_WEAK}` | liq_count ≤ {CNT_THRESHOLD} |",
        f"| `{TAG_HIGH_SHARE}` | max_single_share ≥ {SHARE_THRESHOLD:.0f}% |",
        "",
        "## Performance Summary",
        "",
        "| Group | N | WR | Median bps | Cum bps | SL | SL% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, s in [
        (f"cnt≤{CNT_THRESHOLD}", st_cnt),
        (f"share≥{SHARE_THRESHOLD:.0f}%", st_share),
        ("Either tag", st_any),
        ("Clean (no tag)", st_clean),
        ("All ETH 500K", st_all),
    ]:
        if s["n"] == 0:
            lines.append(f"| {label} | 0 | — | — | — | — | — |")
        else:
            lines.append(
                f"| {label} | {s['n']} | {s['wr']*100:.0f}% | {s['median']} "
                f"| {s['cum']} | {s['sl']} | {s['sl_pct']} |"
            )

    lines += [
        "",
        "## Status",
        "",
        f"- `{TAG_CNT_WEAK}` DB rows: {tag_cnt_db}",
        f"- `{TAG_HIGH_SHARE}` DB rows: {tag_share_db}",
        "",
        "Minimum N=30 tagged trades needed before any live evaluation.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {OUT_MD}")


if __name__ == "__main__":
    main()
