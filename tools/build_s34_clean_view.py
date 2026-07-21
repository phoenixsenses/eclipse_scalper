"""S34 — research_results -> temiz/canonical view + trust skoru + temiz leaderboard.
SADECE S34_ALL.db uzerinde calisir (live'a dokunmaz).
Cikti: S34_ALL.db'ye research_clean tablosu + reports/research/s34/S34_ALL_CLEAN.md
"""
from __future__ import annotations
import sqlite3, statistics
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/"reports"/"research"/"s34"
DB=RES/"S34_ALL.db"; OUT=RES/"S34_ALL_CLEAN.md"
con=sqlite3.connect(DB); cur=con.cursor()

# non-bps / bozuk raporlar (overfit-dedektorun yakaladigi: total bps degil)
BAD_REPORTS=("S34_V02_EVENT_CHAIN_PUZZLE_TESTS","FUNDING_EXTREME_MEAN_REVERSION","S34_NAVIGATION_BRIDGE",
             "FUNDING_NONOVERLAP","NONPREDICTIVE_CARRY_PROVISION","ETH_PROVISION_REALISM_CORE","ETH_PROVISION_REALISM_FULL")

cur.execute("DROP TABLE IF EXISTS research_clean")
cur.execute("""CREATE TABLE research_clean AS
  SELECT *,
    CASE WHEN wr IS NULL THEN NULL WHEN wr<=1.0 THEN wr*100 ELSE wr END AS wr_norm,
    (CASE WHEN (total_bps IS NULL OR ABS(total_bps)<50000) AND (avg_bps IS NULL OR ABS(avg_bps)<2000)
              AND (wr IS NULL OR wr<=100) THEN 1 ELSE 0 END) AS bps_ok
  FROM research_results
  WHERE report NOT IN {bad}""".format(bad=BAD_REPORTS))
# trust skoru 0-4
cur.execute("ALTER TABLE research_clean ADD COLUMN trust INTEGER")
cur.execute("""UPDATE research_clean SET trust =
   (CASE WHEN n>=15 THEN 1 ELSE 0 END)
 + (CASE WHEN mc_p IS NOT NULL AND mc_p<=0.05 THEN 1 ELSE 0 END)
 + (CASE WHEN bps_ok=1 THEN 1 ELSE 0 END)
 + (CASE WHEN ho_avg IS NOT NULL OR key LIKE '%TEST%' OR key LIKE '%noov%' OR key LIKE '%holdout%' THEN 1 ELSE 0 END)""")
con.commit()
cur.execute("CREATE INDEX ix_rc_trust ON research_clean(trust)")
con.commit()

def q(sql,p=()): return cur.execute(sql,p).fetchall()
ncl=q("SELECT COUNT(*) FROM research_clean")[0][0]
nbad=q("SELECT COUNT(*) FROM research_results")[0][0]-ncl
bps_ok=q("SELECT COUNT(*) FROM research_clean WHERE bps_ok=1")[0][0]

L=["# S34 — Temiz / Canonical Research View","",
   f"> `research_clean` tablosu S34_ALL.db'ye eklendi. {ncl} satir (bozuk-rapor {nbad} atildi). bps-temiz: {bps_ok}. ",
   f"> Uretim: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Live/.env/sizing DOKUNULMADI.","",
   "**Filtreler:** non-bps raporlar atildi; wr 0-1 ise *100 normalize; bps_ok = |total|<50k & |avg|<2k & wr<=100. ",
   "**trust (0-4):** N>=15 + mc_p<=.05 + bps_ok + (OOS/noov/holdout kanit).","",
   "## En Guvenilir Sonuclar (trust=4, bps-temiz, total'a gore)","",
   "| Report | key | dir | N | /ay | WR | avg | total | mc_p |","|---|---|---|--:|--:|--:|--:|--:|--:|"]
for r in q("SELECT report,key,direction,n,per_month,wr_norm,avg_bps,total_bps,mc_p FROM research_clean WHERE trust=4 AND bps_ok=1 AND total_bps IS NOT NULL ORDER BY total_bps DESC LIMIT 30"):
    L.append("| %s | %s | %s | %d | %s | %s | %s | %+.0f | %s |"%(r[0],str(r[1])[:40],r[2] or "",r[3],("%.1f"%r[4]) if r[4] else "-",("%.0f%%"%r[5]) if r[5] is not None else "-",("%+.0f"%r[6]) if r[6] is not None else "-",r[7],("%.3f"%r[8]) if r[8] is not None else "-"))

L+=["","## En Guvenilir WR (trust=4, N>=20)","","| Report | key | dir | N | WR | avg | total |","|---|---|---|--:|--:|--:|--:|"]
for r in q("SELECT report,key,direction,n,wr_norm,avg_bps,total_bps FROM research_clean WHERE trust=4 AND bps_ok=1 AND n>=20 AND wr_norm IS NOT NULL ORDER BY wr_norm DESC LIMIT 20"):
    L.append("| %s | %s | %s | %d | %.0f%% | %s | %s |"%(r[0],str(r[1])[:40],r[2] or "",r[3],r[4],("%+.0f"%r[5]) if r[5] is not None else "-",("%+.0f"%r[6]) if r[6] is not None else "-"))

# trust dagilimi
L+=["","## Trust Dagilimi","","| trust | satir |","|--:|--:|"]
for r in q("SELECT trust,COUNT(*) FROM research_clean GROUP BY trust ORDER BY trust DESC"):
    L.append("| %s | %d |"%(r[0],r[1]))

# canonical registry (alpha_families + clean evidence sayisi)
L+=["","## Canonical Alpha Registry (aile + temiz kanit sayisi)","",
    "| Signal | dir | durum | temiz-kanit(trust>=3) | not |","|---|---|---|--:|---|"]
for a in q("SELECT signal,direction,status,note FROM alpha_families ORDER BY status,signal"):
    # kaba kanit: signal token'i clean'de kac trust>=3 satirda geciyor
    tok=a[0].lower().split("_")[0]
    ev=q("SELECT COUNT(*) FROM research_clean WHERE trust>=3 AND (LOWER(key)||' '||LOWER(label)) LIKE ?",("%"+tok+"%",))[0][0]
    L.append("| %s | %s | %s | %d | %s |"%(a[0],a[1],a[2],ev,a[3][:50]))

L+=["","---","*Uretim: tools/build_s34_clean_view.py — S34_ALL.db research_clean tablosu.*"]
OUT.write_text("\n".join(L),encoding="utf-8")
con.commit(); con.close()
print("research_clean:",ncl,"satir | trust=4 var. Rapor:",OUT)
