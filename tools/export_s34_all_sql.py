"""S34 — TUM alpha ailesi verisini SQL'e export et (tum session'lar).

Kaynaklar:
  - reports/research/s34/*.json   (tum research raporlari, tum session'lar)
  - reports/shadow/*.jsonl        (gercek paper trade ledger'lari)
Ciktilar:
  - reports/research/s34/S34_ALL.db          (SQLite)
  - reports/research/s34/S34_ALL.sql         (CREATE+INSERT dump — her yere import)
  - reports/research/s34/S34_ALL_ANALYSIS.md (karsilastirma/siralama analizi)

Tablolar:
  research_results  : her rapordaki her stat-blok (report, key, label, n, wr, avg, total, mc_p, ...)
  paper_trades      : ledger'lardaki her CLOSE trade (signal, direction, net_bps, ...)
  paper_signal_pnl  : sinyal basina toplu PnL (gercek paper)
  alpha_families    : curated aile taksonomisi (yon, universe, live/paper durum)
"""
from __future__ import annotations
import json, sqlite3, sys, glob
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/"reports"/"research"/"s34"
SHD=ROOT/"reports"/"shadow"
DB=RES/"S34_ALL.db"; SQL=RES/"S34_ALL.sql"; MD=RES/"S34_ALL_ANALYSIS.md"

def num(x):
    try:
        if x is None: return None
        if isinstance(x,bool): return None
        return float(x)
    except: return None

def direction_of(*strs):
    s=" ".join(str(x) for x in strs).upper()
    if "SHORT" in s or "FADE" in s or "_SELL_" in s and "LONG" not in s: return "SHORT"
    if "LONG" in s or "BOUNCE" in s or "REVERSAL_LONG" in s: return "LONG"
    return ""

def is_stat(d):
    if not isinstance(d,dict): return False
    if "n" not in d: return False
    nn=num(d.get("n"))
    if nn is None or nn<=0: return False
    return any(k in d for k in ("wr","avg","avg_bps","total","sum_bps","per_month","win_rate","mean_net_bps"))

def extract(d,report,path,rows):
    """recursive: her stat-blok dict'i research_results satirina cevir."""
    if isinstance(d,dict):
        if is_stat(d):
            wr=d.get("wr");
            if wr is None and d.get("win_rate") is not None: wr=num(d["win_rate"])*100
            rows.append({
                "report":report,"key":path[:180],"label":str(d.get("label") or "")[:200],
                "direction":direction_of(path,d.get("label")),
                "n":int(num(d.get("n")) or 0),
                "per_month":num(d.get("per_month")),
                "wr":num(wr),
                "avg_bps":num(d.get("avg") if d.get("avg") is not None else (d.get("avg_bps") if d.get("avg_bps") is not None else d.get("mean_net_bps"))),
                "total_bps":num(d.get("total") if d.get("total") is not None else (d.get("sum_bps") if d.get("sum_bps") is not None else d.get("cum_net_bps"))),
                "worst_bps":num(d.get("worst")),
                "tail_n":int(num(d.get("tail_n")) or 0) if d.get("tail_n") is not None else None,
                "mdd_bps":num(d.get("mdd")),
                "mc_p":num(d.get("mc_p")),
                "wf":str(d.get("wf") or "") or None,
                "ho_avg":num(d.get("ho_avg")),
            })
        for k,v in d.items():
            if isinstance(v,(dict,list)) and not k.startswith("_meta"):
                extract(v,report,f"{path}.{k}" if path else str(k),rows)
    elif isinstance(d,list):
        for i,v in enumerate(d):
            if isinstance(v,(dict,list)):
                # named row?
                nm=(v.get("key") or v.get("name") or v.get("config") or i) if isinstance(v,dict) else i
                extract(v,report,f"{path}[{nm}]",rows)

def load_research(rows):
    files=sorted(glob.glob(str(RES/"*.json")))
    ok=0; bad=0
    for f in files:
        name=Path(f).stem
        if name in ("S34_ALL","visualization_export"): continue
        try:
            with open(f,encoding="utf-8") as fh: data=json.load(fh)
            before=len(rows); extract(data,name,"",rows)
            ok+=1
        except Exception as e:
            bad+=1
    return ok,bad,len(files)

def load_ledgers(trades):
    for led in sorted(glob.glob(str(SHD/"*.jsonl"))):
        lname=Path(led).name
        try:
            with open(led,encoding="utf-8") as fh:
                for line in fh:
                    try: r=json.loads(line)
                    except: continue
                    if r.get("event")!="CLOSE": continue
                    trades.append({
                        "ledger":lname,"signal":str(r.get("signal") or "?"),
                        "direction":str(r.get("direction") or direction_of(r.get("signal"))),
                        "status":str(r.get("status") or ""),
                        "anchor_ts_ms":r.get("anchor_ts_ms"),"anchor_utc":r.get("anchor_utc") or r.get("opened_utc"),
                        "entry_ts_ms":r.get("entry_ts_ms"),"entry_price":num(r.get("entry_price")),
                        "exit_ts_ms":r.get("exit_ts_ms"),"exit_price":num(r.get("exit_price")),
                        "outcome_bps":num(r.get("outcome_bps")),"net_bps":num(r.get("net_bps")),
                        "close_reason":str(r.get("close_reason") or ""),"session":str(r.get("session") or ""),
                        "dow":r.get("dow"),"conviction_score":r.get("conviction_score"),
                        "running_notional":num(r.get("running_notional")),"closed_utc":r.get("closed_utc"),
                    })
        except Exception: pass

ALPHA_FAMILIES=[
    ("LONG_HOUR17_HOLD6H","LONG","ETH SELL 200K","LIVE","Ana canli alpha: hour>=17 UTC + regime, T0, 6h, 300bps stop, 15x"),
    ("LONG_HOUR17_COMPOSITE","LONG","ETH SELL 200K","PAPER","Composite conviction 0-8 (score>=3) + funding-veto"),
    ("LONG_HOUR17_100K_COMPOSITE","LONG","ETH SELL 100-200K","PAPER","Frekans genisletme, composite score>=3"),
    ("LONG_T15_BOUNCE","LONG","ETH SELL 200K","LEGACY_LIVE","T+15 bounce confirm (eski live LONG legi)"),
    ("SHORT_NEITHER","SHORT","ETH SELL 200K","LIVE","BTC SELL>=2M confirm -> SHORT 2h"),
    ("SHORT_NOISY_BTC1M_D5_H180","SHORT","ETH SELL 200K","PAPER","noisy + BTC1M confirm, 180m hold (diversifier)"),
    ("SHORT_BTC1M_H4","SHORT","ETH SELL 200K","PAPER","BTC1M confirm 4h"),
    ("SHORT_BTC1M_D10_H3","SHORT","ETH SELL 200K","PAPER","BTC1M delay10 3h"),
    ("LONG_ECHO_45_120_SILENCE","LONG","ETH SELL 200K","PAPER","echo 45-120m + silence"),
    ("LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE","LONG","ETH SELL 200K","PAPER","prebuild>=2 double cascade"),
    ("LONG_OFI_SILENCE_BUYERS","LONG","ETH SELL 200K","PAPER","silence + OFI buyers"),
    ("BUY_FADE_SHORT_H45_SL75","SHORT","ETH BUY 200K","PAPER","BUY-side fade short 45m/75bps"),
    ("LONG_SILENCE","LONG","ETH SELL 200K","LEGACY_PAPER","silence LONG (lookahead cikti — arsiv)"),
]

def main():
    if DB.exists(): DB.unlink()
    con=sqlite3.connect(DB); cur=con.cursor()
    cur.executescript("""
    CREATE TABLE research_results(id INTEGER PRIMARY KEY, report TEXT, key TEXT, label TEXT, direction TEXT,
      n INTEGER, per_month REAL, wr REAL, avg_bps REAL, total_bps REAL, worst_bps REAL, tail_n INTEGER,
      mdd_bps REAL, mc_p REAL, wf TEXT, ho_avg REAL);
    CREATE TABLE paper_trades(id INTEGER PRIMARY KEY, ledger TEXT, signal TEXT, direction TEXT, status TEXT,
      anchor_ts_ms INTEGER, anchor_utc TEXT, entry_ts_ms INTEGER, entry_price REAL, exit_ts_ms INTEGER,
      exit_price REAL, outcome_bps REAL, net_bps REAL, close_reason TEXT, session TEXT, dow INTEGER,
      conviction_score INTEGER, running_notional REAL, closed_utc TEXT);
    CREATE TABLE paper_signal_pnl(signal TEXT PRIMARY KEY, direction TEXT, n INTEGER, wins INTEGER, wr REAL,
      sum_net_bps REAL, avg_net_bps REAL, best_bps REAL, worst_bps REAL);
    CREATE TABLE alpha_families(signal TEXT PRIMARY KEY, direction TEXT, universe TEXT, status TEXT, note TEXT);
    """)
    # research
    rows=[]; ok,bad,tot=load_research(rows)
    cur.executemany("INSERT INTO research_results(report,key,label,direction,n,per_month,wr,avg_bps,total_bps,worst_bps,tail_n,mdd_bps,mc_p,wf,ho_avg) VALUES(:report,:key,:label,:direction,:n,:per_month,:wr,:avg_bps,:total_bps,:worst_bps,:tail_n,:mdd_bps,:mc_p,:wf,:ho_avg)",rows)
    # ledgers
    trades=[]; load_ledgers(trades)
    cur.executemany("INSERT INTO paper_trades(ledger,signal,direction,status,anchor_ts_ms,anchor_utc,entry_ts_ms,entry_price,exit_ts_ms,exit_price,outcome_bps,net_bps,close_reason,session,dow,conviction_score,running_notional,closed_utc) VALUES(:ledger,:signal,:direction,:status,:anchor_ts_ms,:anchor_utc,:entry_ts_ms,:entry_price,:exit_ts_ms,:exit_price,:outcome_bps,:net_bps,:close_reason,:session,:dow,:conviction_score,:running_notional,:closed_utc)",trades)
    # paper signal pnl
    cur.execute("""INSERT INTO paper_signal_pnl
      SELECT signal, MAX(direction), COUNT(*), SUM(CASE WHEN net_bps>0 THEN 1 ELSE 0 END),
             ROUND(100.0*SUM(CASE WHEN net_bps>0 THEN 1 ELSE 0 END)/COUNT(*),1),
             ROUND(SUM(net_bps),1), ROUND(AVG(net_bps),1), ROUND(MAX(net_bps),1), ROUND(MIN(net_bps),1)
      FROM paper_trades WHERE net_bps IS NOT NULL GROUP BY signal""")
    cur.executemany("INSERT INTO alpha_families VALUES(?,?,?,?,?)",ALPHA_FAMILIES)
    con.commit()

    # indexes
    cur.executescript("CREATE INDEX ix_rr_report ON research_results(report); CREATE INDEX ix_rr_dir ON research_results(direction); CREATE INDEX ix_pt_sig ON paper_trades(signal);")
    con.commit()

    # dump .sql
    with open(SQL,"w",encoding="utf-8") as f:
        for line in con.iterdump(): f.write(line+"\n")

    # ---- ANALYSIS ----
    def q(sql,p=()): return cur.execute(sql,p).fetchall()
    nres=q("SELECT COUNT(*) FROM research_results")[0][0]
    nrep=q("SELECT COUNT(DISTINCT report) FROM research_results")[0][0]
    ntr=q("SELECT COUNT(*) FROM paper_trades")[0][0]
    L=["# S34 — Tum Alpha Ailesi: SQL Export + Analiz","",
       f"> Uretim: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
       f"> Kaynaklar: {tot} research JSON ({ok} okundu, {bad} atlandi) + {len(list(glob.glob(str(SHD/'*.jsonl'))))} ledger.",
       f"> **research_results: {nres} satir / {nrep} rapor · paper_trades: {ntr} gercek trade**","",
       "DB: `S34_ALL.db` · SQL dump: `S34_ALL.sql` · bu analiz: `S34_ALL_ANALYSIS.md`","",
       "## Gercek Paper Trade PnL (ledger — sinyal basina)","",
       "| Signal | dir | N | WR | sum bps | avg bps | best | worst |","|---|---|--:|--:|--:|--:|--:|--:|"]
    for r in q("SELECT signal,direction,n,wr,sum_net_bps,avg_net_bps,best_bps,worst_bps FROM paper_signal_pnl ORDER BY n DESC"):
        L.append("| %s | %s | %d | %s%% | %+.0f | %+.1f | %+.0f | %+.0f |"%(r[0],r[1] or "",r[2],r[3],r[4] or 0,r[5] or 0,r[6] or 0,r[7] or 0))
    L+=["","## En Yuksek TOTAL PnL — Research (N>=15, mc_p<=0.05)","","| Report | Key | dir | N | /ay | WR | avg | TOTAL | mc_p |","|---|---|---|--:|--:|--:|--:|--:|--:|"]
    for r in q("SELECT report,key,direction,n,per_month,wr,avg_bps,total_bps,mc_p FROM research_results WHERE n>=15 AND mc_p<=0.05 AND total_bps IS NOT NULL ORDER BY total_bps DESC LIMIT 30"):
        L.append("| %s | %s | %s | %d | %s | %s | %s | %+.0f | %s |"%(r[0],str(r[1])[:46],r[2] or "",r[3],("%.1f"%r[4]) if r[4] else "-",("%.0f%%"%r[5]) if r[5] else "-",("%+.0f"%r[6]) if r[6] is not None else "-",r[7],("%.3f"%r[8]) if r[8] is not None else "-"))
    L+=["","## En Yuksek WR — Research (N>=20)","","| Report | Key | dir | N | WR | avg | TOTAL |","|---|---|---|--:|--:|--:|--:|"]
    for r in q("SELECT report,key,direction,n,wr,avg_bps,total_bps FROM research_results WHERE n>=20 AND wr IS NOT NULL ORDER BY wr DESC LIMIT 25"):
        L.append("| %s | %s | %s | %d | %.0f%% | %s | %s |"%(r[0],str(r[1])[:46],r[2] or "",r[3],r[4],("%+.0f"%r[5]) if r[5] is not None else "-",("%+.0f"%r[6]) if r[6] is not None else "-"))
    # LONG vs SHORT
    L+=["","## LONG vs SHORT (research, N>=10, mc_p<=0.05)","","| dir | test sayisi | ort WR | ort avg | ort TOTAL |","|---|--:|--:|--:|--:|"]
    for r in q("SELECT direction,COUNT(*),ROUND(AVG(wr),1),ROUND(AVG(avg_bps),1),ROUND(AVG(total_bps),0) FROM research_results WHERE n>=10 AND mc_p<=0.05 AND direction<>'' GROUP BY direction"):
        L.append("| %s | %d | %s%% | %+s | %+s |"%(r[0],r[1],r[2],r[3],r[4]))
    # alpha families
    L+=["","## Alpha Aileleri (taksonomi + gercek paper PnL)","","| Signal | dir | universe | durum | paper N | paper WR | paper sum | not |","|---|---|---|---|--:|--:|--:|---|"]
    for a in q("SELECT af.signal,af.direction,af.universe,af.status,af.note,pp.n,pp.wr,pp.sum_net_bps FROM alpha_families af LEFT JOIN paper_signal_pnl pp ON af.signal=pp.signal ORDER BY af.status,af.signal"):
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |"%(a[0],a[1],a[2],a[3],a[5] if a[5] is not None else "-",("%.0f%%"%a[6]) if a[6] is not None else "-",("%+.0f"%a[7]) if a[7] is not None else "-",a[4]))
    # per-report count
    L+=["","## Rapor Basina Sonuc Sayisi (en cok 25)","","| Report | sonuc |","|---|--:|"]
    for r in q("SELECT report,COUNT(*) c FROM research_results GROUP BY report ORDER BY c DESC LIMIT 25"):
        L.append("| %s | %d |"%(r[0],r[1]))
    L+=["","## Ornek SQL sorgulari","","```sql",
        "-- en iyi LONG'lar:",
        "SELECT report,key,n,wr,avg_bps,total_bps,mc_p FROM research_results WHERE direction='LONG' AND n>=15 AND mc_p<=0.05 ORDER BY total_bps DESC;",
        "-- gercek paper trade'ler (hour17 composite):",
        "SELECT * FROM paper_trades WHERE signal LIKE '%HOUR17%' ORDER BY closed_utc;",
        "-- bir raporun tum sonuclari:",
        "SELECT key,label,n,wr,avg_bps,total_bps FROM research_results WHERE report='S34_CONVICTION_COMPOSITE';",
        "```",""]
    MD.write_text("\n".join(L),encoding="utf-8")
    con.close()
    print(f"research_results: {nres} satir / {nrep} rapor ({ok} ok, {bad} bad)")
    print(f"paper_trades: {ntr}")
    print(f"DB:  {DB}")
    print(f"SQL: {SQL}")
    print(f"MD:  {MD}")

if __name__=="__main__": main()
