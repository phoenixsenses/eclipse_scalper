"""S34 — Konsensus-feature kompoziti: lean vs weighted vs v3-equal8. Holdout.
hour17 200K, 8 feature. Konsensus (S34_ALL): whale/shelf/rv5m/sync/funding en guvenilir.
Cikti: reports/research/s34/S34_CONSENSUS_COMPOSITE.json + .md
"""
from __future__ import annotations
import bisect, json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors
from ami.storage import production as PR
from ami.storage import research_reader as RR
DB=ROOT/"data"/"microstructure.db"; OUT=ROOT/"reports"/"research"/"s34"
OJ=OUT/"S34_CONSENSUS_COMPOSITE.json"; OM=OUT/"S34_CONSENSUS_COMPOSITE.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=1000; HOLD=6*3600_000; TM=4.5; TRAIN=0.70
CT={"sync":0.5421,"rv":0.0304,"d24":5.0,"be_lo":0.2195,"be_hi":2.0,"imb":0.2633,"shelf":2_775_000.0,"whale":6440.0}
random.seed(42)
def _s(c,q,p=()):
    r=c.execute(q,p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0
def lsum(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def lmax(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def lcnt(c,s,sd,lo,hi,t): return int(_s(c,"SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",(s,sd,lo,hi,t)))
def mbps(c,s,ts,lb):
    """Direct-SQL oracle -- kept as the parity reference for `mbps_v2`
    (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V1).
    No longer called by build(); the reader-backed path is used instead."""
    a=c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(s,ts-lb)).fetchone()
    b=c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(s,ts)).fetchone()
    return (float(b[0])-float(a[0]))/float(a[0])*1e4 if a and b and float(a[0])>0 else None
def mbps_v2(root,s,ts,lb,source_db_path=None):
    """Reader-backed replacement for `mbps`, via lookup_latest_at_or_before."""
    a=RR.lookup_latest_at_or_before(root,table="mark_prices",symbol=s,ts_ms=ts-lb,columns=("mark_price",),source_db_path=source_db_path)
    b=RR.lookup_latest_at_or_before(root,table="mark_prices",symbol=s,ts_ms=ts,columns=("mark_price",),source_db_path=source_db_path)
    return (float(b.row[0])-float(a.row[0]))/float(a.row[0])*1e4 if a.found and b.found and float(a.row[0])>0 else None
def rv5(c,ts):
    # OUT OF SCOPE for this migration: `vol_state` is not in the research-
    # reader's 3-table allowlist (mark_prices/agg_trades/book_ticker) --
    # stays direct-SQL, unchanged, per the gate's explicit instruction to
    # leave out-of-scope tables on their existing path.
    r=c.execute("SELECT rv_5m FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
def bimb(c,ts):
    """Direct-SQL oracle -- kept as the parity reference for `bimb_v2`."""
    r=c.execute("SELECT book_imbalance,ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None and (ts-int(r[1]))<=5*60_000 else None
def bimb_v2(root,ts,source_db_path=None):
    """Reader-backed replacement for `bimb`. The 5-minute staleness check
    stays a Python-side post-lookup guard (it compares the found row's
    own ts_ms to the query ts, not a fixed filter value)."""
    r=RR.lookup_latest_at_or_before(root,table="book_ticker",symbol="ETHUSDT",ts_ms=ts,columns=("book_imbalance","ts_ms"),source_db_path=source_db_path)
    return float(r.row[0]) if r.found and r.row[0] is not None and (ts-int(r.row[1]))<=5*60_000 else None
def ofir(c,lo,hi):
    # OUT OF SCOPE for this migration: a bounded-RANGE aggregate (SUM/
    # COUNT over a window), not an `ORDER BY ts_ms DESC LIMIT 1` point
    # lookup -- belongs to the range-read helper (`execute_read`),
    # already exercised by the two prior consumer-migration gates, not
    # this gate's point-lookup primitive. Stays direct-SQL, unchanged.
    r=c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),SUM(notional),COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?",(lo,hi)).fetchone()
    if not r or r[0] is None: return None,None
    b,se=float(r[0]),float(r[1]); t=b+se; return ((b-se)/t if t>0 else 0.0),((float(r[2])/int(r[3])) if r[3] else None)
def nf(c,ts):
    """Direct-SQL oracle -- kept as the parity reference for `nf_v2`."""
    r=c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return int(r[0]) if r and r[0] else None
def nf_v2(root,ts,source_db_path=None):
    """Reader-backed replacement for `nf`, using the helper's `filters`
    for the `next_funding_time_ms IS NOT NULL` predicate."""
    r=RR.lookup_latest_at_or_before(root,table="mark_prices",symbol="ETHUSDT",ts_ms=ts,columns=("next_funding_time_ms",),filters=(("next_funding_time_ms","!=",None),),source_db_path=source_db_path)
    return int(r.row[0]) if r.found and r.row[0] else None
def hod(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).hour
def sxn(ts):
    h=hod(ts); return "EUROPE" if 7<=h<13 else ("US" if 13<=h<21 else "OFF")
def ep(m,ts):
    r=m.at_or_after(ts); return (int(r[0]),float(r[1])) if r and float(r[1])>0 else None
def lret(m,ts,h):
    e=ep(m,ts)
    if not e: return None
    r=m.at_or_before(ts+h); return (float(r[1])-e[1])/e[1]*1e4 if r else None
def mcp(v,a):
    if len(v)<4: return None
    r=random.Random(0); ct=sum(1 for _ in range(MC) if sum(r.choice([-1,1])*abs(x) for x in v)/len(v)>=a); return round(ct/MC,3)
def stat(g,label="",months=None):
    m=months or TM
    if not g: return {"label":label,"n":0}
    net=[x-FEE for x in g]; n=len(net); w=sum(1 for x in net if x>0); a=sum(net)/n; sv=sorted(net)
    return {"label":label,"n":n,"wr":round(100*w/n,1),"avg":round(a,1),"total":round(sum(net),0),"per_month":round(n/m,1),"worst":round(sv[0],1),"mc_p":mcp(net,a)}
def ps(k,v):
    if not v or v.get("n",0)==0: print("    %-24s N=0"%k[:24]); return
    print("    %-24s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s mc_p=%s"%(k[:24],v["n"],v.get("per_month",0),str(v["wr"])+"%",str(v["avg"])+"bps",str(v.get("total")),v.get("mc_p","?")))
def noov(pairs,h=HOLD):
    busy=-1;o=[]
    for ts,v in sorted(pairs):
        if ts>=busy: o.append(v);busy=ts+h
    return o

def hits(f):
    return {"sync":f["sync"]>=CT["sync"],"rv":f["rv"] is not None and f["rv"]>=CT["rv"],"d24":f["d24"]>=CT["d24"],
            "ofi":f["ofi"] is not None and f["ofi"]>=0,"be":CT["be_lo"]<=f["be"]<CT["be_hi"],
            "imb":f["imb"] is not None and f["imb"]<=CT["imb"],"shelf":f["shelf"]>=CT["shelf"],
            "whale":f["whale"] is not None and f["whale"]<CT["whale"]}
# konsensus agirliklari (S34_ALL feature-vote'tan: whale/shelf/rv/sync yuksek)
WGT={"whale":3,"shelf":2,"rv":2,"sync":2,"be":1,"d24":1,"ofi":1,"imb":1}
def equal8(f): return sum(1 for v in hits(f).values() if v)
def lean4(f): h=hits(f); return sum(int(h[k]) for k in ("sync","rv","shelf","whale"))
def min3(f): h=hits(f); return sum(int(h[k]) for k in ("sync","shelf","whale"))
def weighted(f): h=hits(f); return sum(WGT[k] for k,v in h.items() if v)  # 0-13

def build(conn,root,m,now,start):
    ev=[]
    for a in reconstruct_anchors(load_liquidations(conn,"ETHUSDT","SELL",start,now),bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30):
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps_v2(root,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps_v2(root,"BTCUSDT",ts,7*24*3600_000) or 0
        if ((mbps_v2(root,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sxn(ts)=="EUROPE" or not(b4<0 or b7<0) or hod(ts)<17: continue
        sk=lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
        of,whale=ofir(conn,ts-5*60_000,ts); e=ep(m,ts)
        shelf=_s(conn,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?",(ts-24*3600_000,ts,e[1]*0.98,e[1])) if e else 0
        f={"sync":sk/rn if rn>0 else 0,"rv":rv5(conn,ts),"d24":lcnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,200_000),
           "ofi":of,"be":lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts)/rn if rn>0 else 0,"imb":bimb_v2(root,ts),"shelf":shelf,"whale":whale}
        nff=nf_v2(root,ts); m2f=((nff-ts)/60_000) if nff else None
        y=lret(m,ts,HOLD)
        if y is None: continue
        ev.append({"ts":ts,"f":f,"veto":(m2f is not None and m2f<60),"y":y})
    ev.sort(key=lambda x:x["ts"]); return ev

def evalgate(ev,scorefn,K,name):
    n=len(ev); cut=int(n*TRAIN); te=ev[cut:]
    full=[(e["ts"],e["y"]) for e in ev if scorefn(e["f"])>=K and not e["veto"]]
    R={}
    R[name+"_full"]=stat([v for _,v in full],name+" full",TM)
    R[name+"_TEST"]=stat([e["y"] for e in te if scorefn(e["f"])>=K and not e["veto"]],name+" TEST",TM*(1-TRAIN))
    nv=noov(full); s=stat(nv,name+" noov",TM); s["per_month"]=round(len(nv)/TM,1); R[name+"_noov"]=s
    return R

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 Consensus Composite ===")
    root,_root_source=PR.resolve_production_root()
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        m=load_mark_index(conn,"ETHUSDT"); print("build..."); ev=build(conn,root,m,now,start)
        sp=[e["ts"] for e in ev]; TM=max(1.0,(sp[-1]-sp[0])/86_400_000/30.0); print(f"  events={len(ev)} months={TM:.2f}")
        R={}
        print("\n  v3 equal8 (mevcut):")
        for K in (3,4): R.update(evalgate(ev,equal8,K,f"v3_equal8_ge{K}")); ps(f"v3_equal8_ge{K}_full",R[f"v3_equal8_ge{K}_full"]); ps(f"v3_equal8_ge{K}_TEST",R[f"v3_equal8_ge{K}_TEST"]); ps(f"v3_equal8_ge{K}_noov",R[f"v3_equal8_ge{K}_noov"])
        print("\n  lean4 (sync+rv+shelf+whale):")
        for K in (2,3): R.update(evalgate(ev,lean4,K,f"lean4_ge{K}")); ps(f"lean4_ge{K}_full",R[f"lean4_ge{K}_full"]); ps(f"lean4_ge{K}_TEST",R[f"lean4_ge{K}_TEST"]); ps(f"lean4_ge{K}_noov",R[f"lean4_ge{K}_noov"])
        print("\n  min3 (sync+shelf+whale):")
        for K in (2,3): R.update(evalgate(ev,min3,K,f"min3_ge{K}")); ps(f"min3_ge{K}_full",R[f"min3_ge{K}_full"]); ps(f"min3_ge{K}_TEST",R[f"min3_ge{K}_TEST"])
        print("\n  weighted (konsensus agirlik, 0-13):")
        for K in (5,6,7): R.update(evalgate(ev,weighted,K,f"weighted_ge{K}")); ps(f"weighted_ge{K}_full",R[f"weighted_ge{K}_full"]); ps(f"weighted_ge{K}_TEST",R[f"weighted_ge{K}_TEST"]); ps(f"weighted_ge{K}_noov",R[f"weighted_ge{K}_noov"])
    meta={"n":len(ev),"months":round(TM,2)}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"results":R,"meta":meta},indent=2,default=str),encoding="utf-8")
    lines=[f"# S34 Consensus Composite (lean vs weighted vs v3)","",f"> hour17 200K {len(ev)} event {TM:.1f} ay. Holdout 70/30. Tarih {datetime.now(timezone.utc):%Y-%m-%d}","",
           "| Variant | N | /ay | WR | avg | total | mc_p |","|---|--:|--:|--:|--:|--:|--:|"]
    for k,v in R.items():
        if v.get("n",0)>0: lines.append("| %s | %d | %.1f | %.1f%% | %+.0f | %+.0f | %s |"%(k,v["n"],v.get("per_month",0),v["wr"],v["avg"],v.get("total"),v.get("mc_p","?")))
    lines+=["","---","*Script: tools/research_s34_consensus_composite.py*"]
    OM.write_text("\n".join(lines),encoding="utf-8"); print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")
if __name__=="__main__": main()
