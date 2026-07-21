"""S34 Master Navigator — conviction-priority scheduler (yollari bagla, tam detay).

Sorun (Puzzle D1): tek-slot naive FIFO'da dusuk-conviction 100K route slotu kapip
yuksek-conviction composite'i pre-empt ediyor -> birlesik < long_comp tek basina.
Cozum: conviction-oncelikli admission (dusuk-kalite route'a yuksek bar) + route priority.
LOOKAHEAD YOK: her event geldiginde ANINDA karar (gelecegi bilmeden).

Route havuzu:
  LONG_comp  : hour17 200K, score8>=3, 6h hold,  priority=score8 (3-8)
  LONG_100k  : hour17 100-200K mini, score8>=3, 6h, priority=score8 (100k penalize edilebilir)
  SHORT_1317 : SHORT confirm 13-17 UTC BTC>=1M, 180m, priority=5 (+1 if BTC>=2M)

Politikalar:
  P0 base_long_comp        (sadece long_comp, no-overlap)
  P1 fifo_union            (hepsi, ilk gelen alir)  [Puzzle D1]
  P2 admit>=4 / admit>=5   (hepsi ama priority esigi)
  P3 route_priority        (long_comp>=3, short her, 100k YALNIZ >=5)
  P4 route_priority_strict (long_comp>=4, short her, 100k >=6)
  + conviction-weighted sizing en iyi politikada (unit=priority-2)

Cikti: reports/research/s34/S34_MASTER_NAVIGATOR.json + .md
"""
from __future__ import annotations
import bisect, json, random, sqlite3, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB=ROOT/"data"/"microstructure.db"; OUT=ROOT/"reports"/"research"/"s34"
OJ=OUT/"S34_MASTER_NAVIGATOR.json"; OM=OUT/"S34_MASTER_NAVIGATOR.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=1000; HOLD=6*3600_000; SHOLD=180*60_000; TM=4.5
CT={"sync":0.5421,"rv":0.0304,"d24":5.0,"be_lo":0.2195,"be_hi":2.0,"imb":0.2633,"shelf":2_775_000.0,"whale":6440.0}
random.seed(42)

def _s(c,q,p=()):
    r=c.execute(q,p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0
def lsum(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def lmax(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def lcnt(c,s,sd,lo,hi,t): return int(_s(c,"SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",(s,sd,lo,hi,t)))
def lfirst(c,s,sd,lo,hi,t):
    r=c.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1",(s,sd,lo,hi,t)).fetchone()
    return int(r[0]) if r else None
def mbps(c,s,ts,lb):
    a=c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(s,ts-lb)).fetchone()
    b=c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(s,ts)).fetchone()
    return (float(b[0])-float(a[0]))/float(a[0])*1e4 if a and b and float(a[0])>0 else None
def rv5(c,ts):
    r=c.execute("SELECT rv_5m FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
def book_imb(c,ts):
    r=c.execute("SELECT book_imbalance,ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None and (ts-int(r[1]))<=5*60_000 else None
def ofir(c,lo,hi):
    r=c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),SUM(notional),COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?",(lo,hi)).fetchone()
    if not r or r[0] is None: return None,None
    b,se=float(r[0]),float(r[1]); t=b+se; whale=(float(r[2])/int(r[3])) if r[3] else None
    return ((b-se)/t if t>0 else 0.0),whale
def nextfund(c,ts):
    r=c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return int(r[0]) if r and r[0] else None
def hod(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).hour
def sxn(ts):
    h=hod(ts); return "EUROPE" if 7<=h<13 else ("US" if 13<=h<21 else "OFF")
def ep(m,ts):
    r=m.at_or_after(ts); return (int(r[0]),float(r[1])) if r and float(r[1])>0 else None
def lret(m,ts,hold):
    e=ep(m,ts)
    if not e: return None
    r=m.at_or_before(ts+hold); return (float(r[1])-e[1])/e[1]*1e4 if r else None
def sret(m,ts,hold):
    e=ep(m,ts)
    if not e: return None
    r=m.at_or_before(ts+hold); return -(float(r[1])-e[1])/e[1]*1e4 if r else None

def mcp(v,a):
    if len(v)<4: return None
    r=random.Random(0); ct=sum(1 for _ in range(MC) if sum(r.choice([-1,1])*abs(x) for x in v)/len(v)>=a); return round(ct/MC,3)
def stat(net,label="",months=None):
    m=months or TM
    if not net: return {"label":label,"n":0}
    n=len(net); w=sum(1 for x in net if x>0); sv=sorted(net); a=sum(net)/n
    cum=pk=mdd=0.0
    for x in net: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return {"label":label,"n":n,"wr":round(100*w/n,1),"avg":round(a,1),"total":round(sum(net),0),"per_month":round(n/m,1),"worst":round(sv[0],1),"mdd":round(mdd,0),"mc_p":mcp(net,a)}
def ps(k,v):
    if not v or v.get("n",0)==0: print("    %-30s N=0"%k[:30]); return
    print("    %-30s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s mdd=%-7s mc_p=%s"%(k[:30],v["n"],v.get("per_month",0),str(v["wr"])+"%",str(v["avg"])+"bps",str(v.get("total")),str(v.get("mdd")),v.get("mc_p","?")))

def score8(f):
    s=(int(f["sync"]>=CT["sync"])+int(f["rv"] is not None and f["rv"]>=CT["rv"])+int(f["d24"]>=CT["d24"])
       +int(f["ofi"] is not None and f["ofi"]>=0)+int(CT["be_lo"]<=f["be"]<CT["be_hi"])
       +int(f["imb"] is not None and f["imb"]<=CT["imb"])+int(f["shelf"]>=CT["shelf"]))
    if f["whale"] is not None and f["whale"]<CT["whale"]: s+=1
    return s
def feats(conn,m,ts,rn):
    sk=lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
    of,whale=ofir(conn,ts-5*60_000,ts); e=ep(m,ts)
    shelf=_s(conn,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?",(ts-24*3600_000,ts,e[1]*0.98,e[1])) if e else 0
    return {"sync":sk/rn if rn>0 else 0,"rv":rv5(conn,ts),"d24":lcnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,200_000),
            "ofi":of,"be":lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts)/rn if rn>0 else 0,"imb":book_imb(conn,ts),"shelf":shelf,"whale":whale}

def build_pool(conn,m,now,start):
    """Tum route event'lerini tek havuzda: (ts, route, priority, hold, dir, net_bps)."""
    pool=[]
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    # LONG comp 200K + LONG 100k mini
    for thr,route,mini in ((200_000.0,"LONG_comp",False),(100_000.0,"LONG_100k",True)):
        for a in reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(thr,),accel_window_sec=30):
            ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
            if rn<thr or (mini and rn>=200_000) or m.at_or_after(ts) is None: continue
            b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
            if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sxn(ts)=="EUROPE" or not(b4<0 or b7<0) or hod(ts)<17: continue
            f=feats(conn,m,ts,rn); sc=score8(f)
            if sc<3: continue
            nf=nextfund(conn,ts); m2f=((nf-ts)/60_000) if nf else None
            if m2f is not None and m2f<60: continue  # funding veto
            y=lret(m,ts,HOLD)
            if y is None: continue
            pool.append({"ts":ts,"route":route,"priority":sc,"hold":HOLD,"dir":"LONG","net":y-FEE})
    # SHORT 13-17 confirm BTC>=1M
    for a in reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30):
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sxn(ts)=="EUROPE": continue
        nt=lfirst(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP)
        if nt is None or not (13<=hod(nt)<17): continue
        conf=lfirst(conn,"BTCUSDT","SELL",nt+5*60_000,nt+30*60_000,1_000_000.0)
        if conf is None: continue
        btc=lmax(conn,"BTCUSDT","SELL",nt+5*60_000,nt+30*60_000)
        y=sret(m,conf,SHOLD)
        if y is None: continue
        pri=6 if btc>=2_000_000 else 5
        pool.append({"ts":conf,"route":"SHORT_1317","priority":pri,"hold":SHOLD,"dir":"SHORT","net":y-FEE})
    pool.sort(key=lambda x:x["ts"])
    return pool

def schedule(pool,admit=None,weighted=False,months=TM):
    """LOOKAHEAD-YOK sequential scheduler. admit(ev)->bool. weighted: unit=priority-2."""
    busy=-1; taken=[]; mix=defaultdict(int)
    for ev in pool:  # kronolojik
        if admit and not admit(ev): continue
        if ev["ts"]>=busy:
            taken.append(ev); busy=ev["ts"]+ev["hold"]; mix[ev["route"]]+=1
    net=[e["net"] for e in taken]
    s=stat(net,"",months); s["per_month"]=round(len(taken)/months,1); s["mix"]=dict(mix)
    if weighted:
        wsum=sum(e["net"]*(max(1,e["priority"]-2)) for e in taken)
        units=sum(max(1,e["priority"]-2) for e in taken)
        s["weighted_total"]=round(wsum,0); s["weighted_per_unit"]=round(wsum/units,1) if units else None; s["units"]=units
    return s

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 Master Navigator ===")
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        m=load_mark_index(conn,"ETHUSDT")
        print("build route pool..."); pool=build_pool(conn,m,now,start)
        span=[e["ts"] for e in pool]; TM=max(1.0,(span[-1]-span[0])/86_400_000/30.0)
        cnt=defaultdict(int)
        for e in pool: cnt[e["route"]]+=1
        print(f"  havuz={len(pool)} event ({dict(cnt)}) months={TM:.2f}")

        R={}
        # P0 base long_comp only
        R["P0_long_comp_only"]=schedule([e for e in pool if e["route"]=="LONG_comp"],months=TM); ps("P0_long_comp_only",R["P0_long_comp_only"])
        # P1 naive FIFO union
        R["P1_fifo_union"]=schedule(pool,months=TM); ps("P1_fifo_union",R["P1_fifo_union"]); print("       mix:",R["P1_fifo_union"]["mix"])
        # P2 admission bars
        R["P2_admit_ge4"]=schedule(pool,admit=lambda e:e["priority"]>=4,months=TM); ps("P2_admit_ge4",R["P2_admit_ge4"]); print("       mix:",R["P2_admit_ge4"]["mix"])
        R["P2_admit_ge5"]=schedule(pool,admit=lambda e:e["priority"]>=5,months=TM); ps("P2_admit_ge5",R["P2_admit_ge5"]); print("       mix:",R["P2_admit_ge5"]["mix"])
        # P3 route priority: long_comp>=3, short always, 100k only >=5
        def rp(e):
            if e["route"]=="LONG_comp": return e["priority"]>=3
            if e["route"]=="SHORT_1317": return True
            if e["route"]=="LONG_100k": return e["priority"]>=5
            return False
        R["P3_route_priority"]=schedule(pool,admit=rp,months=TM); ps("P3_route_priority",R["P3_route_priority"]); print("       mix:",R["P3_route_priority"]["mix"])
        def rp2(e):
            if e["route"]=="LONG_comp": return e["priority"]>=4
            if e["route"]=="SHORT_1317": return True
            if e["route"]=="LONG_100k": return e["priority"]>=6
            return False
        R["P4_route_priority_strict"]=schedule(pool,admit=rp2,months=TM); ps("P4_route_priority_strict",R["P4_route_priority_strict"]); print("       mix:",R["P4_route_priority_strict"]["mix"])
        # best + conviction-weighted sizing
        print("\n  Conviction-weighted sizing (unit=priority-2):")
        for nm,adm in (("P3_route_priority",rp),("P2_admit_ge4",lambda e:e["priority"]>=4)):
            w=schedule(pool,admit=adm,weighted=True,months=TM)
            R[nm+"_weighted"]=w
            print("    %-24s flat_total=%s weighted_total=%s per_unit=%s units=%s"%(nm,w.get("total"),w.get("weighted_total"),w.get("weighted_per_unit"),w.get("units")))

    meta={"pool":len(pool),"by_route":dict(cnt),"months":round(TM,2)}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"results":R,"meta":meta},indent=2,default=str),encoding="utf-8")
    lines=[f"# S34 Master Navigator — conviction-priority scheduler","",
           f"> Route havuzu {len(pool)} event {dict(cnt)}, {TM:.1f} ay. LOOKAHEAD YOK. Tarih {datetime.now(timezone.utc):%Y-%m-%d}","",
           "| Politika | N | /ay | WR | avg | TOTAL | mdd | mc_p | mix |","|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    for k,v in R.items():
        if v.get("n",0)>0 and "wr" in v:
            lines.append("| %s | %d | %.1f | %.1f%% | %+.0f | %+.0f | %+.0f | %s | %s |"%(k,v["n"],v.get("per_month",0),v["wr"],v["avg"],v.get("total"),v.get("mdd"),v.get("mc_p","?"),v.get("mix","")))
    lines+=["","**Conviction-weighted sizing:**"]
    for k,v in R.items():
        if v.get("weighted_total") is not None:
            lines.append("- %s: flat=%s weighted=%s per_unit=%s"%(k,v.get("total"),v.get("weighted_total"),v.get("weighted_per_unit")))
    lines+=["","---","*Script: tools/research_s34_master_navigator.py*"]
    OM.write_text("\n".join(lines),encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")

if __name__=="__main__": main()
