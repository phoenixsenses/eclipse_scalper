"""S34 Frontier Gauntlet — prediction / navigation / in-data / signal-search / meta.

hour17 200K composite baz. LOOKAHEAD YASAK. no-overlap + MC.
P prediction: P1 100K-erken giris composite (mini->big)
N navigation: N1 state grid (hour x regime x score), N2 sequence (winner->next)
D in-data:    D1 storm-day, D2 cross-asset simultane, D3 multi-horizon
S signal:     S1 whale trade size, S2 time-since-last-cascade
M meta:       M1 regime x score, M2 threshold sweep

Cikti: reports/research/s34/S34_FRONTIER.json + .md
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
OJ=OUT/"S34_FRONTIER.json"; OM=OUT/"S34_FRONTIER.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=500; HOLD=6*3600_000; TM=4.5
CT={"sync_ratio":0.5421,"rv5m":0.0304,"density24":5.0,"be_lo":0.2195,"be_hi":2.0,"imb":0.2633,"shelf":2_775_000.0}
random.seed(42)

def _s(c,q,p=()):
    r=c.execute(q,p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0
def lsum(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def lmax(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def lcnt(c,s,sd,lo,hi,t): return int(_s(c,"SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",(s,sd,lo,hi,t)))
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
def ofi(c,lo,hi):
    r=c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?",(lo,hi)).fetchone()
    if not r or r[0] is None: return None,None
    b,se=float(r[0]),float(r[1]); t=b+se; n=int(r[2])
    return ((b-se)/t if t>0 else 0.0),(t/n if n>0 else None)
def nextfund(c,ts):
    r=c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return int(r[0]) if r and r[0] else None
def hod(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).hour
def day(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).strftime("%Y-%m-%d")
def se(ts):
    h=hod(ts); return "EUROPE" if 7<=h<13 else ("US" if 13<=h<21 else "OFF")
def ep(m,ts):
    r=m.at_or_after(ts); return (int(r[0]),float(r[1])) if r and float(r[1])>0 else None
def lret(m,ts,hold):
    e=ep(m,ts)
    if not e: return None
    r=m.at_or_before(ts+hold); return (float(r[1])-e[1])/e[1]*1e4 if r else None

def mcp(v,a):
    if len(v)<4: return None
    r=random.Random(0); ct=sum(1 for _ in range(MC) if sum(r.choice([-1,1])*abs(x) for x in v)/len(v)>=a); return round(ct/MC,3)
def stat(g,label="",months=None,fee=FEE):
    m=months or TM
    if not g: return {"label":label,"n":0}
    net=[x-fee for x in g]; n=len(net); w=sum(1 for x in net if x>0); sv=sorted(net); a=sum(net)/n
    return {"label":label,"n":n,"wr":round(100*w/n,1),"avg":round(a,1),"total":round(sum(net),0),"per_month":round(n/m,1),"worst":round(sv[0],1),"mc_p":mcp(net,a)}
def ps(k,v):
    if not v or v.get("n",0)==0: print("    %-38s N=0"%k[:38]); return
    print("    %-38s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s mc_p=%s"%(k[:38],v["n"],v.get("per_month",0),str(v["wr"])+"%",str(v["avg"])+"bps",str(v.get("total")),v.get("mc_p","?")))
def med(x): s=sorted(x); return s[len(s)//2] if s else 0
def noov(pairs,hold=HOLD):
    busy=-1;o=[]
    for ts,v in sorted(pairs):
        if ts>=busy: o.append(v);busy=ts+hold
    return o

def composite_score(conn,m,ts,rn):
    sk=lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
    sync=sk/rn if rn>0 else 0; rv=rv5(conn,ts); d24=lcnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,200_000)
    of,whale=ofi(conn,ts-5*60_000,ts); be=lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts)/rn if rn>0 else 0
    imb=book_imb(conn,ts); e=ep(m,ts)
    shelf=_s(conn,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?",(ts-24*3600_000,ts,e[1]*0.98,e[1])) if e else 0
    sc=(int(sync>=CT["sync_ratio"])+int(rv is not None and rv>=CT["rv5m"])+int(d24>=CT["density24"])
        +int(of is not None and of>=0)+int(CT["be_lo"]<=be<CT["be_hi"])+int(imb is not None and imb<=CT["imb"])+int(shelf>=CT["shelf"]))
    return sc,{"sync":sync,"rv":rv,"d24":d24,"whale":whale,"be":be,"shelf":shelf}

def build(conn,m,now,start):
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    all_ts=sorted(int(a.anchor_ts_ms) for a in ancs)
    ev=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or se(ts)=="EUROPE" or not(b4<0 or b7<0) or hod(ts)<17: continue
        sc,comp=composite_score(conn,m,ts,rn)
        nf=nextfund(conn,ts); m2f=((nf-ts)/60_000) if nf else None
        # time since last cascade
        i=bisect.bisect_left(all_ts,ts); tsl=((ts-all_ts[i-1])/60_000) if i>0 else None
        # storm: same-day cascade count
        # cross-asset simultaneity
        btc_conc=lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts+10*60_000); sol_conc=lmax(conn,"SOLUSDT","SELL",ts-10*60_000,ts+10*60_000)
        e={"ts":ts,"rn":rn,"score":sc,"comp":comp,"b7":b7,"hour":hod(ts),"day":day(ts),
           "m2f":m2f,"tsl":tsl,"btc_conc":btc_conc,"sol_conc":sol_conc,
           "veto":(m2f is not None and m2f<60)}
        e["y2"]=lret(m,ts,2*3600_000); e["y4"]=lret(m,ts,4*3600_000); e["y6"]=lret(m,ts,HOLD)
        e["y8"]=lret(m,ts,8*3600_000); e["y12"]=lret(m,ts,12*3600_000)
        if e["y6"] is None: continue
        ev.append(e)
    ev.sort(key=lambda x:x["ts"])
    dc=defaultdict(int)
    for e in ev: dc[e["day"]]+=1
    for e in ev: e["day_count"]=dc[e["day"]]
    return ev,all_ts

# P1: 100K early entry
def run_P(conn,m,now,start):
    print("\n=== P1: 100K erken giris (mini->big) composite ===")
    R={}
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(100_000.0,),accel_window_sec=30)
    big_ts=sorted(int(a.anchor_ts_ms) for a in reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30))
    ev=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<100_000 or rn>=200_000 or m.at_or_after(ts) is None: continue  # 100-200K mini only
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or se(ts)=="EUROPE" or not(b4<0 or b7<0) or hod(ts)<17: continue
        # did a 200K follow within 30m?
        i=bisect.bisect_left(big_ts,ts); grew=(i<len(big_ts) and big_ts[i]-ts<=30*60_000)
        sc,_=composite_score(conn,m,ts,rn)
        y=lret(m,ts,HOLD)
        if y is None: continue
        ev.append({"ts":ts,"score":sc,"grew":grew,"y":y})
    R["P1_mini_all"]=stat([e["y"] for e in ev],"100K mini all (hour17)",TM); ps("P1_mini_all",R["P1_mini_all"])
    R["P1_mini_grew"]=stat([e["y"] for e in ev if e["grew"]],"100K mini that grew to 200K",TM); ps("P1_mini_grew",R["P1_mini_grew"])
    R["P1_mini_fizzled"]=stat([e["y"] for e in ev if not e["grew"]],"100K mini fizzled",TM); ps("P1_mini_fizzled",R["P1_mini_fizzled"])
    R["P1_mini_score3"]=stat([e["y"] for e in ev if e["score"]>=3],"100K mini score>=3",TM); ps("P1_mini_score3",R["P1_mini_score3"])
    print("  grow-rate: %d/%d (%.0f%%)"%(sum(1 for e in ev if e['grew']),len(ev),100*sum(1 for e in ev if e['grew'])/max(1,len(ev))))
    return R

def run_N(ev):
    print("\n=== N: NAVIGATION ===")
    R={}
    print("  N1 state grid (hour x regime x score):")
    for hl,hc in (("h17-19",lambda e:e["hour"]<20),("h20-23",lambda e:e["hour"]>=20)):
        for rl,rc in (("deep7d",lambda e:e["b7"]<-300),("mild7d",lambda e:e["b7"]>=-300)):
            for sl,sc in (("s<4",lambda e:e["score"]<4),("s>=4",lambda e:e["score"]>=4)):
                g=[e["y6"] for e in ev if hc(e) and rc(e) and sc(e)]
                k=f"N1_{hl}_{rl}_{sl}"; R[k]=stat(g,k,TM)
                if R[k].get("n",0)>=5: ps(k,R[k])
    print("  N2 sequence (kazanan/kaybeden sonrasi):")
    aw=[]; al=[]
    for i in range(1,len(ev)):
        prev=ev[i-1]["y6"]-FEE; cur=ev[i]["y6"]
        (aw if prev>0 else al).append(cur)
    R["N2_after_win"]=stat(aw,"after winner",TM); R["N2_after_loss"]=stat(al,"after loser",TM)
    ps("N2_after_win",R["N2_after_win"]); ps("N2_after_loss",R["N2_after_loss"])
    return R

def run_D(conn,ev):
    print("\n=== D: IN-DATA ===")
    R={}
    print("  D1 storm-day (day cascade count):")
    for lbl,cond in (("isolated(1-2/day)",lambda e:e["day_count"]<=2),("busy(3-4)",lambda e:3<=e["day_count"]<=4),("storm(5+)",lambda e:e["day_count"]>=5)):
        g=[e["y6"] for e in ev if cond(e)]; R[f"D1_{lbl}"]=stat(g,lbl,TM); ps(f"D1_{lbl}",R[f"D1_{lbl}"])
    print("  D2 cross-asset simultane:")
    for lbl,cond in (("ETH+BTC(>=500K)",lambda e:e["btc_conc"]>=500_000),("ETH-only(btc<500K)",lambda e:e["btc_conc"]<500_000),("ETH+SOL(>=100K)",lambda e:e["sol_conc"]>=100_000)):
        g=[e["y6"] for e in ev if cond(e)]; R[f"D2_{lbl}"]=stat(g,lbl,TM); ps(f"D2_{lbl}",R[f"D2_{lbl}"])
    print("  D3 multi-horizon (score>=3):")
    for hk,hl in (("y2","2h"),("y4","4h"),("y6","6h"),("y8","8h"),("y12","12h")):
        g=[e[hk] for e in ev if e["score"]>=3 and e.get(hk) is not None]
        R[f"D3_{hl}"]=stat(g,f"score>=3 {hl}",TM); ps(f"D3_{hl}",R[f"D3_{hl}"])
    return R

def run_S(ev):
    print("\n=== S: SIGNAL SEARCH ===")
    R={}
    print("  S1 whale (pre-5m avg trade size) hi/lo:")
    vals=[(e["comp"].get("whale"),e["y6"]) for e in ev if e["comp"].get("whale") is not None]
    if vals:
        mm=med([v for v,_ in vals]); hi=[y for v,y in vals if v>=mm]; lo=[y for v,y in vals if v<mm]
        R["S1_whale_hi"]=stat(hi,"whale hi",TM); R["S1_whale_lo"]=stat(lo,"whale lo",TM)
        ps("S1_whale_hi",R["S1_whale_hi"]); ps("S1_whale_lo",R["S1_whale_lo"])
    print("  S2 time-since-last-cascade:")
    for lbl,cond in (("<2h",lambda t:t<120),("2-12h",lambda t:120<=t<720),(">12h",lambda t:t>=720)):
        g=[e["y6"] for e in ev if e.get("tsl") is not None and cond(e["tsl"])]
        R[f"S2_tsl_{lbl}"]=stat(g,lbl,TM); ps(f"S2_tsl_{lbl}",R[f"S2_tsl_{lbl}"])
    return R

def run_M(ev):
    print("\n=== M: META ===")
    R={}
    print("  M1 regime x score:")
    for rl,rc in (("deep7d",lambda e:e["b7"]<-300),("mild7d",lambda e:e["b7"]>=-300)):
        for sl,sc in (("s>=3",lambda e:e["score"]>=3),("s>=4",lambda e:e["score"]>=4)):
            g=[e["y6"] for e in ev if rc(e) and sc(e)]; k=f"M1_{rl}_{sl}"; R[k]=stat(g,k,TM); ps(k,R[k])
    print("  M2 threshold sweep (full + no-overlap):")
    for K in (2,3,4,5):
        full=[(e["ts"],e["y6"]) for e in ev if e["score"]>=K and not e["veto"]]
        R[f"M2_s{K}_full"]=stat([v for _,v in full],f"score>={K} (no-veto)",TM)
        nv=noov(full); s=stat(nv,f"score>={K} noov",TM); s["per_month"]=round(len(nv)/TM,1); R[f"M2_s{K}_noov"]=s
        ps(f"M2_s{K}_full",R[f"M2_s{K}_full"]); ps(f"M2_s{K}_noov",s)
    return R

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 Frontier Gauntlet ===")
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        m=load_mark_index(conn,"ETHUSDT")
        print("build..."); ev,_=build(conn,m,now,start)
        span=[e["ts"] for e in ev]; TM=max(1.0,(span[-1]-span[0])/86_400_000/30.0)
        print(f"  events={len(ev)} months={TM:.2f}")
        R={}
        R["P"]=run_P(conn,m,now,start); R["N"]=run_N(ev); R["D"]=run_D(conn,ev)
        R["S"]=run_S(ev); R["M"]=run_M(ev)
    meta={"n":len(ev),"months":round(TM,2)}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"results":R,"meta":meta},indent=2,default=str),encoding="utf-8")
    lines=[f"# S34 Frontier Gauntlet","",f"> hour17 200K composite {len(ev)} event {TM:.1f} ay. Tarih {datetime.now(timezone.utc):%Y-%m-%d}",""]
    for q,sec in R.items():
        lines+=[f"## {q}",""]
        for k,v in sec.items():
            if isinstance(v,dict) and v.get("n",0)>0 and "wr" in v:
                lines.append("- **%s**: N=%d /ay=%.1f WR=%.1f%% avg=%+.1f TOT=%s mc_p=%s"%(k,v["n"],v.get("per_month",0),v["wr"],v["avg"],v.get("total"),v.get("mc_p","?")))
        lines.append("")
    lines+=["---","*Script: tools/research_s34_frontier_gauntlet.py*"]
    OM.write_text("\n".join(lines),encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")

if __name__=="__main__": main()
