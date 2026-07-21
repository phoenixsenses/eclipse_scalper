"""S34 SHORT Conviction Gauntlet — SHORT_NOISY diversifier'a conviction skoru + yeni testler.

Q4: SHORT'un kendi sinyalleri = btc_size (dominant, delta+155), be_ratio(+56), sync_ratio(+50).
Baz: ETH SELL 200K -> noisy follow-on(1-30m) -> BTC SELL confirm -> SHORT at noisy_ts.

Sections:
  S1 BTC-esik x hold sweep (500K/1M/2M x 90/120/180/240m)
  S2 SHORT conviction skoru (btc_size+be_ratio+sync_ratio) monoton? gate+holdout+noov+MC
  S3 SHORT saat etkisi (hour matter mi?)
  S4 SHORT funding timing (<60m veto LONG'da; short'ta?)
  S5 Entry timing (noisy_ts vs BTC-confirm-ts) + delay sweep
  S6 liq-shelf-above (short yakit = ustte kumelenmis likidasyon)
  S7 Portfoy: LONG composite + SHORT composite non-overlap

Cikti: reports/research/s34/S34_SHORT_CONVICTION.json + .md
"""
from __future__ import annotations
import bisect, json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB=ROOT/"data"/"microstructure.db"; OUT=ROOT/"reports"/"research"/"s34"
OJ=OUT/"S34_SHORT_CONVICTION.json"; OM=OUT/"S34_SHORT_CONVICTION.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=500; TM=4.5; TRAIN=0.70
# LONG composite thresholds (for portfolio leg)
CT={"sync_ratio":0.5421,"rv5m":0.0304,"density24":5.0,"be_lo":0.2195,"be_hi":2.0,"imb":0.2633,"shelf":2_775_000.0}
random.seed(42)

def _s(c,q,p=()):
    r=c.execute(q,p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0
def lsum(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def lmax(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def lcnt(c,s,sd,lo,hi,t): return int(_s(c,"SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",(s,sd,lo,hi,t)))
def lfirst(c,s,sd,lo,hi,t):
    r=c.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1",(s,sd,lo,hi,t)).fetchone()
    return int(r[0]) if r else None
def lfirst_above(c,s,sd,lo,hi,t):
    r=c.execute("SELECT ts_ms,notional FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1",(s,sd,lo,hi,t)).fetchone()
    return (int(r[0]),float(r[1])) if r else None
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
def ofi(c,s,lo,hi):
    r=c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<?",(s,lo,hi)).fetchone()
    if not r or r[0] is None: return None
    b,se=float(r[0]),float(r[1]); t=b+se; return (b-se)/t if t>0 else 0.0
def nextfund(c,ts):
    r=c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return int(r[0]) if r and r[0] else None
def hod(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).hour
def se(ts):
    h=hod(ts); return "EUROPE" if 7<=h<13 else ("US" if 13<=h<21 else "OFF")
def eprice(m,ts):
    r=m.at_or_after(ts); return (int(r[0]),float(r[1])) if r and float(r[1])>0 else None
def shortret(m,ts,hold):
    e=eprice(m,ts)
    if not e: return None
    r=m.at_or_before(ts+hold); return -(float(r[1])-e[1])/e[1]*1e4 if r else None
def longret(m,ts,hold):
    e=eprice(m,ts)
    if not e: return None
    r=m.at_or_before(ts+hold); return (float(r[1])-e[1])/e[1]*1e4 if r else None

def mcp(v,a):
    if len(v)<4: return None
    r=random.Random(0); ct=sum(1 for _ in range(MC) if sum(r.choice([-1,1])*abs(x) for x in v)/len(v)>=a); return round(ct/MC,3)
def wf(v,k=5):
    n=len(v); return "%d/%d"%(sum(1 for i in range(k) if sum(v[i*n//k:(i+1)*n//k])>0),k) if n>=k else None
def stat(g,label="",months=None,fee=FEE):
    m=months or TM
    if not g: return {"label":label,"n":0}
    net=[x-fee for x in g]; n=len(net); w=sum(1 for x in net if x>0); sv=sorted(net); a=sum(net)/n
    cum=pk=mdd=0.0
    for x in net: cum+=x; pk=max(pk,cum); mdd=min(mdd,cum-pk)
    return {"label":label,"n":n,"wr":round(100*w/n,1),"avg":round(a,1),"total":round(sum(net),0),
            "per_month":round(n/m,1),"worst":round(sv[0],1),"mdd":round(mdd,0),"mc_p":mcp(net,a),"wf":wf(net)}
def ps(k,v):
    if not v or v.get("n",0)==0: print("    %-36s N=0"%k[:36]); return
    print("    %-36s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s mc_p=%s wf=%s"%(k[:36],v["n"],v.get("per_month",0),str(v["wr"])+"%",str(v["avg"])+"bps",str(v.get("total")),v.get("mc_p","?"),v.get("wf")))
def med(x): s=sorted(x); return s[len(s)//2] if s else 0
def noov(pairs,hold):
    busy=-1;o=[]
    for ts,v in sorted(pairs):
        if ts>=busy: o.append(v);busy=ts+hold
    return o

# ---- build SHORT events (BTC>=500K to get N; filter later) ----
def build_short(conn,m,now,start):
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    ev=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or se(ts)=="EUROPE": continue
        nt=lfirst(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP)
        if nt is None: continue
        # BTC confirm: first BTC SELL >=500K after noisy+5m within 30m
        conf=lfirst_above(conn,"BTCUSDT","SELL",nt+5*60_000,nt+30*60_000,500_000.0)
        if conf is None: continue
        conf_ts,_=conf
        btc_max=lmax(conn,"BTCUSDT","SELL",nt+5*60_000,nt+30*60_000)
        f={"btc_size":btc_max,
           "be_ratio":btc_max/rn if rn>0 else 0,
           "sync_ratio":(lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts))/rn if rn>0 else 0,
           "rv5m":rv5(conn,nt),"imb":book_imb(conn,nt),"ofi":ofi(conn,"ETHUSDT",nt,nt+5*60_000),
           "hour":hod(nt),"conf_delay_min":(conf_ts-nt)/60_000}
        nf=nextfund(conn,nt); f["min_to_fund"]=((nf-nt)/60_000) if nf else None
        # shelf ABOVE price (short fuel): 24h ETH SELL notional in [price, price*1.02]
        ep=eprice(m,nt)
        f["shelf_above"]=_s(conn,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?",(nt-24*3600_000,nt,ep[1],ep[1]*1.02)) if ep else 0.0
        ev.append({"anchor_ts":ts,"ts":nt,"conf_ts":conf_ts,"rn":rn,"f":f,"btc_max":btc_max})
    ev.sort(key=lambda e:e["ts"])
    return ev

def build_long_composite(conn,m,now,start):
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    ev=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or se(ts)=="EUROPE" or not(b4<0 or b7<0) or hod(ts)<17: continue
        sk=lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
        ep=eprice(m,ts); shelf=_s(conn,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?",(ts-24*3600_000,ts,ep[1]*0.98,ep[1])) if ep else 0
        sync_ratio=sk/rn if rn>0 else 0; rv=rv5(conn,ts); d24=lcnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,200_000)
        of=ofi(conn,"ETHUSDT",ts-5*60_000,ts); be=lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts)/rn if rn>0 else 0
        imb=book_imb(conn,ts); nf=nextfund(conn,ts); m2f=((nf-ts)/60_000) if nf else None
        sc=(int(sync_ratio>=CT["sync_ratio"])+int(rv is not None and rv>=CT["rv5m"])+int(d24>=CT["density24"])
            +int(of is not None and of>=0)+int(CT["be_lo"]<=be<CT["be_hi"])+int(imb is not None and imb<=CT["imb"])+int(shelf>=CT["shelf"]))
        veto=(m2f is not None and m2f<60)
        y=longret(m,ts,6*3600_000)
        if y is None: continue
        ev.append({"ts":ts,"score":sc,"veto":veto,"y":y})
    ev.sort(key=lambda e:e["ts"])
    return ev

def run_S1(m,ev):
    print("\n=== S1: BTC-esik x hold sweep ===")
    R={}
    for thr in (500_000,1_000_000,2_000_000):
        for hold_m in (90,120,180,240):
            sub=[e for e in ev if e["btc_max"]>=thr]
            g=[shortret(m,e["ts"],hold_m*60_000) for e in sub]; g=[x for x in g if x is not None]
            k=f"S1_btc{thr//1000}K_h{hold_m}"; R[k]=stat(g,k,TM)
            if R[k].get("n",0)>=8: ps(k,R[k])
    return R

def run_S2(m,ev):
    print("\n=== S2: SHORT conviction skoru (btc_size+be_ratio+sync_ratio) ===")
    R={}
    sub=[e for e in ev if e["btc_max"]>=1_000_000]
    for e in sub: e["y"]=shortret(m,e["ts"],180*60_000)
    sub=[e for e in sub if e["y"] is not None]
    n=len(sub); cut=int(n*TRAIN); tr=sub[:cut]; te=sub[cut:]
    mb=med([e["f"]["btc_size"] for e in tr]); mbe=med([e["f"]["be_ratio"] for e in tr]); msy=med([e["f"]["sync_ratio"] for e in tr])
    def sc(e): return int(e["f"]["btc_size"]>=mb)+int(e["f"]["be_ratio"]>=mbe)+int(e["f"]["sync_ratio"]>=msy)
    for e in sub: e["sscore"]=sc(e)
    print(f"  N={n} train={cut} test={n-cut}")
    for s in range(0,4):
        g=[e["y"] for e in sub if e["sscore"]==s]; R[f"S2_score{s}"]=stat(g,f"score={s}",TM); ps(f"S2_score{s}",R[f"S2_score{s}"])
    for K in (2,3):
        full=[(e["ts"],e["y"]) for e in sub if e["sscore"]>=K]
        R[f"S2_gate{K}_full"]=stat([v for _,v in full],f"score>={K} full",TM); ps(f"S2_gate{K}_full",R[f"S2_gate{K}_full"])
        teg=[e["y"] for e in te if e["sscore"]>=K]; R[f"S2_gate{K}_TEST"]=stat(teg,f"score>={K} TEST",TM*(1-TRAIN)); ps(f"S2_gate{K}_TEST",R[f"S2_gate{K}_TEST"])
        nv=noov(full,180*60_000); s2=stat(nv,f"score>={K} noov",TM); s2["per_month"]=round(len(nv)/TM,1); R[f"S2_gate{K}_noov"]=s2; ps(f"S2_gate{K}_noov",s2)
    return R, sub

def run_S3(m,ev):
    print("\n=== S3: SHORT saat etkisi ===")
    R={}
    sub=[e for e in ev if e["btc_max"]>=1_000_000]
    for e in sub: e["y"]=shortret(m,e["ts"],180*60_000)
    sub=[e for e in sub if e["y"] is not None]
    for lbl,cond in (("h0-8",lambda h:0<=h<8),("h13-17",lambda h:13<=h<17),("h17-24",lambda h:h>=17),("US13-21",lambda h:13<=h<21)):
        g=[e["y"] for e in sub if cond(e["f"]["hour"])]; R[f"S3_{lbl}"]=stat(g,lbl,TM); ps(f"S3_{lbl}",R[f"S3_{lbl}"])
    return R

def run_S4(m,ev):
    print("\n=== S4: SHORT funding timing ===")
    R={}
    sub=[e for e in ev if e["btc_max"]>=1_000_000 and e["f"].get("min_to_fund") is not None]
    for e in sub: e["y"]=shortret(m,e["ts"],180*60_000)
    sub=[e for e in sub if e["y"] is not None]
    for lbl,cond in (("<60m",lambda t:t<60),("60-240m",lambda t:60<=t<240),(">240m",lambda t:t>=240)):
        g=[e["y"] for e in sub if cond(e["f"]["min_to_fund"])]; R[f"S4_{lbl}"]=stat(g,lbl,TM); ps(f"S4_{lbl}",R[f"S4_{lbl}"])
    return R

def run_S5(m,ev):
    print("\n=== S5: Entry timing (noisy vs BTC-confirm) + delay ===")
    R={}
    sub=[e for e in ev if e["btc_max"]>=1_000_000]
    gn=[shortret(m,e["ts"],180*60_000) for e in sub]; gn=[x for x in gn if x is not None]
    gc=[shortret(m,e["conf_ts"],180*60_000) for e in sub]; gc=[x for x in gc if x is not None]
    R["S5_entry_noisy"]=stat(gn,"entry @noisy_ts",TM); R["S5_entry_confirm"]=stat(gc,"entry @BTC-confirm",TM)
    ps("S5_entry_noisy",R["S5_entry_noisy"]); ps("S5_entry_confirm",R["S5_entry_confirm"])
    for lbl,cond in (("delay<10m",lambda d:d<10),("delay>=10m",lambda d:d>=10)):
        g=[shortret(m,e["ts"],180*60_000) for e in sub if cond(e["f"]["conf_delay_min"])]; g=[x for x in g if x is not None]
        R[f"S5_{lbl}"]=stat(g,lbl,TM); ps(f"S5_{lbl}",R[f"S5_{lbl}"])
    return R

def run_S6(m,ev):
    print("\n=== S6: liq-shelf-above (short yakit) ===")
    R={}
    sub=[e for e in ev if e["btc_max"]>=1_000_000]
    for e in sub: e["y"]=shortret(m,e["ts"],180*60_000)
    sub=[(e["f"]["shelf_above"],e["y"]) for e in sub if e["y"] is not None and e["f"].get("shelf_above") is not None]
    mm=med([v for v,_ in sub]); hi=[y for v,y in sub if v>=mm]; lo=[y for v,y in sub if v<mm]
    R["S6_shelf_above_hi"]=stat(hi,"shelf-above hi",TM); R["S6_shelf_above_lo"]=stat(lo,"shelf-above lo",TM)
    ps("S6_shelf_above_hi",R["S6_shelf_above_hi"]); ps("S6_shelf_above_lo",R["S6_shelf_above_lo"])
    return R

def run_S7(m,ev,longev):
    print("\n=== S7: Portfoy LONG-composite + SHORT-composite (non-overlap) ===")
    R={}
    # SHORT leg: score>=2 (from S2), BTC>=1M, hold 180m, entry noisy_ts
    sub=[e for e in ev if e["btc_max"]>=1_000_000]
    n=len(sub); cut=int(n*TRAIN); tr=sub[:cut]
    mb=med([e["f"]["btc_size"] for e in tr]); mbe=med([e["f"]["be_ratio"] for e in tr]); msy=med([e["f"]["sync_ratio"] for e in tr])
    shorts=[]
    for e in sub:
        y=shortret(m,e["ts"],180*60_000)
        if y is None: continue
        s=int(e["f"]["btc_size"]>=mb)+int(e["f"]["be_ratio"]>=mbe)+int(e["f"]["sync_ratio"]>=msy)
        if s>=2 and not (e["f"].get("min_to_fund") is not None and e["f"]["min_to_fund"]<60):
            shorts.append((e["ts"],y,180*60_000,"S"))
    # LONG leg: composite score>=3, no funding veto, hold 6h
    longs=[(e["ts"],e["y"],6*3600_000,"L") for e in longev if e["score"]>=3 and not e["veto"]]
    R["S7_long_only"]=stat([v for _,v,_,_ in longs],"LONG composite only",TM); ps("S7_long_only",R["S7_long_only"])
    R["S7_short_only"]=stat([v for _,v,_,_ in shorts],"SHORT composite only",TM); ps("S7_short_only",R["S7_short_only"])
    allp=sorted(longs+shorts); busy=-1; combo=[]
    for ts,v,hold,_ in allp:
        if ts>=busy: combo.append(v); busy=ts+hold
    s=stat(combo,"LONG+SHORT portfolio",TM); s["per_month"]=round(len(combo)/TM,1)
    R["S7_portfolio"]=s; ps("S7_portfolio",s)
    return R

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 SHORT Conviction Gauntlet ===")
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        m=load_mark_index(conn,"ETHUSDT")
        print("build SHORT events..."); ev=build_short(conn,m,now,start)
        span=[e["ts"] for e in ev]; TM=max(1.0,(span[-1]-span[0])/86_400_000/30.0) if len(span)>1 else 4.5
        print(f"  SHORT events (BTC>=500K)={len(ev)} months={TM:.2f}")
        print("build LONG composite (portfolio leg)..."); longev=build_long_composite(conn,m,now,start)
        print(f"  LONG composite events={len(longev)}")
        R={}
        R["S1"]=run_S1(m,ev); (r2,_)=run_S2(m,ev); R["S2"]=r2
        R["S3"]=run_S3(m,ev); R["S4"]=run_S4(m,ev); R["S5"]=run_S5(m,ev)
        R["S6"]=run_S6(m,ev); R["S7"]=run_S7(m,ev,longev)
    meta={"n_short":len(ev),"n_long":len(longev),"months":round(TM,2)}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"results":R,"meta":meta},indent=2,default=str),encoding="utf-8")
    lines=[f"# S34 SHORT Conviction Gauntlet","",f"> SHORT_NOISY {len(ev)} event (BTC>=500K), {TM:.1f} ay, FEE={int(FEE)}. Tarih {datetime.now(timezone.utc):%Y-%m-%d}",""]
    for q,sec in R.items():
        lines+=[f"## {q}",""]
        for k,v in sec.items():
            if isinstance(v,dict) and v.get("n",0)>0 and "wr" in v:
                lines.append("- **%s**: N=%d /ay=%.1f WR=%.1f%% avg=%+.1f TOT=%s mc_p=%s wf=%s"%(k,v["n"],v.get("per_month",0),v["wr"],v["avg"],v.get("total"),v.get("mc_p","?"),v.get("wf","-")))
        lines.append("")
    lines+=["---","*Script: tools/research_s34_short_conviction.py*"]
    OM.write_text("\n".join(lines),encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")

if __name__=="__main__": main()
