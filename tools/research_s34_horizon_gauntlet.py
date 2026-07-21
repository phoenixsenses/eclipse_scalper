"""S34 Horizon Gauntlet — v3 aday dogrulama + ufuk genisletme.

V  whale_lo + 100K holdout dogrulama (v3'e girmeye deger mi)
I  feature interaction (ikili: her ikisi favorable = super sinyal mi)
Z  meta-veto (composite score>=3 KAYBEDENLERINI ongoren T0 sinyali)
H  swing horizon (6/12/24/48h)
R  block-bootstrap robustluk (otokorelasyon-aware CI)
T  time-of-day machine (LONG 17-23 + SHORT 13-17 tam-gun portfoy)

hour17 200K composite baz. FEE=5. no-overlap + holdout + MC.
Cikti: reports/research/s34/S34_HORIZON.json + .md
"""
from __future__ import annotations
import bisect, json, random, sqlite3, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB=ROOT/"data"/"microstructure.db"; OUT=ROOT/"reports"/"research"/"s34"
OJ=OUT/"S34_HORIZON.json"; OM=OUT/"S34_HORIZON.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=500; HOLD=6*3600_000; TM=4.5; TRAIN=0.70
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
def lfirst_above(c,s,sd,lo,hi,t):
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
def stat(g,label="",months=None,fee=FEE):
    m=months or TM
    if not g: return {"label":label,"n":0}
    net=[x-fee for x in g]; n=len(net); w=sum(1 for x in net if x>0); a=sum(net)/n
    return {"label":label,"n":n,"wr":round(100*w/n,1),"avg":round(a,1),"total":round(sum(net),0),"per_month":round(n/m,1),"mc_p":mcp(net,a)}
def ps(k,v):
    if not v or v.get("n",0)==0: print("    %-36s N=0"%k[:36]); return
    print("    %-36s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s mc_p=%s"%(k[:36],v["n"],v.get("per_month",0),str(v["wr"])+"%",str(v["avg"])+"bps",str(v.get("total")),v.get("mc_p","?")))
def med(x): s=sorted(x); return s[len(s)//2] if s else 0
def noov(pairs,hold=HOLD):
    busy=-1;o=[]
    for ts,v in sorted(pairs):
        if ts>=busy: o.append(v);busy=ts+hold
    return o

def feats(conn,m,ts,rn):
    sk=lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
    of,whale=ofir(conn,ts-5*60_000,ts); e=ep(m,ts)
    shelf=_s(conn,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?",(ts-24*3600_000,ts,e[1]*0.98,e[1])) if e else 0
    return {"sync":sk/rn if rn>0 else 0,"rv":rv5(conn,ts),"d24":lcnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,200_000),
            "ofi":of,"be":lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts)/rn if rn>0 else 0,"imb":book_imb(conn,ts),
            "shelf":shelf,"whale":whale}
def hits(f):
    return {"sync":f["sync"]>=CT["sync"],"rv":f["rv"] is not None and f["rv"]>=CT["rv"],
            "d24":f["d24"]>=CT["d24"],"ofi":f["ofi"] is not None and f["ofi"]>=0,
            "be":CT["be_lo"]<=f["be"]<CT["be_hi"],"imb":f["imb"] is not None and f["imb"]<=CT["imb"],
            "shelf":f["shelf"]>=CT["shelf"],"whale_lo":f["whale"] is not None and f["whale"]<CT["whale"]}
def score7(f): h=hits(f); return sum(1 for k in ("sync","rv","d24","ofi","be","imb","shelf") if h[k])
def score8(f): return score7(f)+(1 if hits(f)["whale_lo"] else 0)

def build(conn,m,now,start,hour_gate=17):
    ancs=reconstruct_anchors(load_liquidations(conn,"ETHUSDT","SELL",start,now),bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    ev=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sxn(ts)=="EUROPE" or not(b4<0 or b7<0) or hod(ts)<hour_gate: continue
        f=feats(conn,m,ts,rn); nf=nextfund(conn,ts); m2f=((nf-ts)/60_000) if nf else None
        e={"ts":ts,"rn":rn,"f":f,"s7":score7(f),"s8":score8(f),"b7":b7,"hour":hod(ts),"veto":(m2f is not None and m2f<60)}
        e["y6"]=lret(m,ts,HOLD); e["y12"]=lret(m,ts,12*3600_000); e["y24"]=lret(m,ts,24*3600_000); e["y48"]=lret(m,ts,48*3600_000)
        if e["y6"] is None: continue
        ev.append(e)
    ev.sort(key=lambda x:x["ts"])
    return ev

def run_V(conn,m,ev,now,start):
    print("\n=== V: whale + 100K holdout dogrulama ===")
    R={}
    n=len(ev); cut=int(n*TRAIN); te=ev[cut:]
    # whale as filter (holdout: threshold fixed CT, just report TEST)
    R["V_whale_lo_TEST"]=stat([e["y6"] for e in te if hits(e["f"])["whale_lo"]],"whale_lo TEST",TM*(1-TRAIN)); ps("V_whale_lo_TEST",R["V_whale_lo_TEST"])
    R["V_whale_hi_TEST"]=stat([e["y6"] for e in te if not hits(e["f"])["whale_lo"]],"whale_hi TEST",TM*(1-TRAIN)); ps("V_whale_hi_TEST",R["V_whale_hi_TEST"])
    # score8 vs score7 gate>=3/4 (full + TEST)
    for lbl,sf in (("s7",lambda e:e["s7"]),("s8",lambda e:e["s8"])):
        for K in (3,4):
            R[f"V_{lbl}_ge{K}_full"]=stat([e["y6"] for e in ev if sf(e)>=K and not e["veto"]],f"{lbl}>={K} full",TM)
            R[f"V_{lbl}_ge{K}_TEST"]=stat([e["y6"] for e in te if sf(e)>=K and not e["veto"]],f"{lbl}>={K} TEST",TM*(1-TRAIN))
            ps(f"V_{lbl}_ge{K}_full",R[f"V_{lbl}_ge{K}_full"]); ps(f"V_{lbl}_ge{K}_TEST",R[f"V_{lbl}_ge{K}_TEST"])
    # 100K composite (build separate)
    ancs=reconstruct_anchors(load_liquidations(conn,"ETHUSDT","SELL",start,now),bucket_sec=300,min_gap_sec=900,thresholds=(100_000.0,),accel_window_sec=30)
    ev100=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<100_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sxn(ts)=="EUROPE" or not(b4<0 or b7<0) or hod(ts)<17: continue
        f=feats(conn,m,ts,rn); y=lret(m,ts,HOLD)
        if y is None: continue
        ev100.append({"ts":ts,"s7":score7(f),"y":y})
    ev100.sort(key=lambda x:x["ts"]); n2=len(ev100); c2=int(n2*TRAIN); te2=ev100[c2:]
    R["V_100K_s3_full"]=stat([e["y"] for e in ev100 if e["s7"]>=3],"100K s>=3 full",TM); ps("V_100K_s3_full",R["V_100K_s3_full"])
    R["V_100K_s3_TEST"]=stat([e["y"] for e in te2 if e["s7"]>=3],"100K s>=3 TEST",TM*(1-TRAIN)); ps("V_100K_s3_TEST",R["V_100K_s3_TEST"])
    nv=noov([(e["ts"],e["y"]) for e in ev100 if e["s7"]>=3]); s=stat(nv,"100K s>=3 noov",TM); s["per_month"]=round(len(nv)/TM,1)
    R["V_100K_s3_noov"]=s; ps("V_100K_s3_noov",s)
    return R

def run_I(ev):
    print("\n=== I: feature interaction (ikili favorable) ===")
    R={}
    base=stat([e["y6"] for e in ev],"baz tum",TM); R["I_base"]=base; ps("I_base",base)
    keys=["sync","rv","shelf","be","whale_lo"]
    import itertools
    for a,b in itertools.combinations(keys,2):
        g=[e["y6"] for e in ev if hits(e["f"]).get(a) and hits(e["f"]).get(b)]
        R[f"I_{a}+{b}"]=stat(g,f"{a}&{b} both",TM)
        if R[f"I_{a}+{b}"].get("n",0)>=8: ps(f"I_{a}+{b}",R[f"I_{a}+{b}"])
    return R

def run_Z(ev):
    print("\n=== Z: meta-veto (score>=3 kaybedenleri) ===")
    R={}
    sub=[e for e in ev if e["s7"]>=3 and not e["veto"]]
    R["Z_base_s3"]=stat([e["y6"] for e in sub],"score>=3 base",TM); ps("Z_base_s3",R["Z_base_s3"])
    # for each feature, does unfavorable side concentrate losers?
    for k in ("sync","rv","shelf","be","whale_lo","d24"):
        veto_side=[e["y6"] for e in sub if not hits(e["f"]).get(k)]  # feature NOT hit
        keep=[e["y6"] for e in sub if hits(e["f"]).get(k)]
        R[f"Z_veto_not_{k}"]={"kept_n":len(keep),"kept":stat(keep,f"keep {k}-hit",TM),"removed_avg":round(statistics.mean([x-FEE for x in veto_side]),1) if veto_side else None,"removed_n":len(veto_side)}
        kv=R[f"Z_veto_not_{k}"]["kept"]
        print("    veto !%-9s -> keep N=%d WR=%s avg=%s (removed N=%d avg=%s)"%(k,kv.get("n",0),str(kv.get("wr")),str(kv.get("avg")),len(veto_side),str(R[f"Z_veto_not_{k}"]["removed_avg"])))
    return R

def run_H(ev):
    print("\n=== H: swing horizon (score>=4) ===")
    R={}
    for hk,hl in (("y6","6h"),("y12","12h"),("y24","24h"),("y48","48h")):
        g=[e[hk] for e in ev if e["s7"]>=4 and e.get(hk) is not None]
        R[f"H_{hl}"]=stat(g,f"s>=4 {hl}",TM); ps(f"H_{hl}",R[f"H_{hl}"])
    return R

def run_R(ev):
    print("\n=== R: block-bootstrap robustluk (score>=3 noov) ===")
    R={}
    nv=noov([(e["ts"],e["y6"]) for e in ev if e["s7"]>=3 and not e["veto"]])
    net=[x-FEE for x in nv]; n=len(net)
    if n<10: R["R_note"]={"n":n}; print("    az veri"); return R
    rng=random.Random(1); bs=5; avgs=[]
    for _ in range(2000):
        samp=[]
        while len(samp)<n:
            i=rng.randint(0,n-1); samp+=net[i:i+bs]
        samp=samp[:n]; avgs.append(sum(samp)/n)
    avgs.sort()
    R["R_bootstrap"]={"n":n,"obs_avg":round(sum(net)/n,1),"ci5":round(avgs[int(0.05*len(avgs))],1),"ci95":round(avgs[int(0.95*len(avgs))],1),"p_below0":round(sum(1 for a in avgs if a<0)/len(avgs),3)}
    print("    N=%d obs_avg=%.1f  block-boot 5%%=%.1f 95%%=%.1f  P(avg<0)=%.3f"%(n,sum(net)/n,R["R_bootstrap"]["ci5"],R["R_bootstrap"]["ci95"],R["R_bootstrap"]["p_below0"]))
    return R

def run_T(conn,m,ev,now,start):
    print("\n=== T: time-of-day machine (LONG 17-23 + SHORT 13-17) ===")
    R={}
    longs=[(e["ts"],e["y6"],HOLD) for e in ev if e["s7"]>=3 and not e["veto"]]
    # SHORT 13-17: ETH SELL 200K -> noisy -> BTC>=1M confirm -> entry confirm_ts hold 180m, hour(nt) 13-17
    ancs=reconstruct_anchors(load_liquidations(conn,"ETHUSDT","SELL",start,now),bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    shorts=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sxn(ts)=="EUROPE": continue
        nt=lfirst(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP)
        if nt is None or not (13<=hod(nt)<17): continue
        conf=lfirst_above(conn,"BTCUSDT","SELL",nt+5*60_000,nt+30*60_000,1_000_000.0)
        if conf is None: continue
        y=sret(m,conf,180*60_000)
        if y is not None: shorts.append((conf,y,180*60_000))
    R["T_long_only"]=stat([v for _,v,_ in longs],"LONG 17-23 composite",TM); ps("T_long_only",R["T_long_only"])
    R["T_short_only"]=stat([v for _,v,_ in shorts],"SHORT 13-17 confirm",TM); ps("T_short_only",R["T_short_only"])
    allp=sorted(longs+shorts); busy=-1; combo=[]
    for tsx,v,hold in allp:
        if tsx>=busy: combo.append(v); busy=tsx+hold
    s=stat(combo,"time-machine portfolio",TM); s["per_month"]=round(len(combo)/TM,1)
    R["T_portfolio"]=s; ps("T_portfolio",s)
    return R

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 Horizon Gauntlet ===")
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        m=load_mark_index(conn,"ETHUSDT")
        print("build..."); ev=build(conn,m,now,start)
        span=[e["ts"] for e in ev]; TM=max(1.0,(span[-1]-span[0])/86_400_000/30.0)
        print(f"  events={len(ev)} months={TM:.2f}")
        R={}
        R["V"]=run_V(conn,m,ev,now,start); R["I"]=run_I(ev); R["Z"]=run_Z(ev)
        R["H"]=run_H(ev); R["R"]=run_R(ev); R["T"]=run_T(conn,m,ev,now,start)
    meta={"n":len(ev),"months":round(TM,2)}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"results":R,"meta":meta},indent=2,default=str),encoding="utf-8")
    lines=[f"# S34 Horizon Gauntlet","",f"> hour17 200K composite {len(ev)} event {TM:.1f} ay. Tarih {datetime.now(timezone.utc):%Y-%m-%d}",""]
    for q,sec in R.items():
        lines+=[f"## {q}",""]
        for k,v in sec.items():
            if isinstance(v,dict) and v.get("n",0)>0 and "wr" in v:
                lines.append("- **%s**: N=%d /ay=%.1f WR=%.1f%% avg=%+.1f TOT=%s mc_p=%s"%(k,v["n"],v.get("per_month",0),v["wr"],v["avg"],v.get("total"),v.get("mc_p","?")))
        lines.append("")
    lines+=["---","*Script: tools/research_s34_horizon_gauntlet.py*"]
    OM.write_text("\n".join(lines),encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")

if __name__=="__main__": main()
