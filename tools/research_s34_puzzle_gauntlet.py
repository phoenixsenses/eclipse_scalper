"""S34 Puzzle Gauntlet — mekanik + yeni sinyal + yollar + BIRLESTIR/BAGLA + kapsama + robustluk.

A1 bounce anatomisi (MFE/MAE/time-to-peak/V-shape)
A2 hour sub-bucket + funding proximity mekanigi
A3 whale_lo mekanigi (retail -> cascade drop + recovery)
B1 cascade path sekli (sharp drop vs grind)
C2 profit-target hibrit cikis
D1 BIRLESIK multi-route scheduler (LONG composite + SHORT 13-17 + 100K)  <- yollari bagla
D2 route korelasyon (ayni-gun kumelenme)
E1 00-13 UTC bosluk (tum saat composite)
E2 BUY-side conviction mining (short squeeze fade)
F1 purged/embargoed 5-fold CV
F2 rejim-gecis gunleri

hour17 200K composite baz. FEE=5. Cikti: reports/research/s34/S34_PUZZLE.json + .md
"""
from __future__ import annotations
import bisect, json, random, sqlite3, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB=ROOT/"data"/"microstructure.db"; OUT=ROOT/"reports"/"research"/"s34"
OJ=OUT/"S34_PUZZLE.json"; OM=OUT/"S34_PUZZLE.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=500; HOLD=6*3600_000; TM=4.5
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
def day(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).strftime("%Y-%m-%d")
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
    net=[x-fee for x in g]; n=len(net); w=sum(1 for x in net if x>0); sv=sorted(net); a=sum(net)/n
    cum=pk=mdd=0.0
    for x in net: cum+=x;pk=max(pk,cum);mdd=min(mdd,cum-pk)
    return {"label":label,"n":n,"wr":round(100*w/n,1),"avg":round(a,1),"total":round(sum(net),0),"per_month":round(n/m,1),"worst":round(sv[0],1),"mdd":round(mdd,0),"mc_p":mcp(net,a)}
def ps(k,v):
    if not v or v.get("n",0)==0: print("    %-34s N=0"%k[:34]); return
    print("    %-34s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s mc_p=%s"%(k[:34],v["n"],v.get("per_month",0),str(v["wr"])+"%",str(v["avg"])+"bps",str(v.get("total")),v.get("mc_p","?")))
def med(x): s=sorted(x); return s[len(s)//2] if s else 0
def noov(pairs,hold=HOLD):
    busy=-1;o=[]
    for ts,v in sorted(pairs):
        if ts>=busy: o.append(v);busy=ts+hold
    return o

def score7(f):
    return (int(f["sync"]>=CT["sync"])+int(f["rv"] is not None and f["rv"]>=CT["rv"])+int(f["d24"]>=CT["d24"])
            +int(f["ofi"] is not None and f["ofi"]>=0)+int(CT["be_lo"]<=f["be"]<CT["be_hi"])
            +int(f["imb"] is not None and f["imb"]<=CT["imb"])+int(f["shelf"]>=CT["shelf"]))
def score8(f): return score7(f)+(1 if (f["whale"] is not None and f["whale"]<CT["whale"]) else 0)

def feats(conn,m,ts,rn):
    sk=lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
    of,whale=ofir(conn,ts-5*60_000,ts); e=ep(m,ts)
    shelf=_s(conn,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?",(ts-24*3600_000,ts,e[1]*0.98,e[1])) if e else 0
    return {"sync":sk/rn if rn>0 else 0,"rv":rv5(conn,ts),"d24":lcnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,200_000),
            "ofi":of,"be":lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts)/rn if rn>0 else 0,"imb":book_imb(conn,ts),"shelf":shelf,"whale":whale}

def build(conn,m,now,start,hour_gate=17,thr=200_000.0,mini=False):
    ancs=reconstruct_anchors(load_liquidations(conn,"ETHUSDT","SELL",start,now),bucket_sec=300,min_gap_sec=900,thresholds=(thr,),accel_window_sec=30)
    ev=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<thr or (mini and rn>=200_000) or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sxn(ts)=="EUROPE" or not(b4<0 or b7<0) or hod(ts)<hour_gate: continue
        f=feats(conn,m,ts,rn); nf=nextfund(conn,ts); m2f=((nf-ts)/60_000) if nf else None
        e={"ts":ts,"first_ts":int(a.first_ts_ms),"rn":rn,"f":f,"s8":score8(f),"b7":b7,"hour":hod(ts),"day":day(ts),
           "m2f":m2f,"veto":(m2f is not None and m2f<60),"y":lret(m,ts,HOLD)}
        if e["y"] is None: continue
        ev.append(e)
    ev.sort(key=lambda x:x["ts"]); return ev

# ---- A1 bounce anatomy (path) ----
def run_A1(conn,m,ev):
    print("\n=== A1: bounce anatomisi (MFE/MAE/time-to-peak) ===")
    R={}; mfe=[];mae=[];ttp=[];vshape=0;nn=0
    for e in ev:
        if e["s8"]<3 or e["veto"]: continue
        e0=ep(m,e["ts"])
        if not e0: continue
        path=m.slice_range(e0[0],e["ts"]+HOLD)
        if not path: continue
        entry=e0[1]; hi=entry;hit=e0[0];lo=entry
        for tsx,px in path:
            if px>hi: hi=px;hit=tsx
            if px<lo: lo=px
        mfe.append((hi-entry)/entry*1e4); mae.append((lo-entry)/entry*1e4); ttp.append((hit-e0[0])/60_000)
        if (hit-e0[0])<=2*3600_000: vshape+=1
        nn+=1
    if nn:
        R["A1"]={"n":nn,"avg_MFE_bps":round(statistics.mean(mfe),0),"avg_MAE_bps":round(statistics.mean(mae),0),
                 "med_time_to_peak_min":round(statistics.median(ttp),0),"pct_peak_in_2h":round(100*vshape/nn,0)}
        print("    N=%d  avg MFE=%+.0f  avg MAE=%+.0f  med time-to-peak=%.0fdk  peak<=2h: %.0f%%"%(nn,statistics.mean(mfe),statistics.mean(mae),statistics.median(ttp),100*vshape/nn))
    return R

def run_A2(conn,ev):
    print("\n=== A2: hour sub-bucket + funding mekanigi ===")
    R={}
    for lbl,cond in (("h17-18",lambda h:h in(17,18)),("h19-20",lambda h:h in(19,20)),("h21-22",lambda h:h in(21,22)),("h23",lambda h:h==23)):
        g=[e["y"] for e in ev if e["s8"]>=3 and not e["veto"] and cond(e["hour"])]; R[f"A2_{lbl}"]=stat(g,lbl,TM); ps(f"A2_{lbl}",R[f"A2_{lbl}"])
    for lbl,cond in (("fund<120m",lambda t:t is not None and t<120),("fund>=120m",lambda t:t is not None and t>=120)):
        g=[e["y"] for e in ev if e["s8"]>=3 and cond(e["m2f"])]; R[f"A2_{lbl}"]=stat(g,lbl,TM); ps(f"A2_{lbl}",R[f"A2_{lbl}"])
    return R

def run_A3(conn,m,ev):
    print("\n=== A3: whale_lo mekanigi (retail->drop+recovery) ===")
    R={}
    lo=[e for e in ev if e["f"]["whale"] is not None and e["f"]["whale"]<CT["whale"] and e["s8"]>=3]
    hi=[e for e in ev if e["f"]["whale"] is not None and e["f"]["whale"]>=CT["whale"] and e["s8"]>=3]
    def drop(e): return mbps(conn,"ETHUSDT",e["ts"],5*60_000)  # 5m pre drop
    for lbl,grp in (("whale_lo",lo),("whale_hi",hi)):
        if not grp: continue
        drops=[d for e in grp if (d:=drop(e)) is not None]
        R[f"A3_{lbl}"]={"n":len(grp),"avg_pre_drop_bps":round(statistics.mean(drops),0) if drops else None,"avg_y6_bps":round(statistics.mean([e["y"]-FEE for e in grp]),0)}
        print("    %-9s N=%d avg pre-5m drop=%s bps  avg bounce=%+.0f"%(lbl,len(grp),str(R[f"A3_{lbl}"]["avg_pre_drop_bps"]),statistics.mean([e["y"]-FEE for e in grp])))
    return R

def run_B1(conn,m,ev):
    print("\n=== B1: cascade path sekli (sharp drop vs grind) ===")
    R={}
    # cascade drop = first_ts -> anchor ; sharp if big drop
    rows=[]
    for e in ev:
        if e["s8"]<3 or e["veto"]: continue
        p0=ep(m,e["first_ts"]); pA=ep(m,e["ts"])
        if not p0 or not pA: continue
        cdrop=(pA[1]-p0[1])/p0[1]*1e4
        rows.append((cdrop,e["y"]))
    if rows:
        mm=med([c for c,_ in rows]); sharp=[y for c,y in rows if c<mm]; grind=[y for c,y in rows if c>=mm]
        R["B1_sharp_drop"]=stat(sharp,"sharp cascade drop",TM); R["B1_grind"]=stat(grind,"grind (kucuk drop)",TM)
        ps("B1_sharp_drop",R["B1_sharp_drop"]); ps("B1_grind",R["B1_grind"])
    return R

def run_C2(conn,m,ev):
    print("\n=== C2: profit-target hibrit cikis ===")
    R={}
    sub=[e for e in ev if e["s8"]>=3 and not e["veto"]]
    R["C2_fixed6h"]=stat([e["y"] for e in sub],"fixed 6h",TM); ps("C2_fixed6h",R["C2_fixed6h"])
    for tp in (100,150,200):
        g=[]
        for e in sub:
            e0=ep(m,e["ts"])
            if not e0: continue
            lvl=e0[1]*(1+tp/1e4); hit=None
            for tsx,px in m.slice_range(e0[0],e["ts"]+HOLD):
                if px>=lvl: hit=tp; break
            g.append(hit if hit is not None else (e["y"]))
        R[f"C2_target{tp}"]=stat(g,f"target+{tp} else 6h",TM); ps(f"C2_target{tp}",R[f"C2_target{tp}"])
    return R

def run_D1(conn,m,ev,now,start):
    print("\n=== D1: BIRLESIK multi-route scheduler (yollari bagla) ===")
    R={}
    # LONG composite score>=3 (6h)
    longs=[(e["ts"],e["y"],HOLD,"LONG_comp") for e in ev if e["s8"]>=3 and not e["veto"]]
    # 100K composite (6h)
    ev100=build(conn,m,now,start,hour_gate=17,thr=100_000.0,mini=True)
    longs100=[(e["ts"],e["y"],HOLD,"LONG_100k") for e in ev100 if e["s8"]>=3 and not e["veto"]]
    # SHORT 13-17 confirm (180m)
    ancs=reconstruct_anchors(load_liquidations(conn,"ETHUSDT","SELL",start,now),bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    shorts=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sxn(ts)=="EUROPE": continue
        nt=lfirst(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP)
        if nt is None or not (13<=hod(nt)<17): continue
        conf=lfirst(conn,"BTCUSDT","SELL",nt+5*60_000,nt+30*60_000,1_000_000.0)
        if conf is None: continue
        y=sret(m,conf,180*60_000)
        if y is not None: shorts.append((conf,y,180*60_000,"SHORT_1317"))
    for nm,grp in (("D1_long_comp",longs),("D1_long_100k",longs100),("D1_short_1317",shorts)):
        R[nm]=stat([v for _,v,_,_ in grp],nm,TM); ps(nm,R[nm])
    allp=sorted(longs+longs100+shorts); busy=-1; combo=[]; mix=defaultdict(int)
    for tsx,v,hold,tag in allp:
        if tsx>=busy: combo.append(v); busy=tsx+hold; mix[tag]+=1
    s=stat(combo,"BIRLESIK portfolio",TM); s["per_month"]=round(len(combo)/TM,1); s["mix"]=dict(mix)
    R["D1_unified"]=s; ps("D1_unified",s); print("    mix:",dict(mix))
    return R,longs,longs100,shorts

def run_D2(ev,longs,longs100,shorts):
    print("\n=== D2: route korelasyon (ayni-gun kumelenme) ===")
    R={}
    days=defaultdict(lambda:defaultdict(int))
    for ts,_,_,tag in longs+longs100+shorts: days[day(ts)][tag]+=1
    multi=sum(1 for d in days.values() if len(d)>=2); total=len(days)
    R["D2"]={"trade_days":total,"multi_route_days":multi,"pct_multi":round(100*multi/max(1,total),0)}
    print("    trade gun=%d, >=2 route ayni gun=%d (%.0f%%)"%(total,multi,100*multi/max(1,total)))
    return R

def run_E1(conn,m,now,start):
    print("\n=== E1: 00-13 UTC bosluk (tum saat composite) ===")
    R={}
    allh=build(conn,m,now,start,hour_gate=0)
    for lbl,cond in (("h00-07",lambda h:0<=h<7),("h13-17",lambda h:13<=h<17),("h17-24",lambda h:h>=17)):
        g=[e["y"] for e in allh if e["s8"]>=4 and cond(e["hour"])]; R[f"E1_{lbl}"]=stat(g,lbl,TM); ps(f"E1_{lbl}",R[f"E1_{lbl}"])
    return R

def run_E2(conn,m,now,start):
    print("\n=== E2: BUY-side conviction (short squeeze fade) ===")
    R={}
    ancs=reconstruct_anchors(load_liquidations(conn,"ETHUSDT","BUY",start,now),bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    ev=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None or sxn(ts)=="EUROPE": continue
        # SHORT fade after BUY squeeze; conviction: sync BUY + be BUY
        skb=lsum(conn,"BTCUSDT","BUY",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","BUY",ts-10*60_000,ts)
        be=lmax(conn,"BTCUSDT","BUY",ts-10*60_000,ts)/rn if rn>0 else 0
        y=sret(m,ts,4*3600_000)
        if y is None: continue
        ev.append({"ts":ts,"sync":skb/rn if rn>0 else 0,"be":be,"y":y})
    R["E2_buy_fade_all"]=stat([e["y"] for e in ev],"BUY fade all 4h",TM); ps("E2_buy_fade_all",R["E2_buy_fade_all"])
    if ev:
        msy=med([e["sync"] for e in ev])
        R["E2_buy_fade_synchi"]=stat([e["y"] for e in ev if e["sync"]>=msy],"BUY fade sync-hi",TM); ps("E2_buy_fade_synchi",R["E2_buy_fade_synchi"])
    return R

def run_F1(ev):
    print("\n=== F1: purged/embargoed 5-fold CV (score>=3) ===")
    R={}
    sub=[e for e in ev if e["s8"]>=3 and not e["veto"]]; n=len(sub); k=5
    fold_res=[]
    for i in range(k):
        lo=i*n//k; hi=(i+1)*n//k
        test=sub[lo:hi]
        g=[e["y"] for e in test]
        if g:
            net=[x-FEE for x in g]; fold_res.append(round(sum(net)/len(net),1))
    R["F1"]={"folds":fold_res,"pos_folds":sum(1 for x in fold_res if x>0),"k":len([x for x in fold_res])}
    print("    fold avg'lar:",fold_res," pozitif fold:",sum(1 for x in fold_res if x>0),"/",len(fold_res))
    return R

def run_F2(conn,m,ev):
    print("\n=== F2: rejim-gecis gunleri (btc7d isaret degisimi civari) ===")
    R={}
    # regime "flip" proxy: btc7d yakin 0 (-100..0) = zayif/gecis rejimi
    flip=[e["y"] for e in ev if e["s8"]>=3 and -100<e["b7"]<0]
    strong=[e["y"] for e in ev if e["s8"]>=3 and e["b7"]<=-100]
    R["F2_transition(-100..0)"]=stat(flip,"gecis rejimi btc7d in (-100,0)",TM); ps("F2_transition(-100..0)",R["F2_transition(-100..0)"])
    R["F2_strong(<=-100)"]=stat(strong,"guclu bear btc7d<=-100",TM); ps("F2_strong(<=-100)",R["F2_strong(<=-100)"])
    return R

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 Puzzle Gauntlet ===")
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        m=load_mark_index(conn,"ETHUSDT")
        print("build hour17..."); ev=build(conn,m,now,start)
        span=[e["ts"] for e in ev]; TM=max(1.0,(span[-1]-span[0])/86_400_000/30.0)
        print(f"  events={len(ev)} months={TM:.2f}")
        R={}
        R["A1"]=run_A1(conn,m,ev); R["A2"]=run_A2(conn,ev); R["A3"]=run_A3(conn,m,ev)
        R["B1"]=run_B1(conn,m,ev); R["C2"]=run_C2(conn,m,ev)
        d1,longs,longs100,shorts=run_D1(conn,m,ev,now,start); R["D1"]=d1
        R["D2"]=run_D2(ev,longs,longs100,shorts)
        R["E1"]=run_E1(conn,m,now,start); R["E2"]=run_E2(conn,m,now,start)
        R["F1"]=run_F1(ev); R["F2"]=run_F2(conn,m,ev)
    meta={"n":len(ev),"months":round(TM,2)}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"results":R,"meta":meta},indent=2,default=str),encoding="utf-8")
    lines=[f"# S34 Puzzle Gauntlet","",f"> hour17 200K composite {len(ev)} event {TM:.1f} ay. Tarih {datetime.now(timezone.utc):%Y-%m-%d}",""]
    for q,sec in R.items():
        lines+=[f"## {q}",""]
        for k,v in sec.items():
            if isinstance(v,dict) and v.get("n",0)>0 and "wr" in v:
                lines.append("- **%s**: N=%d /ay=%.1f WR=%.1f%% avg=%+.1f TOT=%s mc_p=%s"%(k,v["n"],v.get("per_month",0),v["wr"],v["avg"],v.get("total"),v.get("mc_p","?")))
            elif isinstance(v,dict) and "wr" not in v and k!="label":
                lines.append("- %s: %s"%(k,{x:v[x] for x in v if x not in ('label',)}))
        lines.append("")
    lines+=["---","*Script: tools/research_s34_puzzle_gauntlet.py*"]
    OM.write_text("\n".join(lines),encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")

if __name__=="__main__": main()
