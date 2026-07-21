"""S34 Deep Questions Q1-Q8 — composite'i live'a tasima onundeki engeller + yeni alpha yuzeyi.

Q1 Ablasyon: 6 sinyalin hangisi tasiyor; equal vs weighted skor
Q2 Gercek limit-fill: book_ticker ask ile -10/-20/-30bps fill-orani + EV/signal
Q3 Conviction exit + 15x drawdown sim: yuksek-skor uzun tut; weighted sizing hesabi tutuyor mu
Q4 SHORT conviction mining: SHORT_NOISY icin T0 sinyal taramasi
Q5 Skor hour17'siz: tum saatlerde score>=4 = daha cok frekans ayni kalite?
Q6 Liq-shelf: fiyat altinda kumelenmis likidasyon = daha cok yakit? (heatmap statik, proxy)
Q7 Expanding walk-forward + son-donem decay
Q8 Funding-cycle timing: bounce kalitesi funding'e yakinlikla degisiyor mu

Cikti: reports/research/s34/S34_DEEP_QUESTIONS.json + .md
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
OJ=OUT/"S34_DEEP_QUESTIONS.json"; OM=OUT/"S34_DEEP_QUESTIONS.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=500; HOLD=6*3600_000; TM=4.5; TRAIN=0.70; LEV=15
random.seed(42)
CT={"sync_ratio":0.5421,"rv5m":0.0304,"density24":5.0,"be_lo":0.2195,"be_hi":2.0,"imb":0.2633}

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
def book_imb(c,ts):
    r=c.execute("SELECT book_imbalance,ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None and (ts-int(r[1]))<=5*60_000 else None
def min_ask(c,lo,hi):
    r=c.execute("SELECT MIN(ask_price) FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?",(lo,hi)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
def ofi(c,s,lo,hi):
    r=c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<?",(s,lo,hi)).fetchone()
    if not r or r[0] is None: return None
    b,se=float(r[0]),float(r[1]); t=b+se; return (b-se)/t if t>0 else 0.0
def rv5(c,ts):
    r=c.execute("SELECT rv_5m FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
def nextfund(c,ts):
    r=c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return int(r[0]) if r and r[0] else None
def hod(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).hour
def sess(ts):
    h=hod(ts); return "EUROPE" if 7<=h<13 else ("US" if 13<=h<21 else "OFF")
def eprice(m,ts):
    r=m.at_or_after(ts); return (int(r[0]),float(r[1])) if r and float(r[1])>0 else None
def holdret(m,ts,hold=HOLD):
    e=eprice(m,ts)
    if not e: return None
    r=m.at_or_before(ts+hold); return (float(r[1])-e[1])/e[1]*1e4 if r else None
def shortret(m,ts,hold):
    e=eprice(m,ts)
    if not e: return None
    r=m.at_or_before(ts+hold); return -(float(r[1])-e[1])/e[1]*1e4 if r else None

def mcp(v,a):
    if len(v)<4: return None
    r=random.Random(0); ct=sum(1 for _ in range(MC) if sum(r.choice([-1,1])*abs(x) for x in v)/len(v)>=a); return round(ct/MC,3)
def stat(g,label="",months=None,fee=FEE):
    m=months or TM
    if not g: return {"label":label,"n":0}
    net=[x-fee for x in g]; n=len(net); w=sum(1 for x in net if x>0); sv=sorted(net); a=sum(net)/n
    return {"label":label,"n":n,"wr":round(100*w/n,1),"avg":round(a,1),"total":round(sum(net),0),
            "per_month":round(n/m,1),"worst":round(sv[0],1),"mc_p":mcp(net,a)}
def ps(k,v):
    if not v or v.get("n",0)==0: print("    %-38s N=0"%k[:38]); return
    print("    %-38s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s mc_p=%s"%(k[:38],v["n"],v.get("per_month",0),str(v["wr"])+"%",str(v["avg"])+"bps",str(v.get("total")),v.get("mc_p","?")))
def med(x): s=sorted(x); return s[len(s)//2] if s else 0
def noov(pairs,hold=HOLD):
    busy=-1;o=[]
    for ts,v in sorted(pairs):
        if ts>=busy: o.append(v);busy=ts+hold
    return o

SIGS=["sync_ratio","rv5m","density24","ofi_pre","be_ratio","ask_heavy"]
def sig_hits(f):
    h={}
    h["sync_ratio"]=f.get("sync_ratio") is not None and f["sync_ratio"]>=CT["sync_ratio"]
    h["rv5m"]=f.get("rv5m") is not None and f["rv5m"]>=CT["rv5m"]
    h["density24"]=f.get("density24") is not None and f["density24"]>=CT["density24"]
    h["ofi_pre"]=f.get("ofi_pre") is not None and f["ofi_pre"]>=0
    h["be_ratio"]=f.get("be_ratio") is not None and CT["be_lo"]<=f["be_ratio"]<CT["be_hi"]
    h["ask_heavy"]=f.get("imb") is not None and f["imb"]<=CT["imb"]
    return h
def score_of(f,drop=None):
    h=sig_hits(f); return sum(1 for k,v in h.items() if v and k!=drop)

def build(conn,m,now,start,hour_gate=True):
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    evs=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sess(ts)=="EUROPE" or not (b4<0 or b7<0): continue
        if hour_gate and hod(ts)<17: continue
        sk=lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
        pT=eprice(m,ts)
        f={"sync_ratio":sk/rn if rn>0 else 0,"rv5m":rv5(conn,ts),
           "density24":float(lcnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,200_000)),
           "ofi_pre":ofi(conn,"ETHUSDT",ts-5*60_000,ts),
           "be_ratio":lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts)/rn if rn>0 else 0,
           "imb":book_imb(conn,ts)}
        # liq-shelf (Q6): prior-24h ETH SELL notional in price band just below current (0-2%)
        shelf=0.0
        if pT:
            lo_px=pT[1]*0.98
            shelf=_s(conn,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?",(ts-24*3600_000,ts,lo_px,pT[1]))
        f["shelf"]=shelf
        nf=nextfund(conn,ts); f["min_to_fund"]=((nf-ts)/60_000) if nf else None
        ev={"ts":ts,"rn":rn,"f":f,"y":holdret(m,ts),"y8":holdret(m,ts+0,8*3600_000)}
        if ev["y"] is None: continue
        ev["score"]=score_of(f)
        evs.append(ev)
    evs.sort(key=lambda e:e["ts"])
    return evs

# ---- Q1 ablation ----
def run_Q1(evs):
    print("\n=== Q1: Ablasyon (score>=3 gate) ===")
    R={}
    base=[e["y"] for e in evs if e["score"]>=3]; R["Q1_full6"]=stat(base,"tum 6 sinyal",TM); ps("Q1_full6",R["Q1_full6"])
    for d in SIGS:
        g=[e["y"] for e in evs if score_of(e["f"],drop=d)>=3]
        R[f"Q1_drop_{d}"]=stat(g,f"-{d}",TM); ps(f"Q1_drop_{d}",R[f"Q1_drop_{d}"])
    # solo lift: each signal hi vs lo (avg delta)
    print("  solo katki (hi-lo avg):")
    for d in SIGS:
        hi=[e["y"]-FEE for e in evs if sig_hits(e["f"])[d]]; lo=[e["y"]-FEE for e in evs if not sig_hits(e["f"])[d]]
        if hi and lo:
            R[f"Q1_solo_{d}"]={"hi_avg":round(sum(hi)/len(hi),1),"lo_avg":round(sum(lo)/len(lo),1),"delta":round(sum(hi)/len(hi)-sum(lo)/len(lo),1),"hi_n":len(hi)}
            print("    %-12s hi=%+.1f(N%d) lo=%+.1f delta=%+.1f"%(d,sum(hi)/len(hi),len(hi),sum(lo)/len(lo),sum(hi)/len(hi)-sum(lo)/len(lo)))
    return R

# ---- Q2 limit fill realism ----
def run_Q2(conn,m,evs):
    print("\n=== Q2: Gercek limit-fill (book ask) + EV/signal ===")
    R={}
    subs=[e for e in evs if e["score"]>=3]
    for k in (10,20,30):
        fills=0; g=[]
        for e in subs:
            ep=eprice(m,e["ts"])
            if not ep: continue
            lvl=ep[1]*(1-k/1e4)
            ma=min_ask(conn,ep[0],e["ts"]+15*60_000)
            if ma is not None and ma<=lvl:
                fills+=1; r=m.at_or_before(e["ts"]+HOLD)
                if r: g.append((float(r[1])-lvl)/lvl*1e4)
        nsub=len(subs); frate=fills/nsub if nsub else 0
        s=stat(g,f"limit -{k}bps filled",TM)
        # EV per signal: filled avg * fill_rate (no-fill = 0 opportunity)
        ev_sig=(s.get("avg") or 0)*frate
        R[f"Q2_limit{k}"]={**s,"fill_rate":round(100*frate,1),"ev_per_signal":round(ev_sig,1),"n_signals":nsub}
        print("    limit -%dbps: fill=%.0f%% (%d/%d) filled_avg=%s EV/signal=%+.1f mc=%s"%(k,100*frate,fills,nsub,str(s.get("avg")),ev_sig,s.get("mc_p")))
    # market ref
    g=[e["y"] for e in subs]; R["Q2_market_ref"]=stat(g,"market T0 ref",TM); ps("Q2_market_ref",R["Q2_market_ref"])
    return R

# ---- Q3 conviction exit + 15x drawdown sim ----
def run_Q3(m,evs):
    print("\n=== Q3: Conviction exit + 15x drawdown sim ===")
    R={}
    # conviction-conditional hold: score>=4 -> 8h else 6h
    g_cond=[]
    for e in evs:
        if e["score"]<3: continue
        y = e["y8"] if (e["score"]>=4 and e.get("y8") is not None) else e["y"]
        g_cond.append(y)
    R["Q3_cond_exit"]=stat(g_cond,"score>=4 hold8h else6h (gate>=3)",TM); ps("Q3_cond_exit",R["Q3_cond_exit"])
    R["Q3_fixed_ref"]=stat([e["y"] for e in evs if e["score"]>=3],"fixed 6h ref",TM); ps("Q3_fixed_ref",R["Q3_fixed_ref"])
    # equity sim (no-overlap chronological), leverage 15x
    def sim(weighted,unit_pct):
        eq=1.0; peak=1.0; mdd=0.0; busy=-1; n=0
        for e in sorted(evs,key=lambda x:x["ts"]):
            if e["score"]<3 or e["ts"]<busy: continue
            units=(e["score"]-2) if weighted else 1   # 1..4 weighted, else 1
            marg=min(eq*unit_pct*units, eq*0.90)
            pnl=marg*LEV*((e["y"]-FEE)/1e4)
            eq+=pnl; peak=max(peak,eq); mdd=min(mdd,(eq-peak)/peak); busy=e["ts"]+HOLD; n+=1
        return {"n":n,"final_mult":round(eq,3),"max_dd_pct":round(100*mdd,1)}
    for wl,wname in ((False,"flat"),(True,"weighted")):
        for up in (0.05,0.10):
            r=sim(wl,up); R[f"Q3_sim_{wname}_{int(up*100)}pct"]=r
            print("    sizing=%-8s unit_pct=%d%%: trades=%d final_mult=%.2fx max_DD=%.1f%%"%(wname,int(up*100),r["n"],r["final_mult"],r["max_dd_pct"]))
    return R

# ---- Q4 SHORT conviction mining ----
def run_Q4(conn,m,now,start):
    print("\n=== Q4: SHORT conviction mining (SHORT_NOISY BTC1M) ===")
    R={}
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    ev=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or m.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sess(ts)=="EUROPE": continue
        nt=lfirst(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP)
        if nt is None: continue
        btc=lmax(conn,"BTCUSDT","SELL",nt+5*60_000,nt+30*60_000)
        if btc<1_000_000: continue
        y=shortret(m,nt,180*60_000)
        if y is None: continue
        f={"sync_ratio":(lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts))/rn if rn>0 else 0,
           "rv5m":rv5(conn,nt),"be_ratio":btc/rn if rn>0 else 0,"imb":book_imb(conn,nt),
           "ofi_post":ofi(conn,"ETHUSDT",nt,nt+5*60_000),"btc_size":btc,"hour":hod(nt)}
        ev.append({"ts":nt,"f":f,"y":y})
    ev.sort(key=lambda e:e["ts"])
    R["Q4_base"]=stat([e["y"] for e in ev],"SHORT_NOISY base",TM); ps("Q4_base",R["Q4_base"])
    print("  SHORT feature solo (hi-lo avg, N=%d):"%len(ev))
    for k in ("sync_ratio","rv5m","be_ratio","imb","ofi_post","btc_size"):
        vals=[(e["f"].get(k),e["y"]) for e in ev if e["f"].get(k) is not None]
        if len(vals)<8: continue
        mm=med([v for v,_ in vals]); hi=[y-FEE for v,y in vals if v>=mm]; lo=[y-FEE for v,y in vals if v<mm]
        if hi and lo:
            R[f"Q4_{k}"]={"hi_avg":round(sum(hi)/len(hi),1),"lo_avg":round(sum(lo)/len(lo),1),"delta":round(sum(hi)/len(hi)-sum(lo)/len(lo),1)}
            print("    %-11s hi=%+.1f lo=%+.1f delta=%+.1f"%(k,sum(hi)/len(hi),sum(lo)/len(lo),sum(hi)/len(hi)-sum(lo)/len(lo)))
    return R

# ---- Q5 score without hour gate ----
def run_Q5(conn,m,now,start):
    print("\n=== Q5: Skor hour17'siz (tum saatler) ===")
    R={}
    allh=build(conn,m,now,start,hour_gate=False)
    print("  all-hours events=%d"%len(allh))
    for K in (3,4):
        pairs=[(e["ts"],e["y"]) for e in allh if e["score"]>=K]
        R[f"Q5_allhours_s{K}_full"]=stat([v for _,v in pairs],f"all-hours score>={K}",TM); ps(f"Q5_allhours_s{K}_full",R[f"Q5_allhours_s{K}_full"])
        nv=noov(pairs); s=stat(nv,f"all-hours score>={K} noov",TM); s["per_month"]=round(len(nv)/TM,1)
        R[f"Q5_allhours_s{K}_noov"]=s; ps(f"Q5_allhours_s{K}_noov",s)
    # split: hour>=17 subset vs hour<17 subset within score>=4
    for lbl,cond in (("hour>=17",lambda t:hod(t)>=17),("hour<17",lambda t:hod(t)<17)):
        g=[e["y"] for e in allh if e["score"]>=4 and cond(e["ts"])]
        R[f"Q5_s4_{lbl}"]=stat(g,f"score>=4 {lbl}",TM); ps(f"Q5_s4_{lbl}",R[f"Q5_s4_{lbl}"])
    return R

# ---- Q6 liq-shelf ----
def run_Q6(evs):
    print("\n=== Q6: Liq-shelf (fiyat altinda kumelenmis likidasyon = yakit) ===")
    R={}
    vals=[(e["f"]["shelf"],e["y"]) for e in evs if e["f"].get("shelf") is not None]
    mm=med([v for v,_ in vals]); hi=[y-FEE for v,y in vals if v>=mm]; lo=[y-FEE for v,y in vals if v<mm]
    R["Q6_shelf_hi"]=stat([y for v,y in vals if v>=mm],"shelf hi (cok yakit)",TM)
    R["Q6_shelf_lo"]=stat([y for v,y in vals if v<mm],"shelf lo",TM)
    ps("Q6_shelf_hi",R["Q6_shelf_hi"]); ps("Q6_shelf_lo",R["Q6_shelf_lo"])
    if hi and lo: print("    delta hi-lo = %+.1f"%(sum(hi)/len(hi)-sum(lo)/len(lo)))
    return R

# ---- Q7 walk-forward + decay ----
def run_Q7(evs):
    print("\n=== Q7: Expanding walk-forward + son-donem decay ===")
    R={}
    sub=[e for e in evs if e["score"]>=3]
    n=len(sub); k=4
    print("  expanding folds (score>=3):")
    for i in range(1,k):
        te=sub[i*n//k:(i+1)*n//k]; g=[e["y"] for e in te]
        s=stat(g,f"fold{i+1}",TM/k); R[f"Q7_fold{i+1}"]=s
        print("    fold%d N=%d WR=%s avg=%s"%(i+1,s.get("n",0),str(s.get("wr")),str(s.get("avg"))))
    # recent-half vs first-half
    R["Q7_first_half"]=stat([e["y"] for e in sub[:n//2]],"ilk yari",TM/2)
    R["Q7_recent_half"]=stat([e["y"] for e in sub[n//2:]],"son yari",TM/2)
    ps("Q7_first_half",R["Q7_first_half"]); ps("Q7_recent_half",R["Q7_recent_half"])
    return R

# ---- Q8 funding timing ----
def run_Q8(evs):
    print("\n=== Q8: Funding-cycle timing ===")
    R={}
    sub=[e for e in evs if e["f"].get("min_to_fund") is not None]
    for lbl,cond in (("<60m to fund",lambda mt:mt<60),("60-240m",lambda mt:60<=mt<240),(">240m",lambda mt:mt>=240)):
        g=[e["y"] for e in sub if cond(e["f"]["min_to_fund"])]
        R[f"Q8_{lbl}"]=stat(g,lbl,TM); ps(f"Q8_{lbl}",R[f"Q8_{lbl}"])
    return R

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 Deep Questions Q1-Q8 ===")
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        m=load_mark_index(conn,"ETHUSDT")
        print("build hour17 events..."); evs=build(conn,m,now,start,hour_gate=True)
        span=[e["ts"] for e in evs]; TM=max(1.0,(span[-1]-span[0])/86_400_000/30.0)
        print(f"  events={len(evs)} months={TM:.2f}")
        R={}
        R["Q1"]=run_Q1(evs); R["Q2"]=run_Q2(conn,m,evs); R["Q3"]=run_Q3(m,evs)
        R["Q4"]=run_Q4(conn,m,now,start); R["Q5"]=run_Q5(conn,m,now,start)
        R["Q6"]=run_Q6(evs); R["Q7"]=run_Q7(evs); R["Q8"]=run_Q8(evs)
    meta={"n":len(evs),"months":round(TM,2)}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"results":R,"meta":meta},indent=2,default=str),encoding="utf-8")
    lines=["# S34 Deep Questions Q1-Q8","",f"> hour17 200K baz {len(evs)} event {TM:.1f} ay. FEE={int(FEE)} lev={LEV}x. Tarih {datetime.now(timezone.utc):%Y-%m-%d}",""]
    for q,sec in R.items():
        lines+=[f"## {q}",""]
        for k,v in sec.items():
            if isinstance(v,dict) and v.get("n",0)>0 and "wr" in v:
                lines.append("- **%s**: N=%d /ay=%.1f WR=%.1f%% avg=%+.1f TOT=%s mc_p=%s"%(k,v["n"],v.get("per_month",0),v["wr"],v["avg"],v.get("total"),v.get("mc_p","?")))
            elif isinstance(v,dict) and ("delta" in v or "final_mult" in v or "fill_rate" in v):
                lines.append("- %s: %s"%(k,{x:v[x] for x in v if x not in ('label',)}))
        lines.append("")
    lines+=["---","*Script: tools/research_s34_deep_questions.py*"]
    OM.write_text("\n".join(lines),encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")

if __name__=="__main__": main()
