"""S34 Early-Signal + Management Gauntlet.

hour17 alpha'yi "perfect"e yaklastirmak icin A-F test bankasi.
Baz: hour17 200K LONG (not bull, not EU, regime btc4h<0 OR btc7d<0, hour>=17),
     entry T0, hold 6h. FEE=5bps. Metodoloji: no-overlap + holdout + MC + WF.
     LOOKAHEAD YASAK: sadece T0-knowable feature ile filtre/karar.

A EARLY  : erken/anticipatory giris (kismi anchor 100/150K, first-liq, hiz/accel)
B ENTRY  : giris kalitesi (sub-minute, limit/maker, micro-confirm)
C MGMT   : yonetim (breakeven, trailing, kismi cikis, funding-exit, adaptif hold, scale-in)
D SIZING : conviction-weighted (funding=lo + sync=hi + accel) + magnitude
E REGIME : btc7d derinlik, ay stabilite, funding-saat yakinligi
F PORT   : hour17 + SHORT_NOISY portfoy; SHORT hour; cross-asset SOL/BTC kendi cascade

Cikti: reports/research/s34/S34_EARLY_MGMT.json + .md
"""
from __future__ import annotations
import bisect, json, math, random, sqlite3, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors
from tools.research_s34_wave_absorption import book_features_at

DB=ROOT/"data"/"microstructure.db"; OUT=ROOT/"reports"/"research"/"s34"
OJ=OUT/"S34_EARLY_MGMT.json"; OM=OUT/"S34_EARLY_MGMT.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=500; HOLD=6*3600_000; TM=4.5
random.seed(42)

def _s(conn,sql,p=()):
    r=conn.execute(sql,p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0
def liq_sum(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def liq_max(c,s,sd,lo,hi): return _s(c,"SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",(s,sd,lo,hi))
def liq_cnt(c,s,sd,lo,hi,t): return int(_s(c,"SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",(s,sd,lo,hi,t)))
def liq_first(c,s,sd,lo,hi,t):
    r=c.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1",(s,sd,lo,hi,t)).fetchone()
    return int(r[0]) if r else None
def mbps(c,s,ts,lb):
    a=c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(s,ts-lb)).fetchone()
    b=c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(s,ts)).fetchone()
    return (float(b[0])-float(a[0]))/float(a[0])*1e4 if a and b and float(a[0])>0 else 0.0
def funding(c,ts):
    r=c.execute("SELECT funding_rate FROM funding_rates WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else 0.0
def ofi(c,s,lo,hi):
    r=c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<?",(s,lo,hi)).fetchone()
    if not r or r[0] is None: return 0.0
    b,se=float(r[0]),float(r[1]); t=b+se; return (b-se)/t if t>0 else 0.0
def sess(ts):
    h=datetime.fromtimestamp(ts/1000,tz=timezone.utc).hour
    return "EUROPE" if 7<=h<13 else ("US" if 13<=h<21 else "OFF")
def hod(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).hour
def dow(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).weekday()
def mon(ts): d=datetime.fromtimestamp(ts/1000,tz=timezone.utc); return f"{d.year}-{d.month:02d}"
def next_funding(ts):
    d=datetime.fromtimestamp(ts/1000,tz=timezone.utc)
    for h in (0,8,16,24):
        slot=d.replace(hour=h%24,minute=0,second=0,microsecond=0)
        if h==24: slot=slot.replace(day=d.day)+ (datetime(1,1,2)-datetime(1,1,1))
        base=datetime(d.year,d.month,d.day,tzinfo=timezone.utc)
        cand=base.timestamp()*1000+h*3600_000
        if cand>ts: return int(cand)
    return ts+8*3600_000

# ---- stats ----
def mcp(v,a):
    if len(v)<4: return None
    r=random.Random(0); ct=sum(1 for _ in range(MC) if sum(r.choice([-1,1])*abs(x) for x in v)/len(v)>=a)
    return round(ct/MC,3)
def wf(v,k=5):
    n=len(v)
    return "%d/%d"%(sum(1 for i in range(k) if sum(v[i*n//k:(i+1)*n//k])>0),k) if n>=k else None
def stat(gross,label="",months=None,fee=FEE):
    m=months or TM
    if not gross: return {"label":label,"n":0}
    net=[g-fee for g in gross]; n=len(net); w=sum(1 for x in net if x>0); sv=sorted(net); a=sum(net)/n
    cum=pk=mdd=0.0
    for x in net: cum+=x; pk=max(pk,cum); mdd=min(mdd,cum-pk)
    return {"label":label,"n":n,"wr":round(100*w/n,1),"avg":round(a,1),"total":round(sum(net),0),
            "per_month":round(n/m,1),"worst":round(sv[0],1),"tail_n":sum(1 for x in net if x<-100),
            "mdd":round(mdd,0),"mc_p":mcp(net,a),"wf":wf(net)}
def noov(pairs,hold=HOLD):
    busy=-1; out=[]
    for ts,v in sorted(pairs):
        if ts>=busy: out.append(v); busy=ts+hold
    return out
def ps(k,v):
    if not v or v.get("n",0)==0: print("    %-40s N=0"%k[:40]); return
    print("    %-40s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s worst=%-7s mc_p=%s wf=%s"%(
        k[:40],v["n"],v.get("per_month",0),str(v["wr"])+"%",str(v["avg"])+"bps",str(v.get("total")),str(v.get("worst")),v.get("mc_p","?"),v.get("wf")))

# ---- price path helpers ----
def path_between(marks,a,b): return [px for _,px in marks.slice_range(a,b)]
def entry_px(marks,ts):
    r=marks.at_or_after(ts); return (int(r[0]),float(r[1])) if r and float(r[1])>0 else None
def gross_hold(marks,ts,hold=HOLD):
    e=entry_px(marks,ts)
    if not e: return None
    r=marks.at_or_before(ts+hold)
    return (float(r[1])-e[1])/e[1]*1e4 if r else None

# ---- management sims on path (LONG) ----
def m_fixed(path,ep): return (path[-1]-ep)/ep*1e4 if path else None
def m_stop(path,ep,sb):
    lvl=ep*(1-sb/1e4)
    for px in path:
        if px<=lvl: return -sb
    return (path[-1]-ep)/ep*1e4
def m_breakeven(path,ep,trig):
    armed=False; be=ep
    for px in path:
        if not armed and px>=ep*(1+trig/1e4): armed=True
        if armed and px<=be: return 0.0
    return (path[-1]-ep)/ep*1e4
def m_trail(path,ep,arm,trail):
    armed=False; hi=ep
    for px in path:
        if px>hi: hi=px
        if not armed and px>=ep*(1+arm/1e4): armed=True
        if armed and px<=hi*(1-trail/1e4): return (hi*(1-trail/1e4)-ep)/ep*1e4
    return (path[-1]-ep)/ep*1e4
def m_partial(path,ep,tp):
    # half at tp first-touch, half at end
    hit=None
    for px in path:
        if px>=ep*(1+tp/1e4): hit=ep*(1+tp/1e4); break
    end=(path[-1]-ep)/ep*1e4
    if hit is None: return end
    return 0.5*tp + 0.5*end
def m_scalein(path,ep,dip):
    # add 2nd unit if price dips to ep*(1-dip); exit both at end; return per-unit avg bps
    lvl=ep*(1-dip/1e4); added=None
    for px in path:
        if px<=lvl: added=lvl; break
    end_px=path[-1]
    if added is None: return (end_px-ep)/ep*1e4
    pnl1=(end_px-ep)/ep*1e4; pnl2=(end_px-added)/added*1e4
    return (pnl1+pnl2)/2.0

# ---- build ETH events ----
def build(conn,marks,thr,now,start,rich=True):
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(float(thr),),accel_window_sec=30)
    evs=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<thr or marks.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000); b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000)
        e1=mbps(conn,"ETHUSDT",ts,3600_000); bull=e1>20 and b4>50; s=sess(ts)
        if bull or s=="EUROPE": continue
        if not (b4<0 or b7<0): continue
        if hod(ts)<17: continue   # hour17 gate
        ev={"ts":ts,"rn":rn,"first_ts":int(a.first_ts_ms),"accel":float(a.running_accel),
            "rate":float(a.running_liq_count)/max(1.0,(ts-int(a.first_ts_ms))/1000.0),
            "liq_count":float(a.running_liq_count),"btc4h":b4,"btc7d":b7,"hour":hod(ts),
            "dow":dow(ts),"month":mon(ts)}
        if rich:
            sk=liq_sum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+liq_sum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
            ev["sync_sell_pre"]=sk; ev["sync_ratio"]=sk/rn if rn>0 else 0.0
            bc=liq_max(conn,"BTCUSDT","SELL",ts-10*60_000,ts); ev["be_ratio_pre"]=bc/rn if rn>0 else 0.0
            ev["btc5m"]=mbps(conn,"BTCUSDT",ts,5*60_000); ev["funding"]=funding(conn,ts)
            ev["ofi_pre"]=ofi(conn,"ETHUSDT",ts-5*60_000,ts)
            nt=liq_first(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP); ev["noisy_ts"]=nt
        evs.append(ev)
    evs.sort(key=lambda e:e["ts"])
    return evs

def regime(e): return True  # already gated in build

# =====================================================================
def run_A(conn,marks,ev200,months,now,start):
    print("\n=== A: EARLY / anticipatory entry ===")
    R={}
    # baseline 200K T0
    R["A0_base_200K_T0"]=stat([g for e in ev200 if (g:=gross_hold(marks,e["ts"])) is not None],"200K T0 6h",months)
    ps("A0_base_200K_T0",R["A0_base_200K_T0"])
    # A1 partial anchor 100/150K
    for thr in (100_000,150_000):
        evs=build(conn,marks,thr,now,start,rich=False)
        pairs=[(e["ts"],g) for e in evs if (g:=gross_hold(marks,e["ts"])) is not None]
        R[f"A1_{thr//1000}K_T0_raw"]=stat([v for _,v in pairs],f"{thr//1000}K T0 raw",months)
        s=stat(noov(pairs),f"{thr//1000}K no-overlap",months); s["per_month"]=round(len(noov(pairs))/months,1)
        R[f"A1_{thr//1000}K_noov"]=s
        ps(f"A1_{thr//1000}K_T0_raw",R[f"A1_{thr//1000}K_T0_raw"]); ps(f"A1_{thr//1000}K_noov",s)
    # A2 first-liq entry (200K)
    R["A2_firstliq_entry"]=stat([g for e in ev200 if (g:=gross_hold(marks,e["first_ts"])) is not None],"200K enter at burst first_ts",months)
    ps("A2_firstliq_entry",R["A2_firstliq_entry"])
    # A3 velocity/accel split (200K)
    for key in ("accel","rate","liq_count"):
        vals=sorted(e[key] for e in ev200); med=vals[len(vals)//2] if vals else 0
        for lbl,cond in (("hi",lambda e,k=key,m=med:e[k]>=m),("lo",lambda e,k=key,m=med:e[k]<m)):
            g=[gg for e in ev200 if cond(e) and (gg:=gross_hold(marks,e["ts"])) is not None]
            R[f"A3_{key}_{lbl}"]=stat(g,f"200K {key} {lbl}",months)
        ps(f"A3_{key}_hi",R[f"A3_{key}_hi"]); ps(f"A3_{key}_lo",R[f"A3_{key}_lo"])
    return R

def run_B(marks,ev200,months):
    print("\n=== B: ENTRY quality ===")
    R={}
    for off,lbl in ((0,"T0"),(10_000,"T+10s"),(30_000,"T+30s"),(60_000,"T+1m"),(120_000,"T+2m"),(300_000,"T+5m")):
        pairs=[(e["ts"],g) for e in ev200 if (g:=gross_hold(marks,e["ts"]+off,HOLD-off)) is not None]
        s=stat(noov(pairs),f"entry {lbl}",months); s["per_month"]=round(len(noov(pairs))/months,1)
        R[f"B1_{lbl}"]=s; ps(f"B1_{lbl}",s)
    # B2 limit entry below anchor (fill if dips within 15m)
    for k in (10,20,30):
        g=[]
        for e in ev200:
            ep=entry_px(marks,e["ts"])
            if not ep: continue
            lvl=ep[1]*(1-k/1e4); path15=path_between(marks,ep[0],e["ts"]+15*60_000)
            if any(px<=lvl for px in path15):
                r=marks.at_or_before(e["ts"]+HOLD)
                if r: g.append((r[1]-lvl)/lvl*1e4)
        R[f"B2_limit_{k}bps"]=stat(g,f"limit -{k}bps fill<15m hold6h",months); ps(f"B2_limit_{k}bps",R[f"B2_limit_{k}bps"])
    # B3 micro-confirm: enter at first up-tick > anchor within 5m
    g=[]
    for e in ev200:
        ep=entry_px(marks,e["ts"])
        if not ep: continue
        sl=marks.slice_range(ep[0],e["ts"]+5*60_000); ent=None
        for tsx,px in sl:
            if px>ep[1]: ent=(tsx,px); break
        if ent:
            r=marks.at_or_before(ent[0]+HOLD)
            if r: g.append((r[1]-ent[1])/ent[1]*1e4)
    R["B3_micro_confirm"]=stat(g,"enter first up-tick<5m",months); ps("B3_micro_confirm",R["B3_micro_confirm"])
    return R

def run_C(marks,ev200,months):
    print("\n=== C: MANAGEMENT ===")
    R={}
    # precompute path per event
    paths={}
    for e in ev200:
        ep=entry_px(marks,e["ts"])
        if not ep: continue
        p6=path_between(marks,ep[0],e["ts"]+HOLD)
        p8=path_between(marks,ep[0],e["ts"]+8*3600_000)
        if p6: paths[e["ts"]]=(ep[1],p6,p8)
    def collect(fn): return [fn(v[0],v[1]) for v in paths.values()]
    R["C0_fixed6h"]=stat(collect(lambda ep,p:m_fixed(p,ep)),"fixed 6h",months); ps("C0_fixed6h",R["C0_fixed6h"])
    R["C_stop300"]=stat(collect(lambda ep,p:m_stop(p,ep,300)),"stop300 (live)",months); ps("C_stop300",R["C_stop300"])
    for tr in (50,100):
        R[f"C1_be{tr}"]=stat(collect(lambda ep,p,t=tr:m_breakeven(p,ep,t)),f"breakeven@+{tr}",months); ps(f"C1_be{tr}",R[f"C1_be{tr}"])
    for arm,tl in ((100,50),(150,75),(200,100)):
        R[f"C1_trail{arm}_{tl}"]=stat(collect(lambda ep,p,a=arm,t=tl:m_trail(p,ep,a,t)),f"trail arm{arm} trail{tl}",months); ps(f"C1_trail{arm}_{tl}",R[f"C1_trail{arm}_{tl}"])
    for tp in (50,100,150):
        R[f"C2_partial{tp}"]=stat(collect(lambda ep,p,t=tp:m_partial(p,ep,t)),f"partial 50%@+{tp}",months); ps(f"C2_partial{tp}",R[f"C2_partial{tp}"])
    # C4 adaptive: at 6h if last6>entry use 8h else 6h
    def adapt(ep,p6,p8):
        if p6[-1]>ep and p8: return (p8[-1]-ep)/ep*1e4
        return (p6[-1]-ep)/ep*1e4
    R["C4_adaptive_8h"]=stat([adapt(v[0],v[1],v[2]) for v in paths.values()],"adaptive hold 6/8h",months); ps("C4_adaptive_8h",R["C4_adaptive_8h"])
    # C5 scale-in
    for dip in (100,150):
        R[f"C5_scalein{dip}"]=stat(collect(lambda ep,p,d=dip:m_scalein(p,ep,d)),f"scale-in @-{dip}",months); ps(f"C5_scalein{dip}",R[f"C5_scalein{dip}"])
    return R

def run_C3(conn,marks,ev200,months):
    print("  C3: funding-aware exit")
    R={}
    g_fund=[]; g_fix=[]
    for e in ev200:
        ep=entry_px(marks,e["ts"])
        if not ep: continue
        nf=next_funding(ep[0]); exit_ts=min(nf,e["ts"]+HOLD)
        rf=marks.at_or_before(exit_ts); rx=marks.at_or_before(e["ts"]+HOLD)
        if rf: g_fund.append((rf[1]-ep[1])/ep[1]*1e4)
        if rx: g_fix.append((rx[1]-ep[1])/ep[1]*1e4)
    R["C3_funding_exit"]=stat(g_fund,"exit at next funding or 6h",months)
    R["C3_fix_ref"]=stat(g_fix,"fixed 6h ref",months)
    ps("C3_funding_exit",R["C3_funding_exit"]); ps("C3_fix_ref",R["C3_fix_ref"])
    return R

def run_D(marks,ev200,months):
    print("\n=== D: CONVICTION sizing ===")
    R={}
    fund=sorted(e["funding"] for e in ev200); fmed=fund[len(fund)//2]
    sr=sorted(e["sync_ratio"] for e in ev200); smed=sr[len(sr)//2]
    ac=sorted(e["accel"] for e in ev200); amed=ac[len(ac)//2]
    def score(e): return int(e["funding"]<fmed)+int(e["sync_ratio"]>=smed)+int(e["accel"]>=amed)
    for sc in (0,1,2,3):
        g=[gg for e in ev200 if score(e)==sc and (gg:=gross_hold(marks,e["ts"])) is not None]
        R[f"D1_score{sc}"]=stat(g,f"conviction score={sc}",months); ps(f"D1_score{sc}",R[f"D1_score{sc}"])
    # weighted portfolio: size = score+1 units, no-overlap
    pairs=[(e["ts"],(gross_hold(marks,e["ts"]),score(e)+1)) for e in ev200 if gross_hold(marks,e["ts"]) is not None]
    busy=-1; wnet=0.0; units=0; flatnet=0.0; fcnt=0
    for ts,(g,u) in sorted(pairs):
        if ts>=busy:
            wnet+=(g-FEE)*u; units+=u; flatnet+=(g-FEE); fcnt+=1; busy=ts+HOLD
    R["D1_weighted_vs_flat"]={"label":"conviction-weighted vs flat (no-overlap)","n":fcnt,
        "weighted_total_units":round(wnet,0),"weighted_avg_per_unit":round(wnet/units,1) if units else None,
        "flat_total":round(flatnet,0),"flat_avg":round(flatnet/fcnt,1) if fcnt else None}
    print("    D1_weighted: flat_total=%s  weighted_total=%s (units=%d)"%(round(flatnet,0),round(wnet,0),units))
    # D2 magnitude: corr of feature with l6h (winners), rank by |mean(hi)-mean(lo)|
    print("  D2 feature vs bounce magnitude (hi-lo avg bps):")
    for key in ("funding","sync_ratio","accel","btc5m","ofi_pre","be_ratio_pre","rate"):
        vals=sorted((e[key],gross_hold(marks,e["ts"])) for e in ev200 if gross_hold(marks,e["ts"]) is not None)
        vals=[(k,v) for k,v in vals if v is not None]
        if len(vals)<8: continue
        med=sorted(k for k,_ in vals)[len(vals)//2]
        hi=[v-FEE for k,v in vals if k>=med]; lo=[v-FEE for k,v in vals if k<med]
        if hi and lo:
            d=sum(hi)/len(hi)-sum(lo)/len(lo)
            R[f"D2_{key}"]={"hi_avg":round(sum(hi)/len(hi),1),"lo_avg":round(sum(lo)/len(lo),1),"delta":round(d,1)}
            print("    %-14s hi=%+.1f lo=%+.1f delta=%+.1f"%(key,sum(hi)/len(hi),sum(lo)/len(lo),d))
    return R

def run_E(marks,ev200,months):
    print("\n=== E: REGIME robustness ===")
    R={}
    for lbl,cond in (("deep_btc7d<-300",lambda e:e["btc7d"]<-300),("mild_btc7d>=-300",lambda e:e["btc7d"]>=-300)):
        g=[gg for e in ev200 if cond(e) and (gg:=gross_hold(marks,e["ts"])) is not None]
        R[f"E3_{lbl}"]=stat(g,lbl,months); ps(f"E3_{lbl}",R[f"E3_{lbl}"])
    # E2 month + recent half
    bym=defaultdict(list)
    for e in ev200:
        g=gross_hold(marks,e["ts"])
        if g is not None: bym[e["month"]].append(g-FEE)
    print("  E2 ay stabilite:")
    for m in sorted(bym):
        v=bym[m]; print("    %s N=%-3d WR=%-5.0f%% avg=%+.1f sum=%+.0f"%(m,len(v),100*sum(1 for x in v if x>0)/len(v),sum(v)/len(v),sum(v)))
    # E1 funding-time proximity: hours histogram
    print("  E1 saat dagilimi (hour>=17):")
    hh=defaultdict(int)
    for e in ev200: hh[e["hour"]]+=1
    for h in range(17,24): print("    %02d:00 N=%d"%(h,hh.get(h,0)))
    return R

def run_F(conn,marks,ev200,months,now,start):
    print("\n=== F: PORTFOLIO / diversification ===")
    R={}
    # F1 SHORT_NOISY BTC1M d5 h180 (paper diversifier) trades
    def short_out(ts,hold):
        e=entry_px(marks,ts)
        if not e: return None
        r=marks.at_or_before(ts+hold)
        return -(float(r[1])-e[1])/e[1]*1e4 if r else None
    sn=[]
    for e in ev200:
        nt=e.get("noisy_ts")
        if nt is None: continue
        btc=liq_max(conn,"BTCUSDT","SELL",nt+5*60_000,nt+30*60_000)
        if btc>=1_000_000:
            g=short_out(nt,180*60_000)
            if g is not None: sn.append((nt,g))
    R["F1_short_noisy"]=stat(noov([(t,v) for t,v in sn],180*60_000),"SHORT_NOISY BTC1M d5 h180 no-overlap",months)
    R["F1_short_noisy"]["per_month"]=round(len(noov(sn,180*60_000))/months,1); ps("F1_short_noisy",R["F1_short_noisy"])
    # F1 portfolio: hour17 LONG + short_noisy non-overlapping (single slot)
    longp=[(e["ts"],gross_hold(marks,e["ts"]),HOLD,"L") for e in ev200 if gross_hold(marks,e["ts"]) is not None]
    shortp=[(t,v,180*60_000,"S") for t,v in sn]
    allp=sorted(longp+shortp)
    busy=-1; combo=[]
    for ts,v,hold,_ in allp:
        if v is None: continue
        if ts>=busy: combo.append(v); busy=ts+hold
    R["F1_portfolio"]=stat(combo,"hour17 LONG + SHORT_NOISY portfolio",months)
    R["F1_portfolio"]["per_month"]=round(len(combo)/months,1); ps("F1_portfolio",R["F1_portfolio"])
    # F2 SHORT hour split
    for lbl,cond in (("h17-19",lambda t:17<=hod(t)<20),("h20-23",lambda t:hod(t)>=20)):
        g=[v for t,v in sn if cond(t)]
        R[f"F2_short_{lbl}"]=stat(g,f"short_noisy {lbl}",months); ps(f"F2_short_{lbl}",R[f"F2_short_{lbl}"])
    # F3 cross-asset: SOL & BTC own SELL cascade -> LONG hour17 6h
    for sym,thr in (("SOLUSDT",200_000),("BTCUSDT",1_000_000)):
        liqs=load_liquidations(conn,sym,"SELL",start,now)
        if not liqs: print(f"    {sym}: no liq"); continue
        ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(float(thr),),accel_window_sec=30)
        m2=load_mark_index(conn,sym)
        pairs=[]
        for a in ancs:
            ts=int(a.anchor_ts_ms)
            if float(a.running_notional)<thr or m2.at_or_after(ts) is None or hod(ts)<17: continue
            b4=mbps(conn,"BTCUSDT",ts,4*3600_000); b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000)
            if not (b4<0 or b7<0): continue
            e=m2.at_or_after(ts); r=m2.at_or_before(ts+HOLD)
            if e and r and float(e[1])>0: pairs.append((ts,(float(r[1])-float(e[1]))/float(e[1])*1e4))
        R[f"F3_{sym}_own"]=stat(noov(pairs),f"{sym} own cascade LONG hour17 6h",months)
        R[f"F3_{sym}_own"]["per_month"]=round(len(noov(pairs))/months,1); ps(f"F3_{sym}_own",R[f"F3_{sym}_own"])
    return R

def md(sec,meta):
    L=["# S34 Early-Signal + Management Gauntlet","",
       f"> Baz: hour17 200K LONG (T0, 6h). Evren {meta['n']} event, {meta['months']:.1f} ay, FEE={int(FEE)}bps.",
       f"> LOOKAHEAD YASAK · no-overlap+holdout+MC. Tarih {datetime.now(timezone.utc):%Y-%m-%d}","",
       "Kolon: N /ay WR Avg TOTAL Worst mc_p WF",""]
    hdr="| Test | N | /ay | WR | Avg | TOTAL | Worst | mc_p | WF |"; sepp="|---|--:|--:|--:|--:|--:|--:|--:|--:|"
    titles={"A":"A) Erken/Anticipatory Giris","B":"B) Giris Kalitesi","C":"C) Management",
            "D":"D) Conviction Sizing","E":"E) Regime Robustness","F":"F) Portfolio"}
    for s in ["A","B","C","D","E","F"]:
        L+=[f"## {titles[s]}","",hdr,sepp]
        for k,v in sec.get(s,{}).items():
            if isinstance(v,dict) and v.get("n",0)>0 and "wr" in v:
                L.append("| %s | %d | %.1f | %.1f%% | %+.1f | %s | %s | %s | %s |"%(k,v["n"],v.get("per_month",0),v["wr"],v["avg"],v.get("total"),v.get("worst"),v.get("mc_p","?"),v.get("wf","-")))
        L.append("")
    L+=["---","*Script: tools/research_s34_early_mgmt_gauntlet.py*"]
    return "\n".join(L)

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 Early+Mgmt Gauntlet ===")
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        marks=load_mark_index(conn,"ETHUSDT")
        print("build 200K events...")
        ev=build(conn,marks,200_000,now,start,rich=True)
        span=[e["ts"] for e in ev]; TM=max(1.0,(span[-1]-span[0])/86_400_000/30.0)
        print(f"  events={len(ev)} months={TM:.2f}")
        sec={}
        sec["A"]=run_A(conn,marks,ev,TM,now,start)
        sec["B"]=run_B(marks,ev,TM)
        sec["C"]=run_C(marks,ev,TM); sec["C"].update(run_C3(conn,marks,ev,TM))
        sec["D"]=run_D(marks,ev,TM)
        sec["E"]=run_E(marks,ev,TM)
        sec["F"]=run_F(conn,marks,ev,TM,now,start)
    meta={"n":len(ev),"months":round(TM,2)}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"sections":sec,"meta":meta},indent=2,default=str),encoding="utf-8")
    OM.write_text(md(sec,meta),encoding="utf-8")
    print(f"\nJSON: {OJ}\nMD:   {OM}\nDone.")

if __name__=="__main__": main()
