"""S34 Signal Mining — cascade etrafinda ONCESI / ANI / SONRASI microstructure taramasi.

hour17 200K baz (LONG, hold 6h). Sezgi otesi: bid/ask, book imbalance, spread,
OI (stale-atlandi), basis, agg-trade flow, liq-cascade sekli, vol, funding, price-action.
Her feature holdout (kron. 70/30) ile skorlanir: TRAIN favorable yon secer, TEST raporlar.
POST feature'lar T+5 delayed-entry ile (tradeable), PRE/AT feature'lar T0 ile.

Cikti: reports/research/s34/S34_SIGNAL_MINING.json + .md
"""
from __future__ import annotations
import bisect, json, math, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB=ROOT/"data"/"microstructure.db"; OUT=ROOT/"reports"/"research"/"s34"
OJ=OUT/"S34_SIGNAL_MINING.json"; OM=OUT/"S34_SIGNAL_MINING.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=500; HOLD=6*3600_000; TM=4.5; TRAIN=0.70
random.seed(42)

def _s(c,sql,p=()):
    r=c.execute(sql,p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0
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
def funding(c,ts):
    r=c.execute("SELECT funding_rate FROM funding_rates WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
def book_at(c,s,ts):
    r=c.execute("SELECT bid_qty,ask_qty,spread_pct,book_imbalance,bid_depth_usd,ts_ms FROM book_ticker WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(s,ts)).fetchone()
    if not r or (ts-int(r[5]))>5*60_000: return None
    return {"bid_qty":float(r[0]),"ask_qty":float(r[1]),"spread_pct":float(r[2] or 0),
            "imb":float(r[3] or 0),"bid_depth":float(r[4] or 0),
            "ba_ratio":(float(r[0])/float(r[1])) if float(r[1])>0 else None}
def ofiv(c,s,lo,hi):
    r=c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),COUNT(*) FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<?",(s,lo,hi)).fetchone()
    if not r or r[0] is None: return None,None
    b,se=float(r[0]),float(r[1]); t=b+se
    return ((b-se)/t if t>0 else 0.0), int(r[2])
def basis(c,ts):
    sp=c.execute("SELECT spot_price,ts_ms FROM spot_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    mk=c.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
    if not sp or not mk or (ts-int(sp[1]))>10*60_000 or float(sp[0])<=0: return None
    return (float(mk[0])-float(sp[0]))/float(sp[0])*1e4
def hod(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).hour
def sess(ts):
    h=hod(ts); return "EUROPE" if 7<=h<13 else ("US" if 13<=h<21 else "OFF")

def mkidx_price(marks,ts):
    r=marks.at_or_after(ts); return float(r[1]) if r and float(r[1])>0 else None
def hold_from(marks,ts,hold=HOLD):
    e=marks.at_or_after(ts)
    if not e or float(e[1])<=0: return None
    r=marks.at_or_before(ts+hold)
    return (float(r[1])-float(e[1]))/float(e[1])*1e4 if r else None

# ---- stats ----
def mcp(v,a):
    if len(v)<4: return None
    r=random.Random(0); ct=sum(1 for _ in range(MC) if sum(r.choice([-1,1])*abs(x) for x in v)/len(v)>=a)
    return round(ct/MC,3)
def blk(vals,fee=FEE):
    if not vals: return {"n":0}
    net=[x-fee for x in vals]; n=len(net); w=sum(1 for x in net if x>0); a=sum(net)/n
    return {"n":n,"wr":round(100*w/n,1),"avg":round(a,1),"total":round(sum(net),0),"mc_p":mcp(net,a)}

# ---- feature list (window, key) ----
FEATS=[
    # ONCESI (pre)
    ("pre","book_imb","bid/ask imbalance @T0"),
    ("pre","spread","spread_pct @T0"),
    ("pre","bid_depth","bid depth USD @T0"),
    ("pre","ba_ratio","bid_qty/ask_qty @T0"),
    ("pre","ofi_5m","agg-trade OFI pre-5m"),
    ("pre","ofi_15m","agg-trade OFI pre-15m"),
    ("pre","trades_5m","agg-trade count pre-5m"),
    ("pre","eth_5m","ETH ret pre-5m"),
    ("pre","eth_15m","ETH ret pre-15m"),
    ("pre","eth_1h","ETH ret pre-1h"),
    ("pre","rv5m","realized vol 5m"),
    ("pre","vol_dec","vol_decile"),
    ("pre","funding","funding_rate"),
    ("pre","basis","spot-perp basis bps"),
    ("pre","btc5m","BTC ret 5m"),
    ("pre","prebuild","prebuildup 30m count"),
    ("pre","density24","24h cascade density"),
    ("pre","sync_ratio","sync_sell/rn"),
    ("pre","be_ratio","BTC_conc/rn"),
    # ANI (at cascade)
    ("at","rn","cascade running_notional"),
    ("at","accel","running_accel"),
    ("at","rate","liq/sec rate"),
    ("at","liq_count","liq count"),
    ("at","dominance","max_single/rn"),
    ("at","max_single","max single liq"),
    ("at","casc_drop","ETH drop during cascade bps"),
    # SONRASI (post, T+5 delayed entry)
    ("post","ofi_post5","agg-trade OFI post-5m (buyers)"),
    ("post","reclaim5","price reclaim vs anchor @T+5 bps"),
    ("post","imb_post5","book imbalance @T+5"),
    ("post","biddep_rec5","bid depth change 0->5m"),
    ("post","followon5","follow-on liq 1-5m notional"),
    ("post","btc_post5","BTC ret 0->5m post"),
]

def build(conn,marks,now,start):
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    evs=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or marks.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
        e1=mbps(conn,"ETHUSDT",ts,3600_000) or 0
        if (e1>20 and b4>50) or sess(ts)=="EUROPE" or not (b4<0 or b7<0) or hod(ts)<17: continue
        f={}
        bk=book_at(conn,"ETHUSDT",ts)
        f["book_imb"]=bk["imb"] if bk else None; f["spread"]=bk["spread_pct"] if bk else None
        f["bid_depth"]=bk["bid_depth"] if bk else None; f["ba_ratio"]=bk["ba_ratio"] if bk else None
        o5,t5=ofiv(conn,"ETHUSDT",ts-5*60_000,ts); o15,_=ofiv(conn,"ETHUSDT",ts-15*60_000,ts)
        f["ofi_5m"]=o5; f["ofi_15m"]=o15; f["trades_5m"]=float(t5) if t5 is not None else None
        f["eth_5m"]=mbps(conn,"ETHUSDT",ts,5*60_000); f["eth_15m"]=mbps(conn,"ETHUSDT",ts,15*60_000); f["eth_1h"]=e1
        vs=conn.execute("SELECT rv_5m,vol_decile FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
        f["rv5m"]=float(vs[0]) if vs and vs[0] is not None else None; f["vol_dec"]=float(vs[1]) if vs and vs[1] is not None else None
        f["funding"]=funding(conn,ts); f["basis"]=basis(conn,ts); f["btc5m"]=mbps(conn,"BTCUSDT",ts,5*60_000)
        f["prebuild"]=float(lcnt(conn,"ETHUSDT","SELL",ts-30*60_000,ts-1000,PROP))
        f["density24"]=float(lcnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,200_000))
        sk=lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
        f["sync_ratio"]=sk/rn if rn>0 else 0; f["be_ratio"]=lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts)/rn if rn>0 else 0
        f["rn"]=rn; f["accel"]=float(a.running_accel); f["liq_count"]=float(a.running_liq_count)
        f["rate"]=float(a.running_liq_count)/max(1.0,(ts-int(a.first_ts_ms))/1000.0)
        f["dominance"]=float(a.running_single_liq_dominance); f["max_single"]=float(a.max_single_notional)
        pA=mkidx_price(marks,int(a.first_ts_ms)); pT=mkidx_price(marks,ts)
        f["casc_drop"]=(pT-pA)/pA*1e4 if pA and pT else None
        # POST
        op5,_=ofiv(conn,"ETHUSDT",ts,ts+5*60_000); f["ofi_post5"]=op5
        p5=marks.at_or_before(ts+5*60_000)
        f["reclaim5"]=(float(p5[1])-pT)/pT*1e4 if p5 and pT else None
        bk5=book_at(conn,"ETHUSDT",ts+5*60_000)
        f["imb_post5"]=bk5["imb"] if bk5 else None
        f["biddep_rec5"]=(bk5["bid_depth"]-bk["bid_depth"]) if (bk5 and bk) else None
        f["followon5"]=lsum(conn,"ETHUSDT","SELL",ts+60_000,ts+5*60_000)
        f["btc_post5"]=(lambda a,b:(b-a)/a*1e4 if a and b else None)(mkidx_price(marks,ts),(marks.at_or_before(ts+5*60_000) or [None,None])[1])
        ev={"ts":ts,"f":f}
        ev["y_t0"]=hold_from(marks,ts); ev["y_t5"]=hold_from(marks,ts+5*60_000,HOLD)
        if ev["y_t0"] is None: continue
        evs.append(ev)
    evs.sort(key=lambda e:e["ts"])
    return evs

def screen(evs):
    n=len(evs); cut=int(n*TRAIN); tr,te=evs[:cut],evs[cut:]
    rows=[]
    for win,key,desc in FEATS:
        ykey="y_t5" if win=="post" else "y_t0"
        tv=[(e["f"].get(key),e[ykey]) for e in tr if e["f"].get(key) is not None and e[ykey] is not None]
        ev2=[(e["f"].get(key),e[ykey]) for e in te if e["f"].get(key) is not None and e[ykey] is not None]
        allv=[(e["f"].get(key),e[ykey]) for e in evs if e["f"].get(key) is not None and e[ykey] is not None]
        if len(tv)<10 or len(ev2)<6 or len(set(v for v,_ in tv))<3:
            continue
        med=sorted(v for v,_ in tv)[len(tv)//2]
        tr_hi=[y-FEE for v,y in tv if v>=med]; tr_lo=[y-FEE for v,y in tv if v<med]
        fav="hi" if (sum(tr_hi)/len(tr_hi) if tr_hi else -9e9)>=(sum(tr_lo)/len(tr_lo) if tr_lo else -9e9) else "lo"
        te_fav=[y for v,y in ev2 if (v>=med)==(fav=="hi")]
        te_oth=[y for v,y in ev2 if (v>=med)!=(fav=="hi")]
        all_fav=[y for v,y in allv if (v>=med)==(fav=="hi")]
        sb=blk(te_fav); so=blk(te_oth); sa=blk(all_fav)
        delta=round((sb.get("avg") or 0)-(so.get("avg") or 0),1)
        rows.append({"win":win,"key":key,"desc":desc,"fav":fav,"cut":round(med,4),
                     "test":sb,"test_other_avg":so.get("avg"),"delta":delta,"full":sa})
    rows.sort(key=lambda r:-(r["delta"]))
    return rows

def make_md(rows,meta):
    L=["# S34 Signal Mining — cascade oncesi/ani/sonrasi","",
       f"> hour17 200K baz, {meta['n']} event, {meta['months']:.1f} ay. Holdout 70/30 (TRAIN yon secer, TEST raporlar).",
       f"> PRE/AT=T0 entry, POST=T+5 delayed. FEE={int(FEE)}bps. Tarih {datetime.now(timezone.utc):%Y-%m-%d}","",
       "Delta = TEST(favorable yon avg) - TEST(diger yon avg). Full = tum-veri favorable yon.","",
       "| Rank | Window | Signal | Fav | TEST N | TEST WR | TEST avg | Delta | Full N | Full avg | Full mc_p |",
       "|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for i,r in enumerate(rows,1):
        t=r["test"]; fu=r["full"]
        L.append("| %d | %s | %s | %s | %d | %s | %s | %+.1f | %d | %s | %s |"%(
            i,r["win"],r["desc"],r["fav"],t.get("n",0),
            (str(t.get("wr"))+"%") if t.get("wr") is not None else "-",
            ("%+.1f"%t["avg"]) if t.get("avg") is not None else "-",
            r["delta"],fu.get("n",0),
            ("%+.1f"%fu["avg"]) if fu.get("avg") is not None else "-",
            fu.get("mc_p","?")))
    L+=["","---","*Script: tools/research_s34_signal_mining.py*"]
    return "\n".join(L)

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 Signal Mining ===")
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        marks=load_mark_index(conn,"ETHUSDT")
        print("build events + features...")
        evs=build(conn,marks,now,start)
        span=[e["ts"] for e in evs]; TM=max(1.0,(span[-1]-span[0])/86_400_000/30.0)
        print(f"  events={len(evs)} months={TM:.2f}")
        rows=screen(evs)
    meta={"n":len(evs),"months":round(TM,2)}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"rows":rows,"meta":meta},indent=2,default=str),encoding="utf-8")
    OM.write_text(make_md(rows,meta),encoding="utf-8")
    print("\n=== RANKED (delta desc) ===")
    print("  %-5s %-26s %-4s %6s %7s %7s %7s %6s"%("win","signal","fav","TEST_N","TESTwr","TESTavg","delta","fullmc"))
    for r in rows:
        t=r["test"]; fu=r["full"]
        print("  %-5s %-26s %-4s %6d %6s%% %+6.1f %+7.1f %6s"%(
            r["win"],r["desc"][:26],r["fav"],t.get("n",0),str(t.get("wr")),
            t.get("avg") or 0,r["delta"],str(fu.get("mc_p"))))
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")

if __name__=="__main__": main()
