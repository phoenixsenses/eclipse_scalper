"""S34 Conviction Composite — robust sinyalleri TEK skora birlestir + validate.

Mining/early-mgmt'ten gecen robust sinyaller:
  sync_ratio(hi), rv5m(hi), density24(hi), ofi_pre(buyers), be_ratio(moderate; extreme veto),
  book ask-heavy(imbalance lo). POST teyit: bid-depth rebuild 0->5m(hi).
Baz: hour17 200K LONG hold 6h. Kron. 70/30 holdout (esik TRAIN, rapor TEST).

Testler:
  1  Skor dagilimi -> WR/avg/total (monoton mu?)
  2  Gate score>=K: full + TEST-OOS + no-overlap + MC
  3  Conviction-weighted sizing (unit=score) vs flat
  4  + LIMIT entry -20bps (score>=3)
  5  + POST bid-rebuild teyit (score>=3, T+5 delayed)
  6  FINAL onerilen composite + verdict

Cikti: reports/research/s34/S34_CONVICTION_COMPOSITE.json + .md
"""
from __future__ import annotations
import bisect, json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB=ROOT/"data"/"microstructure.db"; OUT=ROOT/"reports"/"research"/"s34"
OJ=OUT/"S34_CONVICTION_COMPOSITE.json"; OM=OUT/"S34_CONVICTION_COMPOSITE.md"
PROP=50_000.0; LB=400*24*3600_000; FEE=5.0; MC=1000; HOLD=6*3600_000; TM=4.5; TRAIN=0.70
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
def book_at(c,s,ts):
    r=c.execute("SELECT book_imbalance,bid_depth_usd,ts_ms FROM book_ticker WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(s,ts)).fetchone()
    if not r or (ts-int(r[2]))>5*60_000: return None
    return {"imb":float(r[0] or 0),"bid_depth":float(r[1] or 0)}
def ofi(c,s,lo,hi):
    r=c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<?",(s,lo,hi)).fetchone()
    if not r or r[0] is None: return None
    b,se=float(r[0]),float(r[1]); t=b+se; return (b-se)/t if t>0 else 0.0
def hod(ts): return datetime.fromtimestamp(ts/1000,tz=timezone.utc).hour
def sess(ts):
    h=hod(ts); return "EUROPE" if 7<=h<13 else ("US" if 13<=h<21 else "OFF")
def eprice(marks,ts):
    r=marks.at_or_after(ts); return (int(r[0]),float(r[1])) if r and float(r[1])>0 else None
def hold_from(marks,ts,hold=HOLD):
    e=eprice(marks,ts)
    if not e: return None
    r=marks.at_or_before(ts+hold); return (float(r[1])-e[1])/e[1]*1e4 if r else None

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
            "per_month":round(n/m,1),"worst":round(sv[0],1),"tail_n":sum(1 for x in net if x<-100),
            "mdd":round(mdd,0),"mc_p":mcp(net,a),"wf":wf(net)}
def noov(pairs,hold=HOLD):
    busy=-1; out=[]
    for ts,v in sorted(pairs):
        if ts>=busy: out.append(v); busy=ts+hold
    return out
def ps(k,v):
    if not v or v.get("n",0)==0: print("    %-34s N=0"%k[:34]); return
    print("    %-34s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s worst=%-7s mc_p=%s wf=%s"%(
        k[:34],v["n"],v.get("per_month",0),str(v["wr"])+"%",str(v["avg"])+"bps",str(v.get("total")),str(v.get("worst")),v.get("mc_p","?"),v.get("wf")))

def build(conn,marks,now,start):
    liqs=load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs=reconstruct_anchors(liqs,bucket_sec=300,min_gap_sec=900,thresholds=(200_000.0,),accel_window_sec=30)
    evs=[]
    for a in ancs:
        ts=int(a.anchor_ts_ms); rn=float(a.running_notional)
        if rn<200_000 or marks.at_or_after(ts) is None: continue
        b4=mbps(conn,"BTCUSDT",ts,4*3600_000) or 0; b7=mbps(conn,"BTCUSDT",ts,7*24*3600_000) or 0
        if ((mbps(conn,"ETHUSDT",ts,3600_000) or 0)>20 and b4>50) or sess(ts)=="EUROPE" or not (b4<0 or b7<0) or hod(ts)<17: continue
        sk=lsum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+lsum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
        bk=book_at(conn,"ETHUSDT",ts); bk5=book_at(conn,"ETHUSDT",ts+5*60_000)
        vs=conn.execute("SELECT rv_5m FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts,)).fetchone()
        f={"sync_ratio":sk/rn if rn>0 else 0,
           "rv5m":float(vs[0]) if vs and vs[0] is not None else None,
           "density24":float(lcnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,200_000)),
           "ofi_pre":ofi(conn,"ETHUSDT",ts-5*60_000,ts),
           "be_ratio":lmax(conn,"BTCUSDT","SELL",ts-10*60_000,ts)/rn if rn>0 else 0,
           "imb":bk["imb"] if bk else None,
           "bid_rebuild":(bk5["bid_depth"]-bk["bid_depth"]) if (bk and bk5) else None}
        nt=lfirst(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP)
        ev={"ts":ts,"rn":rn,"f":f,"noisy_ts":nt,"y_t0":hold_from(marks,ts),"y_t5":hold_from(marks,ts+5*60_000)}
        if ev["y_t0"] is None: continue
        evs.append(ev)
    evs.sort(key=lambda e:e["ts"])
    return evs

def med(vals): s=sorted(vals); return s[len(s)//2] if s else 0

def build_score(tr):
    """TRAIN'de medyanlari sec -> score fonksiyonu dondur."""
    m={}
    for k in ("sync_ratio","rv5m","density24","ofi_pre","be_ratio"):
        vals=[e["f"][k] for e in tr if e["f"].get(k) is not None]
        m[k]=med(vals)
    im=[e["f"]["imb"] for e in tr if e["f"].get("imb") is not None]; m["imb"]=med(im)
    def score(e):
        f=e["f"]; s=0
        if f.get("sync_ratio") is not None and f["sync_ratio"]>=m["sync_ratio"]: s+=1
        if f.get("rv5m") is not None and f["rv5m"]>=m["rv5m"]: s+=1
        if f.get("density24") is not None and f["density24"]>=m["density24"]: s+=1
        if f.get("ofi_pre") is not None and f["ofi_pre"]>=0: s+=1  # buyers
        # be_ratio: moderate iyi, extreme(>=2) veto
        if f.get("be_ratio") is not None and f["be_ratio"]>=m["be_ratio"] and f["be_ratio"]<2.0: s+=1
        if f.get("imb") is not None and f["imb"]<=m["imb"]: s+=1  # ask-heavy
        return s
    return score,m

def limit_fill(marks,ts,k_bps):
    e=eprice(marks,ts)
    if not e: return None
    lvl=e[1]*(1-k_bps/1e4)
    for _,px in marks.slice_range(e[0],ts+15*60_000):
        if px<=lvl:
            r=marks.at_or_before(ts+HOLD); return (float(r[1])-lvl)/lvl*1e4 if r else None
    return None  # fill olmadi

def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass
    print("=== S34 Conviction Composite ===")
    with sqlite3.connect(f"file:{DB}?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now=int(datetime.now(tz=timezone.utc).timestamp()*1000); start=now-LB
        marks=load_mark_index(conn,"ETHUSDT")
        print("build..."); evs=build(conn,marks,now,start)
        span=[e["ts"] for e in evs]; TM=max(1.0,(span[-1]-span[0])/86_400_000/30.0)
        print(f"  events={len(evs)} months={TM:.2f}")
        cut=int(len(evs)*TRAIN); tr,te=evs[:cut],evs[:cut and cut:] and evs[cut:] or evs[cut:]
        te=evs[cut:]
        score,mth=build_score(tr)
        for e in evs: e["score"]=score(e)
        R={}

        print("\n[1] Skor dagilimi (T0 hold6h):")
        for sc in range(0,7):
            g=[e["y_t0"] for e in evs if e["score"]==sc]
            R[f"score_{sc}"]=stat(g,f"score={sc}",TM); ps(f"score_{sc}",R[f"score_{sc}"])

        print("\n[2] Gate score>=K (full + TEST-OOS + no-overlap):")
        for K in (3,4):
            full=[(e["ts"],e["y_t0"]) for e in evs if e["score"]>=K]
            R[f"gate{K}_full"]=stat([v for _,v in full],f"score>={K} full",TM); ps(f"gate{K}_full",R[f"gate{K}_full"])
            teg=[e["y_t0"] for e in te if e["score"]>=K]
            R[f"gate{K}_TEST"]=stat(teg,f"score>={K} TEST-OOS",TM*(1-TRAIN)); ps(f"gate{K}_TEST",R[f"gate{K}_TEST"])
            nv=noov(full); s=stat(nv,f"score>={K} no-overlap",TM); s["per_month"]=round(len(nv)/TM,1)
            R[f"gate{K}_noov"]=s; ps(f"gate{K}_noov",s)

        print("\n[3] Conviction-weighted sizing (unit=score+1) vs flat, no-overlap:")
        pairs=[(e["ts"],(e["y_t0"],e["score"]+1)) for e in evs]
        busy=-1; wn=0.0; un=0; fn=0.0; fc=0
        for ts,(y,u) in sorted(pairs):
            if ts>=busy: wn+=(y-FEE)*u; un+=u; fn+=(y-FEE); fc+=1; busy=ts+HOLD
        R["weighted"]={"label":"weighted vs flat","n":fc,"flat_total":round(fn,0),
                       "weighted_total":round(wn,0),"weighted_per_unit":round(wn/un,1) if un else None,"units":un}
        print(f"    flat_total={round(fn,0)}  weighted_total={round(wn,0)}  units={un}  per_unit={round(wn/un,1) if un else None}")

        print("\n[4] score>=3 + LIMIT entry -20bps:")
        lp=[(e["ts"],g) for e in evs if e["score"]>=3 and (g:=limit_fill(marks,e["ts"],20)) is not None]
        R["gate3_limit20_full"]=stat([v for _,v in lp],"score>=3 limit-20 full",TM); ps("gate3_limit20_full",R["gate3_limit20_full"])
        nv=noov(lp); s=stat(nv,"score>=3 limit-20 no-overlap",TM); s["per_month"]=round(len(nv)/TM,1)
        R["gate3_limit20_noov"]=s; ps("gate3_limit20_noov",s)

        print("\n[5] score>=3 + POST bid-rebuild teyit (T+5 delayed):")
        brm=med([e["f"]["bid_rebuild"] for e in tr if e["f"].get("bid_rebuild") is not None])
        pp=[(e["ts"],e["y_t5"]) for e in evs if e["score"]>=3 and e["f"].get("bid_rebuild") is not None and e["f"]["bid_rebuild"]>=brm and e["y_t5"] is not None]
        R["gate3_bidrebuild_full"]=stat([v for _,v in pp],"score>=3 + bid-rebuild T+5",TM); ps("gate3_bidrebuild_full",R["gate3_bidrebuild_full"])
        nv=noov(pp); s=stat(nv,"score>=3 bid-rebuild no-overlap",TM); s["per_month"]=round(len(nv)/TM,1)
        R["gate3_bidrebuild_noov"]=s; ps("gate3_bidrebuild_noov",s)

        print("\n[6] Karsilastirma: mevcut live (tum hour17) vs composite:")
        base=[(e["ts"],e["y_t0"]) for e in evs]
        nb=noov(base); sb=stat(nb,"hour17 base no-overlap",TM); sb["per_month"]=round(len(nb)/TM,1)
        R["base_noov"]=sb; ps("base_noov",sb)

    meta={"n":len(evs),"months":round(TM,2),"medians":{k:round(v,4) for k,v in mth.items()}}
    OUT.mkdir(parents=True,exist_ok=True)
    OJ.write_text(json.dumps({"results":R,"meta":meta},indent=2,default=str),encoding="utf-8")
    lines=["# S34 Conviction Composite","",f"> hour17 200K, {len(evs)} event, {TM:.1f} ay. Holdout 70/30. FEE={int(FEE)}bps.",
           f"> Skor: sync_ratio+rv5m+density24+ofi_pre+be_ratio(mod)+ask-heavy (0-6). Tarih {datetime.now(timezone.utc):%Y-%m-%d}","",
           "| Test | N | /ay | WR | Avg | TOTAL | Worst | mc_p | WF |","|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k,v in R.items():
        if isinstance(v,dict) and v.get("n",0)>0 and "wr" in v:
            lines.append("| %s | %d | %.1f | %.1f%% | %+.1f | %s | %s | %s | %s |"%(k,v["n"],v.get("per_month",0),v["wr"],v["avg"],v.get("total"),v.get("worst"),v.get("mc_p","?"),v.get("wf","-")))
    lines+=["","Weighted sizing: flat_total=%s weighted_total=%s (units=%s)"%(R["weighted"]["flat_total"],R["weighted"]["weighted_total"],R["weighted"]["units"]),
            "","---","*Script: tools/research_s34_conviction_composite.py*"]
    OM.write_text("\n".join(lines),encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")

if __name__=="__main__": main()
