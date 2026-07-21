"""S34 — S34_ALL.db uzerinde kapsamli meta-analiz. SADECE SQL okur, live'a dokunmaz.
Cikti: reports/research/s34/S34_ALL_INSIGHTS.md
Bolumler: konsensus, overfit-dedektor, celiski, mezarlik, research-vs-paper,
          feature-konsensus, kapsama haritasi, Kelly/risk kalibrasyonu.
"""
from __future__ import annotations
import sqlite3, statistics
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/"reports"/"research"/"s34"
DB=RES/"S34_ALL.db"; OUT=RES/"S34_ALL_INSIGHTS.md"
con=sqlite3.connect(DB); cur=con.cursor()

# tum research satirlari (bir kere cek)
ROWS=cur.execute("SELECT report,key,label,direction,n,per_month,wr,avg_bps,total_bps,worst_bps,tail_n,mdd_bps,mc_p,wf,ho_avg FROM research_results").fetchall()
COL=["report","key","label","direction","n","per_month","wr","avg","total","worst","tail","mdd","mc_p","wf","ho_avg"]
R=[dict(zip(COL,r)) for r in ROWS]
def txt(r): return (str(r["key"] or "")+" "+str(r["label"] or "")).lower()

CONCEPTS={
 "silence":["silence","sil "],"echo":["echo"],"sync":["sync"],"whale":["whale"],
 "shelf":["shelf"],"rv5m":["rv5","rv_"],"density":["density","d24","dens"],"ofi":["ofi"],
 "funding":["funding","fund"],"hour17":["hour17","hour>=17","h17","hour 17"],"regime":["regime","btc4h","btc7d"],
 "prebuild":["prebuild"],"score>=3":["score>=3","s>=3","ge3","score3","score_3"],
 "score>=4":["score>=4","s>=4","ge4","score4","score_4"],"score>=5":["score>=5","s>=5","ge5","score_5"],
 "composite":["composite","conviction"],"100k":["100k","100 k"],"short":["short"],"noisy":["noisy"],
 "btc1m":["btc1m","btc>=1m","btc1000","1_000_000"],"btc2m":["btc2m","btc>=2m","btc2000"],
 "cross-asset":["eth+btc","eth+sol","cross","sol","btc-led"],"deep7d":["deep7d","deep","btc7d<"],
 "T15/bounce":["t15","bounce","t+15"],"reversal":["reversal"],"fade":["fade"],"buy-side":["buy_","eth buy","buy-side","buy fade"],
 "prof-target":["target","trailing","breakeven"],"limit-entry":["limit"],"scale-in":["scale"],
 "vol-compress":["vol_dec","compress","rv_low"],"navigation":["navig","nav_","state grid","n1_","after_win"],
}
def rows_with(toks): return [r for r in R if any(t in txt(r) for t in toks)]

L=["# S34 — Tum SQL Data: Kapsamli Meta-Analiz","",
   f"> Kaynak: `S34_ALL.db` (SADECE okundu). {len(R)} research satir / {len(set(r['report'] for r in R))} rapor. ",
   f"> Uretim: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Live/.env/sizing DOKUNULMADI.",""]

# ---- 1. KONSENSUS (concept x raporlar) ----
L+=["## 1) Konsensus Tarayici (concept, raporlar arasi)","",
    "Bir concept ne kadar cok RAPORDA + yuksek WR + yuksek anlamlilik ile geciyorsa o kadar guvenilir.","",
    "| Concept | #satir | #rapor | ort WR | med total | %anlamli(mc<=.05) | %pozitif |","|---|--:|--:|--:|--:|--:|--:|"]
cons=[]
for name,toks in CONCEPTS.items():
    rr=[r for r in rows_with(toks) if r["n"] and r["n"]>=8]
    if len(rr)<3: continue
    reps=len(set(r["report"] for r in rr))
    wrs=[r["wr"] for r in rr if r["wr"] is not None]; tots=[r["total"] for r in rr if r["total"] is not None]
    mcs=[r["mc_p"] for r in rr if r["mc_p"] is not None]
    sig=sum(1 for x in mcs if x<=0.05)/len(mcs) if mcs else 0
    pos=sum(1 for r in rr if (r["total"] or 0)>0)/len(rr)
    cons.append((name,len(rr),reps,statistics.mean(wrs) if wrs else 0,statistics.median(tots) if tots else 0,sig,pos))
cons.sort(key=lambda x:-(x[5]*x[2]))  # anlamlilik x rapor-yayginligi
for name,ns,reps,wr,mt,sig,pos in cons:
    L.append("| %s | %d | %d | %.0f%% | %+.0f | %.0f%% | %.0f%% |"%(name,ns,reps,wr,mt,100*sig,100*pos))

# ---- 2. OVERFIT DEDEKTOR ----
L+=["","## 2) Overfit / Suphe Dedektoru","",
    "Kucuk-N dev-sayi, cok-yuksek WR dusuk-N, veya mc_p olmayan buyuk-total.","",
    "| Report | key | N | WR | total | mc_p | flag |","|---|---|--:|--:|--:|--:|---|"]
ov=[]
for r in R:
    if not r["n"]: continue
    fl=[]
    if r["n"]<10 and (r["total"] or 0)>800: fl.append("kucuk-N-dev-total")
    if (r["wr"] or 0)>=92 and r["n"]<15: fl.append("WR>=92-dusuk-N")
    if r["mc_p"] is None and (r["total"] or 0)>1200: fl.append("mc_p-yok-buyuk-total")
    if fl: ov.append((r,";".join(fl)))
ov.sort(key=lambda x:-(x[0]["total"] or 0))
for r,fl in ov[:20]:
    L.append("| %s | %s | %d | %s | %+.0f | %s | %s |"%(r["report"],str(r["key"])[:40],r["n"],("%.0f%%"%r["wr"]) if r["wr"] is not None else "-",r["total"] or 0,("%.3f"%r["mc_p"]) if r["mc_p"] is not None else "yok",fl))
L.append(f"\n> Toplam suphe flag'i: {len(ov)} satir.")

# ---- 3. CELISKI BULUCU ----
L+=["","## 3) Celiski Bulucu (ayni concept, zit sonuc)","",
    "Hem guclu-pozitif hem olu sonucu olan concept'ler = metodoloji hassas (dikkat).","",
    "| Concept | anlamli-pozitif | olu(mc>.5 veya total<0) | celiski? |","|---|--:|--:|---|"]
for name,toks in CONCEPTS.items():
    rr=[r for r in rows_with(toks) if r["n"] and r["n"]>=8]
    if len(rr)<4: continue
    sigpos=sum(1 for r in rr if r["mc_p"] is not None and r["mc_p"]<=0.05 and (r["total"] or 0)>0)
    dead=sum(1 for r in rr if (r["mc_p"] is not None and r["mc_p"]>0.5) or (r["total"] or 0)<0)
    if sigpos>=3 and dead>=3:
        L.append("| %s | %d | %d | EVET |"%(name,sigpos,dead))

# ---- 4. MEZARLIK (don't retest) ----
grave=[r for r in R if r["n"] and r["n"]>=15 and ((r["mc_p"] is not None and r["mc_p"]>0.6) or (r["avg"] is not None and r["avg"]<-8))]
seen=set(); gl=[]
for r in sorted(grave,key=lambda r:(r["total"] or 0)):
    lab=(r["label"] or r["key"] or "")[:60]
    if lab in seen: continue
    seen.add(lab); gl.append(r)
L+=["","## 4) Mezarlik — Reddedilen Hipotezler (bir daha test etme)","",
    f"mc_p>0.6 veya avg<-8bps (N>=15). Toplam {len(grave)} satir, {len(gl)} benzersiz.","",
    "| Report | label | N | WR | avg | mc_p |","|---|---|--:|--:|--:|--:|"]
for r in gl[:20]:
    L.append("| %s | %s | %d | %s | %+.0f | %s |"%(r["report"],(r["label"] or r["key"])[:44],r["n"],("%.0f%%"%r["wr"]) if r["wr"] is not None else "-",r["avg"] or 0,("%.3f"%r["mc_p"]) if r["mc_p"] is not None else "-"))

# ---- 5. RESEARCH vs PAPER ----
L+=["","## 5) Research vs Gercek Paper (tahmin-gerceklik)","",
    "| Paper signal | paper N | paper WR | paper avg | ~research WR | ~research avg | not |","|---|--:|--:|--:|--:|--:|---|"]
PP=cur.execute("SELECT signal,n,wr,avg_net_bps FROM paper_signal_pnl").fetchall()
SIGMATCH={"LONG_SILENCE":["silence"],"SHORT_NOISY":["short","noisy"],"SHORT_NEITHER":["short","neither","btc2m","btc>=2m"]}
for sig,n,wr,avg in PP:
    toks=SIGMATCH.get(sig,[sig.lower()])
    rr=[r for r in rows_with(toks) if r["n"] and r["n"]>=15 and r["wr"] is not None]
    rwr=statistics.median([r["wr"] for r in rr]) if rr else None
    ravg=statistics.median([r["avg"] for r in rr if r["avg"] is not None]) if rr else None
    note="gap: gercek<research (silence lookahead)" if sig=="LONG_SILENCE" else ("gercek~research" if rwr and abs((wr or 0)-rwr)<12 else "fark var")
    L.append("| %s | %d | %.0f%% | %+.1f | %s | %s | %s |"%(sig,n,wr,avg or 0,("%.0f%%"%rwr) if rwr else "-",("%+.0f"%ravg) if ravg is not None else "-",note))

# ---- 6. FEATURE KONSENSUS (kazanan config'lerde vote) ----
win=[r for r in R if r["n"] and r["n"]>=15 and (r["wr"] or 0)>=70 and r["mc_p"] is not None and r["mc_p"]<=0.05]
L+=["","## 6) Feature Konsensus Siralamasi (kazanan config'lerde geçme)","",
    f"Kazanan config = N>=15, WR>=70, mc_p<=0.05 ({len(win)} satir). Feature ne kadar cok geciyorsa o kadar konsensus.","",
    "| Feature | kazananda #satir | tum-veride #satir | lift |","|---|--:|--:|--:|"]
feat_toks={"sync":["sync"],"whale":["whale"],"rv5m":["rv5","rv_"],"shelf":["shelf"],"be_ratio":["be_ratio","be "],
 "density":["density","d24"],"ofi":["ofi"],"hour17":["hour17","h17","hour>=17"],"regime":["regime","btc7d","btc4h"],
 "silence":["silence"],"echo":["echo"],"funding":["funding"],"prebuild":["prebuild"],"score":["score","conviction","composite"]}
fr=[]
for f,toks in feat_toks.items():
    w=sum(1 for r in win if any(t in txt(r) for t in toks))
    a=sum(1 for r in R if r["n"] and any(t in txt(r) for t in toks))
    lift=(w/max(1,len(win)))/(a/max(1,len(R))) if a else 0
    fr.append((f,w,a,lift))
fr.sort(key=lambda x:-x[3])
for f,w,a,lift in fr:
    L.append("| %s | %d | %d | %.2fx |"%(f,w,a,lift))

# ---- 7. KAPSAMA HARITASI ----
L+=["","## 7) Kapsama Haritasi (asset x yon)","","| Asset | LONG satir | SHORT satir | notr |","|---|--:|--:|--:|"]
for asset,ts in (("ETH",["eth"]),("BTC",["btc"]),("SOL",["sol"])):
    rr=[r for r in R if any(t in txt(r) for t in ts) and r["n"]]
    lo=sum(1 for r in rr if r["direction"]=="LONG"); sh=sum(1 for r in rr if r["direction"]=="SHORT"); nu=len(rr)-lo-sh
    L.append("| %s | %d | %d | %d |"%(asset,lo,sh,nu))

# ---- 8. KELLY / RISK KALIBRASYONU (canli/paper adaylari) ----
L+=["","## 8) Kelly / Risk Kalibrasyonu (conviction sleeve'leri)","",
    "Kelly-yaklasik f* = WR - (1-WR)/R, R=avg_kazanc/avg_kayip (worst proxy). Sadece rehber.","",
    "| Config | N | WR | avg | worst | onerilen risk-tier |","|---|--:|--:|--:|--:|---|"]
KELLY=[("S34_HORIZON","results.V.V_s8_ge3_full"),("S34_HORIZON","results.V.V_s8_ge4_full"),
 ("S34_FRONTIER","results.M.M2_s5_full"),("S34_HORIZON","results.V.V_100K_s3_noov"),
 ("S34_CONVICTION_COMPOSITE","results.gate4_noov")]
for rep,key in KELLY:
    r=next((x for x in R if x["report"]==rep and x["key"]==key),None)
    if not r: continue
    wr=(r["wr"] or 0)/100; worst=abs(r["worst"] or 200); avg=r["avg"] or 0
    R_ratio=(avg+ (1-wr)*worst)/worst if worst>0 else 1  # kaba kazanc/kayip
    f=wr-(1-wr)/max(0.5,R_ratio)
    tier="AGGRESSIVE" if f>0.4 else ("MODERATE" if f>0.2 else "CONSERVATIVE")
    L.append("| %s | %d | %.0f%% | %+.0f | %+.0f | %s (f~%.2f) |"%(key.split(".")[-1],r["n"],r["wr"] or 0,avg,r["worst"] or 0,tier,max(0,f)))

L+=["","---","*Uretim: tools/analyze_s34_all_sql.py — sadece S34_ALL.db okundu.*"]
OUT.write_text("\n".join(L),encoding="utf-8")
con.close()
print("Rapor:",OUT,"| satir:",len(L))
