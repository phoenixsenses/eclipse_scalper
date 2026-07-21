"""S34 — Tum alpha ailesi KONSOLIDE karar raporu. SADECE SQL'den okur (S34_ALL.db).
Block-bootstrap CI SQL semasinda kolon degil -> S34_HORIZON.json'dan cekilir (okuma).
Live/.env/sizing'e DOKUNMAZ. Cikti: reports/research/s34/S34_ALPHA_DECISION_REPORT.md
"""
from __future__ import annotations
import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/"reports"/"research"/"s34"
DB=RES/"S34_ALL.db"; OUT=RES/"S34_ALPHA_DECISION_REPORT.md"

con=sqlite3.connect(DB); cur=con.cursor()
def row(report,key):
    r=cur.execute("SELECT n,per_month,wr,avg_bps,total_bps,worst_bps,tail_n,mdd_bps,mc_p,wf,ho_avg FROM research_results WHERE report=? AND key=?",(report,key)).fetchone()
    if not r: return None
    return dict(zip(["n","per_month","wr","avg","total","worst","tail","mdd","mc_p","wf","ho_avg"],r))

# block-bootstrap from horizon json
BOOT={}
try:
    h=json.load(open(RES/"S34_HORIZON.json",encoding="utf-8"))
    BOOT=h.get("results",{}).get("R",{}).get("R_bootstrap",{})
except Exception: pass

# in-sample yield ledger real paper
PP={r[0]:r for r in cur.execute("SELECT signal,n,wr,sum_net_bps,avg_net_bps,best_bps,worst_bps FROM paper_signal_pnl")}

def flags(oos,lookahead,noov,realfill,doublecount):
    return f"IS/OOS:{oos} · lookahead:{lookahead} · no-ov:{noov} · fill:{realfill} · tekrar-sayim:{doublecount}"

# FAMILIES: (family_title, decision, [ (report,key,label,oos,lookahead,noov,realfill,double) ... ], extra_note)
FAM=[
 ("1) LONG composite v3 (hour17 + 8-sinyal conviction)","LIVE ADAYI (once paper-forward)",[
   ("S34_HORIZON","results.V.V_s8_ge3_full","score>=3 full (v3, 8-sig)","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_HORIZON","results.V.V_s8_ge3_TEST","score>=3 TEST","OOS/holdout","yok","HAYIR","mark","hayir"),
   ("S34_HORIZON","results.V.V_s8_ge4_full","score>=4 full","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_HORIZON","results.V.V_s8_ge4_TEST","score>=4 TEST","OOS/holdout","yok","HAYIR","mark","hayir"),
   ("S34_CONVICTION_COMPOSITE","results.gate3_noov","score>=3 no-overlap","in-sample","yok","EVET","mark","hayir"),
   ("S34_CONVICTION_COMPOSITE","results.gate4_noov","score>=4 no-overlap","in-sample","yok","EVET","mark","hayir"),
   ("S34_CONVICTION_COMPOSITE","results.base_noov","hour17 baz (composite yok) noov","in-sample","yok","EVET","mark","hayir"),
 ],"Block-bootstrap (S34_HORIZON): score>=3 noov N=%s obs_avg=%s 5%%CI=%s 95%%CI=%s P(avg<0)=%s. Weighted-sizing (composite) flat->4.2x (S34_CONVICTION_COMPOSITE.results.weighted)."%(BOOT.get('n'),BOOT.get('obs_avg'),BOOT.get('ci5'),BOOT.get('ci95'),BOOT.get('p_below0'))),

 ("2) 100K composite (frekans genisletme)","PAPER (holdout gecti, frekans 2x)",[
   ("S34_HORIZON","results.V.V_100K_s3_full","100K score>=3 full","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_HORIZON","results.V.V_100K_s3_TEST","100K score>=3 TEST","OOS/holdout","yok","HAYIR","mark","hayir"),
   ("S34_HORIZON","results.V.V_100K_s3_noov","100K score>=3 no-overlap","in-sample","yok","EVET","mark","hayir"),
   ("S34_FRONTIER","results.P.P1_mini_fizzled","100K mini fizzled (buyumeyen)","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.P.P1_mini_grew","100K mini grew (200K'ya ulasan)","in-sample","yok","HAYIR","mark","EVET(overlap)"),
 ],"Fizzle eden mini'ler buyuyenden iyi bounce yapiyor (P1)."),

 ("3) Conviction sleeves score>=2/3/4/5 (esik sweep)","score>=3/4 PAPER; >=5 premium sleeve",[
   ("S34_FRONTIER","results.M.M2_s2_full","score>=2 full","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.M.M2_s2_noov","score>=2 no-overlap","in-sample","yok","EVET","mark","hayir"),
   ("S34_FRONTIER","results.M.M2_s3_full","score>=3 full","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.M.M2_s3_noov","score>=3 no-overlap","in-sample","yok","EVET","mark","hayir"),
   ("S34_FRONTIER","results.M.M2_s4_full","score>=4 full","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.M.M2_s4_noov","score>=4 no-overlap","in-sample","yok","EVET","mark","hayir"),
   ("S34_FRONTIER","results.M.M2_s5_full","score>=5 full","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.M.M2_s5_noov","score>=5 no-overlap","in-sample","yok","EVET","mark","hayir"),
   ("S34_CONVICTION_COMPOSITE","results.score_2","score=2 (dagilim)","in-sample","yok","HAYIR","mark","hayir"),
   ("S34_CONVICTION_COMPOSITE","results.score_5","score=5 (dagilim)","in-sample","yok","HAYIR","mark","hayir"),
 ],"Monoton: score arttikca WR/avg yukselir, frekans duser."),

 ("4) whale_lo (retail = kucuk trade sinyali)","PAPER (holdout WR94, v3'e alindi)",[
   ("S34_HORIZON","results.V.V_whale_lo_TEST","whale_lo TEST","OOS/holdout","yok","HAYIR","mark","hayir"),
   ("S34_HORIZON","results.V.V_whale_hi_TEST","whale_hi TEST (karsit)","OOS/holdout","yok","HAYIR","mark","hayir"),
   ("S34_FRONTIER","results.S.S1_whale_lo","whale_lo full","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_HORIZON","results.I.I_sync+whale_lo","sync & whale_lo (interaction)","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_HORIZON","results.I.I_rv+whale_lo","rv & whale_lo","in-sample","yok","HAYIR","mark","EVET(overlap)"),
 ],"OOS'ta WR94 (N=17). Interaction: iki sinyal birden WR80-87."),

 ("5) SHORT confirm-entry 13-17 (time-machine sleeve)","PAPER (kucuk-N, kirilgan)",[
   ("S34_SHORT_CONVICTION","results.S3.S3_h13-17","SHORT hour 13-17","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_SHORT_CONVICTION","results.S3.S3_h17-24","SHORT hour 17-24 (olu)","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_SHORT_CONVICTION","results.S5.S5_entry_confirm","entry@BTC-confirm (tradeable)","in-sample","YOK","HAYIR","mark","EVET(overlap)"),
   ("S34_SHORT_CONVICTION","results.S5.S5_entry_noisy","entry@noisy (LOOKAHEAD)","in-sample","VAR","HAYIR","mark","EVET(overlap)"),
   ("S34_SHORT_CONVICTION","results.S1.S1_btc2000K_h120","BTC>=2M h120","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_HORIZON","results.T.T_short_only","SHORT 13-17 (horizon)","in-sample","yok","HAYIR","mark","EVET(overlap)"),
 ],"KRITIK: tradeable confirm-entry (+32,mc0.176) noisy-entry LOOKAHEAD'inden (+91) cok zayif. N kucuk."),

 ("6) Cross-asset sync (senkron cascade)","RESEARCH-ONLY (composite sync_ratio zaten iceriyor)",[
   ("S34_FRONTIER","results.D.D2_ETH+BTC(>=500K)","ETH+BTC simultane","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.D.D2_ETH+SOL(>=100K)","ETH+SOL simultane","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.D.D2_ETH-only(btc<500K)","ETH-only","in-sample","yok","HAYIR","mark","EVET(overlap)"),
 ],"sync_ratio composite'te var; ayri route degil, dogrulayici."),

 ("7) deep7d navigation (rejim x skor haritasi)","RESEARCH-ONLY (navigasyon kurali)",[
   ("S34_FRONTIER","results.N.N1_h17-19_deep7d_s>=4","h17-19 deep + s>=4","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.N.N1_h17-19_deep7d_s<4","h17-19 deep + s<4 (olu)","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.N.N1_h20-23_deep7d_s>=4","h20-23 deep + s>=4","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.M.M1_deep7d_s>=4","deep7d & s>=4","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.N.N2_after_win","kazanan sonrasi (momentum)","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_FRONTIER","results.N.N2_after_loss","kaybeden sonrasi","in-sample","yok","HAYIR","mark","EVET(overlap)"),
 ],"Derin rejimde yuksek-skor sart (s<4 deep = olu). Kazanctan sonra momentum."),

 ("8) Funding veto (<60m = olu)","LIVE-KURAL (composite'e veto olarak alindi)",[
   ("S34_DEEP_QUESTIONS","results.Q8.Q8_<60m to fund","<60m funding (olu)","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_DEEP_QUESTIONS","results.Q8.Q8_60-240m","60-240m","in-sample","yok","HAYIR","mark","EVET(overlap)"),
   ("S34_DEEP_QUESTIONS","results.Q8.Q8_>240m",">240m","in-sample","yok","HAYIR","mark","EVET(overlap)"),
 ],"<60m ölü -> veto. Composite paper route'da uygulandi."),

 ("9) 6h horizon (hold suresi)","LIVE-KURAL (6h kesin optimal)",[
   ("S34_HORIZON","results.H.H_6h","score>=4 6h","in-sample","yok","HAYIR","mark","event-reuse(coklu-horizon)"),
   ("S34_HORIZON","results.H.H_12h","score>=4 12h","in-sample","yok","HAYIR","mark","event-reuse"),
   ("S34_HORIZON","results.H.H_24h","score>=4 24h","in-sample","yok","HAYIR","mark","event-reuse"),
   ("S34_HORIZON","results.H.H_48h","score>=4 48h (olu)","in-sample","yok","HAYIR","mark","event-reuse"),
   ("S34_FRONTIER","results.D.D3_6h","score>=3 6h","in-sample","yok","HAYIR","mark","event-reuse"),
 ],"WR 6h'te tepe (48h olu). Ayni event coklu-horizon'da tekrar olculuyor (satirlar arasi)."),
]

def fmt(v,fmtstr,default="-"):
    return (fmtstr%v) if v is not None else default

L=["# S34 — Tum Alpha Ailesi: Konsolide Karar Raporu","",
   f"> Kaynak: `S34_ALL.db` (SADECE SQL'den okundu). Block-bootstrap: `S34_HORIZON.json`.",
   f"> Uretim: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Live/.env/sizing'e DOKUNULMADI.","",
   "**Metrik notu:** median bps ve top3-removed cumulative research_results semasinda YOK (mark ‘-’). ",
   "block-bootstrap CI sadece composite score>=3 icin var (S34_HORIZON R). no-overlap /ay = `_noov` satirlarindaki /ay.","",
   "**Flag lejantı:** IS=in-sample, OOS=holdout/test. lookahead ‘VAR’ = geleceğe bakiyor (guvenilmez). ",
   "no-ov EVET = tek-pozisyon uygulandi. fill=mark (gercek borsa fill degil; paper=mark-fiyat). tekrar-sayim EVET(overlap)=ayni event zaman-penceresinde ust uste sayilabilir.","",
]
hdr="| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |"
sep="|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
for title,decision,items,note in FAM:
    L+=[f"## {title}",f"**Karar: {decision}**","",hdr,sep]
    flagrows=[]
    for report,key,label,oos,la,noov,rf,dc in items:
        r=row(report,key)
        if not r:
            L.append(f"| {label} (BULUNAMADI) | - | - | - | - | - | - | - | - | - |"); continue
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |"%(
            label,r["n"],fmt(r["per_month"],"%.1f"),fmt(r["wr"],"%.0f%%"),fmt(r["avg"],"%+.0f"),
            fmt(r["total"],"%+.0f"),fmt(r["worst"],"%+.0f"),(str(r["tail"]) if r["tail"] is not None else "-"),
            fmt(r["mc_p"],"%.3f"),(r["wf"] or "-")))
        flagrows.append((label,oos,la,noov,rf,dc))
    L+=["","**Provenance / risk flag'leri:**","","| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |","|---|---|---|---|---|---|"]
    for lb,oos,la,noov,rf,dc in flagrows:
        L.append(f"| {lb} | {oos} | {la} | {noov} | {rf} | {dc} |")
    L+=["",f"> {note}",""]

# gercek paper (ledger)
L+=["## Gercek Paper Trade PnL (ledger — mark-fill, gercege en yakin)","",
    "| Signal | N | WR | cum bps | avg bps | best | worst |","|---|--:|--:|--:|--:|--:|--:|"]
for sig,r in sorted(PP.items(),key=lambda x:-(x[1][1] or 0)):
    L.append("| %s | %d | %s%% | %+.0f | %+.1f | %+.0f | %+.0f |"%(sig,r[1],r[2],r[3] or 0,r[4] or 0,r[5] or 0,r[6] or 0))
L.append("> Not: yeni route'lar (hour17/composite/100K) henuz paper trade biriktirmedi (restart bekliyor).")

# NET KARAR TABLOSU
L+=["","---","# NET KARAR TABLOSU","",
 "| Alpha / route | Durum | Neden |","|---|---|---|",
 "| **LONG composite v3 (score>=3)** | **LIVE ADAYI** (once paper-forward) | OOS TEST WR78-82, block-boot 5%%CI +26.3 P(<0)=0.0, monoton, 6-sinyal additive |",
 "| **LONG composite score>=4/5 sleeve** | **LIVE ADAYI (premium, dusuk frekans)** | OOS WR85-90, worst kucuk; sleeve olarak |",
 "| **100K composite** | **SHADOW/PAPER** | Holdout gecti (TEST WR72, noov 11.8/ay) ama in-sample tail; forward gerek |",
 "| **whale_lo (8. sinyal)** | **SHADOW/PAPER (v3'e alindi)** | Holdout WR94 (N=17 kucuk); forward dogrulama |",
 "| **funding veto (<60m)** | **LIVE-KURAL** | <60m ölü; risksiz veto |",
 "| **6h horizon** | **LIVE-KURAL** | 6h WR tepe, 48h olu; kesin |",
 "| **deep7d navigation** | **RESEARCH-ONLY** | Navigasyon kurali (rejimde skor sart); ayri route degil |",
 "| **cross-asset sync** | **RESEARCH-ONLY** | sync_ratio zaten composite'te; dogrulayici |",
 "| **SHORT confirm-entry 13-17** | **SADECE SHADOW (kirilgan)** | Tradeable-entry zayif (+32 mc0.176), noisy-entry lookahead; N kucuk |",
 "| SHORT conviction skoru (gate2/3) | **REDDEDILECEK / OVERFIT SUPHESI** | N=6, TEST N=2; OOS dogrulanamaz |",
 "| Limit-entry -20bps | **REDDEDILECEK** | Q2 gercek-fill: %36 fill -> EV/signal +34.5 < market +74.2 |",
 "| score>=2 all-hours / hour<17 | **REDDEDILECEK** | hour<17 skor olu (mc0.458); hour gate sart |",
 "| 48h horizon | **REDDEDILECEK** | avg -31 mc0.706 (edge decay) |",
 "| LONG_SILENCE (arsiv) | **REDDEDILDI (lookahead)** | silence T+30'da bilinir; provisional -137 mc0.514 |",
 "",
 "**Ozet:** Ana deploy hatti = hour17 LONG (canli) -> composite v3 score>=3 (paper-forward, live aday) + funding-veto + 6h + weighted-sizing. ",
 "Frekans genisletme = 100K composite (paper). Premium = score>=4/5 sleeve. SHORT hala kirilgan (shadow). ",
 "Reddedilenler: limit-entry, SHORT-skor-gate, all-hours-skor, 48h, silence-lookahead.",
 "","---","*Uretim: tools/report_s34_alpha_decision.py — sadece S34_ALL.db + S34_HORIZON.json okundu.*"]

OUT.write_text("\n".join(L),encoding="utf-8")
con.close()
print("Rapor:",OUT)
print("Satir:",len(L))
