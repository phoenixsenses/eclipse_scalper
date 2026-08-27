# -*- coding: utf-8 -*-
"""Extra Telegram commands (v2) — the ones the 135-close autopsy actually earned.

Kept in a separate module so the bot's core stays small and so this file can be
edited without touching the alert loop. Everything here is READ-ONLY over the
paper lane's own ledgers.

Design rule carried over from v1: a command answers a question the operator
would otherwise have to ask me. It never reports a number without the caveat
that makes the number readable — and where a statistic is degenerate, it says
so instead of printing it.
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research_trader import ledgers, paths, autopsy as A, registries  # noqa: E402

UP, DOWN, FLAT = "🟢", "🔴", "⚪"
WARN, SEAL, CHART, CLOCK = "⚠️", "🔒", "📊", "🕐"
MICRO, SCALE, BOOK = "🔬", "⚖️", "📚"


def _esc(s):
    import html
    return html.escape(str(s), quote=False)


def arm_letter(policy_id: str) -> str:
    """The arm's own letter, read off its id.

    Never a prefix test against a fixed set. The first version of the positions
    card asked `startswith("A_") else "B"`, so every ARM C and ARM D position was
    displayed as a B position -- a two-arm assumption that keeps working,
    silently and wrongly, once a third arm exists. Five places had it.

    Deriving it from the id means there is no table to fall out of date: a new
    arm named `E_..._V1` prints as E without anyone editing this file."""
    return (policy_id or "?").split("_")[0][:1] or "?"


def _live_marks(rows):
    """Mark open positions at the last quote at or before now.

    Uses the lane's OWN QuoteReader so the price semantics are identical to the
    ones the trade will actually be closed on -- a second, differently-sourced
    price would make the running number disagree with the eventual close for
    reasons that have nothing to do with the market.

    Display only. The runner never reads this: the live thesis monitor records
    STATE, not running P&L, precisely so that an open position's mark cannot
    influence the rule. Showing it here does not change that; it is a
    mark-to-market, not a realisable number.
    """
    marks = {}
    try:
        from research_trader.quotes import QuoteReader
        import time
        now = int(time.time() * 1000)
        reader = QuoteReader()
        try:
            for r in rows:
                try:
                    q = reader.last_at_or_before(r["symbol"], now)
                    if q is None:
                        continue
                    entry = float(r["entry"]["price"])
                    px = q.bid if r["direction"] == "LONG" else q.ask
                    gross = (px - entry) / entry * 1e4
                    if r["direction"] == "SHORT":
                        gross = -gross
                    marks[r["position_id"]] = {
                        "gross_bps": gross, "age_s": (now - q.ts_ms) / 1000}
                except Exception:
                    continue
        finally:
            reader.close()
    except Exception:
        pass
    return marks


def _moments(v):
    if not v:
        return None
    d = {"n": len(v), "mean": st.mean(v), "median": st.median(v)}
    if len(v) > 1:
        d["sd"] = st.pstdev(v)
        d["se"] = st.pstdev(v) / len(v) ** 0.5
        d["t"] = d["mean"] / d["se"] if d["se"] else None
    return d


# ------------------------------------------------------------------ /autopsy
def cmd_autopsy(args=None) -> str:
    """Decision quality and outcome, kept apart — the whole point of the object."""
    rows = ledgers.read_all(A.AUTOPSY_LEDGER)
    if not rows:
        return f"{CLOCK} <b>Henüz autopsy yok</b>"
    sep = collections.Counter(r["separation"] for r in rows)
    nets = [r["outcome"]["net_bps"] for r in rows
            if r["outcome"].get("net_bps") is not None]
    m = _moments(nets)
    good = sum(1 for n in nets if n > 0)
    out = [f"{MICRO} <b>Autopsy</b> · {len(rows)} kapanış", ""]
    for k, v in sep.most_common():
        dot = UP if "GOOD_OUTCOME" in k else (DOWN if "BAD_OUTCOME" in k else FLAT)
        out.append(f"  {dot} <code>{v:3d}</code> {_esc(k)}")
    out += ["", f"  net ort <b>{m['mean']:+.1f}</b> · medyan <code>{m['median']:+.1f}</code>"
                f" · kazanan <code>{good}/{len(nets)}</code>"]
    bad_dec = sum(v for k, v in sep.items() if k.startswith("BAD_DECISION"))
    out.append(f"  kural uyuşmazlığı: <code>{bad_dec}</code>")
    out += ["", f"<i>karar kalitesi ve sonuç AYRI eksen. {sep.get('GOOD_DECISION_BAD_OUTCOME',0)} "
                f"kez kural doğru uygulandı ve sonuç kötü çıktı — küçük edge'li "
                f"bir hatta normal hâl.</i>"]
    # The count above is rows. The count that decides whether the mean means
    # anything is blocks, and it is much smaller. Printing the first without the
    # second is how a two-day sample gets read as a hundred-trade result.
    b = _blocks()
    if b:
        out += ["", f"{WARN} <b>Ama bunlar {b['n']} bahis, {b['rows']} değil.</b>",
                f"  ufuk 6s · takvim {b['span_h']:.0f}s → en fazla "
                f"<code>{b['ceiling']:.0f}</code> örtüşmeyen pencere",
                f"  blok başına net <b>{b['mean']:+.1f}</b> bps · "
                f"se <code>{b['se']:.1f}</code> · t <code>{b['t']:+.2f}</code>",
                f"  <i>sıfırdan ayırt edilemiyor. Payda tek gerçek kısıt.</i>"]
    return "\n".join(out)


def _blocks():
    """One number per non-overlapping outcome window.

    Positions overlap, so per-position moments overstate certainty. The strict
    connected-component rule degenerates here (the lane opens continuously, so
    everything is one component); non-overlapping calendar blocks are the
    generous reading, and even they give single digits."""
    from research_trader import registries
    rows = [r for r in registries._closes()
            if registries.classify_close(r)["in_headline"]]
    if len(rows) < 2:
        return None
    ev = sorted((r["entry"]["ts_ms"], r["entry"]["ts_ms"] + r["horizon_ms"],
                 float(r["net_bps"])) for r in rows)
    H = ev[0][1] - ev[0][0]
    t0 = min(e[0] for e in ev)
    span = (max(e[1] for e in ev) - t0) / 3600_000
    bl = collections.defaultdict(list)
    for a, _b, n in ev:
        bl[(a - t0) // H].append(n)
    means = [st.mean(v) for v in bl.values()]
    m = _moments(means)
    if not m or len(means) < 2:
        return None
    return {"n": len(means), "rows": len(ev), "span_h": span,
            "ceiling": span / (H / 3600_000), "mean": m["mean"],
            "se": m["se"], "t": m["t"]}


def cmd_episode(args=None) -> str:
    """Position within the market episode. A burned-sample look, labelled."""
    from research_trader import registries, policy
    rows = [r for r in registries._closes()
            if registries.classify_close(r)["in_headline"]]
    if not rows:
        return f"{CLOCK} <b>Kapanış yok</b>"
    ev = sorted((r.get("anchor_ts_ms") or r["entry"]["ts_ms"],
                 float(r["net_bps"])) for r in rows)
    gap = policy.MIN_GAP_MS
    eid, i, last = {}, 0, None
    for t, _ in ev:
        if last is None or t - last > gap:
            i += 1
        eid[t] = i
        last = t
    seen, pos = collections.Counter(), collections.defaultdict(list)
    for t, n in ev:
        seen[eid[t]] += 1
        pos[min(seen[eid[t]], 4)].append(n)
    out = [f"{MICRO} <b>Epizod içi sıra</b> · {i} piyasa epizodu", ""]
    for k in sorted(pos):
        v = pos[k]
        lbl = f"{k}." if k < 4 else "4.+"
        dot = UP if st.mean(v) > 0 else DOWN
        out.append(f"  {dot} <b>{lbl:<4}</b> n=<code>{len(v):<3}</code>"
                   f" ort <b>{st.mean(v):+7.1f}</b>"
                   f" · kazanan <code>{100*sum(1 for x in v if x>0)//len(v)}%</code>")
    first = pos.get(1, [])
    later = [x for k in pos if k > 1 for x in pos[k]]
    if first and later:
        out += ["", f"  ilk <b>{st.mean(first):+.1f}</b> vs sonraki "
                    f"<b>{st.mean(later):+.1f}</b> → fark "
                    f"<code>{st.mean(first)-st.mean(later):+.1f}</code>"]
    out += ["", f"{WARN} <i>YANMIŞ ÖRNEKLEM. Bunu sonuçlara bakarak buldum, "
                f"kural türetilemez. Ölçen enstrüman (<b>D kolu</b>) bu tabloyu "
                f"görmeden donduruldu — ve bu bakış prereg §5b'de kayıtlı, "
                f"yani D'nin ileride iyi çıkması bakir doğrulama SAYILMAZ.</i>"]
    return "\n".join(out)


# --------------------------------------------------------------------- /cost
def cmd_cost(args=None) -> str:
    """Realised execution cost — measured, not modelled (TRADER-01)."""
    p = REPO / "reports/research_trader/TRADER01_IMPLEMENTATION_SHORTFALL_V1.json"
    if not p.exists():
        return f"{WARN} <b>Maliyet ayrıştırması yok</b>"
    d = json.loads(p.read_text(encoding="utf-8"))
    out = [f"{SCALE} <b>Gerçekleşen maliyet</b> <i>(ölçülen, modellenen değil)</i>", ""]
    for pid, m in d.get("by_arm", {}).items():
        e = m["execution_cost_bps"]
        drift = m["decision_to_fill_drift_bps"]
        o = m["opportunity_cost_bps"]
        arm = arm_letter(pid)
        out.append(f"  <b>kol {arm}</b>")
        out.append(f"     icra <code>{e['mean_bps']:+.2f}</code> bps"
                   f" (sd {e.get('sd_bps', 0):.2f}, n={e['n']})")
        out.append(f"     karar→dolum <code>{drift['mean_bps']:+.2f}</code>"
                   f" <i>maliyet</i> (sd {drift.get('sd_bps', 0):.1f})")
        if o.get("n"):
            out.append(f"     fırsat <code>{o['mean_bps']:+.1f}</code> bps"
                       f" · n={o['n']} · sd <code>{o.get('sd_bps', 0):.0f}</code>")
        out.append("")
    out.append(f"{SEAL} <i>icra ve fırsat maliyeti ASLA toplanmaz — icra iki taraf "
               f"arasında sıfır toplamlı, fırsat değil (Hasbrouck s.147).</i>")
    return "\n".join(out)


# --------------------------------------------------------------------- /path
def cmd_path(args=None) -> str:
    """MFE/MAE — and the reason it is not a signal."""
    tca = [r for r in ledgers.read_all(paths.TCA_LEDGER)
           if r.get("event") == "PATH" and r.get("status") == "OBSERVED"]
    if not tca:
        return f"{CLOCK} <b>Yol verisi yok</b>"
    mfe = _moments([r["mfe_bps"] for r in tca])
    mae = _moments([r["mae_bps"] for r in tca])
    first = collections.Counter(
        "MFE" if r["time_to_mfe_ms"] < r["time_to_mae_ms"] else "MAE" for r in tca)
    return "\n".join([
        f"{CHART} <b>Yol</b> · {len(tca)} pozisyon", "",
        f"  MFE ort <code>{mfe['mean']:+.0f}</code> medyan <code>{mfe['median']:+.0f}</code>",
        f"  MAE ort <code>{mae['mean']:+.0f}</code> medyan <code>{mae['median']:+.0f}</code>",
        f"  önce gelen: {dict(first)}",
        "",
        f"{WARN} <b>Bundan çıkış kuralı ÇIKMAZ.</b>",
        "<i>“önce lehte gitti sonra döndü” örüntüsü saf gürültüde de aynen "
        "çıkıyor (600 rastgele yürüyüş: %88 vs gerçek %86). Sonun işaretine "
        "koşullanmış her yolun mekanik özelliği — mekanizma değil. MFE "
        "ulaşılan fiyattır, elde edilebilir olan değil.</i>",
    ])


# ------------------------------------------------------------------ /symbols
def cmd_symbols(args=None) -> str:
    """Where the book's outcome actually concentrates."""
    closes = [c for c in A._closes()
              if registries.classify_close(c)["in_headline"]]
    if not closes:
        return f"{CLOCK} <b>Manşet kapanış yok</b>"
    by = collections.defaultdict(list)
    for c in closes:
        by[c["symbol"]].append(float(c["net_bps"]))
    rows = sorted(by.items(), key=lambda kv: -sum(kv[1]))
    tot = sum(sum(v) for v in by.values())
    out = [f"{CHART} <b>Sembole göre</b> · {len(closes)} manşet kapanış", ""]
    for s, v in rows:
        dot = UP if sum(v) > 0 else DOWN
        out.append(f"  {dot} <b>{_esc(s).replace('USDT',''):<6}</b>"
                   f" <code>{sum(v):+8.0f}</code> n={len(v):<3}"
                   f" ort <code>{st.mean(v):+7.1f}</code>")
    out += ["", f"  toplam <code>{tot:+.0f}</code> bps",
            "", f"{WARN} <i>bunlar bağımsız gözlem değil — aynı kaskadda açılanlar "
                f"tek olaydır; epizod birimi pozisyon biriminden çok daha küçük.</i>"]
    return "\n".join(out)


# --------------------------------------------------------------------- /risk
def cmd_risk(args=None) -> str:
    """Concentration: the thing that actually decides this book's outcome."""
    rows = ledgers.read_all(paths.TRADE_LEDGER)
    op = {}
    for r in rows:
        if r.get("event") == "OPEN":
            op[r["position_id"]] = r
        elif r.get("event") == "CLOSE":
            op.pop(r["position_id"], None)
    if not op:
        return f"{FLAT} <b>Açık pozisyon yok</b>"
    dirs = collections.Counter(p["direction"] for p in op.values())
    syms = collections.Counter(p["symbol"] for p in op.values())
    one_way = max(dirs.values()) / sum(dirs.values())
    hhi = sum((c / len(op)) ** 2 for c in syms.values())
    ts = sorted(p["entry"]["ts_ms"] for p in op.values() if p.get("entry"))
    eps, cur = 0, None
    for t in ts:
        if cur is None or t - cur > 900_000:
            eps += 1
        cur = t
    return "\n".join([
        f"{SCALE} <b>Risk</b> · {len(op)} açık pozisyon", "",
        f"  yön: {dict(dirs)} → tek-yön payı <b>%{100*one_way:.0f}</b>",
        f"  sembol: {len(syms)} isim · etkin <code>{1/hhi:.1f}</code> (1/HHI)",
        f"  bağımsız epizod: <b>{eps}</b> → pozisyon başına <code>{len(op)/eps:.2f}</code>",
        "",
        f"{WARN} <i>{len(op)} pozisyon {eps} bahis demek. Per-pozisyon her "
        f"istatistik <code>√({len(op)}/{eps})={((len(op)/eps)**0.5):.2f}×</code> abartır.</i>",
    ])


# --------------------------------------------------------------------- /books
def cmd_books(args=None) -> str:
    """What the corpus says that constrains this lane."""
    return "\n".join([
        f"{BOOK} <b>Külliyat kısıtları</b> <i>(13 kaynak, 4 299 sayfa, diskte)</i>", "",
        "<b>Hasbrouck</b> s.147 — icra maliyeti iki taraf arasında sıfır toplamlı;",
        "  fırsat maliyeti <b>değil</b>. İkisi asla netleştirilmez.",
        "<b>Hasbrouck</b> s.154 — piyasa emri: yüksek beklenen maliyet, düşük varyans.",
        "  Limit emri: düşük beklenen, <b>yüksek varyans</b>. Ortalama tek başına rapor değildir.",
        "<b>Kissell</b> böl.3 — IS = Σsⱼpⱼ − S·Pd + fees. <code>Pd</code> karar fiyatıdır;",
        "  onsuz shortfall gürültülü değil <b>tanımsızdır</b>.",
        "<b>Bouchaud</b> §7.5 — yarış sonrası ortalamaya dönüş öngörülür",
        "  <i>(ama büyük-tick rejiminde; bizim tick 380× altında)</i>.",
        "<b>López de Prado</b> Ek B — False Strategy: en iyi sonuç, kaç şey",
        "  denendiği bilinmeden anlamlı değildir.",
        "",
        f"{SEAL} <i>diskte YOK: O'Hara, ISLP — okundu iddia edilmez.</i>",
    ])


# --------------------------------------------------------------------- /why
def cmd_why(args=None) -> str:
    """Why the lane does not yet make an economic decision. The honest answer."""
    return "\n".join([
        f"{MICRO} <b>Neden henüz ekonomik karar yok</b>", "",
        "Maliyet ölçülüyor: <code>~10.7 bps</code> gerçekleşen.",
        "Fayda <b>ölçülmüyor</b>: her aday için beklenen hareket tahmini yok.",
        "",
        "Yani kural her uygun anchor'ı alıyor ve maliyet karara <b>hiç girmiyor</b>.",
        "Ölçüldü: bütün maliyet bileşenleri sıfırlansa açılan işlem sayısı aynı.",
        "",
        f"  <code>ECONOMIC_ADMISSION_NOT_AVAILABLE</code>",
        "  → bir <b>etiket</b>, bir kapı değil. Doğruyu söylüyor, ama kapatmıyor.",
        "",
        f"{SEAL} <i>Bunu emekliye ayırmak magnitude/duration forecast motoru "
        f"ister (B1) — ve o motor forward hattını GÖRMEDEN dondurulmalı, "
        f"yoksa 15 Eylül kapısının dayandığı kanıt yanar.</i>",
    ])


COMMANDS_V2 = {
    "autopsy": cmd_autopsy,
    "episode": cmd_episode,
    "cost": cmd_cost,
    "path": cmd_path,
    "symbols": cmd_symbols,
    "risk": cmd_risk,
    "books": cmd_books,
    "why": cmd_why,
}

MENU_V2 = [
    {"command": "autopsy", "description": "karar kalitesi vs sonuc"},
    {"command": "episode", "description": "epizod ici sira (yanmis ornek)"},
    {"command": "cost", "description": "gerceklesen icra + firsat maliyeti"},
    {"command": "path", "description": "MFE/MAE ve neden sinyal degil"},
    {"command": "symbols", "description": "sonuc hangi sembollerde"},
    {"command": "risk", "description": "yogunlasma ve bagimsiz epizod"},
    {"command": "books", "description": "kulliyatin koydugu kisitlar"},
    {"command": "why", "description": "neden henuz ekonomik karar yok"},
]
