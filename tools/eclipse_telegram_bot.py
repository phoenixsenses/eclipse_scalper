# -*- coding: utf-8 -*-
"""Eclipse Autonomous Research Trader -- Telegram alert bot.

A READ-ONLY notification surface over the paper lane's own append-only ledgers.
It cannot open, close, size or alter a position: it imports no broker, no
policy writer and no ledger-of-record writer, and every file it touches is
opened for reading.

Deliberately lives in tools/ and NOT in research_trader/. The trader's code
identity hashes every package file, so a notification module living there would
trip CODE_DRIFT and force a restart of a lane holding open positions -- the
exact cost recorded in SYSTEM_STATE 416.

The bot token is read from OUTSIDE the repo (D:/eclipse_secrets/telegram_bot.json)
because this tree has a public mirror. The token is never logged, never printed
and never written into any repo file.

Design brief from the operator: "göz yormayacak, renkli menkli" -- colour comes
from a small, consistent emoji vocabulary rather than from noise, one card per
event, and OPENs are summarised rather than announced one by one.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
SECRETS = Path("D:/eclipse_secrets/telegram_bot.json")
STATE = REPO / "runtime" / "telegram_bot_state.json"      # runtime/*_state.json is gitignored
TRADER = REPO / "data" / "research_trader"
HEARTBEAT = REPO / "runtime" / "research_trader" / "heartbeat.json"

API = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT = 20

# ---------------------------------------------------------------- vocabulary
# One meaning per symbol. A reader should never have to decode a legend.
UP, DOWN, FLAT = "🟢", "🔴", "⚪"
OPEN_, CLOSE_, VOID_ = "🔵", "🏁", "🚫"
WARN, HALT, OK_ = "⚠️", "🛑", "✅"
CLOCK, CHART, SEAL = "🕐", "📊", "🔒"


class BotRefused(RuntimeError):
    pass


# -------------------------------------------------------------------- config
def load_cfg() -> dict:
    if not SECRETS.exists():
        raise BotRefused(
            f"no bot config at {SECRETS}. Create it with "
            '{"bot_token": "...", "chat_id": null}')
    cfg = json.loads(SECRETS.read_text(encoding="utf-8"))
    if not cfg.get("bot_token"):
        raise BotRefused("bot_token missing from the config")
    return cfg


def save_cfg(cfg: dict) -> None:
    SECRETS.write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def _call(cfg: dict, method: str, _http_timeout: int | None = None, **params):
    """One Telegram call.

    `_http_timeout` must EXCEED any long-poll timeout in `params`, or the HTTP
    read times out while Telegram is still legitimately holding the connection
    open. The first version used a flat 20s against a 45s long poll, so every
    single getUpdates raised a read timeout: commands still worked (the updates
    stay queued) but every cycle logged an error and command latency rose."""
    r = requests.post(API.format(token=cfg["bot_token"], method=method),
                      json=params, timeout=_http_timeout or TIMEOUT)
    try:
        return r.json()
    except Exception:
        return {"ok": False, "description": f"non-JSON response {r.status_code}"}


def send(cfg: dict, text: str, *, silent: bool = False) -> dict:
    if not cfg.get("chat_id"):
        raise BotRefused("chat_id is not set; run --connect after messaging the bot")
    return _call(cfg, "sendMessage", chat_id=cfg["chat_id"], text=text,
                 parse_mode="HTML", disable_web_page_preview=True,
                 disable_notification=silent)


# --------------------------------------------------------------- ledger read
def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def trades() -> list[dict]:
    return _rows(TRADER / "trade_ledger.jsonl")


def heartbeat() -> dict:
    if not HEARTBEAT.exists():
        return {}
    try:
        return json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _depth_excess(row: dict) -> float:
    e = (row.get("entry") or {}).get("cost_parts") or {}
    x = (row.get("exit") or {}).get("cost_parts") or {}
    return max(e.get("depth_excess_multiple", 0.0), x.get("depth_excess_multiple", 0.0))


def classify(row: dict) -> str:
    """Same three classes the registries use. Kept in step deliberately."""
    if row.get("net_bps") is None:
        return "VOID"
    if _depth_excess(row) > 1.0:
        return "EXECUTION_NOT_IDENTIFIED"
    return "HEADLINE"


# ------------------------------------------------------------------ styling
def _bar(bps: float, width: int = 10) -> str:
    """A tiny magnitude bar. Caps at 300 bps so one outlier cannot fill it."""
    n = min(width, int(abs(bps) / 300 * width + 0.5))
    ch = "▰" if bps >= 0 else "▰"
    return (ch * n) + ("▱" * (width - n))


def _sign(bps: float) -> str:
    return UP if bps > 0 else (DOWN if bps < 0 else FLAT)


def _dur(ms: int) -> str:
    m = int(ms // 60000)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def arm_letter(policy_id: str) -> str:
    """The arm's own letter, read off its id -- never a prefix test against a
    fixed set. `"A" if startswith("A_") else "B"` silently displayed every ARM C
    and ARM D row as a B row. Deriving it means no table can fall out of date.

    Defined in the core rather than the optional extension: the alert loop
    renders cards, and a missing extension must not break an alert."""
    return (policy_id or "?").split("_")[0][:1] or "?"


def _esc(s) -> str:
    return html.escape(str(s), quote=False)


def _hhmm(utc: str | None) -> str:
    if not utc:
        return "--:--"
    return _esc(str(utc)[11:16])


# -------------------------------------------------------------------- cards
def card_close(row: dict) -> str:
    cls = classify(row)
    sym = _esc(row.get("symbol", "?"))
    arm = arm_letter(row.get("policy_id", ""))
    side = _esc(row.get("direction", "?"))

    if cls == "VOID":
        return (f"{VOID_} <b>{sym}</b> · <code>{arm}</code> {side}\n"
                f"    <i>void — {_esc(row.get('exit_reason'))}</i>\n"
                f"    <i>excluded from every statistic</i>")

    net = float(row.get("net_bps") or 0.0)
    gross = float(row.get("gross_bps") or 0.0)
    cost = gross - net
    mfe, mae = row.get("mfe_bps"), row.get("mae_bps")
    dot = _sign(net)

    lines = [
        f"{CLOSE_} <b>{sym}</b> · <code>arm {arm}</code> · {side}",
        f"{dot} <b>{net:+.1f} bps</b>  <code>{_bar(net)}</code>",
        f"    gross <code>{gross:+.1f}</code> · cost <code>{cost:.1f}</code>"
        f" · held <code>{_dur(row.get('horizon_ms') or 0)}</code>",
    ]
    if mfe is not None and mae is not None:
        lines.append(f"    path <code>{float(mfe):+.0f}</code> / "
                     f"<code>{float(mae):+.0f}</code> bps")
    if cls == "EXECUTION_NOT_IDENTIFIED":
        lines.append(f"{WARN} <i>execution not identified at this size "
                     f"({_depth_excess(row):.0f}× displayed depth) — "
                     f"kept in history, out of headline</i>")
    return "\n".join(lines)


def card_opens(rows: list[dict]) -> str:
    """OPENs are summarised, not announced one by one -- the lane opens roughly
    four an hour and a card each would be unreadable."""
    if not rows:
        return ""
    by_sym: dict[str, list[str]] = {}
    for r in rows:
        arm = arm_letter(r.get("policy_id", ""))
        by_sym.setdefault(_esc(r.get("symbol", "?")), []).append(arm)
    bits = [f"<b>{s}</b><code>{''.join(sorted(set(a)))}</code>"
            for s, a in sorted(by_sym.items())]
    n = len(rows)
    return (f"{OPEN_} <b>{n} new position{'s' if n != 1 else ''}</b>\n"
            f"    {' · '.join(bits)}")


def arms_block() -> str:
    rows = [r for r in trades() if r.get("event") == "CLOSE"]
    out = []
    for pid in _arm_ids():
        label = ARM_LABELS.get(pid, pid).replace("·", "").strip()
        scored = [r for r in rows
                  if r.get("policy_id") == pid and classify(r) == "HEADLINE"]
        if not scored:
            out.append(f"  <code>{label:11}</code> <i>no scored closes</i>")
            continue
        nets = [float(r["net_bps"]) for r in scored]
        mean = sum(nets) / len(nets)
        wr = 100.0 * sum(1 for x in nets if x > 0) / len(nets)
        out.append(f"  {_sign(mean)} <code>{label:11}</code> "
                   f"n=<code>{len(nets)}</code> "
                   f"avg <b>{mean:+.1f}</b> · win <code>{wr:.0f}%</code>")
    return "\n".join(out)


def card_status() -> str:
    hb = heartbeat()
    rows = trades()
    closes = [r for r in rows if r.get("event") == "CLOSE"]
    headline = [r for r in closes if classify(r) == "HEADLINE"]

    alive = bool(hb)
    age = "?"
    if hb.get("utc"):
        try:
            t = datetime.fromisoformat(hb["utc"].replace("Z", "+00:00"))
            age = f"{(datetime.now(timezone.utc) - t).total_seconds():.0f}s"
            alive = (datetime.now(timezone.utc) - t).total_seconds() < 180
        except Exception:
            pass
    head = OK_ if alive else HALT
    feed = (hb.get("anchor_feed") or {}).get("state", "?")

    return "\n".join([
        f"{CHART} <b>Eclipse · paper trader</b>",
        f"{head} runner <code>{'alive' if alive else 'NOT ALIVE'}</code>"
        f" · beat <code>{age}</code> · feed <code>{_esc(feed)}</code>",
        f"{OPEN_} open <b>{hb.get('open_positions', '?')}</b>"
        f" · closed <b>{len(headline)}</b> scored"
        f" (<code>{len(closes) - len(headline)}</code> excluded)",
        "",
        arms_block(),
        "",
        f"{SEAL} <i>paper only · no real order path</i>",
    ])


# -------------------------------------------------------------------- state
def load_state() -> dict:
    if not STATE.exists():
        return {"seen_open": [], "seen_close": [], "last_digest": None}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"seen_open": [], "seen_close": [], "last_digest": None}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def _key(r: dict) -> str:
    return f"{r.get('position_id')}|{r.get('event')}"


# --------------------------------------------------------------------- poll
def poll_once(cfg: dict, *, dry: bool = False, announce_opens: bool = True) -> dict:
    st = load_state()
    seen_o, seen_c = set(st["seen_open"]), set(st["seen_close"])
    rows = trades()

    new_opens = [r for r in rows if r.get("event") == "OPEN" and _key(r) not in seen_o]
    new_closes = [r for r in rows if r.get("event") == "CLOSE" and _key(r) not in seen_c]

    sent = []
    # Closes first: they are the information. Opens are context.
    for r in new_closes:
        msg = card_close(r)
        if dry:
            print(msg, "\n" + "-" * 46)
        else:
            send(cfg, msg)
        sent.append(("CLOSE", r.get("symbol")))
        seen_c.add(_key(r))

    if new_opens and announce_opens:
        msg = card_opens(new_opens)
        if dry:
            print(msg, "\n" + "-" * 46)
        else:
            send(cfg, msg, silent=True)     # quiet: context, not an event
        sent.append(("OPEN", len(new_opens)))
    for r in new_opens:
        seen_o.add(_key(r))

    st["seen_open"], st["seen_close"] = sorted(seen_o), sorted(seen_c)
    if not dry:
        save_state(st)
    return {"new_opens": len(new_opens), "new_closes": len(new_closes), "sent": sent}


LOCK = REPO / "runtime" / "telegram_bot.lock"


def _pid_alive(pid: int) -> bool:
    """Real existence check. See research_trader/runner.py for the full story:
    on Windows `os.kill(pid, 0)` is NOT an existence check (CTRL_C_EVENT == 0),
    it fails for live processes that do not share our console, and it can send
    a real Ctrl-C. Using it here reclaimed a LIVE lock and duplicated this
    watcher."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x00100000 | 0x1000, False, pid)   # SYNCHRONIZE|QUERY_LIMITED
        if not h:
            return k.GetLastError() == 5                     # ACCESS_DENIED -> exists
        try:
            code = ctypes.c_ulong()
            if k.GetExitCodeProcess(h, ctypes.byref(code)):
                return code.value == 259                     # STILL_ACTIVE
            return True
        finally:
            k.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def acquire_lock() -> None:
    """One watcher, exactly. Two would double every alert and race the state
    file. The .venv launcher on this machine spawns a child that also matches
    the command line, so 'did I start it twice?' is not answerable by looking
    at the process list -- it has to be settled structurally."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "utc": _now_iso()})
    for attempt in (1, 2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            try:
                prev = int(json.loads(LOCK.read_text(encoding="utf-8"))["pid"])
            except Exception:
                prev = -1
            if prev == os.getpid():
                return
            if prev > 0 and _pid_alive(prev):
                raise BotRefused(f"a watcher is already running as PID {prev}")
            if attempt == 1:
                print(f"reclaiming stale lock from dead PID {prev}")
                try:
                    LOCK.unlink()
                except OSError:
                    pass
                continue
            raise BotRefused("could not acquire the watcher lock")
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            return


def release_lock() -> None:
    try:
        if LOCK.exists() and int(json.loads(
                LOCK.read_text(encoding="utf-8"))["pid"]) == os.getpid():
            LOCK.unlink()
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ==========================================================================
# COMMANDS
#
# Every command is a READ. There is deliberately no command that opens, closes,
# sizes, promotes, or changes an arm -- the dispatch table below is the whole
# surface, and none of it writes to a ledger of record or a policy.
# ==========================================================================
def open_positions() -> list[dict]:
    """Rebuild the open set from the append-only ledger, same as the runner."""
    opened, closed = {}, set()
    for r in trades():
        pid = r.get("position_id")
        if r.get("event") == "OPEN":
            opened[pid] = r
        elif r.get("event") == "CLOSE":
            closed.add(pid)
    return [v for k, v in opened.items() if k not in closed]


def _age_ms(row: dict) -> int:
    ts = (row.get("entry") or {}).get("ts_ms")
    return int(time.time() * 1000) - int(ts) if ts else 0


def cmd_help() -> str:
    return "\n".join([
        f"{CHART} <b>Komutlar</b>",
        "",
        "<code>/status</code>   — genel durum kartı",
        "<code>/positions</code> — açık pozisyonlar, kalan süre",
        "<code>/arms</code>     — A kontrol vs B şampiyon",
        "<code>/last</code>     — son kapanışlar (<code>/last 8</code>)",
        "<code>/today</code>    — bugünün kapanışları",
        "<code>/health</code>   — runner + feed sağlığı",
        "<code>/thesis</code>   — tez/autopsy durumu",
        "<code>/science</code>  — bilim köprüsü",
        "<code>/help</code>     — bu liste",
        "",
        "<code>/autopsy</code>  — karar kalitesi vs sonuç",
        "<code>/cost</code>     — gerçekleşen icra + fırsat maliyeti",
        "<code>/path</code>     — MFE/MAE ve neden sinyal değil",
        "<code>/symbols</code>  — sonuç hangi sembollerde",
        "<code>/risk</code>     — yoğunlaşma, bağımsız epizod",
        "<code>/books</code>    — külliyatın koyduğu kısıtlar",
        "<code>/why</code>      — neden henüz ekonomik karar yok",
        "",
        f"{SEAL} <i>hepsi salt-okunur · emir veren komut yok</i>",
    ])


def cmd_positions() -> str:
    rows = open_positions()
    if not rows:
        return f"{FLAT} <b>Açık pozisyon yok</b>"
    marks = _v2._live_marks(rows) if _v2 else {}
    out = [f"{OPEN_} <b>{len(rows)} açık pozisyon</b>", ""]
    # One line per POSITION, not per symbol. Grouping by symbol hid which arm
    # held what -- and the arm is the whole experiment.
    tot = 0.0
    for r in sorted(rows, key=lambda x: (x.get("policy_id", ""), x.get("symbol", ""))):
        letter = arm_letter(r.get("policy_id", "")) if _v2 else "?"
        age = _age_ms(r)
        left = max(0, int(r.get("horizon_ms") or 0) - age)
        m = marks.get(r.get("position_id"))
        if m is None:
            cell = f"<code>   —  </code> <i>fiyat yok</i>"
        else:
            g = m["gross_bps"]
            tot += g
            cell = f"{_sign(g)} <b>{g:+7.1f}</b>"
        out.append(f"  <code>{letter}</code> <b>{_esc(r.get('symbol','?')).replace('USDT',''):<6}</b>"
                   f" {_esc(r.get('direction','?')):<5} {cell}"
                   f" · <code>{_dur(left)}</code> kaldı")
    if marks:
        # NOT one grand total. The arms share anchors, so the same market bet
        # appears once per arm holding it -- summing across arms counts one
        # event up to four times and reads as portfolio exposure it is not.
        per: dict[str, list[float]] = {}
        for r in rows:
            m = marks.get(r.get("position_id"))
            if m is not None:
                per.setdefault(arm_letter(r.get("policy_id", "")),
                               []).append(m["gross_bps"])
        out.append("")
        for L in sorted(per):
            v = per[L]
            out.append(f"  <code>{L}</code> {_sign(sum(v))} <b>{sum(v):+7.0f}</b>"
                       f" bps <i>({len(v)} poz)</i>")
        out += [f"{WARN} <i>kollar aynı anchor'ları paylaşır — toplamlar AYRI "
                f"verilir, tek toplam aynı bahsi 4 kez sayardı. Bu GROSS "
                f"mark-to-market: maliyet (~10.7 bps) düşülmedi ve kural bu "
                f"sayıyı GÖRMÜYOR. Kapanış ufukta, bu rakamda değil.</i>"]
    return "\n".join(out)


def _arm_ids() -> list[str]:
    """The arm set as the RUNNING policy module defines it. Reading it here
    means a newly frozen arm appears in this report without the bot being
    edited -- and an arm that was retired disappears instead of leaving a
    stale row behind."""
    try:
        sys.path.insert(0, str(REPO))
        from research_trader import policy as _pol
        return [p.policy_id for p in _pol.ARMS]
    except Exception:
        return ["A_ALL_ANCHORS_V1", "B_ONE_PER_EPISODE_V1"]


# Labels are declared, not derived, so an arm nobody labelled shows up under
# its raw id rather than vanishing. An arm at N=0 must still be visible:
# absence from a table is how a live arm gets forgotten.
ARM_LABELS = {"A_ALL_ANCHORS_V1": "A · kontrol",
              "B_ONE_PER_EPISODE_V1": "B · şampiyon",
              "C_DEPTH_ADMITTED_V1": "C · derinlik kapısı",
              "D_MARKET_EPISODE_V1": "D · piyasa epizodu"}


def cmd_arms() -> str:
    rows = [r for r in trades() if r.get("event") == "CLOSE"]
    out = [f"{CHART} <b>Kollar</b>  <i>(sadece manşet kapanışlar)</i>", ""]
    ids = _arm_ids()
    for pid in ids:
        label = ARM_LABELS.get(pid, pid)
        scored = [r for r in rows
                  if r.get("policy_id") == pid and classify(r) == "HEADLINE"]
        if not scored:
            out += [f"{FLAT} <b>{label}</b>",
                    "  <code>N=0</code> <i>— donduruldu, henüz kapanış yok</i>", ""]
            continue
        nets = [float(r["net_bps"]) for r in scored]
        mean = sum(nets) / len(nets)
        wins = [x for x in nets if x > 0]
        out += [
            f"{_sign(mean)} <b>{label}</b>",
            f"  n=<code>{len(nets)}</code> · ort <b>{mean:+.1f}</b> bps"
            f" · toplam <code>{sum(nets):+.0f}</code>",
            f"  kazanan <code>{len(wins)}/{len(nets)}</code>"
            f" · en iyi <code>{max(nets):+.0f}</code>"
            f" · en kötü <code>{min(nets):+.0f}</code>",
            "",
        ]
    out.append(f"{WARN} <i>n küçük — bunlar betimleyici, iddia değil. "
               f"Promosyon kapısı 200 epizod · 21 gün · 8 sembol.</i>")
    if "C_DEPTH_ADMITTED_V1" in ids or "D_MARKET_EPISODE_V1" in ids:
        out.append(f"{SEAL} <i>C ve D 2026-08-27'de donduruldu "
                   f"(ARM_CD_PREREGISTRATION_V1). İkisi de yalnız işlem ELER, "
                   f"edge üretemez — dürüst öncül sıfır beklenti + daha az "
                   f"varyans. A ve B'nin N'i sıfırlanmadı.</i>")
    return "\n".join(out)


def cmd_last(n: int = 5) -> str:
    cl = [r for r in trades() if r.get("event") == "CLOSE"]
    if not cl:
        return f"{FLAT} <b>Henüz kapanış yok</b>"
    n = max(1, min(n, 12))
    return "\n\n".join(card_close(r) for r in cl[-n:])


def cmd_today() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cl = [r for r in trades() if r.get("event") == "CLOSE"
          and str(r.get("closed_utc", "")).startswith(today)]
    head = [h for h in cl if classify(h) == "HEADLINE"]
    if not cl:
        return f"{FLAT} <b>Bugün kapanış yok</b> <code>({today})</code>"
    lines = [f"{CHART} <b>Bugün</b> <code>{today}</code>",
             f"  {len(cl)} kapanış · <b>{len(head)}</b> manşette", ""]
    if head:
        nets = [float(h["net_bps"]) for h in head]
        tot = sum(nets)
        lines.append(f"{_sign(tot)} toplam <b>{tot:+.0f}</b> bps"
                     f" · ort <code>{tot / len(nets):+.1f}</code>")
        lines.append("")
    for r in cl:
        cls = classify(r)
        if cls == "HEADLINE":
            net = float(r["net_bps"])
            lines.append(f"  {_sign(net)} <b>{_esc(r.get('symbol')):<9}</b>"
                         f" <code>{net:+8.1f}</code>")
        else:
            mark = VOID_ if cls == "VOID" else WARN
            lines.append(f"  {mark} <b>{_esc(r.get('symbol')):<9}</b>"
                         f" <i>hariç</i>")
    return "\n".join(lines)


def cmd_health() -> str:
    hb = heartbeat()
    if not hb:
        return f"{HALT} <b>Heartbeat yok</b> — runner çalışmıyor olabilir"
    try:
        t = datetime.fromisoformat(hb["utc"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - t).total_seconds()
    except Exception:
        age = -1
    alive = 0 <= age < 180
    feed = hb.get("anchor_feed") or {}
    c = hb.get("counts") or {}
    return "\n".join([
        f"{OK_ if alive else HALT} <b>Sağlık</b>",
        "",
        f"  runner <code>{'ALIVE' if alive else 'NOT ALIVE'}</code>"
        f" · pid <code>{hb.get('pid')}</code> · beat <code>{age:.0f}s</code>",
        f"  feed <code>{_esc(feed.get('state'))}</code>"
        f" · {feed.get('symbols_1h', '?')} sembol/saat"
        f" · yaş <code>{int((feed.get('age_ms') or 0) / 1000)}s</code>",
        f"  halt <code>{_esc(hb.get('halt_reason') or 'yok')}</code>",
        "",
        f"  anchor görülen <code>{c.get('anchors_seen', 0)}</code>"
        f" · no-trade epizod <code>{c.get('no_trade_episode', 0)}</code>",
        f"  E3 reddi <code>{c.get('no_trade_e3_protected', 0)}</code>"
        f" · fill reddi <code>{c.get('no_trade_fill_refused', 0)}</code>",
        "",
        f"{SEAL} <i>{_esc((hb.get('paper_only') or {}).get('verdict'))}</i>",
    ])


def cmd_thesis() -> str:
    th = {t.get("trade_id") or t.get("candidate_id")
          for t in _rows(TRADER / "trade_thesis_ledger.jsonl")}
    cl = [r for r in trades() if r.get("event") == "CLOSE"]
    post = [r for r in cl if r.get("position_id") in th]
    pending = [r for r in open_positions() if r.get("position_id") in th]
    lines = [f"{CLOCK} <b>Tez durumu</b>", "",
             f"  tez yazılmış pozisyon: <b>{len(th)}</b>",
             f"  post-tez kapanış: <b>{len(post)}</b>", ""]
    if post:
        lines.append(f"{OK_} <b>Autopsy tetiklendi</b> — ilk gerçek "
                     f"post-kontrat kapanış geldi.")
    else:
        lines.append(f"{CLOCK} <i>Autopsy bekliyor. Mevcut kapanışların hepsi "
                     f"tez kontratı ÖNCESİ; onlardan autopsy üretmek, "
                     f"denetlediğini iddia ettiği inançları uydurmak olurdu.</i>")
        if pending:
            soon = min(max(0, int(p.get("horizon_ms") or 0) - _age_ms(p))
                       for p in pending)
            lines.append("")
            lines.append(f"  ilk uygun kapanışa <b>~{_dur(soon)}</b>")
    return "\n".join(lines)


def cmd_science() -> str:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sb", REPO / "research_trader" / "science_bridge.py")
        sb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sb)
        s = sb.status()
    except Exception as exc:
        return f"{WARN} <b>Köprü okunamadı</b>: <code>{_esc(exc)}</code>"
    return "\n".join([
        f"{SEAL} <b>Bilim → Trader köprüsü</b>", "",
        f"  durum <code>{_esc(s['state'])}</code>",
        f"  diskteki release <b>{s['releases_on_disk']}</b>"
        f" · işlenmemiş <b>{len(s['unseen'])}</b>",
        f"  son tüketilen <code>{_esc(s['last_consumed_science_release_id'])}</code>",
        f"  doğan soru <code>{s['n_research_questions']}</code>"
        f" · challenger <code>{s['n_challenger_research']}</code>",
        "",
        f"<i>Bilim yalnız COMPLETE + HASH + LINEAGE doğrulanmış release ile "
        f"girer ve yalnız ARAŞTIRMA doğurur — şampiyonu asla değiştiremez.</i>",
    ])


COMMANDS = {
    "start": lambda a: cmd_help(),
    "help": lambda a: cmd_help(),
    "status": lambda a: card_status(),
    "positions": lambda a: cmd_positions(),
    "pos": lambda a: cmd_positions(),
    "arms": lambda a: cmd_arms(),
    "last": lambda a: cmd_last(int(a[0]) if a and a[0].isdigit() else 5),
    "today": lambda a: cmd_today(),
    "health": lambda a: cmd_health(),
    "thesis": lambda a: cmd_thesis(),
    "science": lambda a: cmd_science(),
}

# v2 commands live in their own module so the alert loop stays small and the
# command surface can grow without touching it. Loaded defensively: a broken
# extension must never take the bot down.
try:
    import importlib.util as _il
    _spec = _il.spec_from_file_location(
        "eclipse_bot_commands_v2", REPO / "tools" / "eclipse_bot_commands_v2.py")
    _v2 = _il.module_from_spec(_spec)
    _spec.loader.exec_module(_v2)
    COMMANDS.update({k: (lambda f: (lambda a: f(a)))(v)
                     for k, v in _v2.COMMANDS_V2.items()})
    _MENU_V2 = _v2.MENU_V2
except Exception as _exc:                     # extension is optional, never fatal
    print("v2 commands unavailable:", _exc)
    _v2 = None                                # cmd_positions degrades, never raises
    _MENU_V2 = []

MENU = [
    {"command": "status", "description": "genel durum"},
    {"command": "positions", "description": "acik pozisyonlar"},
    {"command": "arms", "description": "A kontrol vs B sampiyon"},
    {"command": "last", "description": "son kapanislar"},
    {"command": "today", "description": "bugunun kapanislari"},
    {"command": "health", "description": "runner + feed sagligi"},
    {"command": "thesis", "description": "tez / autopsy durumu"},
    {"command": "science", "description": "bilim koprusu"},
    {"command": "help", "description": "komut listesi"},
] + _MENU_V2


def install_menu(cfg: dict) -> dict:
    return _call(cfg, "setMyCommands", commands=MENU)


def handle_update(cfg: dict, upd: dict) -> str | None:
    msg = upd.get("message") or upd.get("edited_message") or {}
    chat = (msg.get("chat") or {}).get("id")
    text = (msg.get("text") or "").strip()
    if not text.startswith("/"):
        return None
    # Only the connected operator gets answers. Anyone can find a bot; the
    # trading state is not theirs to read.
    if str(chat) != str(cfg.get("chat_id")):
        return None
    parts = text[1:].split()
    name = parts[0].split("@")[0].lower()
    fn = COMMANDS.get(name)
    if fn is None:
        _call(cfg, "sendMessage", chat_id=chat, parse_mode="HTML",
              text=f"{WARN} bilinmeyen komut <code>/{_esc(name)}</code>\n"
                   f"<code>/help</code> yaz")
        return name
    try:
        body = fn(parts[1:])
    except Exception as exc:
        body = f"{WARN} <b>{_esc(name)}</b> hata: <code>{_esc(exc)}</code>"
    _call(cfg, "sendMessage", chat_id=chat, text=body, parse_mode="HTML",
          disable_web_page_preview=True)
    return name


def health_alert(cfg: dict) -> str | None:
    """Alert on a CHANGE in lane health, once per transition.

    Added after the anchor feed sat silent for 930s with nobody told
    (SYSTEM_STATE 422). Everything behaved correctly -- the trader refused a
    stale feed, the supervisor declared a crash loop -- and the operator still
    only found out because someone happened to run a freshness check. Correct
    components do not add up to an informed operator."""
    hb = heartbeat()
    if not hb:
        return None
    feed = (hb.get("anchor_feed") or {})
    now = {"feed": feed.get("state"), "tradeable": bool(feed.get("tradeable")),
           "halt": hb.get("halt_reason")}
    st = load_state()
    prev = st.get("last_health")
    st["last_health"] = now
    save_state(st)
    if prev is None or prev == now:
        return None

    ok = now["tradeable"] and now["feed"] == "FULL" and not now["halt"]
    if ok:
        return (f"{OK_} <b>Lane normale döndü</b>\n"
                f"    feed <code>{_esc(now['feed'])}</code> · anchor alımı açık")
    bits = [f"{HALT} <b>Lane anchor alamıyor</b>"]
    if now["feed"] != "FULL":
        bits.append(f"    feed <code>{_esc(now['feed'])}</code>"
                    f" · {int((feed.get('age_ms') or 0)/1000)}s sessiz")
    if now["halt"]:
        bits.append(f"    halt <code>{_esc(now['halt'])}</code>")
    bits.append("    <i>açık pozisyonlar etkilenmez; yeni giriş durur</i>")
    return "\n".join(bits)


def watch(cfg: dict, interval: int = 60, digest_hours: int = 6) -> None:
    """Long-poll for commands AND diff the ledger.

    getUpdates blocks up to `interval` seconds, so a command is answered in
    well under a second while the ledger is still checked every cycle -- one
    loop, no second thread, no second process to keep alive."""
    acquire_lock()
    install_menu(cfg)
    print(f"watching (long-poll {interval}s, digest {digest_hours}h) -- ctrl-c to stop")
    st = load_state()
    offset = st.get("update_offset")
    while True:
        try:
            r = poll_once(cfg)
            if r["sent"]:
                print(time.strftime("%H:%M:%S"), "sent", r["sent"])

            alert = health_alert(cfg)
            if alert:
                send(cfg, alert)
                print(time.strftime("%H:%M:%S"), "health alert sent")

            st = load_state()
            last = st.get("last_digest") or 0
            if time.time() - last >= digest_hours * 3600:
                send(cfg, card_status(), silent=True)
                st["last_digest"] = time.time()
                save_state(st)

            upd = _call(cfg, "getUpdates", _http_timeout=interval + 15,
                        offset=offset, timeout=interval)
            if upd.get("ok"):
                for u in upd["result"]:
                    offset = u["update_id"] + 1
                    name = handle_update(cfg, u)
                    if name:
                        print(time.strftime("%H:%M:%S"), "cmd", name)
                if upd["result"]:
                    st = load_state()
                    st["update_offset"] = offset
                    save_state(st)
            else:
                time.sleep(5)
        except BotRefused:
            release_lock()
            raise
        except KeyboardInterrupt:
            release_lock()
            raise
        except Exception as exc:                      # a poll failure is not fatal
            print("poll error:", exc)
            time.sleep(5)


# ------------------------------------------------------------------ commands
def verify(cfg: dict) -> dict:
    r = _call(cfg, "getMe")
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("description")}
    u = r["result"]
    return {"ok": True, "bot": u.get("username"), "name": u.get("first_name")}


def connect(cfg: dict) -> dict:
    """Discover the chat id from whoever messaged the bot."""
    r = _call(cfg, "getUpdates")
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("description")}
    chats = {}
    for upd in r["result"]:
        m = upd.get("message") or upd.get("channel_post") or {}
        c = m.get("chat") or {}
        if c.get("id"):
            chats[c["id"]] = c.get("first_name") or c.get("title") or c.get("username")
    if not chats:
        return {"ok": False, "reason": "no chats yet -- send the bot any message first"}
    cid, who = sorted(chats.items())[0]
    cfg["chat_id"] = cid
    save_cfg(cfg)
    return {"ok": True, "chat_id": cid, "who": who}


def main(argv=None) -> int:
    # The Windows console here is cp1254, which cannot encode the emoji the
    # cards use. That is a CONSOLE limit, not a message limit -- Telegram gets
    # UTF-8 either way -- so only the local preview needs re-encoding.
    for stream in ("stdout", "stderr"):
        try:
            getattr(__import__("sys"), stream).reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Eclipse paper-trader Telegram alerts")
    ap.add_argument("--verify", action="store_true", help="check the token")
    ap.add_argument("--connect", action="store_true", help="discover and save chat_id")
    ap.add_argument("--status", action="store_true", help="send a status card now")
    ap.add_argument("--preview", action="store_true", help="print cards, send nothing")
    ap.add_argument("--once", action="store_true", help="one poll, then exit")
    ap.add_argument("--watch", action="store_true", help="poll continuously")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--digest-hours", type=int, default=6)
    a = ap.parse_args(argv)

    if a.preview:                       # works with no token at all
        print(card_status())
        print("\n" + "=" * 46 + "\n")
        cl = [r for r in trades() if r.get("event") == "CLOSE"]
        for r in cl[-4:]:
            print(card_close(r), "\n" + "-" * 46)
        op = [r for r in trades() if r.get("event") == "OPEN"][-5:]
        print(card_opens(op))
        return 0

    cfg = load_cfg()
    if a.verify:
        print(json.dumps(verify(cfg), indent=1))
        return 0
    if a.connect:
        print(json.dumps(connect(cfg), indent=1))
        return 0
    if a.status:
        print(json.dumps(send(cfg, card_status()), indent=1)[:300])
        return 0
    if a.once:
        print(json.dumps(poll_once(cfg), indent=1))
        return 0
    if a.watch:
        watch(cfg, a.interval, a.digest_hours)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
