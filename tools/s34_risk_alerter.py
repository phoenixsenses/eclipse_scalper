"""S34 Risk Alerter — observation-only, send-only.

Reads the execution-management audit and live executor state, evaluates
critical conditions, and sends a Telegram message to the operator.

Conditions checked (all OBSERVE -> NOTIFY, no live actions):
  - OVERSIZE   : live margin >> tail-budget margin (>10x = URGENT)
  - TAIL_EVENT : recent shadow trade with net < -100 bps
  - KILL_TRIP  : runtime/KILL_SWITCH file exists
  - ATOMICITY  : NOT_ATOMIC finding (entry before stop) — static, always warn once
  - SL_GAP     : worst real fill > nominal stop (gap-through)

Run:
    python tools/s34_risk_alerter.py --check-only   # print conditions, no send
    python tools/s34_risk_alerter.py                # send if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID set
    python tools/s34_risk_alerter.py --force-send   # send even if all conditions green
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXECUTION_AUDIT = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_EXECUTION_MANAGEMENT_AUDIT.json"
LIVE_STATE      = ROOT / "runtime" / "s34_v_engine_live_state.json"
SHADOW_LEDGER   = ROOT / "reports" / "shadow" / "s34_realtime_shadow.jsonl"
KILL_SWITCH     = ROOT / "runtime" / "KILL_SWITCH"
ALERT_STATE     = ROOT / "runtime" / "s34_risk_alerter_state.json"

OVERSIZE_URGENT_X   = 10.0   # flag if live margin > 10x tail-budget
TAIL_LOSS_BPS       = -100.0 # recent shadow trade below this = tail event
RECENT_TRADE_LIMIT  = 20     # check last N shadow trades for tail events


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_alert_state() -> dict:
    return load_json(ALERT_STATE, {"last_sent_utc": None, "conditions_last": {}})


def save_alert_state(state: dict) -> None:
    ALERT_STATE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def evaluate_conditions() -> dict:
    """Return dict of condition_name -> bool (True = alert condition met)."""
    audit = load_json(EXECUTION_AUDIT, {})
    live  = load_json(LIVE_STATE, {})
    conds: dict[str, bool] = {}
    details: dict[str, str] = {}

    # --- OVERSIZE ---
    env = audit.get("live_env") or {}
    margin_env    = float(env.get("margin_usdt") or 0.0)
    budget_margin = float(env.get("max_budget_margin_usdt") or 0.0)
    oversize = (margin_env / budget_margin) if budget_margin > 0 else 0.0
    conds["OVERSIZE"] = oversize > OVERSIZE_URGENT_X
    details["OVERSIZE"] = f"margin_env={margin_env}usdt budget={budget_margin}usdt oversize={oversize:.1f}x"

    # --- ATOMICITY (static, always flag as risk awareness) ---
    atom = audit.get("atomicity_audit") or {}
    is_not_atomic = "NOT_ATOMIC" in str(atom.get("finding") or "")
    conds["ATOMICITY"] = is_not_atomic
    details["ATOMICITY"] = str(atom.get("finding") or "UNKNOWN") + f" poll={atom.get('poll_sec')}s"

    # --- SL GAP THROUGH ---
    gap = audit.get("gap_through") or {}
    nominal = float(gap.get("current_stop_nominal_bps") or 150.0)
    worst   = float(gap.get("current_stop_research_max_loss_bps") or 0.0)
    conds["SL_GAP"] = worst < -nominal
    details["SL_GAP"] = f"nominal={nominal}bps worst_fill={worst}bps gap={gap.get('gap_plus_fee_bps')}bps PARTIAL_PROTECTION"

    # --- KILL_TRIP ---
    kill_active = KILL_SWITCH.exists()
    conds["KILL_TRIP"] = kill_active
    details["KILL_TRIP"] = "KILL_SWITCH_FILE_EXISTS" if kill_active else "inactive"

    # --- TAIL_EVENT in recent shadow trades ---
    tail_found = False
    tail_desc  = "none in last 20 shadow trades"
    if SHADOW_LEDGER.exists():
        lines: list[str] = []
        try:
            with SHADOW_LEDGER.open(encoding="utf-8") as fh:
                for raw in fh:
                    lines.append(raw)
        except Exception:
            pass
        recent = lines[-RECENT_TRADE_LIMIT:]
        for raw in recent:
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            if rec.get("event") != "CLOSE":
                continue
            net = rec.get("net_bps")
            if net is not None and math.isfinite(float(net)) and float(net) < TAIL_LOSS_BPS:
                tail_found = True
                tail_desc = f"net={net}bps signal={rec.get('signal')} anchor={rec.get('anchor_ts_ms')}"
                break
    conds["TAIL_EVENT"] = tail_found
    details["TAIL_EVENT"] = tail_desc

    # --- ACTIVE POSITION check ---
    active = live.get("active")
    has_pos = active is not None
    pos_desc = "no_active_position"
    if has_pos and isinstance(active, dict):
        pos_desc = (f"pos_amt={active.get('pos_amt')} entry={active.get('entry_price')} "
                    f"sil_resolved={active.get('silence_gate_resolved')}")
    details["ACTIVE_POSITION"] = pos_desc

    return {"conditions": conds, "details": details, "oversize_x": oversize}


def build_message(conds: dict, details: dict, oversize_x: float) -> str:
    triggered = [k for k, v in conds.items() if v]
    all_clear  = len(triggered) == 0
    header = "S34 RISK ALERT" if triggered else "S34 Risk Check — ALL_CLEAR"
    lines = [f"[{header}]  {utc_now()[:19]}Z", ""]
    for k, v in conds.items():
        icon = "[ALERT]" if v else "[OK]"
        lines.append(f"{icon} {k}: {details.get(k, '')}")
    lines.append("")
    if triggered:
        lines.append(f"TRIGGERED: {', '.join(triggered)}")
        if "OVERSIZE" in triggered:
            lines.append(f"  =>Reduce per-trade MARGIN to tail-budget (currently {oversize_x:.1f}x over budget) or disarm executor.")
        if "KILL_TRIP" in triggered:
            lines.append("  =>KILL_SWITCH file exists; executor is blocking new entries.")
        if "TAIL_EVENT" in triggered:
            lines.append("  =>Recent shadow tail loss detected. Review position size.")
    else:
        lines.append("No urgent conditions. Monitor as usual.")
    return "\n".join(lines)


def send_telegram(text: str, token: str, chat_id: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        print(f"Telegram HTTP error: {exc.code} {exc.reason}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"Telegram send error: {exc}", file=sys.stderr)
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S34 risk alerter — observation only, no live actions.")
    p.add_argument("--check-only", action="store_true", help="Print conditions without sending.")
    p.add_argument("--force-send", action="store_true", help="Send even if all conditions green.")
    p.add_argument("--token", default="", help="Telegram bot token (default: TELEGRAM_BOT_TOKEN env).")
    p.add_argument("--chat-id", default="", help="Telegram chat id (default: TELEGRAM_CHAT_ID env).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    result = evaluate_conditions()
    conds   = result["conditions"]
    details = result["details"]
    oversize_x = float(result.get("oversize_x") or 0.0)

    msg = build_message(conds, details, oversize_x)
    print(msg)

    triggered = [k for k, v in conds.items() if v]
    should_send = bool(triggered) or args.force_send

    if args.check_only:
        print("\n(--check-only: not sending)")
        return 0

    token   = args.token or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or ""
    chat_id = args.chat_id or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID") or ""

    if not token or not chat_id:
        print("No TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID configured; skipping send.", file=sys.stderr)
        state = load_alert_state()
        state["last_checked_utc"] = utc_now()
        state["conditions_last"] = {k: v for k, v in conds.items()}
        state["send_skipped"] = "no_token_or_chat"
        save_alert_state(state)
        return 0

    if not should_send:
        print("All clear — not sending (use --force-send to override).")
        return 0

    ok = send_telegram(msg, token, chat_id)
    state = load_alert_state()
    state["last_checked_utc"] = utc_now()
    state["conditions_last"] = {k: v for k, v in conds.items()}
    if ok:
        state["last_sent_utc"] = utc_now()
        state["last_sent_conditions"] = triggered
        print("Alert sent.")
    else:
        print("Send failed.", file=sys.stderr)
    save_alert_state(state)
    return 0 if ok or not triggered else 1


if __name__ == "__main__":
    raise SystemExit(main())
