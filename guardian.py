# guardian.py — SCALPER ETERNAL — COSMIC GUARDIAN ASCENDANT PRODUCTION — SMALL CAPITAL QUICK PROFIT MODE — 2026 v2.2
# Dated: January 03, 2026
# Philosophy: Elite protection for manual perpetual positions on tiny accounts ($30–$50).
# Prioritizes ultra-quick small absolute profit capture ($1 → $2 → $3 tiers), ironclad loss guards,
# proper order management, and full config integration. No aggressive hacks — pure disciplined preservation.
#
# Key fixes & enhancements in v2.2 (over v2.1):
# • Removed fetch_position_mode() call — CosmicExchangeOracle lacks this method (only set_position_mode exists)
#   Reverted to unconditional hedge mode activation as in original code (wrapper likely logs "already active")
# • Minor logging tweak for hedge mode activation
# • All previous v2.1 robustness (safe_float, staged profit scaling, dust avoidance, etc.) preserved

import asyncio
import time
import os
from datetime import datetime, timezone
from collections import defaultdict

from bot.core import EclipseEternal
from utils.logging import log_core
from execution.emergency import emergency_flat
from strategies.risk import portfolio_heat
from config.settings import Config
from brain.state import Position
from notifications.telegram import Notifier

cfg = Config()

commander = Notifier(token=os.getenv('TELEGRAM_TOKEN'), chat_id=os.getenv('TELEGRAM_CHAT_ID')) if os.getenv('TELEGRAM_TOKEN') else None

# SMALL CAPITAL MODE — Absolute dollar profit targets (quick scalping style)
SMALL_CAPITAL_MODE = True
PROFIT_TARGETS_DOLLARS = [
    (1.0, 0.4),   # +$1 profit → close 40% of current position
    (2.0, 0.3),   # +$2 profit → close 30% of remaining
    (3.0, 1.0),   # +$3 profit → close remaining 100%
]

def safe_float(d, key, default=0.0):
    """Safely convert a dict value to float, handling None, missing keys, or invalid values."""
    val = d.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

class EclipseGuardian:
    def __init__(self):
        self.bot = EclipseEternal()
        self.paused = False
        self.velocity_windows = defaultdict(list)
        self.session_peak_equity = 0.0
        self.protection_actions = []
        self.running = True
        # Track profit tier status per symbol — '1'/'2'/'3' for $1/$2/$3 tiers
        self.profit_tiers_taken = defaultdict(lambda: {'1': False, '2': False, '3': False})

    async def send_divine_message(self, message: str, priority: str = 'critical'):
        log_core.critical(message)
        if commander:
            await commander.speak(message, priority)

    async def load_manual_positions(self):
        positions = await self.bot.ex.fetch_positions()
        restored = 0
        for pos in positions:
            size = safe_float(pos, 'contracts', 0.0)
            if size == 0.0:
                continue
            sym = pos.get('symbol', '')
            if not sym.endswith('USDT'):
                continue

            internal_sym = sym.replace('USDT', '/USDT')
            entry = safe_float(pos, 'entryPrice', 0.0)
            leverage = safe_float(pos, 'leverage', 20.0)

            side = "long" if size > 0 else "short"
            position_obj = Position(
                side=side,
                size=abs(size),  # store absolute size in state
                entry_price=entry,
                atr=0,
                leverage=leverage,
                entry_ts=time.time(),
                confidence=1.0
            )
            # Add tracking attributes
            position_obj.breakeven_moved = False
            position_obj.trailing_active = False
            position_obj.hard_stop_order_id = None

            self.bot.state.positions[internal_sym] = position_obj
            # Reset tier tracking for freshly loaded positions
            self.profit_tiers_taken[internal_sym] = {'1': False, '2': False, '3': False}
            restored += 1

        if restored:
            await self.send_divine_message(f"GUARDIAN RESURRECTION — {restored} manual positions restored")
        else:
            log_core.info("GUARDIAN SCAN — No manual positions found to restore")

    async def apply_management(self):
        positions = await self.bot.ex.fetch_positions()
        for pos in positions:
            contracts = safe_float(pos, 'contracts', 0.0)
            if contracts == 0.0:
                continue

            sym = pos.get('symbol', '')
            internal_sym = sym.replace('USDT', '/USDT')

            if internal_sym not in self.bot.state.positions:
                continue

            state_pos = self.bot.state.positions[internal_sym]
            current_price = safe_float(pos, 'markPrice', state_pos.entry_price)
            unrealized_pnl = safe_float(pos, 'unrealizedPnl', 0.0)

            remaining_size = abs(contracts)
            local_remaining = remaining_size  # For staged profit-taking simulation

            # === QUICK SMALL PROFIT CLOSURE (SMALL CAPITAL MODE) ===
            if SMALL_CAPITAL_MODE and unrealized_pnl > 0:
                for dollars, portion in PROFIT_TARGETS_DOLLARS:
                    tier_key = str(dollars)
                    if (unrealized_pnl >= dollars and 
                        not self.profit_tiers_taken[internal_sym][tier_key] and
                        local_remaining > 0):

                        close_amount = local_remaining * portion
                        # Avoid tiny dust orders
                        if close_amount < 0.0001:
                            close_amount = local_remaining

                        if close_amount > 0:
                            try:
                                await self.bot.ex.create_order(
                                    symbol=sym,
                                    type='market',
                                    side='sell' if state_pos.side == 'long' else 'buy',
                                    amount=close_amount,
                                    params={'reduceOnly': True}
                                )
                                await self.send_divine_message(
                                    f"QUICK PROFIT TIER ${dollars} HIT {internal_sym} — "
                                    f"Closed {portion:.0%} (${unrealized_pnl:.2f} unrealized)"
                                )
                                self.profit_tiers_taken[internal_sym][tier_key] = True
                                local_remaining -= close_amount

                                if portion == 1.0 or local_remaining <= 0:
                                    if internal_sym in self.bot.state.positions:
                                        del self.bot.state.positions[internal_sym]
                                    self.profit_tiers_taken[internal_sym] = {'1': False, '2': False, '3': False}
                                    break
                            except Exception as e:
                                await self.send_divine_message(f"Quick profit close failed {internal_sym}: {e}")

            # === BREAKEVEN MOVE AFTER MEANINGFUL PROFIT ===
            initial_margin_approx = abs(state_pos.size) * state_pos.entry_price / state_pos.leverage if state_pos.entry_price > 0 and state_pos.leverage > 0 else 1
            unrealized_pct = (unrealized_pnl / initial_margin_approx) * 100 if initial_margin_approx > 0 else 0

            if unrealized_pct > 1.0 and not state_pos.breakeven_moved:
                buffer_pct = 0.1
                be_price = state_pos.entry_price * (1 + buffer_pct/100 if state_pos.side == 'long' else 1 - buffer_pct/100)
                try:
                    # Cancel existing hard stop if present
                    if getattr(state_pos, 'hard_stop_order_id', None):
                        await self.bot.ex.cancel_order(state_pos.hard_stop_order_id, sym)

                    new_stop = await self.bot.ex.create_order(
                        symbol=sym,
                        type='STOP_MARKET',
                        side='sell' if state_pos.side == 'long' else 'buy',
                        amount=remaining_size,
                        params={'stopPrice': be_price, 'reduceOnly': True}
                    )
                    state_pos.hard_stop_order_id = new_stop.get('id')
                    state_pos.breakeven_moved = True
                    await self.send_divine_message(f"BREAKEVEN ASCENDED {internal_sym} — Stop moved to +{buffer_pct:.1f}%")
                except Exception as e:
                    await self.send_divine_message(f"Breakeven move failed {internal_sym}: {e}")

            # === SIMPLE TRAILING (activate after +1.5%) ===
            activation_pct = 1.5
            if unrealized_pct > activation_pct and not state_pos.trailing_active:
                callback_rate = int(cfg.TRAILING_CALLBACK_RATE)
                try:
                    await self.bot.ex.create_order(
                        symbol=sym,
                        type='TRAILING_STOP_MARKET',
                        side='sell' if state_pos.side == 'long' else 'buy',
                        amount=remaining_size,
                        params={
                            'callbackRate': callback_rate,
                            'reduceOnly': True,
                            'activationPrice': current_price
                        }
                    )
                    state_pos.trailing_active = True
                    await self.send_divine_message(f"TRAILING STOP ACTIVATED {internal_sym} — {callback_rate/100:.1f}% callback")
                except Exception as e:
                    await self.send_divine_message(f"Trailing activation failed {internal_sym}: {e}")

    async def velocity_drawdown_guard(self):
        current_time = time.time()
        positions = await self.bot.ex.fetch_positions()
        for pos in positions:
            contracts = safe_float(pos, 'contracts', 0.0)
            if contracts == 0.0:
                continue
            sym = pos.get('symbol', '')
            internal_sym = sym.replace('USDT', '/USDT')
            unrealized = safe_float(pos, 'unrealizedPnl', 0.0)

            self.velocity_windows[internal_sym].append((current_time, unrealized))
            # 5-minute window
            self.velocity_windows[internal_sym] = [w for w in self.velocity_windows[internal_sym] if current_time - w[0] < 300]

            if len(self.velocity_windows[internal_sym]) >= 2:
                peak = max(pnl for _, pnl in self.velocity_windows[internal_sym])
                if peak > 0:
                    drop = (peak - unrealized) / peak
                    if drop > cfg.VELOCITY_DRAWDOWN_PCT:
                        await self.send_divine_message(f"VELOCITY BREACH {internal_sym} ({drop:.1%}) — EMERGENCY FLAT")
                        await emergency_flat(self.bot)

    async def run(self):
        await self.send_divine_message("COSMIC GUARDIAN ASCENDANT AWAKENED — Small-capital quick-profit protection engaged")
        await self.bot.ex.fetch_markets()
        
        # Unconditional hedge mode activation (as in original code — wrapper likely detects/logs "already active")
        await self.bot.ex.set_position_mode(True)

        await self.load_manual_positions()

        bal = await self.bot.ex.fetch_balance()
        equity = safe_float(bal.get('USDT', {}), 'total', 0.0) or safe_float(bal.get('total', {}), 'USDT', 0.0)
        self.bot.state.current_equity = self.bot.state.peak_equity = self.session_peak_equity = equity
        self.bot.state.start_of_day_equity = equity
        self.bot.state.current_day = datetime.now(timezone.utc).date()

        await self.send_divine_message(f"GUARDIAN ACTIVE | Equity ≈ ${equity:.2f} | Small-capital quick-profit mode ON")

        while self.running:
            try:
                if self.paused:
                    await asyncio.sleep(10)
                    continue

                bal = await self.bot.ex.fetch_balance()
                total = safe_float(bal.get('USDT', {}), 'total', 0.0) or safe_float(bal.get('total', {}), 'USDT', 0.0)
                self.bot.state.current_equity = total
                self.session_peak_equity = max(self.session_peak_equity, total)

                # Session drawdown protection
                session_dd = (self.session_peak_equity - total) / self.session_peak_equity if self.session_peak_equity > 0 else 0
                if session_dd > cfg.SESSION_EQUITY_PEAK_PROTECTION_PCT:
                    await self.send_divine_message(f"SESSION DD BREACH ({session_dd:.1%}) — EMERGENCY FLAT")
                    await emergency_flat(self.bot)

                # Daily reset
                today = datetime.now(timezone.utc).date()
                if today != self.bot.state.current_day:
                    self.bot.state.start_of_day_equity = total
                    self.bot.state.current_day = today
                    self.session_peak_equity = total
                    self.velocity_windows.clear()
                    await self.send_divine_message("NEW DAY — Protection resets applied")

                self.bot.state.daily_pnl = total - self.bot.state.start_of_day_equity
                if self.bot.state.daily_pnl < -cfg.MAX_DAILY_LOSS_PCT * self.bot.state.start_of_day_equity / 100:
                    await self.send_divine_message("DAILY LOSS BREACH — EMERGENCY FLAT")
                    await emergency_flat(self.bot)

                # Core management
                await self.apply_management()
                await self.velocity_drawdown_guard()

                positions_count = len(self.bot.state.positions)
                log_core.info(
                    f"Guardian Watch | Equity ≈ ${total:.2f} | Daily PnL ${self.bot.state.daily_pnl:+.2f} | "
                    f"Positions: {positions_count} | Mode: SMALL CAPITAL QUICK PROFIT"
                )

                await asyncio.sleep(15)

            except Exception as e:
                await self.send_divine_message(f"GUARDIAN ERROR: {repr(e)}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    guardian = EclipseGuardian()
    try:
        asyncio.run(guardian.run())
    except KeyboardInterrupt:
        log_core.critical("GUARDIAN INTERRUPTED — Shutting down gracefully")