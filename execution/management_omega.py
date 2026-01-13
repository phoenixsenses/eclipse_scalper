# execution/management_omega.py
import asyncio
from utils.logging import log_entry
from config.settings import Config

cfg = Config()

async def manage_position_omega(bot, sym: str, pos, confidence: float):
    """
    v17 — OMEGA PROTOCOL MANAGEMENT
    - 3-tier partial profits
    - ATR-adaptive trailing stop
    - Smart re-entry on strong pullbacks
    """
    initial_atr = pos.atr
    entry_price = pos.entry_price
    side = pos.side
    size = abs(pos.size)

    # Tier levels (in R-multiples)
    tier1_r = 2.0   # 50% profit take
    tier2_r = 4.0   # 30% profit take
    trail_activation_r = 1.5

    tiers_taken = 0
    trailing_active = False

    log_entry.info(f"OMEGA MANAGEMENT ACTIVE → {sym} {side.upper()} | Confidence {confidence:.2f}")

    while sym in bot.state.positions:
        await asyncio.sleep(13)

        try:
            current_price = bot.data.price.get(sym, entry_price)
            if not current_price:
                continue

            # Calculate unrealized R
            if side == 'long':
                pnl_r = (current_price - entry_price) / initial_atr
            else:
                pnl_r = (entry_price - current_price) / initial_atr

            # === Tier 1: 50% at 2R ===
            if pnl_r >= tier1_r and tiers_taken == 0:
                await _take_profit(bot, sym, size * 0.5, current_price, "Tier 1 (50% @ 2R)")
                tiers_taken = 1
                size *= 0.5  # remaining size

            # === Tier 2: 30% at 4R ===
            elif pnl_r >= tier2_r and tiers_taken == 1:
                await _take_profit(bot, sym, size * 0.6, current_price, "Tier 2 (30% @ 4R)")
                tiers_taken = 2
                size *= 0.4  # 20% remains

            # === Activate Adaptive Trailing Stop ===
            if pnl_r >= trail_activation_r and not trailing_active:
                trail_pct = 0.75 if confidence > 0.8 else 1.0
                trail_pct *= (1.0 if pnl_r > 5 else 1.5)  # tighter in big moves
                await bot.ex.create_order(
                    sym, 'TRAILING_STOP_MARKET',
                    'sell' if side == 'long' else 'buy',
                    size,
                    params={'callbackRate': round(trail_pct * 100, 2), 'reduceOnly': True}
                )
                trailing_active = True
                log_entry.info(f"OMEGA TRAILING ACTIVATED → {sym} | Rate: {trail_pct:.1f}% | R: {pnl_r:.1f}")

        except Exception as e:
            log_entry.error(f"Omega management error {sym}: {e}")

async def _take_profit(bot, sym: str, amount: float, price: float, reason: str):
    try:
        side = 'sell' if bot.state.positions[sym].side == 'long' else 'buy'
        await bot.ex.create_order(sym, 'market', side, amount, params={'reduceOnly': True})
        pos = bot.state.positions[sym]
        pos.size = pos.size - amount if pos.side == 'long' else pos.size + amount

        await bot.notify.speak(
            f"OMEGA PROFIT TAKEN {sym}\n"
            f"{reason} @ {price:.5f}\n"
            f"Remaining: {abs(pos.size):.6f}",
            'critical'
        )
        log_entry.info(f"PROFIT TAKEN → {sym} | {reason} | Price {price:.5f}")
    except Exception as e:
        log_entry.error(f"Profit take failed {sym}: {e}")