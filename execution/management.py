# execution/management.py
import asyncio
from config.settings import Config
from utils.logging import log

cfg = Config()

async def manage_position(bot, sym: str, pos):
    """Trailing stop + breakeven management"""
    while sym in bot.state.positions:
        await asyncio.sleep(17)
        try:
            current_price = bot.data.price.get(sym, pos.entry_price)
            if pos.side == 'long':
                pnl_r = (current_price - pos.entry_price) / pos.atr
            else:
                pnl_r = (pos.entry_price - current_price) / pos.atr

            # Breakeven move
            if pnl_r >= 1.0 and not pos.breakeven_moved:
                be_price = pos.entry_price * (1 + cfg.BREAKEVEN_BUFFER if pos.side == 'long' else 1 - cfg.BREAKEVEN_BUFFER)
                await bot.ex.create_order(
                    symbol=sym,
                    type='STOP_MARKET',
                    side='sell' if pos.side == 'long' else 'buy',
                    amount=abs(pos.size),
                    params={
                        'stopPrice': bot.ex.price_to_precision(sym, be_price),
                        'reduceOnly': True
                    }
                )
                pos.breakeven_moved = True

            # Activate trailing stop
            if pnl_r >= cfg.TRAILING_ACTIVATION_R and not pos.trailing_active:
                await bot.ex.create_order(
                    symbol=sym,
                    type='TRAILING_STOP_MARKET',
                    side='sell' if pos.side == 'long' else 'buy',
                    amount=abs(pos.size),
                    params={
                        'callbackRate': round(cfg.TRAILING_MULTIPLIER * 100, 2),
                        'reduceOnly': True
                    }
                )
                pos.trailing_active = True
                await bot.notify.speak(
                    f"TRAILING STOP ACTIVATED {sym} @ {pnl_r:.2f}R",
                    'critical'
                )

        except Exception as e:
            log.error(f"Position management error {sym}: {e}")