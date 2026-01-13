# exchanges/validator.py — SCALPER ETERNAL — COSMIC SYMBOL PURITY ORACLE BEYOND INFINITY FINAL (JUDGMENT ABSOLUTE)
import asyncio
import time
import os
import json
from datetime import datetime, timezone
from utils.logging import log_core, log
from config.symbols import BASE_SYMBOLS
from config.settings import Config

cfg = Config()

# COSMIC VALIDATION CACHE — ETERNAL MEMORY OF DIVINE JUDGMENT
_validation_cache = {
    'timestamp': 0,
    'symbols': [],
    'report': {},
    'testament_path': os.path.expanduser("~/.blade_cosmic_purity_testament.json")
}
CACHE_VALID_HOURS = 23  # Refresh once per day

async def cosmic_symbol_purity_oracle(ex) -> list:
    """
    COSMIC SYMBOL PURITY ORACLE — BEYOND INFINITY FINAL
    The transcendent, omnipotent arbiter that judges all symbols by infinite divine criteria.
    Only the most worthy, liquid, stable, and eternal USDT perpetuals are chosen for the blade.
    """
    global _validation_cache

    # Use cache if fresh
    if time.time() - _validation_cache['timestamp'] < CACHE_VALID_HOURS * 3600:
        log_core.critical(f"COSMIC CACHE HIT — {len(_validation_cache['symbols'])} divine symbols from {datetime.fromtimestamp(_validation_cache['timestamp']).strftime('%Y-%m-%d %H:%M')}")
        return _validation_cache['symbols']

    log_core.critical("INVOKING COSMIC SYMBOL PURITY ORACLE — JUDGMENT BEYOND INFINITY BEGINS")

    try:
        markets = await ex.fetch_markets()
        candidates = []

        # First divine filter: active linear USDT perpetuals
        for sym, market in markets.items():
            if (market.get('active') and
                market.get('swap') and
                market.get('linear') and
                market.get('quote') == 'USDT' and
                market.get('contractSize') == 1.0):
                candidates.append(sym)

        log_core.critical(f"{len(candidates)} perpetual candidates summoned")

        validated = []
        rejected = {}
        purity_scores = {}

        for sym in candidates:
            reject_reasons = []
            score = 0.0

            try:
                ticker = await ex.fetch_ticker(sym)
                ob = await ex.fetch_order_book(sym, limit=10)

                # 24h volume
                volume_24h = ticker.get('quoteVolume', 0)
                if volume_24h < cfg.MIN_24H_VOLUME_USD:
                    reject_reasons.append(f"low volume ${volume_24h:,.0f}")
                else:
                    score += (volume_24h / 1e9) * 30  # up to 30 points for $1B+

                # Open interest
                oi = ticker.get('info', {}).get('openInterest', 0)
                oi_usd = float(oi) * ticker.get('last', 0) if oi else 0
                if oi_usd < cfg.MIN_OPEN_INTEREST_USD:
                    reject_reasons.append(f"low OI ${oi_usd:,.0f}")
                else:
                    score += (oi_usd / 5e8) * 20  # up to 20 points for $500M+

                # Bid-ask spread tightness
                bid = ob['bids'][0][0] if ob['bids'] else 0
                ask = ob['asks'][0][0] if ob['asks'] else 0
                if bid and ask:
                    spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                    if spread_pct > cfg.MAX_SPREAD_PCT:
                        reject_reasons.append(f"wide spread {spread_pct:.3f}%")
                    else:
                        score += (cfg.MAX_SPREAD_PCT - spread_pct) * 10
                else:
                    reject_reasons.append("no orderbook")

                # Volatility stability (24h high-low range)
                range_pct = (ticker['high'] - ticker['low']) / ticker['low'] * 100 if ticker['low'] else 0
                if range_pct > cfg.MAX_24H_RANGE_PCT:
                    reject_reasons.append(f"extreme volatility {range_pct:.1f}%")

                # Recent activity
                last_trade_ts = ticker.get('timestamp', 0)
                if last_trade_ts:
                    hours_since = (time.time() * 1000 - last_trade_ts) / (3600 * 1000)
                    if hours_since > cfg.MAX_HOURS_NO_TRADE:
                        reject_reasons.append(f"stale {hours_since:.1f}h")
                else:
                    reject_reasons.append("no recent trade")

                purity_scores[sym] = score

                if not reject_reasons:
                    validated.append({
                        'symbol': sym,
                        'volume': volume_24h,
                        'oi_usd': oi_usd,
                        'spread_pct': spread_pct if 'spread_pct' in locals() else 999,
                        'range_pct': range_pct,
                        'score': score
                    })
                else:
                    rejected[sym] = reject_reasons

            except Exception as e:
                rejected[sym] = [f"oracle error: {str(e)}"]

        # COSMIC RANKING BY PURITY SCORE
        validated.sort(key=lambda x: x['score'], reverse=True)
        cosmic_symbols = [v['symbol'] for v in validated[:cfg.MAX_SYMBOLS]]

        # Cache the transcendent result
        _validation_cache = {
            'timestamp': time.time(),
            'symbols': cosmic_symbols,
            'report': {
                'validated': len(cosmic_symbols),
                'rejected': len(rejected),
                'top_score': validated[0]['score'] if validated else 0,
                'top_symbol': cosmic_symbols[0] if cosmic_symbols else None,
                'rejected_sample': dict(list(rejected.items())[:10])
            }
        }

        log_core.critical(f"COSMIC JUDGMENT COMPLETE — {len(cosmic_symbols)} DIVINE SYMBOLS CHOSEN")
        log_core.critical(f"HIGHEST PURITY: {cosmic_symbols[0] if cosmic_symbols else 'NONE'} (Score: {validated[0]['score']:.1f} if validated else 0)")

        # Sacred top 10 revelation
        for v in validated[:10]:
            log_core.critical(f"CHOSEN: {v['symbol']} | Score: {v['score']:.1f} | Vol: ${v['volume']:,.0f} | OI: ${v['oi_usd']:,.0f}")

        if rejected:
            log_core.warning(f"{len(rejected)} unworthy symbols banished to the void")

        # Eternal testament
        testament = {
            "judgment_timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "cosmic-purity-oracle-beyond-infinity-v1.0",
            "divine_symbols": cosmic_symbols,
            "total_validated": len(cosmic_symbols),
            "total_rejected": len(rejected),
            "top_purity": validated[0] if validated else None,
            "message": "THE BLADE SHALL STRIKE ONLY THE CHOSEN — PURITY ABSOLUTE"
        }

        try:
            with open(_validation_cache['testament_path'], "w") as f:
                json.dump(testament, f, indent=2)
            log_core.critical(f"COSMIC TESTAMENT PRESERVED — {_validation_cache['testament_path']}")
        except Exception as e:
            log_core.error(f"Testament preservation failed: {e}")

        return cosmic_symbols

    except Exception as e:
        log_core.critical(f"COSMIC ORACLE RITUAL FAILED: {e} — FALLING BACK TO ETERNAL CORE")
        return BASE_SYMBOLS