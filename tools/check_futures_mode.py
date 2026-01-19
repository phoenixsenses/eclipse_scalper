from dotenv import load_dotenv
load_dotenv()
import os, ccxt

ex = ccxt.binance({
    "apiKey": os.getenv("BINANCE_API_KEY"),
    "secret": os.getenv("BINANCE_API_SECRET"),
    "options": {"defaultType": "future"},
    "enableRateLimit": True,
})
ex.load_markets()

print("positionSideDual:", ex.fapiPrivateGetPositionSideDual())     # true=Hedge, false=One-way
print("multiAssetsMargin:", ex.fapiPrivateGetMultiAssetsMargin())   # true=Multi-assets
