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

print("dualSidePosition:", ex.fapiPrivateGetPositionSideDual())
print("multiAssetsMargin:", ex.fapiPrivateGetMultiAssetsMargin())

# What Binance expects in Hedge mode:
print("\nHEDGE MODE PARAMS YOU MUST USE:")
print("OPEN LONG : side=buy  positionSide=LONG")
print("OPEN SHORT: side=sell positionSide=SHORT")
print("CLOSE LONG: side=sell positionSide=LONG (reduceOnly=True)")
print("CLOSE SHORT:side=buy  positionSide=SHORT (reduceOnly=True)")
