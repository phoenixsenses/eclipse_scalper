import json, pathlib
p=pathlib.Path("reports/PASSIVE_POCKET_RANKING_V2_7D_advSweep_fee0p25.json")
d=json.loads(p.read_text(encoding="utf-8"))
r=d["ranking"][0]
print("Pocket:", r["symbol"], r["rule"], "h=", r["horizon_sec"], "imb>=", r["min_imbalance"], "int>=", r["min_trade_intensity"], "spr<=", r["max_spread"])
for k,v in r["evals"].items():
    print(k, "median_filled_avg_net=", v.get("median_filled_avg_net"), "pass_rate=", v.get("pass_rate"))
