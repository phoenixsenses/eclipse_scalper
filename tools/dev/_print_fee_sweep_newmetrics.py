import json, pathlib
p=pathlib.Path("reports/PASSIVE_POCKET_RANKING_V2_7D_feeSweep.json")
d=json.loads(p.read_text(encoding="utf-8"))
r=d["ranking"][0]
print("Pocket:", r["symbol"], r["rule"], "h=", r["horizon_sec"], "imb>=", r["min_imbalance"], "int>=", r["min_trade_intensity"], "spr<=", r["max_spread"])
print("Top fields:", {k:r.get(k) for k in ["score","score_raw_core","score_raw_stress","score_raw_min","robust_core","robust_stress","insufficient_fill_rate","attempt_fill_rate_median","attempts_per_min_median","median_net_per_attempt"]})
for k,v in r["evals"].items():
    print(k,
          "median_net_per_attempt=", v.get("median_net_per_attempt"),
          "attempt_fill_rate_median=", v.get("attempt_fill_rate_median"),
          "attempts_per_min_median=", v.get("attempts_per_min_median"),
          "median_filled_avg_net=", v.get("median_filled_avg_net"),
          "pass_rate=", v.get("pass_rate"))
