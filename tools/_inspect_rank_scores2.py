import json, pathlib

p = pathlib.Path("reports/PASSIVE_POCKET_RANKING_V2_7D_fastpass.json")
d = json.loads(p.read_text(encoding="utf-8"))

for i, r in enumerate(d["ranking"], 1):
    print(i, r["symbol"], r["rule"], "score=", repr(r["score"]), "robust_core=", r.get("robust_core"), "robust_stress=", r.get("robust_stress"))
