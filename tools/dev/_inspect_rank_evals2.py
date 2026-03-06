import json, pathlib, pprint

p = pathlib.Path("reports/PASSIVE_POCKET_RANKING_V2_7D_fastpass.json")
d = json.loads(p.read_text(encoding="utf-8"))

r = d["ranking"][0]
evals = r.get("evals")

print("type(evals):", type(evals))
print("evals keys:", list(evals.keys()) if isinstance(evals, dict) else None)

if isinstance(evals, dict):
    k0 = next(iter(evals.keys()))
    print("\nFirst eval key:", k0)
    print("type(evals[k0]):", type(evals[k0]))
    print("\nSample eval value:")
    pprint.pprint(evals[k0], width=140)
elif isinstance(evals, list) and evals:
    print("\nEval keys:", sorted(evals[0].keys()))
    pprint.pprint(evals[0], width=140)
else:
    print("evals is empty or unknown structure")
