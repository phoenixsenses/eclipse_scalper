import json, pathlib, pprint

p = pathlib.Path("reports/PASSIVE_POCKET_RANKING_V2_7D_fastpass.json")
d = json.loads(p.read_text(encoding="utf-8"))

ranking = d["ranking"]

print("\nRanking count:", len(ranking))

# ilk pocket
first = ranking[0]

print("\nTop-level pocket keys:")
print(sorted(first.keys()))

print("\nScore:", first["score"])
print("Symbol:", first["symbol"])
print("Rule:", first["rule"])

print("\n--- evals structure ---")

evals = first.get("evals", [])

print("evals count:", len(evals))

if evals:
    print("\nEval keys:")
    print(sorted(evals[0].keys()))

    print("\nSample eval item:")
    pprint.pprint(evals[0], width=120)
