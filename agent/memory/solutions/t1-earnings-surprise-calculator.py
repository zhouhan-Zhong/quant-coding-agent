# memory-solution
# unit: t1-earnings-surprise-calculator | category: backtest-factors | tags: earnings-surprise, sue, standardized-unexpected-earnings, yoy-growth, beat-miss, json-aggregation
# deliverables: results.json
# Per company-quarter: surprise$ = A - E; surprise% = (A - E)/|E|*100; SUE = (A - E)/std_est (0 if std == 0);
#   yoy% = (A - prior)/|prior|*100; class = beat if surprise% > 2, miss if < -2, else meet.
# Summary per company = plain means over quarters (beat_rate = share of "beat"); keep input quarter ORDER (checker indexes by i).
# Output keyed by ticker -> {"quarters": [...], "summary": {...}}, floats rounded to 4 dp; guard zero denominators.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

data = json.load(open(os.path.join(DATA_DIR, "earnings.json")))

def pct(num, den):                       # percentage change with |den| normalisation, 0 if den == 0
    return float(num) / abs(float(den)) * 100.0 if den != 0 else 0.0

results = {}
for comp in data["companies"]:
    quarters = []
    for q in comp["quarters"]:
        A, E, sd, prior = float(q["actual_eps"]), float(q["consensus_estimate"]), float(q["std_estimate"]), float(q["prior_year_eps"])
        surprise_pct = pct(A - E, E)
        sue = (A - E) / sd if sd != 0 else 0.0
        cls = "beat" if surprise_pct > 2.0 else ("miss" if surprise_pct < -2.0 else "meet")
        quarters.append({"period": q["period"], "surprise_dollars": round(A - E, 4), "surprise_pct": round(surprise_pct, 4),
                         "sue": round(sue, 4), "yoy_growth": round(pct(A - prior, prior), 4), "classification": cls})
    df = pd.DataFrame(quarters)
    summary = {"avg_surprise_pct": round(float(df["surprise_pct"].mean()), 4),
               "avg_sue": round(float(df["sue"].mean()), 4),
               "beat_rate": round(float((df["classification"] == "beat").mean()), 4),
               "avg_yoy_growth": round(float(df["yoy_growth"].mean()), 4)}
    results[comp["ticker"]] = {"quarters": quarters, "summary": summary}

json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
