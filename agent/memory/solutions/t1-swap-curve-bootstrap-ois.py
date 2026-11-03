# memory-solution
# unit: t1-swap-curve-bootstrap-ois | category: rates-curves | tags: swap-curve, ois, dual-curve, libor, irs, bootstrap, brentq, log-linear-df, 30-360, act-360, pv01, debug-template
# deliverables: ois_discount_curve.csv, libor_forward_curve.csv, repriced_quotes.csv, swap_valuation.json, summary.json
# All times from valuation_date: t = ACT/365.25 for interpolation/zero rates, tau = ACT/360 for OIS/deposit/float accrual.
# OIS & deposit pillars: DF = 1/(1 + r * tau_act360(val, mat)); zero = -ln DF / t (continuous).
# DF interpolation log-linear in t; flat continuous-zero extrapolation on both sides (DF = exp(-z_edge * t)).
# IRS (start = spot = val + 2 business days, unadjusted schedules built backward from maturity in 6M/3M steps):
#   annuity = sum tau_30/360 * DF_OIS;  float = sum tau_act360 * F_j * DF_OIS,  F_j = (DF_L(t_{j-1})/DF_L(t_j) - 1)/tau_act360
#   par = float/annuity; bootstrap LIBOR DF at each IRS pillar sequentially with brentq (OIS discounting, LIBOR forwards).
# Swap: pay_fixed npv = float - fixed; pv01 = |npv(LIBOR zeros + 1bp: DF_L *= exp(-1e-4 t)) - npv|, OIS unbumped.
# Template bugs fixed: fixed leg 30/360 (not ACT/360); OIS discounting inside the IRS bootstrap; OIS built first; npv sign by pay_receive.
import os, json, math, csv
from datetime import date, datetime, timedelta
from scipy.optimize import brentq

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

def rd(name): return list(csv.DictReader(open(os.path.join(DATA_DIR, name))))
conv = json.load(open(os.path.join(DATA_DIR, "conventions.json")))
ois_q, dep_q, irs_q = rd("ois_quotes.csv"), rd("deposits.csv"), rd("irs_3m_quotes.csv")
swap = json.load(open(os.path.join(DATA_DIR, "swap_to_value.json")))
pdate = lambda s: datetime.strptime(s, "%Y-%m-%d").date()
VAL = pdate(conv["valuation_date"])
SPOT = VAL
for _ in range(int(conv.get("spot_lag_business_days", 2))):      # Mon-Fri calendar, no holidays
    SPOT += timedelta(days=1)
    while SPOT.weekday() >= 5: SPOT += timedelta(days=1)

def add_months(d, m):
    mt = d.month - 1 + m; y, mo = d.year + mt // 12, mt % 12 + 1
    nxt = date(y + (mo == 12), 1 if mo == 12 else mo + 1, 1)
    return date(y, mo, min(d.day, (nxt - timedelta(days=1)).day))
act360 = lambda a, b: (b - a).days / 360.0
tyrs = lambda d: (d - VAL).days / 365.25
def d30360(a, b):
    d1 = min(a.day, 30); d2 = 30 if (d1 == 30 and b.day == 31) else min(b.day, 30)
    return (360 * (b.year - a.year) + 30 * (b.month - a.month) + d2 - d1) / 360.0

def df_at(curve, t):                                   # curve: sorted list of (t, DF)
    if t <= 0: return 1.0
    if t <= curve[0][0]: return math.exp(math.log(curve[0][1]) / curve[0][0] * t)
    if t >= curve[-1][0]: return math.exp(math.log(curve[-1][1]) / curve[-1][0] * t)
    for (t0, d0), (t1, d1) in zip(curve[:-1], curve[1:]):
        if t0 <= t <= t1:
            w = (t - t0) / (t1 - t0); return math.exp((1 - w) * math.log(d0) + w * math.log(d1))

def schedule(eff, mat, months):                        # backward, unadjusted, dates strictly after eff
    ds = [mat]
    while add_months(ds[-1], -months) > eff: ds.append(add_months(ds[-1], -months))
    return ds[::-1]

def annuity(eff, mat, ois):
    prev, a = eff, 0.0
    for d in schedule(eff, mat, 6): a += d30360(prev, d) * df_at(ois, tyrs(d)); prev = d
    return a
def float_pv(eff, mat, ois, lib):
    prev, pv = eff, 0.0
    for d in schedule(eff, mat, 3):
        tau = act360(prev, d); f = (df_at(lib, tyrs(prev)) / df_at(lib, tyrs(d)) - 1.0) / tau
        pv += tau * f * df_at(ois, tyrs(d)); prev = d
    return pv
par = lambda eff, mat, ois, lib: float_pv(eff, mat, ois, lib) / annuity(eff, mat, ois)

# 1) OIS curve (closed form), 2) deposits, 3) sequential IRS bootstrap with OIS discounting
ois = sorted((tyrs(pdate(q["maturity_date"])), 1.0 / (1.0 + float(q["par_rate"]) * act360(VAL, pdate(q["maturity_date"])))) for q in ois_q)
lib = sorted((tyrs(pdate(q["maturity_date"])), 1.0 / (1.0 + float(q["rate"]) * act360(VAL, pdate(q["maturity_date"])))) for q in dep_q)
for q in sorted(irs_q, key=lambda q: q["maturity_date"]):
    mat, target = pdate(q["maturity_date"]), float(q["par_rate"]); t = tyrs(mat)
    res = lambda dfx: par(SPOT, mat, ois, sorted(lib + [(t, dfx)])) - target
    lib = sorted(lib + [(t, brentq(res, 1e-6, 1.0, xtol=1e-15, rtol=1e-15, maxiter=200))])

with open(os.path.join(OUT_DIR, "ois_discount_curve.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pillar_date", "time_years", "discount_factor", "zero_rate_continuous"])
    for q in sorted(ois_q, key=lambda q: q["maturity_date"]):
        t = tyrs(pdate(q["maturity_date"])); d = df_at(ois, t)
        w.writerow([q["maturity_date"], repr(t), repr(d), repr(-math.log(d) / t)])
with open(os.path.join(OUT_DIR, "libor_forward_curve.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pillar_date", "time_years", "libor_discount_factor", "zero_rate_continuous", "forward_3m"])
    for md in sorted(q["maturity_date"] for q in dep_q + irs_q):
        m = pdate(md); t = tyrs(m); d = df_at(lib, t); m3 = add_months(m, 3)
        fwd = (d / df_at(lib, tyrs(m3)) - 1.0) / act360(m, m3)
        w.writerow([md, repr(t), repr(d), repr(-math.log(d) / t), repr(fwd)])

rows = []
for kind, qs, key, curve in (("ois", ois_q, "par_rate", ois), ("deposit", dep_q, "rate", lib)):
    for q in qs:
        m = pdate(q["maturity_date"]); model = (1.0 / df_at(curve, tyrs(m)) - 1.0) / act360(VAL, m)
        rows.append([kind, q["tenor_label"], float(q[key]), model, (model - float(q[key])) * 1e4])
for q in irs_q:
    model = par(SPOT, pdate(q["maturity_date"]), ois, lib); r0 = float(q["par_rate"])
    rows.append(["irs_3m", q["tenor_label"], r0, model, (model - r0) * 1e4])
with open(os.path.join(OUT_DIR, "repriced_quotes.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["instrument_type", "tenor_label", "input_par_rate", "model_par_rate", "residual_bps"])
    for r in rows: w.writerow([r[0], r[1], repr(r[2]), repr(r[3]), repr(r[4])])

def value(lib_curve):
    eff, mat, N = pdate(swap["effective_date"]), pdate(swap["maturity_date"]), float(swap["notional"])
    fix = N * float(swap["fixed_rate"]) * annuity(eff, mat, ois); flt = N * float_pv(eff, mat, ois, lib_curve)
    return (flt - fix if swap["pay_receive"] == "pay_fixed" else fix - flt), fix, flt
npv, fix, flt = value(lib)
bump = float(swap.get("pv01_bump_bps", 1.0)) * 1e-4
npv_b = value([(t, d * math.exp(-bump * t)) for t, d in lib])[0]
json.dump({"npv": npv, "fixed_leg_pv": fix, "float_leg_pv": flt, "pv01": abs(npv_b - npv), "pay_receive": swap["pay_receive"]},
          open(os.path.join(OUT_DIR, "swap_valuation.json"), "w"), indent=2)

zero = lambda curve, d: -math.log(df_at(curve, tyrs(d))) / tyrs(d)
m5, m10 = add_months(VAL, 60), add_months(VAL, 120)
irs10 = pdate(next(q["maturity_date"] for q in irs_q if q["tenor_label"] == "10Y"))
json.dump({"ois_5y_zero_rate_continuous": zero(ois, m5), "ois_10y_zero_rate_continuous": zero(ois, m10),
           "libor_5y_zero_rate_continuous": zero(lib, m5), "libor_10y_zero_rate_continuous": zero(lib, m10),
           "libor_10y_par_swap_rate": par(SPOT, irs10, ois, lib),
           "max_bootstrap_residual_bps": max(abs(r[4]) for r in rows)},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
