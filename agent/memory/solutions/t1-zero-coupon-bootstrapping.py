# memory-solution
# unit: t1-zero-coupon-bootstrapping | category: rates-curves | tags: bootstrap, par-curve, zero-rates, discount-factors, forward-rates, linear-zero-interpolation, brentq, continuous-compounding
# deliverables: results.json
# Par bond: 1 = (c/f) sum_{k<n} DF(k/f) + (1 + c/f) DF(T); DF = exp(-z t), zero rates linear in t between pillars, flat outside.
# Coupon dates inside the gap (T_prev, T) depend on the unknown z(T): solve z(T) with brentq so the par equation holds exactly
#   (closed form DF(T) = (1 - c/f sum DF)/(1 + c/f) only when every earlier coupon date is already known).
# z = -ln DF / T; 1y forward fwd(T) = z(T) T - z(T-1)(T-1) with z(T-1) interpolated on the final curve; fwd(first) = z(first). Round to 8 dp.
import os, json, math
import numpy as np
from scipy.optimize import brentq

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

p = json.load(open(os.path.join(DATA_DIR, "curve_data.json")))
mats, rates, freq = [float(m) for m in p["maturities"]], [float(r) for r in p["par_rates"]], int(p["coupon_freq"])
zt, zz = [], []                                            # bootstrapped pillars (t, z)

def zero(t, ts, zs):                                       # linear in t, flat extrapolation
    return float(np.interp(t, ts, zs))

for T, c in sorted(zip(mats, rates)):
    cpn, n = c / freq, int(round(T * freq))
    def par_err(zT):                                       # price - 1 with the trial pillar appended
        ts, zs = zt + [T], zz + [zT]
        return cpn * sum(math.exp(-zero(k / freq, ts, zs) * k / freq) for k in range(1, n)) + (1 + cpn) * math.exp(-zT * T) - 1.0
    zT = brentq(par_err, -0.5, 1.0, xtol=1e-15, rtol=1e-15, maxiter=200)
    zt.append(T); zz.append(zT)

key = lambda T: str(int(T)) if float(T).is_integer() else str(T)
Z = dict(zip(zt, zz))
fwd = {T: (Z[T] if i == 0 else Z[T] * T - zero(T - 1, zt, zz) * (T - 1)) for i, T in enumerate(zt)}
out = {"zero_rates": {key(T): round(Z[T], 8) for T in zt},
       "discount_factors": {key(T): round(math.exp(-Z[T] * T), 8) for T in zt},
       "forward_rates": {key(T): round(fwd[T], 8) for T in zt}}
json.dump(out, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
