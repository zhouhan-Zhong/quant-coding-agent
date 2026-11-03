# DELIVERABLES: results.json
import json, math, os
with open("/app/curve_data.json") as f:
    p = json.load(f)
mats = list(p["maturities"]); rates = list(p["par_rates"]); freq = int(p["coupon_freq"])
known_t, known_z = [], []
def interp_z(t):
    if not known_t:
        return None
    if t <= known_t[0]:
        return known_z[0]
    if t >= known_t[-1]:
        return known_z[-1]
    for i in range(len(known_t) - 1):
        if known_t[i] <= t <= known_t[i + 1]:
            t0, t1 = known_t[i], known_t[i + 1]
            return known_z[i] + (known_z[i + 1] - known_z[i]) * (t - t0) / (t1 - t0)
dfs, zs = {}, {}
for T, c in zip(mats, rates):
    n = int(round(T * freq)); coupon = c / freq
    s = 0.0
    for k in range(1, n):
        tk = k / freq
        z = zs.get(tk)
        if z is None:
            z = interp_z(tk)
            if z is None:
                z = -math.log(1.0 / (1.0 + coupon)) / tk
        s += math.exp(-z * tk)
    df = (1.0 - coupon * s) / (1.0 + coupon)
    z = -math.log(df) / T
    dfs[T] = df; zs[T] = z
    known_t.append(T); known_z.append(z)
fwd = {}
for i, T in enumerate(mats):
    if i == 0:
        fwd[T] = zs[T]
    else:
        Tp = T - 1
        zp = zs.get(Tp, interp_z(Tp))
        fwd[T] = (zs[T] * T - zp * Tp) / 1.0
os.makedirs("/app/output", exist_ok=True)
out = {"zero_rates": {str(T): round(zs[T], 8) for T in mats},
       "discount_factors": {str(T): round(dfs[T], 8) for T in mats},
       "forward_rates": {str(T): round(fwd[T], 8) for T in mats}}
with open("/app/output/results.json", "w") as f:
    json.dump(out, f, indent=2)
