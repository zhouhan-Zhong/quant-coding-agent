---
title: Grader invariants by task category (the organizers' category contract)
category: meta
tags: invariants, grader, checker, category, derivatives pricing, fixed income, bonds, credit, cds, factor research, backtesting, risk management, var, microstructure, execution, fx, nlp, text, sentiment, cross-domain, put-call parity, delta bounds, gamma, vega, dv01, discount factor, survival probability, hazard, dollar neutral, information coefficient, self-financing, drawdown, cvar, expected shortfall, bid ask, trajectory, cip, triangular arbitrage, sentiment range, look-ahead, common mistakes
summary: The structural checks that the Track-1 category contract tells graders to assert - one block per category - plus the common mistakes it lists; verify these on your own written outputs before finishing.
---
## When this applies
Always: every task card belongs to one of ten categories, and its checker asserts the category's "financial
invariants" besides the task-specific numbers. Cross-domain tasks assert the union of their constituent categories'
checks. Run the relevant block against your own outputs (recompute from the written values) before the script exits.

## Invariants by category
derivatives-pricing: put-call parity C - P = S e^{-qT} - K e^{-rT} (1e-3 abs typical); 0 < delta_call < 1, -1 < delta_put < 0;
  gamma > 0 and vega > 0 for Europeans; price >= intrinsic (call >= max(0, S - K e^{-rT})); call <= S, put <= K e^{-rT};
  price non-decreasing in T at fixed K; butterfly convexity d2C/dK2 >= 0 across strikes; PDE residual ~ 0 on interior
  grid points; American >= European; in + out = vanilla.
  Mistakes: theta per year vs per day (/365), sigma as 20 instead of 0.20, missing e^{-rT} on K in parity, instability
  for T < 5 days on coarse grids, hard-coding one numerical method when "any method" is allowed (tests check invariants).
fixed-income: price decreases when yield rises; a bond with coupon = yield prices at par (100); DV01 ~ -dP/dy x 1e-4 within
  0.1%; ModDur x P x 1e-4 ~ DV01; swap NPV = 0 at the par rate; DF(0) = 1 and DF strictly decreasing/positive; zero and
  forward rates finite and consistent with DFs. Mistakes: mixed compounding/day counts (annual vs semi-annual, ACT/ACT vs
  30/360), clean vs dirty (accrued interest), wrong frequency in yield <-> DF conversions, DV01 not scaled by notional.
credit: survival S(T1) > S(T2) for T1 < T2 and S(0) = 1; hazard rates > 0 at every tenor; premium leg PV = protection leg PV
  at the par spread; CVA >= 0; tranche losses sum to the portfolio loss. Mistakes: spread vs upfront quotation, premium
  leg not conditioned on survival, continuous hazard mixed with annually compounded yields.
factor-research: long-short weights sum to ~0 (dollar neutral); IC (Spearman) in [-1, 1]; Sharpe from EXCESS returns with
  the sample std; beta ~ 0 after market neutralisation; signal date strictly before the return period (no look-ahead);
  turnover >= 0. Mistakes: raw-factor sorting without per-date cross-sectional ranking, survivorship/revision bias,
  annualising monthly Sharpe with 12 instead of sqrt(12).
backtesting: self-financing V_n = V_{n-1} (1 + r_n); (1 + r_1)...(1 + r_n) = V_n/V_0; signals use data available at the
  signal date only; max drawdown >= the worst single-period loss in magnitude; turnover >= 0 at every rebalance;
  Sharpe from excess returns. Mistakes: in-sample parameters reused on the same data, risk-free ignored in Sharpe,
  trading at the same close that generated the signal (real trades fill at the next open).
risk-management: CVaR/ES >= VaR at the same level; VaR_99 >= VaR_95; diversified ES <= sum of stand-alone ES
  (subadditivity); net delta ~ 0 after a delta-neutral hedge; stress P&L = sum position_i x scenario_return_i.
  Mistakes: log vs simple returns mixed, ES as the mean of ALL losses instead of losses beyond VaR, 1-day vs h-day
  (sqrt(h) scaling for normal models only).
microstructure: bid <= ask at every tick (crossed quotes are arbitrage); effective spread >= 0; Almgren-Chriss
  liquidation trades >= 0 (no re-buy) and sum to the position; market impact pushes the price against the aggressor;
  VWAP within [low, high]. Mistakes: quoted vs effective spread (2 x |trade - mid|), N + 1 intervals instead of N.
fx: EURUSD x USDJPY / EURJPY ~ 1 within bid-ask (no triangular arbitrage); CIP F = S e^{(r_d - r_f) T} within tolerance;
  Garman-Kohlhagen parity with q = r_foreign; carry sign (long high-yield, short low-yield); digital replication
  boundary conditions. Mistakes: base vs quote currency inverted, GK delta without e^{-r_f T}, wrong cycle direction.
nlp-on-finance: sentiment scores inside the declared range ([-1, 1] or [0, 1]); event_date < return window start;
  extracted figures within +/-1% of the source; labels only from the declared vocabulary; entity/risk counts are
  non-negative ints. Mistakes: millions vs billions / USD vs EUR, boilerplate (safe-harbor text) not stripped when
  required, management and analyst speech pooled when the task separates them.
cross-domain: at least one invariant from EACH constituent category (e.g. parity or delta bounds AND DV01 consistency;
  survival monotone AND CIP; sentiment range AND no look-ahead AND dollar neutrality). Mistakes: solving one
  sub-problem only, passing values between sub-problems in inconsistent units (bp vs decimal), checking only the final
  aggregate instead of the intermediate quantities.

## How to use this
- After writing the deliverables, re-open them and assert the applicable block in code (abs/rel tolerances as in the
  task, else 1e-6 relative / 1e-8 absolute for analytic identities); fix the computation if an assertion fails
  instead of relaxing the check.
- Flags such as parity_holds, is_dollar_neutral, within_tolerance must be computed from the written numbers with the
  task's threshold and cast to Python bool.
- When the task text and a textbook disagree, the task text wins for the numbers; the invariants above still hold.
