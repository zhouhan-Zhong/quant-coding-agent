---
title: Corporate valuation conventions: FCFF and FCFE, WACC with market weights and CAPM cost of equity, levered and unlevered beta, Gordon and exit-multiple terminal values, mid-year discounting, enterprise-to-equity bridge, diluted shares, sensitivity grids and financial ratios
category: portfolio
tags: dcf, discounted cash flow, free cash flow, fcff, fcfe, nopat, ebit, ebitda, capex, depreciation, working capital, wacc, cost of capital, cost of equity, capm, beta, levered beta, unlevered beta, hamada, cost of debt, tax shield, terminal value, gordon growth, perpetuity growth, exit multiple, mid-year convention, enterprise value, equity value, net debt, diluted shares, treasury stock method, sensitivity table, ev/ebitda, p/e, roe, roic, dupont
summary: FCFF = EBIT (1 - t) + D&A - CapEx - change in NWC; FCFE = FCFF - interest (1 - t) + net borrowing; WACC = E/V r_e + D/V r_d (1 - t) with market values, r_e = r_f + beta (E[R_m] - r_f); beta_L = beta_U (1 + (1 - t) D/E); Gordon terminal value at year N = FCF_N (1 + g) / (WACC - g) discounted with the year-N factor (g < WACC); mid-year convention discounts year t at (1 + WACC)^-(t - 0.5); equity = EV - debt + cash (- minorities - preferred); per share with diluted shares (treasury stock method).
triggers: text:discounted cash flow, text:wacc, text:free cash flow to
---
## When this applies
"Value the company with a 5-year DCF", "compute WACC", "terminal value with 2 % perpetual growth / 10x EBITDA exit",
"equity value per share", "sensitivity of the value to WACC and g", "ROE / ROIC / DuPont decomposition".

## Cash flows
- NOPAT = EBIT x (1 - tax rate); FCFF = NOPAT + depreciation & amortisation - capital expenditure - increase in net working
  capital (NWC = operating current assets - operating current liabilities; an increase is a cash outflow).
- FCFE = FCFF - interest x (1 - t) + net borrowing (new debt - repayments); discount FCFE at the cost of equity, FCFF at WACC.
- Forecast items given as % of revenue: compute revenue first, then each line; keep signs consistent (CapEx positive
  number subtracted).

## Discount rates
- Cost of equity (CAPM): r_e = r_f + beta x (market return - r_f) (+ size / country premia only if stated).
- Beta re-levering (Hamada): beta_U = beta_L / (1 + (1 - t) D/E); beta_L(target) = beta_U (1 + (1 - t) D/E target).
- WACC = E/(D + E) r_e + D/(D + E) r_d (1 - t) with MARKET values of equity (price x shares) and debt (book if no market
  value is given); preferred stock adds a term at its own cost.

## Terminal value and discounting
- Gordon growth at year N: TV_N = FCF_{N+1} / (WACC - g) = FCF_N (1 + g) / (WACC - g); requires g < WACC.
  Exit multiple: TV_N = metric_N x multiple (EBITDA_N x EV/EBITDA).
- PV = sum_{t=1..N} FCF_t / (1 + WACC)^t + TV_N / (1 + WACC)^N; mid-year convention uses exponent t - 0.5 for the flows
  (the TV keeps N for a Gordon TV on end-of-year flows unless the task says otherwise).
- Valuation date not at a year end: discount with fractional years (t - stub).

## Enterprise to equity
- Equity value = EV - total debt + cash and equivalents - minority interest - preferred equity + non-operating assets.
- Diluted shares: basic + in-the-money options via the treasury stock method (n x (1 - K / P)) + convertible shares if
  dilutive; value per share = equity value / diluted shares (iterate if P is the computed price, when asked).

## Ratios
- EV/EBITDA, P/E = price / EPS, EPS = net income / weighted shares; ROE = net income / average equity; ROA = net income /
  average assets; ROIC = NOPAT / invested capital (debt + equity - cash); DuPont: ROE = (NI / sales)(sales / assets)
  (assets / equity).  Use averages or year-end balances exactly as the task says.

## Invariants worth printing
- g < WACC; the TV share of EV (commonly 50-80 %); value falls when WACC rises and rises with g in a sensitivity grid;
  EV - net debt reconciles to the equity value; WACC lies between r_d (1 - t) and r_e.
