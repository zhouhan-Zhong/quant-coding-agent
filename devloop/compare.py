"""A/B comparison of evaluation runs: keep a change only if it helps by more than the run-to-run noise.

Per unit, each arm's pass rate is the mean over its repeats.  The arms are compared on the mean number of passed
units per run; the units whose rate moved are tested with an exact two-sided sign test (McNemar's test when each
arm has one run).  Local runs move by about +-2-3 units out of 86 between identical repeats, so the default
decision threshold is a gain of 2 units, and the p-value is reported next to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

from .runs import Run, pass_rates


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact binomial test of k successes in n trials at p = 0.5."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@dataclass
class Comparison:
    label_a: str
    label_b: str
    units: list[str]
    rate_a: dict[str, float]
    rate_b: dict[str, float]
    gained: list[str]
    lost: list[str]
    p_value: float
    min_gain: float
    held_out: list[str] = field(default_factory=list)   # units the change was not derived from

    @property
    def total_a(self) -> float:
        return sum(self.rate_a[u] for u in self.units)

    @property
    def total_b(self) -> float:
        return sum(self.rate_b[u] for u in self.units)

    @property
    def delta(self) -> float:
        return self.total_b - self.total_a

    @property
    def held_out_delta(self) -> float | None:
        if not self.held_out:
            return None
        return sum(self.rate_b[u] - self.rate_a[u] for u in self.held_out)

    @property
    def keep(self) -> bool:
        return self.delta >= self.min_gain


def compare(runs_a: list[Run], runs_b: list[Run], *, min_gain: float = 2.0, sources: list[str] = (),
            label_a: str = "A", label_b: str = "B") -> Comparison:
    ra, rb = pass_rates(runs_a), pass_rates(runs_b)
    units = sorted(set(ra) & set(rb))
    if not units:
        raise ValueError("the two arms have no unit in common")
    gained = [u for u in units if rb[u] > ra[u]]
    lost = [u for u in units if rb[u] < ra[u]]
    held = [u for u in units if u not in set(sources)] if sources else []
    return Comparison(label_a=label_a, label_b=label_b, units=units, rate_a=ra, rate_b=rb, gained=gained, lost=lost,
                      p_value=sign_test_p(len(gained), len(gained) + len(lost)), min_gain=min_gain, held_out=held)


def render(c: Comparison) -> str:
    n = len(c.units)
    lines = [f"# A/B: {c.label_b} vs {c.label_a}", "",
             "| | passed (mean per run) | rate |", "|---|---|---|",
             f"| {c.label_a} | {c.total_a:g}/{n} | {c.total_a / n:.4f} |",
             f"| {c.label_b} | {c.total_b:g}/{n} | {c.total_b / n:.4f} |", "",
             f"Change: **{c.delta:+g} units**; {len(c.gained)} unit(s) gained, {len(c.lost)} lost; "
             f"two-sided sign test p = {c.p_value:.3f}."]
    if c.held_out_delta is not None:
        lines.append(f"On the {len(c.held_out)} units the change was not derived from: {c.held_out_delta:+g} units.")
    lines += ["", f"Decision: **{'keep' if c.keep else 'discard'}** (threshold: a gain of at least {c.min_gain:g} units).", ""]
    if c.gained:
        lines.append("Gained: " + ", ".join(f"{u} ({c.rate_a[u]:g}->{c.rate_b[u]:g})" for u in c.gained))
    if c.lost:
        lines.append("Lost: " + ", ".join(f"{u} ({c.rate_a[u]:g}->{c.rate_b[u]:g})" for u in c.lost))
    return "\n".join(lines) + "\n"
