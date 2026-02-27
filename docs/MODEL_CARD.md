# Model Card — BBO Capstone Optimization Approach

## 1) Overview
**Approach name:** Weekly Hybrid Black-Box Optimizer

**Core components:**
- Ridge regression local surrogate
- Gaussian Process + Expected Improvement (BO-style)
- Trust-region local candidate generation
- Manual directional heuristics / minimal exploration baselines

**Objective direction:** maximize `y` for all functions.

---

## 2) Intended use
### Intended
- Weekly sequential optimization with limited observations.
- Transparent educational workflow for method evolution and reflection.

### Not intended
- Safety-critical deployment.
- Strong global-optimality claims under sparse data.

---

## 3) Week-by-week method evolution
- **Weeks 1–4:** random/manual exploration to initialize data.
- **Week 5:** Ridge used on F2/F5/F7 (stable local direction with tiny datasets).
- **Week 6:** mixed policy by function class:
  - strong ML exploitation: F5/F7
  - local ML refinement: F2/F3/F4/F8
  - manual/exploration: F1/F6
- **Week 7:** GP+EI (BO-style) introduced for F5/F7; Ridge retained for F2/F3/F4/F8.
- **Week 8:** switched to local-only trust-region BO for F5/F7 to avoid damaging global jumps.
- **Week 9:** micro-refinement (smaller step sizes, tighter trust regions, local training emphasis).
- **Week 10:** trust-region local GP+EI planning generalized to F2–F8, with incumbent-safe selection for F1.

---

## 4) Function-level strategy rationale
- **F5, F7:** strong nonlinear trend + high upside from local BO; trust regions reduce jump risk.
- **F2, F3, F4, F8:** Ridge works well for stable local improvements with tiny updates.
- **F6:** directional heuristic remained reliable and simple.
- **F1:** low-priority exploration function (minimal space-filling policy).

---

## 5) Performance summary (Weeks 1–9 observed)

| Function | Week 1 y | Week 9 y | Delta (W9-W1) | Best observed y | Best week |
|---|---:|---:|---:|---:|---:|
| F1 | -3.353e-61 | 1.554e-237 | ~0 (unstable/near-zero) | 4.715e-13 | 3 |
| F2 | 0.420441 | 0.458958 | +0.038517 | 0.627259 | 8 |
| F3 | -0.120807 | -0.041784 | +0.079023 | -0.041784 | 9 |
| F4 | -18.597235 | -12.773487 | +5.823748 | -12.699964 | 5 |
| F5 | 287.434382 | 472.012140 | +184.577758 | 472.012140 | 9 |
| F6 | -1.630453 | -1.175343 | +0.455110 | -1.122296 | 8 |
| F7 | 0.626706 | 1.091104 | +0.464397 | 1.091104 | 9 |
| F8 | 8.633935 | 8.762618 | +0.128683 | 8.762618 | 9 |

**Weekly movement count (functions improved vs previous week):**
- Week 5: 6 up / 2 down
- Week 6: 7 up / 1 down
- Week 7: 4 up / 4 down (affected by global-jump behavior in F5/F7)
- Week 8: 7 up / 1 down (after trust-region correction)
- Week 9: 5 up / 3 down (mixed micro-refinement behavior)

---

## 6) Assumptions and limitations
### Assumptions
- Local surrogate gradients are useful near best-known points.
- Tight trust regions are safer in later-stage optimization.
- Maximization objective is consistent across all functions.

### Limitations
- Very small sample regime (one point/week/function).
- Hyperparameter sensitivity (alpha, sigma, xi, step size).
- No global-optimality guarantee.
- Results depend on candidate generation seeds and local geometry.

---

## 7) Failure modes observed
- Global BO candidate jumps (notably Week 7 on F5/F7) can severely degrade results.
- Over-aggressive step sizes can overshoot narrow basins (F2/F4 sensitivity).
- Near-converged phases need micro-steps; larger moves waste iterations.

---

## 8) Ethical / transparency considerations
- Strategy, code, and documentation are shared for reproducibility.
- Failures are explicitly documented (not only improvements).
- Claims are bounded by sparse-data uncertainty.

---

## 9) Maintenance plan
- Update this card each week when results are available.
- Add Week 10 observed outcomes after portal evaluation.
- Keep changelog links in README for major strategy shifts.
