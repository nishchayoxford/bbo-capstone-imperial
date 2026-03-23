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
- **Week 11:** adaptive local GP+EI continuation with function-specific policy (F1 maximin, F8 fixed-dimension-aware handling).
- **Week 12:** continued late-stage local BO refinement and stable query generation.
- **Week 13:** ensemble upgrade (GP + Gradient Boosting), hybrid UCB/EI scoring, multi-center trust-region search.

---

## 4) Function-level strategy rationale
- **F5, F7:** strong nonlinear trend + high upside from local BO; trust regions reduce jump risk.
- **F2, F3, F4, F8:** Ridge works well for stable local improvements with tiny updates.
- **F6:** directional heuristic remained reliable and simple.
- **F1:** low-priority exploration function (minimal space-filling policy).

---

## 5) Performance summary (Weeks 1–12 observed)

| Function | Week 1 y | Week 12 y | Delta (W12-W1) | Best observed y | Best week |
|---|---:|---:|---:|---:|---:|
| F1 | -3.353e-61 | 4.724e-13 | +4.724e-13 | 4.724e-13 | 12 |
| F2 | 0.420441 | 0.497682 | +0.077241 | 0.627259 | 8 |
| F3 | -0.120807 | -0.044746 | +0.076061 | -0.034103 | 10 |
| F4 | -18.597235 | -12.086855 | +6.510379 | -12.086855 | 12 |
| F5 | 287.434382 | 672.506197 | +385.071815 | 672.506197 | 12 |
| F6 | -1.630453 | -1.112748 | +0.517705 | -1.112748 | 12 |
| F7 | 0.626706 | 1.133930 | +0.507224 | 1.133930 | 12 |
| F8 | 8.633935 | 8.766626 | +0.132691 | 8.766626 | 12 |

**Weekly movement count (functions improved vs previous week):**
- Week 10: 6 up / 2 down
- Week 11: 6 up / 2 down
- Week 12: 6 up / 2 down

(Week 13 is currently a planning stage in this repository snapshot.)

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
- Update this card each week when new portal outcomes are available.
- Keep `data/weekly_results/` synchronized with script/notebook history.
- Keep changelog links in README for major strategy shifts.
