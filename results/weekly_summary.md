# Weekly Summary

## Why this file was empty before
It was intentionally created as a template scaffold so you could fill in your own reflection text.
Now it is populated from the week-to-week script content and observed outcomes.

---

## Week 5
- **Strategy:** Ridge regression surrogate on F2, F5, F7 using Weeks 1–4 data.
- **Highlights:** moved from pure exploration toward model-guided updates.
- **What improved:**
  - F2 reached a new local region after Week 4 best.
  - F5 and F7 maintained upward trend from prior weeks.
- **What failed / risks:** only three functions were model-covered; others remained unoptimized by ML this week.
- **Next-week adjustment:** expand ML policy to all functions with mixed strategy by function behavior.

## Week 6
- **Strategy:** mixed policy
  - Strong ML exploitation: F5, F7
  - Local ML refinement: F2, F3, F4, F8
  - Manual: F1 (maximin exploration), F6 (direction continuation)
- **Highlights:** broad expansion from selective ML to full 8-function policy map.
- **Observed movement vs Week 5:** **7 improved, 1 declined**.
- **What improved most:** F5 and F7 continued strong gains; F3/F8 refined steadily.
- **What failed:** F4 showed sensitivity and did not improve this round.
- **Next-week adjustment:** introduce advanced BO for high-return functions (F5/F7).

## Week 7
- **Strategy:** GP+EI (BO-style) for F5/F7, Ridge for F2/F3/F4/F8, manual for F6, exploration for F1.
- **Highlights:** first use of advanced Bayesian optimization in pipeline.
- **Observed movement vs Week 6:** **4 improved, 4 declined**.
- **What improved:** F2, F6, F8 kept progressing.
- **What failed:** major global-jump failures on **F5 and F7** (clear performance drop).
- **Next-week adjustment:** enforce trust-region local-only BO to prevent destructive global jumps.

## Week 8
- **Strategy:** trust-region BO (local-only) for F5/F7 + Ridge local refinement (F2/F3/F4/F8) + manual F6.
- **Highlights:** successful recovery after Week 7 instability.
- **Observed movement vs Week 7:** **7 improved, 1 declined**.
- **What improved most:**
  - F5 recovered strongly to best-so-far.
  - F7 recovered to best-so-far.
  - F2/F6/F8 reached new best values.
- **What failed:** F3 slightly down from Week 7 best (still close to peak).
- **Next-week adjustment:** enter micro-refinement phase with smaller steps and tighter trust regions.

## Week 9 (planning stage)
- **Strategy:** micro-refinement
  - Ridge micro-steps: F2/F3/F4/F8
  - Trust-region BO micro-exploitation: F5/F7
  - Manual micro-continue: F6
  - Minimal effort track: F1
- **Rationale:** most functions are near current local bests, so precision beats exploration.
- **Status:** Week 9 queries generated; results pending portal evaluation.
- **Planned update:** append Week 9 observed outcomes and final capstone reflection after submission.


## Week 10 (planning stage)
- **Strategy:** trust-region local BO (TuRBO-lite style) for F2–F8 + incumbent-nearby policy for F1.
- **Highlights:** moved from mixed Week 9 micro-methods to a unified local GP+EI planning template for most functions.
- **Design choices:**
  - Local GP is trained on nearest points to incumbent (k-local training).
  - EI is maximized over large trust-region candidate sets.
  - Duplicate-avoidance after rounding protects portal submission uniqueness.
- **Status:** Week 10 queries generated; portal outcomes pending.
- **Next update:** append Week 10 observed y values and delta-vs-Week 9 after evaluation.
