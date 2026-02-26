# Model Card — BBO Capstone Optimization Approach

## 1) Overview
**Model/approach name:** Weekly hybrid black-box optimizer (Ridge + GP/EI trust-region + heuristics)

**Versioning:** evolved week-by-week from exploratory to local exploitation strategies.

## 2) Intended use
### Intended
- Sequential weekly optimization in a constrained black-box setting
- Educational demonstration of iterative surrogate-based search

### Not intended
- Safety-critical optimization
- Claims of globally optimal solutions under tiny-data regimes

## 3) Details (techniques and evolution)
- Weeks 1–4: random/manual exploration to establish initial observations
- Week 5: Ridge surrogate for selected functions (F2/F5/F7)
- Week 6: mixed policy by function group (strong ML vs local ML vs manual)
- Week 7: advanced BO-style GP+EI introduced for F5/F7
- Week 8: trust-region local-only BO to prevent harmful global jumps
- Week 9: micro-refinement with smaller steps and tighter trust regions

Function-specific strategy rationale:
- **F5/F7:** stronger nonlinear local modeling benefit → GP+EI trust-region
- **F2/F3/F4/F8:** stable local linear improvements → Ridge micro/local refinement
- **F1:** low-priority exploration track (space-filling)
- **F6:** directional heuristic maintained due to steady trend

## 4) Performance
Use this section to summarize results across all 8 functions.

Suggested metrics:
- Best-so-far value by week per function
- Week-over-week delta (`y_t - y_{t-1}`)
- Cumulative improvement from first non-random baseline

> Add a table here once your final week data is complete.

## 5) Assumptions and limitations
Assumptions:
- Objective is maximization for all functions.
- Local surrogate direction is informative near best-known points.
- Small trust regions reduce risk of destructive jumps.

Limitations:
- Extremely small sample sizes each week.
- Sensitive to hyperparameter choices (alpha, sigma, xi, step size).
- No guarantee of global optimum.
- Function landscapes are unknown and may be non-stationary from sparse observations.

## 6) Ethical considerations
- Emphasis on transparency and reproducibility through code, datasheet, and method notes.
- Avoid overclaiming model certainty given tiny data.
- Encourage peer review and explicit reporting of failed attempts.

## 7) Maintenance and updates
- Update this card weekly when strategy changes.
- Document major policy shifts in commit messages and README changelog.
