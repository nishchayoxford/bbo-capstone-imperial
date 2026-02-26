# Datasheet for BBO Capstone Dataset

## 1) Motivation
This dataset was created to support a weekly black-box optimization workflow for the Imperial AI/ML capstone.

**Task supported:** for each function (F1–F8), submit one query vector per week and track returned score `y` over time.

---

## 2) Composition
### Core records
Each record represents one evaluation event:
- `function_id` (F1..F8)
- `week` (integer)
- `x` (input vector in `[0,1)^d`, where `d` depends on function)
- `y` (returned objective value; maximization target)
- optional notes: strategy/method used

### Dimensions by function
- F1: 2D
- F2: 2D
- F3: 3D
- F4: 4D
- F5: 4D
- F6: 5D
- F7: 6D
- F8: 8D

### Coverage status in repository
- Weeks 1–8 outcomes are embedded in `scripts/capstoneweek9.py` (`DATA` section).
- Week 9 script contains **proposed Week 9 queries**; Week 9 outcomes are to be appended after portal evaluation.

---

## 3) Collection process
### Data generation process
1. Choose weekly query `x` for each function.
2. Submit queries in course BBO portal.
3. Receive score `y` from black-box evaluator.
4. Log `x` and `y` for next-week modeling.

### Strategy timeline
- **Weeks 1–4:** random/manual exploration.
- **Week 5:** Ridge surrogate for selected functions (F2, F5, F7).
- **Week 6:** mixed strategy:
  - strong ML exploitation: F5, F7
  - local ML refinement: F2, F3, F4, F8
  - manual: F1, F6
- **Week 7:** GP+EI (BO-style) introduced for F5/F7; Ridge for F2/F3/F4/F8.
- **Week 8:** trust-region BO (local-only) for F5/F7 after observing harmful global jumps.
- **Week 9:** micro-refinement phase with tighter steps/trust regions.

---

## 4) Preprocessing and uses
### Preprocessing applied
- Input clipping to `[0, 0.999999]` for portal validity.
- Query string formatting to six decimals (`0.xxxxxx-...`).
- Duplicate-avoidance logic in later weeks (rounding-aware uniqueness checks).
- Optional standardization of `y` in some model-fitting steps (for numerical stability).

### Intended uses
- Educational demonstration of sequential black-box optimization under tiny-data constraints.
- Transparent week-to-week method comparison and reflection.

### Non-intended uses
- Production high-stakes optimization.
- Claims of guaranteed global optimum.
- Safety-critical decision-making.

---

## 5) Distribution and maintenance
- Hosted in a public GitHub repository for peers/facilitators.
- Maintainer: student repository owner.
- Update cadence: weekly (after each new portal result).

---

## 6) Known limitations / gaps
- Very small sample sizes per function (one new point per week).
- Some early-week rationale is retrospective.
- Week 9 outcome values are pending (at time of current document version).

---

## 7) Versioning policy
- Keep one commit per weekly update.
- Optional tags: `week-05`, `week-06`, etc.
- Document strategy shifts clearly in commit messages and `results/weekly_summary.md`.

---

## 8) Related files
- `docs/MODEL_CARD.md` — method details, assumptions, limitations
- `results/weekly_summary.md` — concise week-by-week narrative
- `scripts/` — executable planning scripts by week
