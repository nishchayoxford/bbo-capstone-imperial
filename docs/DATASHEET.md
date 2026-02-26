# Datasheet for BBO Capstone Dataset

## 1) Motivation
This dataset was created to support weekly optimization decisions for an 8-function black-box optimization capstone project.

**Task supported:** selecting one query vector per function per week to maximize function value.

## 2) Composition
The dataset contains:

- Function ID (`F1` ... `F8`)
- Week number
- Input vector `x` (dimension depends on function)
- Returned objective value `y`
- Optional notes (method used, rationale)

**Format:** CSV/Markdown summaries (student-maintained).

**Current known gaps:**
- Early-week rationale notes may be incomplete.
- Some preprocessing/provenance fields may need manual fill-in.

## 3) Collection process
Queries were generated weekly and submitted through the course BBO portal.

- Weeks 1–4: random/manual exploration
- Weeks 5+: model-guided strategies (Ridge, GP+EI/BO, trust-region BO, and heuristic/manual steps)

Evaluation values were returned by the black-box system and logged after each submission.

## 4) Preprocessing and uses
### Preprocessing
- Inputs clipped to `[0, 0.999999]`
- Values formatted to 6 decimals for portal compatibility
- Duplicate-protection logic used in later weeks to avoid repeated rounded query strings

### Intended use
- Educational demonstration of sequential black-box optimization
- Reproducibility and reflection across weekly iterations

### Non-intended use
- High-stakes/production optimization without additional validation
- Fairness/safety-critical deployment decisions

## 5) Distribution and maintenance
- Hosted in a public GitHub repository for peer/facilitator review.
- Maintainer: repository owner (student).
- Update cadence: weekly (or after each iteration).

## 6) Versioning
- Dataset snapshots should be tagged by week (e.g., `week-05`, `week-06`, ...).
- Major methodology changes should be documented in commit messages and README.

## 7) Notes for reviewers
Please see:
- `README.md` for project overview
- `docs/MODEL_CARD.md` for method decisions and limitations
