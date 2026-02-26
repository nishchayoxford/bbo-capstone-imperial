# Imperial BBO Capstone — Black-Box Optimization (8 Functions)

A student-friendly repository for my Imperial College London AI/ML capstone project.

## Project overview
This project optimizes **8 black-box objective functions (F1–F8)** over weekly iterations.

- **Weeks 1–4:** random/manual exploration
- **Weeks 5 onward:** machine-learning-guided optimization (Ridge, BO/GP-EI, trust-region variants, and controlled manual heuristics)

## Repository structure

- `scripts/` → runnable Python scripts by week (source of truth)
- `notebooks/` → Jupyter notebook versions of weekly work
- `docs/DATASHEET.md` → datasheet for capstone dataset (required)
- `docs/MODEL_CARD.md` → model card for optimization approach (required)
- `data/` → query history + evaluation results (add your CSVs here)
- `results/` → weekly summaries and plots

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/capstoneweek9.py
```

## Weekly method evolution (high-level)

- **Week 5:** Ridge surrogate for selected functions (F2, F5, F7)
- **Week 6:** Mixed strategy (strong ML for F5/F7, local ML for F2/F3/F4/F8, manual for F1/F6)
- **Week 7:** Added GP+EI (BO-style) for F5/F7, Ridge for F2/F3/F4/F8
- **Week 8:** Trust-region BO (local only) to avoid harmful global jumps on F5/F7
- **Week 9:** Micro-refinement phase (tiny steps, local trust-region exploitation)

## Required assignment documents

- Datasheet: `docs/DATASHEET.md`
- Model card: `docs/MODEL_CARD.md`

## Reproducibility notes

- Input vectors are clipped to `[0, 0.999999]` for portal compatibility.
- Queries are printed in portal format: `0.xxxxxx-0.xxxxxx-...`
- Comments were standardized for student readability and method rationale.

## TODO before final submission

- [ ] Add your actual dataset files to `data/`
- [ ] Fill missing placeholders in `docs/DATASHEET.md`
- [ ] Fill missing placeholders in `docs/MODEL_CARD.md`
- [ ] Add plots in `results/plots/`
- [ ] Make repo public and submit GitHub link
