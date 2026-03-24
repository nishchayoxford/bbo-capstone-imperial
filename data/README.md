# data/

Place your capstone dataset files here.

Recommended files:
- `bbo_query_history.csv` (function, week, x components, query_string)
- `bbo_function_results.csv` (function, week, y, delta_vs_prev, best_so_far_flag)

If you only have logs in scripts, export tables from notebooks or create CSV manually.

Current snapshot in this repo:
- `data/weekly_results/inputs/` contains cumulative input logs (`1.txt` ... `13.txt`).
- `data/weekly_results/outputs/` contains cumulative output logs (`1.txt` ... `13.txt`).

Professional data hygiene note:
- Raw text logs now include the Week 13 submission and returned outcomes.
- Plotting/documentation scripts should read from the latest cumulative files so summaries stay aligned with the final repository state.
