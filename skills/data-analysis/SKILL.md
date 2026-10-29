---
name: data-analysis
description: Perform reproducible scientific statistics on CSV data with explicit inputs, formulas, and limitations.
---

# Data Analysis

Use this workflow when a question asks for statistics, correlations, regression, or a data interpretation.

1. Identify the input file and columns before calculating.
2. Use `file_tool` to inspect the file and `python_calculate` with `data_file` for computation.
3. Report sample size, missing values, units when available, and whether standard deviation uses `ddof=1` or `ddof=0`.
4. Use `stats.pearsonr` for Pearson correlation and `metrics.r2_score` for R².
5. Distinguish computed values from interpretation. Never infer causation from correlation alone.

Keep Python output concise and reproducible. If the data are too small, truncated, or unsuitable for a test, say so explicitly.
