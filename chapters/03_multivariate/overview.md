# Part IV: Multivariate Analysis

Real materials datasets rarely involve a single variable.
This part introduces methods for extracting structure from high-dimensional data
and building predictive models from many correlated measurements.

Every method here is really answering one of two practical questions: *"I have
many correlated measurements per sample — can I compress them into a handful
of interpretable numbers?"* (PCA, Factor Analysis) or *"can I use many
correlated measurements to predict something I care about?"* (MLR, LDA, PLS).
The [theory page](theory.md) builds the intuition for each with plain
language and chemistry examples before showing any matrix notation — treat
the equations as a reference for what the Python code computes, not
something to memorise.

## Notebooks in this chapter

| Notebook | What you learn |
|----------|---------------|
| 11. Multiple Linear Regression | MLR, VIF, model diagnostics |
| 12. PCA | Dimensionality reduction, scores, loadings, biplots |
| 13. LDA | Supervised classification by discriminant analysis |
| 14. Factor Analysis | Latent constructs, varimax rotation, communalities |
| 15. PLS | Partial Least Squares regression; cross-validation; VIP |
| 16. Live Tutorial 1: PCA & FA | PCA built from scratch in NumPy; parallel analysis; Hotelling's T² |
| 17. Live Tutorial 2: PLS/PCR & LDA | Spectral regression and classification; permutation tests; PLS-DA |

Notebooks 16–17 are self-contained hands-on sessions, each built
around one coherent dataset (a simulated cathode-material database, then
simulated electrolyte spectra) rather than four separate toy examples —
deliberately going further than Notebooks 12–15 with a from-scratch PCA
implementation, parallel analysis, permutation testing, and PLS-DA, plus
substantial self-study exercises.

## When to use which method?

| Question | Method |
|----------|--------|
| Predict a continuous response from many X variables | MLR or PLS |
| Find the main sources of variation in X | PCA |
| Classify samples into known groups | LDA |
| Identify latent factors driving correlations | FA |
| Predict Y from highly collinear X | PLS (usually — PCR can tie or win, see Notebook 15 §15.6–15.8) |
