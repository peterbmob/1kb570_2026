# Part III: Applied Statistics

Statistics provides the foundation for deciding whether an observed effect is real
or simply the result of random variation. This part covers everything from descriptive
summaries to ANOVA and linear regression.

You do not need advanced mathematics for this part — every method answers a
question you already ask informally in the lab: *"is this batch actually
different, or did I just get unlucky with noise?"*, *"how sure am I about
this average?"*, *"does this trend line actually mean something?"* The
[theory page](theory.md) builds the intuition behind each method with plain
language before showing any formula; treat the equations as a reference for
what the Python code is computing, not something to memorise.

## Notebooks in this chapter

| Notebook | What you learn |
|----------|---------------|
| 6. Descriptive Statistics | Mean, spread, outlier detection |
| 7. Probability Distributions | Normal, t, F, χ² — and how to test for normality |
| 8. Hypothesis Testing | t-tests, p-values, statistical power |
| 9. ANOVA | Comparing multiple groups; interaction effects |
| 10. Simple Linear Regression | Fitting lines, R², residual diagnostics |
| 11. Live Tutorial 1: *t*-Tests | A quality-lab case study: one-sample, two-sample, Welch, paired, Mann–Whitney, bootstrap CIs, power |
| 12. Live Tutorial 2: ANOVA & Regression | One-way/blocked/two-way ANOVA, the ANOVA-regression bridge, variable selection, ANCOVA |

Notebooks 11–12 are self-contained hands-on sessions that revisit
Notebooks 8–10 through one running materials-engineering case study each,
going deeper (bootstrap methods, randomized block designs, formal variable
selection) than the topic-by-topic notebooks above have room for, and
finish with substantial self-study exercise sets.

## Key principle: probability vs. statistics

*Probability* asks: "Given a model, what data would we expect?"  
*Statistics* asks: "Given data, what can we infer about the model?"

Both perspectives are needed when evaluating experimental results.
