# Part V: Design of Experiments

Designing experiments efficiently is critical when materials synthesis is costly,
time-consuming, or uses hazardous precursors. This part covers the full DoE workflow:
from screening many factors quickly to finding the optimum with as few experiments as possible.

You do not need advanced mathematics for this part. Every method here answers a
question you already ask in the lab by instinct — *"which factor actually matters?"*,
*"where's the sweet spot?"*, *"how do I balance several goals at once?"* — DoE simply
gives you a systematic, efficient way to answer it with data instead of guesswork.
The [theory page](theory.md) walks through the intuition behind each method with
chemistry examples before showing any formulas; treat the equations there as a
reference to check *what the Python code is computing*, not as something to memorise.

## Notebooks in this chapter

| Notebook | What you learn |
|----------|---------------|
| 16. Introduction to DoE | Why DoE beats OFAT; terminology; design types |
| 17. Full Factorial Design | 2ᵏ designs; main effects; interactions |
| 18. Fractional Factorial Design | Resolution; aliasing; screening many factors |
| 19. Response Surface Methodology | CCD, Box-Behnken; quadratic models; contour plots |
| 20. Optimization using DoE | Steepest ascent; desirability functions |
| 21. Simplex Optimization | Sequential model-free optimization; Nelder-Mead |
| 22. Live Tutorial 1: Two-Factor DoE, Start to Finish | 2² design built by hand; manual effect estimation proven against regression; ANOVA; curvature test; confirmation run |
| 23. Live Tutorial 2: From Minitab to Six Factors | Minitab-style factorial workflow in Python; Pareto-based model reduction; Box & Draper's real 2⁶ textbook data; Yates' method; normal-probability screening |
| 24. Live Tutorial 3: Model Reduction and Fractional Designs | Backward elimination; Resolution III vs. V aliasing bias |
| 25. Live Tutorial 4: Optimization and Mixture Designs | Steepest-ascent bridge; simplex-lattice mixture designs; Scheffé models; ternary contour plots |

Notebooks 22–25 are self-contained **hands-on sessions** (Notebook 24 runs
shorter, since it builds directly on Notebook 22's and Notebook 12's
techniques rather than introducing new ones). Notebook 25 in
particular introduces **mixture designs** (simplex-lattice and
simplex-centroid, Scheffé models, ternary contour plots) — a design type
not covered anywhere else in this course, essential whenever "factors" are
ingredient proportions that must sum to 100% (alloy, glass, or electrolyte
formulation).

## The DoE workflow

```
Define goal → Choose factors & ranges → Select design → Run experiments
     → Build model → Validate model → Optimize → Confirm optimum
```

DoE is not just about statistics — it is an engineering mindset that forces
you to think systematically about what you know, what you do not know,
and what experiments give you the most information per unit cost.

This workflow is a practical rendering of a much older idea: Box and
Wilson's (1951) three-stage strategy of **screening** many candidate
factors down to the vital few, fitting a simple **empirical model** to find
which direction improves the response, and only then building a detailed
**optimisation model** to pinpoint the optimum once you are close enough
for it to matter. The [theory page](theory.md) opens with this "big
picture" before working through each stage's formulas in detail.
