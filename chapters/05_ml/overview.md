# Part VI: Introduction to AI and Machine Learning (Extra Material)

This part extends the statistical and multivariate methods from Parts III–V into the
broader field of **machine learning (ML) and artificial intelligence (AI)**.
The focus remains firmly on materials science applications — every method is
demonstrated on simulated or real datasets typical of synthesis, characterisation,
and property optimisation workflows.

**This part is extra material.** It is not required for the [course
project](../06_project/project.md) and goes beyond the core Parts I–V syllabus —
treat it as an optional, self-paced extension for anyone who wants to see where
the statistical and DoE ideas from earlier in the course lead next, not as
material you need to master to complete the course. Every notebook is written
so that Parts I–V alone are enough background to follow it.

You do not need advanced mathematics for this part. Every method extends an
idea you have already met: clustering groups similar samples the way your
eye would on a scatter plot; Gaussian Processes are regression that also
tells you how *confident* it is; Bayesian optimisation is DoE's steepest
ascent (Part V), but re-run automatically after every experiment; ensemble
methods and neural networks are regression models (Parts III–IV) flexible
enough to bend around nonlinear relationships MLR and PLS cannot capture.
Each notebook builds the intuition in plain language before the code, and
treats any formula as a reference for what's being computed rather than
something to memorise.

---

## Why ML for Materials Science?

Traditional DoE and statistical methods work extremely well when:
- The number of experiments is small relative to the number of parameters
- A clear mechanistic model exists for the response surface
- Only a few (< 10) factors need to be considered simultaneously

Modern materials challenges — high-entropy alloys, combinatorial synthesis,
spectral datasets with thousands of variables, autonomous laboratories — routinely
exceed these limits. Machine learning methods fill the gap by:

1. **Learning complex, nonlinear structure** from large datasets without requiring
   a pre-specified model form
2. **Quantifying uncertainty** (Gaussian Processes, Bayesian methods)
3. **Guiding experiments adaptively** (Bayesian optimisation, active learning)
4. **Extracting latent groupings** from unlabelled data (clustering)
5. **Scaling to thousands of variables** (ensemble methods, neural networks)

---

## How Part VI Connects to the Rest of the Course

Every method here is a direct extension of something you already used earlier
in the course — the table below spells out exactly which prior notebook to
revisit if a Part VI idea feels unfamiliar. Because chapter numbering restarts
within each part, prior-part references below are given as "Notebook N, Part X".

| Part VI Chapter | Builds On |
|---|---|
| 16. Unsupervised Clustering & Dimensionality Reduction | PCA (Notebook 12, Part IV), LDA (Notebook 13, Part IV) — clustering finds groups without the labels LDA is given |
| 17. Gaussian Process Regression (Kriging) | RSM (Notebook 19, Part V) as the deterministic surface it generalises; confidence/prediction intervals (Notebook 10, Part III) as the idea behind the GP's uncertainty band |
| 18. Bayesian Optimisation | GPR (Notebook 17, Part VI) as the surrogate it queries; steepest ascent (Notebook 20, Part V) as the DoE strategy it automates; desirability functions (Notebook 20 §20.2, Part V) for multi-objective framing |
| 19. Ensemble Methods | MLR (Notebook 11, Part IV) and factorial effects (Notebook 2, Part V) as the linear/interaction structure trees and forests capture without a pre-specified model form |
| 20. Neural Networks Basics | PLS (Notebook 15, Part IV) and MLR (Notebook 11, Part IV) as the linear baselines neural networks are compared against |

---

## Chapters in This Part

```{tableofcontents}
```
