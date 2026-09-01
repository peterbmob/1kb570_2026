# Theoretical Background: Design of Experiments

This page explains the *ideas* behind the DoE methods used in Part V before
showing the formulas. You do not need advanced mathematics to use DoE well —
you need to understand what question each method answers, and what its output
tells you about your chemistry. Every section below starts with the intuition
in plain language and chemistry examples; the equations follow afterwards for
reference, clearly marked, in case you want to see exactly what the software
is computing.

---

## The Big Picture: Box and Wilson's Sequential Strategy

### The idea

Suppose you are developing a new LiFePO4 cathode synthesis and want to
maximise the material's capacity. You have a long list of candidate
factors — calcination temperature, sintering time, precursor ratio,
atmosphere, pH, stirring rate — and no idea yet where in that six-dimensional
space the best conditions lie. Running a single, detailed experimental
design over all six factors at once would be prohibitively expensive, and
most of that expense would be wasted: experience says only a handful of
these factors will turn out to matter, and the region you start in is
probably nowhere near the true optimum anyway.

George Box and K. B. Wilson (1951) proposed the solution that is still the
standard playbook today: don't try to answer everything with one design.
Instead, move through **three distinct stages**, each asking a different
question and using a model no more complicated than that question requires:

1. **Screening** — *Which of the many candidate factors actually matter?*
   Run a cheap, low-resolution fractional factorial (Section 2) across all
   six factors at once. Most will turn out to be negligible; you are left
   with the "vital few" — say, calcination temperature and sintering time.
2. **Empirical model** — *Which direction should we move in?* Near the
   starting point, the true response surface is usually so gently curved
   that a flat, first-order **empirical model**
   ($\hat{y} = b_0 + b_1x_1 + b_2x_2$, fitted from a small factorial in just
   the surviving factors) is an adequate local approximation, even though
   you know it cannot be the whole truth far from the optimum. Its
   coefficients give a compass bearing, and **steepest ascent** — a handful
   of confirmation runs walking in that direction — moves the whole
   experimental region toward the optimum for little additional cost.
3. **Optimisation model** — *Exactly where is the optimum, and what shape
   is it?* Once steepest ascent stops improving the response, you have
   walked into the curved region surrounding the true optimum, where the
   flat empirical model breaks down — increasing temperature no longer
   simply increases capacity, because you are now near the top of the
   hill rather than partway up its side. This is the signal to switch to a
   proper second-order **optimisation model** (Section 3, fitted with a
   CCD or Box–Behnken design), which can represent a peak and pinpoint the
   stationary point precisely.

The key habit this teaches is **matching model complexity to how far you
are from the optimum**: a first-order model is cheap and is all you can
justify when you are still far away and mostly need a direction; a
second-order model costs more runs but is only worth that cost once you
are close enough for curvature to matter. Jumping straight to an
optimisation model from the very start — without screening or steepest
ascent first — usually just wastes runs mapping curvature in a region that
never contained the optimum in the first place.

| Box–Wilson stage | Question it answers | Model used | Where in Part V |
|---|---|---|---|
| 1. Screening | Which factors matter? | None, or first-order for ranking only | Section 2 below; Notebooks 17–18 |
| 2. Empirical model + steepest ascent | Which direction improves the response? | First-order: $\hat{y}=b_0+\sum_i b_i x_i$ | Notebook 20, §20.1 |
| 3. Optimisation model | Where exactly is the optimum, and what shape? | Second-order (quadratic) RSM | Section 3 below; Notebook 19 |

This exact scenario — a Li-ion cathode synthesis, screened from several
candidate factors down to calcination temperature and sintering time, then
walked uphill in capacity, then mapped with a quadratic model near the top —
is worked through step by step across Notebooks 18, 20, and 19, so you see
all three Box–Wilson stages applied to the *same* running example rather
than three disconnected case studies. Sections 4 and 5 below (desirability
functions, simplex optimisation) are refinements you reach for *within*
Stage 3, once you already have — or want to avoid needing — an explicit
optimisation model.

---

## 1. The 2$^k$ Factorial Design

### The idea

Imagine you are optimising a sol-gel synthesis and you suspect that both
**temperature** and **precursor concentration** affect the product's particle
size. You want to know three things:

1. Does temperature matter, on average?
2. Does concentration matter, on average?
3. Does the effect of temperature *depend on* which concentration you used
   (and vice versa)? This is called an **interaction**.

A "$2^k$ factorial design" answers all three questions with the smallest
possible number of experiments: test every combination of *k* factors, each
at just two settings — a "low" and a "high" value. With 2 factors that is
$2^2 = 4$ runs; with 3 factors, $2^3 = 8$ runs. Doubling the number of levels
per factor would need far more runs, but two levels are enough to detect
*whether* a factor matters and in *which direction* — you add more levels
later (Section 3) once you know which factors are worth exploring further.

### Coded variables — why we relabel −1 and +1

Every factor has different units: temperature in °C, concentration in mol/L,
time in hours. Comparing "an effect of 50 °C" to "an effect of 0.2 mol/L"
directly is like comparing apples to oranges. **Coding** solves this by
relabelling the low setting of every factor as −1 and the high setting as +1,
regardless of its original units:

$$x_i = \frac{X_i - X_i^0}{(X_i^+ - X_i^-)/2}$$

In words: subtract the centre value $X_i^0$ (halfway between low and high),
then divide by half the range. The result is always −1 at the low setting, 0
at the centre, and +1 at the high setting — for *every* factor, no matter its
units. Once coded, the size of each factor's effect can be compared directly,
and the numbers involved in fitting the model stay small and well-behaved
(computers fit models more reliably with numbers like −1 and +1 than with
numbers like 573.15).

### Estimating an effect — an averaging trick, not calculus

The **main effect** of a factor is simply: *how much does the average response
change when this factor goes from low to high, averaged over all the other
factors' settings?* No calculus is needed — it is an average of differences.
Formally,

$$\hat{E}_j = \frac{1}{2^{k-1}} \mathbf{c}_j^\top \mathbf{y}$$

which just means: take every response value $\mathbf{y}$, multiply it by +1
or −1 according to whether factor $j$ was at its high or low setting in that
run (the vector $\mathbf{c}_j$ records this), add everything up, and divide
by half the number of runs. An **interaction effect** between factors $j$ and
$k$ is calculated the same way, except the +1/−1 pattern used is the *product*
of the two factors' own patterns — this multiplication trick is exactly what
lets you detect "the effect of temperature depends on concentration" from the
same set of experiments, with no extra runs.

### Why the design is "balanced" — no experiment steals information from another

For a $2^3$ design the design matrix (using +1/−1 to denote high/low)

$$\mathbf{X} = \begin{pmatrix} -1 & -1 & -1 \\ +1 & -1 & -1 \\ -1 & +1 & -1 \\ \vdots & \vdots & \vdots \end{pmatrix}$$

has a special property: every column is uncorrelated with every other column
($\mathbf{X}^\top\mathbf{X} = 2^k\mathbf{I}$, i.e. **orthogonal**). Practically,
this means the estimate of the temperature effect is not distorted by the
concentration effect, or by the interaction — each answer comes out "clean."
This is the mathematical reason a full factorial is so efficient: nothing
about the design wastes information.

---

## 2. Fractional Factorial Designs and Aliasing

### The idea

Full factorials get expensive fast: 6 factors already means $2^6 = 64$ runs.
Often, in an early **screening** stage, you don't need the full picture yet —
you just want to know *which few factors, out of many candidates, actually
matter*. A **fractional factorial** runs only a carefully chosen half (or
quarter, or smaller fraction) of the full design, saving time and material at
a cost: some effects become impossible to tell apart from each other.

Think of it like a routine blood panel that reports one combined number for
two related markers instead of testing them separately — cheaper, but if the
combined number is high, you don't immediately know which marker caused it.
In DoE this blending of two effects is called **aliasing** or **confounding**.

### How the aliasing happens

A $2^{k-p}$ fractional design is built by taking a full factorial in fewer
factors and assigning the "extra" factors to columns that were previously
interaction columns. For example, in a $2^{5-1}$ design (half of a $2^5$),
factor E is deliberately set equal to the three-factor interaction of A, B, C,
D:

$$x_5 = x_1 \cdot x_2 \cdot x_3 \cdot x_4 \qquad \text{written as } I = ABCDE$$

This equation, called the **defining relation**, is the "recipe" that tells
you exactly which effects got blended together. Every column in the reduced
design matrix is now the product of some other columns, so two or more
effects always share the same up/down pattern in the data — they are
**aliased**: the data cannot distinguish them.

### Resolution — a "how much can I trust this" rating

Because different fractional designs alias different things, DoE gives every
design a **resolution** rating (in Roman numerals) that tells you, at a
glance, how safe your conclusions will be:

- **Resolution III** — main effects are aliased with two-factor interactions.
  Fine for screening if you are only ranking factors by importance, risky if
  interactions are likely to be real.
- **Resolution IV** — main effects are clean, but two-factor interactions are
  aliased with each other. The most common screening choice — you trust the
  main effects, but treat any interaction as tentative.
- **Resolution V** — main effects *and* two-factor interactions are all clean.
  You get almost the reliability of a full factorial for a fraction of the
  runs.

**Example**: the $2^{5-1}$ design above (generator $I=ABCDE$) is Resolution V:
every two-factor interaction like $AB$ is aliased only with a three-factor
interaction ($AB \equiv CDE$), and three-factor interactions are almost always
negligible in real chemistry — so in practice you lose nothing.

**Rule of thumb**: higher resolution = more trustworthy but more runs. Choose
the lowest resolution that still answers your question; you can always
"resolve" ambiguity later by running a follow-up fraction (an exercise in the
fractional factorial notebook explores this "fold-over" trick).

---

## 3. Response Surface Methodology (RSM)

### The idea

Once screening has told you *which* two or three factors matter, the next
question is *where exactly is the optimum?* A straight-line (main effects)
model can only tell you "more is better" or "less is better" — it cannot
represent a peak or a valley, because a straight line never turns around.
Chemistry is full of "sweet spots": too little catalyst gives low yield, too
much also gives low yield (side reactions, cost, safety) — the best point is
somewhere in the middle. To capture a peak, the model needs a *curve*, which
means adding squared terms:

$$y = \beta_0 + \sum_i \beta_i x_i + \sum_i \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j + \varepsilon$$

In words: the response equals a baseline ($\beta_0$), plus straight-line
contributions from each factor ($\beta_i x_i$), plus curvature terms
($\beta_{ii} x_i^2$, one parabola per factor), plus interaction terms
($\beta_{ij}x_ix_j$), plus experimental noise ($\varepsilon$). This is exactly
the same idea as fitting a parabola through yield-vs-temperature data in
Part III, just extended to several factors at once. In matrix notation this is
written $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$
and solved the same way ordinary regression is solved:
$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$
(see Part III's regression theory for what this means in detail).

### Choosing where to place the extra experiments

A straight-line model only needs corner points (the $2^k$ design from
Section 1). A curved (quadratic) model additionally needs points that are
*not* at the corners, otherwise there is no information about curvature — you
would be trying to fit a parabola through only two points. Two standard
layouts add these extra points differently:

- **Central Composite Design (CCD)**: keeps the corner points and adds
  "star" points that stick out along each axis, plus repeated centre runs.
- **Box–Behnken Design (BBD)**: places points at the *midpoints of the edges*
  of the design cube, plus centre runs, and — importantly — **never at the
  corners**. If the most extreme high-high-high combination of factors is
  physically dangerous or simply doesn't make chemical sense (e.g. maximum
  temperature *and* maximum pressure *and* maximum concentration all at
  once), a Box–Behnken design lets you skip that combination entirely while
  still mapping the curvature of the surface. For 3 factors it needs only
  12 edge points + a few centre points (15–17 runs total), similar to or
  fewer than a CCD.

### Reading the fitted surface: where's the top of the hill?

Once the quadratic model is fitted, finding the optimum is a matter of asking
"where does the surface stop going up and start coming back down in every
direction?" — the top of a hill has zero slope in every direction, exactly
like the peak of a mountain in a topographic map. This point is called the
**stationary point**, found by setting all partial derivatives to zero
(the multi-factor version of "the derivative of a parabola is zero at its
vertex," from Part III):

$$\mathbf{x}_s = -\tfrac{1}{2}\hat{\mathbf{B}}^{-1}\hat{\boldsymbol{\beta}}_L$$

where $\hat{\mathbf{B}}$ collects all the curvature ($\beta_{ii}$) and
interaction ($\beta_{ij}$) coefficients into one matrix. You rarely need to
compute this by hand — `scipy.optimize` does it numerically in the
notebooks — but it helps to know *what* the computer is looking for.

The shape of the stationary point matters chemically, and it is diagnosed by
checking whether the curvature (encoded in $\hat{\mathbf{B}}$'s eigenvalues)
curves the *same way* in every direction:

- All directions curve downward from the point → it is a **maximum** (a
  dome — genuinely the best combination of factors).
- All directions curve upward → it is a **minimum**.
- Some directions curve up, others down → it is a **saddle point** (a
  mountain pass/ridge): there is a *direction* along which you can keep
  improving the response, even though locally it looks like an optimum.
  Recognising a saddle point tells you to explore further along the ridge
  rather than stopping.

---

## 4. Desirability Functions

### The idea

Real syntheses rarely have just one goal. You might want to *maximise*
yield *and* purity, while *minimising* cost and reaction time — all at once,
and these goals often conflict (higher purity might mean lower yield). A
**desirability function** is a way to referee this trade-off automatically:
convert every response to a common 0–1 "grade" (0 = unacceptable, 1 = ideal),
then combine the grades into one overall score to maximise.

$$d_i = \begin{cases} 0 & y_i < L_i \\ \left(\dfrac{y_i - L_i}{T_i - L_i}\right)^{w_i} & L_i \leq y_i \leq T_i \\ \left(\dfrac{U_i - y_i}{U_i - T_i}\right)^{w_i} & T_i < y_i \leq U_i \\ 0 & y_i > U_i \end{cases}$$

In words: you set a lower acceptable bound $L_i$, an ideal target $T_i$, and
an upper bound $U_i$ for each response. Below $L_i$ or above $U_i$, the grade
is 0 (completely unacceptable) no matter how good the other responses are.
Between the bounds, the grade rises smoothly toward 1 as the response
approaches its target; the exponent $w_i$ just controls how quickly it rises
(a shape parameter you rarely need to change from 1).

### Combining the grades — why multiply instead of average

The overall score, the **composite desirability**, is a geometric mean of
the individual grades, not a plain average:

$$D = \left(\prod_{i=1}^r d_i^{w_i}\right)^{1/\sum_i w_i}$$

The reason this matters chemically: if you *averaged* the grades, a
combination that completely fails one requirement (say, purity is
unacceptably low, $d_{\text{purity}}=0$) but is excellent on everything else
could still score 50%. That is misleading — a batch that fails purity is
useless regardless of how good the yield was. Multiplying grades together
means that **if any single response scores zero, the whole combination
scores zero** ($D=0$). $D=1$ only when every response simultaneously hits
its target. This "all requirements must be met" behaviour matches how a real
formulation is judged in practice.

---

## 5. Simplex Optimisation

### The idea

Sometimes you don't want to fit a model at all — you just want to walk,
step by step, toward better conditions, adjusting course as you learn from
each experiment. This is exactly what you would do by intuition in the lab:
try a few combinations, notice which one was worst, and try something
different next time, moving away from the bad result. The **simplex
method** formalises this intuition into an algorithm.

A **simplex** is simply the smallest possible shape that can "surround" a
region in $k$ dimensions: a triangle for 2 factors (3 vertices), a
tetrahedron for 3 factors (4 vertices) — in general, $k+1$ vertices for $k$
factors. Each vertex is one experiment.

### Spendley's fixed-size simplex (1962)

The rule is simple and requires no formulas beyond averaging and reflection:
run the $k+1$ experiments at the simplex's vertices, find the **worst**
result, and replace that vertex by "reflecting" it through the average
(centroid) of the remaining, better vertices — like bouncing off a wall away
from the worst corner:

$$\mathbf{x}_r = 2\bar{\mathbf{x}} - \mathbf{x}_\text{worst}, \qquad \bar{\mathbf{x}} = \frac{1}{k}\sum_{i \neq \text{worst}}\mathbf{x}_i$$

Repeat: evaluate the new vertex, throw out whichever is now worst, reflect
again. The simplex "walks" across the response surface, always moving away
from bad results, and its size never changes.

### Nelder–Mead's adaptive simplex (1965)

The fixed step size is inefficient: large steps are good for covering
ground quickly at first, but too clumsy for fine-tuning near the optimum.
Nelder and Mead added rules so the simplex can **change size as it learns**:

1. **Reflect** — try the mirror-image point, as before.
2. **Expand** — if the reflected point turns out to be the *best point yet*,
   the simplex is moving in a promising direction, so take an even bigger
   step that way.
3. **Contract** — if the reflected point is still bad, the step was too big;
   pull back partway toward the good vertices instead.
4. **Shrink** — if even contracting doesn't help, shrink the whole simplex
   toward its best vertex and try smaller steps all around.

This adaptive behaviour is why Nelder–Mead usually reaches the optimum in
fewer experiments than the fixed-size version, especially once it gets close.

### When to use which

| Method | Step size | Best for |
|---|---|---|
| Spendley | Fixed | Quick initial exploration, simple to hand-calculate |
| Nelder–Mead | Adaptive | Fine-tuning once you are in the right neighbourhood |
| Grid search | Fixed grid | Exhaustive check, good for visualising the whole surface |

Simplex methods never use derivatives — they only ever ask "was this
combination better or worse than that one?" This makes them robust to noisy
lab measurements (unlike gradient-based methods, which can be thrown off by
a single noisy reading), though a smooth mathematical function can sometimes
be optimised faster with gradient methods. A common practical strategy is to
combine ideas from Part V: use a screening design to find the general
direction (steepest ascent, Section 5 of the optimisation notebook), then
finish with Nelder–Mead once you are close to the optimum.
