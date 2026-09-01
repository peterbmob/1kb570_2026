# Theoretical Background: Multivariate Analysis

This page builds intuition for the linear-algebra ideas behind Part IV's
methods before showing the matrix notation. All five methods here are
solving variations of the same underlying problem — you have *many*
correlated measurements per sample (elemental composition, spectral
channels, mechanical properties) and want to compress or reorganise them
into something interpretable. The formulas are included for reference, but
the plain-language explanation in each section is the part worth
understanding first.

---

## 1. Linear Algebra Prerequisites

### Why matrices at all?

A single measurement is a number; a spreadsheet row of eight XRF element
percentages is a *vector* (a list of numbers); a table of eighty samples ×
eight elements is a *matrix* (a grid of numbers). Matrix notation is just
compact bookkeeping — every operation below could be written out as
explicit sums over samples and variables, but the matrix form lets software
(and equations) treat "operate on the whole dataset at once" as a single
symbol instead of a nested loop.

### Eigendecomposition — finding a shape's natural axes

Imagine plotting two correlated variables (say, Cr% and Ni% across many
steel samples) — the cloud of points usually looks like a tilted ellipse,
not a circle, because the variables move together. **Eigendecomposition**
finds that ellipse's natural axes: the direction the cloud is *most*
stretched out (the first eigenvector), the next-most-stretched direction
perpendicular to it (the second eigenvector), and so on. The amount of
stretch along each of these directions is the corresponding **eigenvalue**.

$$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$$

For a covariance matrix $\mathbf{A}=\mathbf{S}$, this is exactly PCA's
starting point (Section 3): the eigenvectors ($\mathbf{V}$) are the new
axes to view the data along, and the eigenvalues
($\boldsymbol{\Lambda}$) tell you how much of the data's total spread each
new axis captures.

### Singular Value Decomposition (SVD) — the same idea, computed more safely

$$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$

SVD factorises the *raw data table* directly (rather than first computing a
covariance matrix and then eigendecomposing it), which is more numerically
stable — squaring numbers to build a covariance matrix can lose precision,
especially with many correlated variables, so most real PCA software (`sklearn.decomposition.PCA`
included) uses SVD under the hood even though the two approaches are
mathematically equivalent. You never need to invoke SVD yourself; it's
worth knowing the name only because you'll see it mentioned in error
messages and documentation.

---

## 2. Multiple Linear Regression (MLR)

### The idea

Multiple linear regression extends simple regression (Part III) to several
predictors at once — instead of one slope, you get one coefficient per
predictor, each answering "holding all the other predictors fixed, how does
the response change as this one predictor changes?"

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

This is the exact same least-squares idea from Part III's regression theory,
just with more columns in $\mathbf{X}$.

### Multicollinearity — when predictors can't be told apart

If two predictors are themselves strongly correlated (e.g. sintering
temperature in °C and in Kelvin — literally the same information twice),
the regression can no longer tell *which one* deserves credit for an
observed effect: infinitely many combinations of their two coefficients fit
the data equally well, so the estimates become unstable and their
individual p-values become meaningless, even though the *model as a whole*
might still predict well. The **Variance Inflation Factor** flags this:

$$\text{VIF}_j = \frac{1}{1 - R^2_j}$$

where $R^2_j$ comes from regressing predictor $j$ on all the *other*
predictors — in other words, "how well can this predictor be predicted from
the rest?" If $R^2_j$ is high (this predictor is redundant given the
others), VIF blows up. VIF > 10 (sometimes > 5) is the usual red flag.
Fixing it means removing one of the redundant predictors, combining them, or
switching to a latent-variable method like PCA or PLS (Sections 3 and 6)
that are explicitly built to handle correlated predictors gracefully.

---

## 3. Principal Component Analysis (PCA)

### The idea

You measured eight elements per steel sample, but do you really have eight
*independent* pieces of information? Probably not — elements that are added
together deliberately (like Cr and Ni in stainless steel formulations) tend
to co-vary, so much of the "spread" across eight columns is really driven
by just a couple of underlying compositional trends. **PCA** finds those
underlying trends directly: new combined axes ("principal components"),
each one a weighted blend of the original variables, ordered so the first
one captures as much of the total variation as mathematically possible, the
second captures as much of what's *left over*, and so on.

$$\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1} \mathbf{v}^\top \mathbf{S} \mathbf{v}$$

This is precisely the eigendecomposition from Section 1, applied to the
covariance matrix $\mathbf{S}$: the first eigenvector is the direction of
maximum spread, and PCA simply calls that direction "PC1."

### Why you must standardise first

If one variable is measured in wt% (values like 0–100) and another in ppm
(values like 0–5000), the ppm variable will dominate the "spread" purely
because of its units, not because it's chemically more important. Scaling
every variable to zero mean and unit variance before PCA (`StandardScaler`)
puts every variable on equal footing, exactly like coding variables in Part
IV — otherwise PCA's "directions of maximum variance" would really just be
"directions where the units happen to produce big numbers."

### The practical workflow

1. Mean-centre and standardise $\mathbf{X}$.
2. Compute the covariance matrix $\mathbf{S}$ and eigendecompose it (or,
   equivalently, run SVD directly on $\mathbf{X}$ — Section 1).
3. **Scores** ($\mathbf{T} = \tilde{\mathbf{X}}\mathbf{V}$) are each sample's
   coordinates along the new PC axes — use these to see which samples are
   similar/different.
4. **Loadings** (the columns of $\mathbf{V}$) tell you how much each
   *original variable* contributes to each PC — use these to interpret what
   a PC axis physically means (Section 12.5 of the PCA notebook covers this
   with a biplot).
5. Keep only the first few PCs that together explain most of the variance
   (typically ≥ 80–90%, checked via the scree plot) — this is the
   "dimensionality reduction" part: dozens of correlated variables become a
   handful of interpretable, uncorrelated scores.

---

## 4. Linear Discriminant Analysis (LDA)

### The idea

PCA doesn't know or care about group labels — it just finds where the data
spreads out the most, which may or may not line up with the groups you
actually care about separating. If you already know each sample's class
(e.g. steel grade) and specifically want axes that **best separate those
known groups**, use LDA instead: it searches for the projection direction
that pushes different groups' cluster centres as far apart as possible
*while also pulling each group's own points as tightly together as
possible* — literally maximising between-group spread relative to
within-group scatter:

$$J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}}$$

$\mathbf{S}_B$ (between-class scatter) measures how far apart the class
*centres* are from the overall centre; $\mathbf{S}_W$ (within-class
scatter) measures how spread out each individual class is around its own
centre. A good discriminant direction has widely separated, tightly
clustered groups — a large numerator, small denominator.

### Assumptions

The Fisher ratio $J(\mathbf{w})$ above is just an optimisation criterion —
it doesn't by itself require anything about how the data are distributed.
But LDA's actual *classification rule* (deciding which class a new sample
belongs to, not just finding the projection) is the optimal rule only under
a specific model: each class is approximately **multivariate normal**, and
all classes share a **common (pooled) covariance matrix**. The second
assumption — equal covariance across classes — is the one worth
remembering, because it's exactly what motivates **QDA** (Quadratic
Discriminant Analysis): when classes clearly have different spreads or
orientations, QDA drops the shared-covariance assumption and fits each
class its own covariance matrix instead, at the cost of needing more data
per class to estimate it reliably.

### PCA vs LDA — unsupervised vs supervised

| Property | PCA | LDA |
|---|---|---|
| Objective | Maximise total variance | Maximise between-class / within-class ratio |
| Supervision | Unsupervised (no labels needed) | Supervised (requires known class labels) |
| Best for | Exploration, noise reduction | Classification, group separation |

Use PCA first to explore an unlabelled dataset; use LDA once you have known
categories and specifically want to classify new samples or understand what
best separates them.

---

## 5. Factor Analysis (FA)

### The idea

FA asks a subtly different question than PCA: instead of "what combined
axes capture the most *total* variance," it asks "are there a small number
of *hidden, unmeasured* properties that, if you knew them, would explain
why all these *measured* variables correlate with each other in the first
place?" In the corrosion-testing case study, you don't measure
"electrochemical activity" directly — but if it exists as a real underlying
property of each sample, it would show up as several separate measurements
(corrosion current, mass loss, pit depth) all moving together, and FA's job
is to recover that hidden common cause from the pattern of correlations.

$$\mathbf{x} = \boldsymbol{\Lambda}\mathbf{f} + \boldsymbol{\varepsilon}$$

Each observed variable $\mathbf{x}$ is modelled as a combination of a few
shared, latent factors $\mathbf{f}$ (the loadings $\boldsymbol{\Lambda}$
say how strongly each variable responds to each factor) plus its own
private leftover noise $\boldsymbol{\varepsilon}$ that isn't shared with
anything else.

### Communality — how much of a variable is "shared"

$$h^2_j = \sum_{k=1}^m \lambda^2_{jk}$$

The **communality** of a variable is the fraction of its variance that the
common factors explain; the remainder ($1-h^2_j$) is that variable's own
private, unshared noise. A variable with low communality isn't well
described by the shared latent structure — it's mostly "doing its own
thing."

### Rotation — making the factors interpretable

The raw mathematical solution for the factors is not unique — you can spin
the whole solution around (rotate it) without changing how well it fits the
data, similar to how you could describe a location using compass directions
rotated 20° and it would still work, just less intuitively. **Varimax
rotation** picks a specific spin that makes each variable load strongly on
as few factors as possible (ideally just one), which makes naming and
interpreting each factor much easier — this is why the loadings in the FA
notebook come out looking like "clean" blocks rather than every variable
loading moderately on every factor.

### FA vs PCA

PCA finds axes that capture the most variance *overall*, without
distinguishing "shared" from "private" variance — every original variable's
full variance is up for grabs. FA is more selective: it deliberately models
*only* the variance that's genuinely shared across variables, setting aside
each variable's private noise. Use PCA when your goal is compact
dimensionality reduction; use FA when your goal is genuinely asking "what
hidden properties are driving these correlations?"

---

## 6. Partial Least Squares (PLS)

### The idea

PCA finds directions of maximum variance in $\mathbf{X}$ alone, with no
regard for whether that variance actually relates to a response $\mathbf{y}$
you care about (e.g. Tg from an NIR spectrum). If the response-relevant
signal is only a small fraction of the total spectral variance, PCA might
"spend" its first few components on variation that has nothing to do with
Tg, forcing you to keep many components before stumbling onto the useful
ones. **PLS** fixes this by finding components that maximise covariance
*between* X and y directly — every component it extracts is explicitly
chosen because it helps predict the response, so PLS typically needs far
fewer components than PCA-then-regress (PCR) to reach the same predictive
accuracy.

### The NIPALS algorithm, in words

The iterative NIPALS algorithm computes one component at a time: it looks
for the combination of X-variables that best tracks the current
response residual, extracts that as a component, then "subtracts out"
(deflates) what that component explains from both X and y before repeating
for the next component. Each new component is therefore chosen specifically
to explain whatever the *previous* components missed about the
relationship between X and y — you rarely need to hand-run the algorithm's
steps yourself (`sklearn.cross_decomposition.PLSRegression` does it), but
knowing that each successive component targets the *leftover* X–y
relationship explains why adding more components keeps improving prediction
up to a point, then stops helping once the real signal is exhausted (this
is exactly what the RMSECV-vs-components plot in the PLS notebook is
showing you).

### VIP scores — which original variables mattered?

$$\text{VIP}_j = \sqrt{\frac{p}{\text{SS}_Y} \sum_{a=1}^A w_{aj}^2 \, \text{SS}_{Y,a}}$$

Because PLS components are combinations of many original variables (e.g.
spectral channels), it's not immediately obvious from the components alone
which *original* variables mattered most. The VIP score re-expresses each
variable's total contribution across all retained components into a single
number; variables with VIP > 1 contributed more than an "average" variable
would and are worth a closer chemical look (e.g. do they correspond to a
known IR absorption band?).

### Comparison of latent-variable methods

| Method | Uses X only? | Uses y during decomposition? | Typically needs |
|---|---|---|---|
| PCA (then regress) | Yes | No | More components (blind to y) |
| PLS | Yes + y | Yes | Fewer components (targeted at y) |
| MLR | Yes + y | No latent variables at all | Struggles when predictors are collinear or numerous |

PLS is usually a strong first choice whenever you have many, possibly
collinear predictors and a response to predict — exactly the situation in
spectroscopy (many correlated wavenumber channels) and most real process
data. It is not automatically better than PCA-then-regress (PCR), though:
when the directions of largest X-variance happen to align with the
directions most predictive of y, the two methods can tie, and with small
samples or a purely exploratory goal PCR's y-blind components can even be
preferable — see Notebook 15 §15.6–15.8 for worked examples of both.
