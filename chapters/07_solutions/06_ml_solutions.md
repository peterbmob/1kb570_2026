# Part VI Solutions: Introduction to AI and Machine Learning

Solutions to the exercises in Notebooks 16–20 (Unsupervised Clustering
through Neural Networks). Code blocks below are written to be pasted
directly into the corresponding notebook (they assume its imports, `rng`,
and variables already exist in your session) — they are not executed as
part of building this page. Every numeric result quoted here was
independently computed with the same seeded `rng = np.random.default_rng(...)`
as the source notebook. This part is marked "Extra Material" in the book,
and these solutions match that lighter touch — the point is to see the
methods working on real (simulated) materials data, not to hedge every
claim as rigorously as Parts II–V do.

## Notebook 16: Unsupervised Clustering and Dimensionality Reduction

**Exercise 1 — Effect of scaling**

> **Hint:** Re-run `KMeans(n_clusters=4, ...)` on `df[elements].values`
> directly (no `StandardScaler`), then compare `adjusted_rand_score` against
> the scaled version from Section 16.2 — and separately look at
> `df[elements].var()` to see which elements would dominate a raw Euclidean
> distance.

:::{dropdown} Show full solution
```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X_raw = df[elements].values

km4_raw = KMeans(n_clusters=4, random_state=0, n_init=20)
km_labels_raw = km4_raw.fit_predict(X_raw)
ari_raw = adjusted_rand_score(grades, km_labels_raw)
print(f'ARI without scaling: {ari_raw:.4f}')
print(f'ARI with StandardScaler (Section 16.2): {ari:.4f}')

print('\nRaw feature variance (drives unscaled Euclidean distance):')
print(df[elements].var().sort_values(ascending=False).round(3))
```
Output: ARI without scaling is **1.0000** — identical to the scaled result.
That's a genuinely different outcome than you'd usually expect (unscaled
K-Means normally suffers badly), and it happens here only because the four
grades are so well separated that even a distance metric dominated by a
handful of elements still keeps them apart. The variance table shows why
scaling would matter on a harder dataset: Ni (var≈9.7), Fe (≈8.6), and Cr
(≈5.3) would completely dominate an unscaled Euclidean distance, while Si
(≈0.007) and N (≈0.003) — despite being just as diagnostic per-grade —
would contribute almost nothing to the raw distance. **The lesson is about
the mechanism, not this particular ARI number**: `StandardScaler` is still
the right default, because you can't know in advance whether your dataset
will be this forgiving.
:::

**Exercise 2 — Perplexity sensitivity**

> **Hint:** Loop `perplexity` over `[5, 15, 50, 100]`, rerun `TSNE(...).fit_transform(X)`
> for each, and score how well the four true grades separate in each 2-D
> embedding with `silhouette_score(Z_tsne, grades)` — a quantitative stand-in
> for "does this look separated" that doesn't depend on you eyeballing four plots.

:::{dropdown} Show full solution
```python
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

for perp in [5, 15, 50, 100]:
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000, random_state=42)
    Z = tsne.fit_transform(X)
    sil = silhouette_score(Z, grades)
    print(f'perplexity={perp:>3}: silhouette (true grades) = {sil:.4f}')
```
Output: `perplexity=5` → 0.766, `15` → **0.907 (best)**, `50` → 0.869,
`100` → 0.852. Perplexity roughly sets "how many neighbours each point
tries to stay close to" in the embedding: too low (5, with 30 points per
grade) makes t-SNE over-focus on very local structure and fragment each
grade's cluster into sub-blobs; too high (100, close to the full dataset
size of 120) forces it to balance far more long-range relationships than
the data really has structure for, blurring the four groups back together
slightly. The default used in Section 16.5 (`perplexity=30`) sits
comfortably in the well-behaved middle of this range — a reasonable
default is usually somewhere between roughly 5% and 50% of the sample
size, and 15–50 here (12–40% of n=120) is exactly that range.
:::

**Exercise 3 — DBSCAN parameter sweep**

> **Hint:** Loop `eps` from 0.5 to 3.0 in steps of 0.25 with `min_samples=5`
> fixed, and for each value record `len(set(labels)) - (1 if -1 in labels else 0)`
> (cluster count) and `np.mean(labels == -1)` (noise fraction) — the same
> two diagnostics Section 16.4 already prints for the single `eps=1.5` case.

:::{dropdown} Show full solution
```python
from sklearn.cluster import DBSCAN

for eps in np.arange(0.5, 3.01, 0.25):
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = np.mean(labels == -1)
    ari = adjusted_rand_score(grades, labels) if n_clusters > 0 else float('nan')
    print(f'eps={eps:.2f}: n_clusters={n_clusters}, noise_frac={noise_frac:.3f}, ARI={ari:.3f}')
```
Output (abbreviated): `eps=0.50` → 5 clusters, 21.7% noise, ARI=0.593
(fragmented — too tight); `eps=0.75`–`1.00` → 4 clusters, <1% noise,
ARI=0.989; **`eps=1.25`–`2.00` → 4 clusters, 0% noise, ARI=1.000 (the
plateau)**; `eps=2.25`–`3.00` → only 3 clusters, ARI drops to 0.709 (too
loose — two grades have merged). **The optimal range is roughly `eps` = 1.25
to 2.0** — a wide, flat plateau where DBSCAN recovers the true structure
exactly, bounded on the low side by fragmentation and on the high side by
merging. Section 16.7's point about DBSCAN needing "similar density between
clusters" is visible here too: a single `eps` works for all four grades
simultaneously only because they were simulated with comparably tight,
independent noise — a real dataset with one denser and one more diffuse
grade would likely have no such wide plateau.
:::

**Exercise 4 — UMAP supervised embedding**

> **Hint:** `umap.UMAP(...).fit_transform(X, y=y_encoded)` needs a *numeric*
> label array — encode the string grades first with
> `sklearn.preprocessing.LabelEncoder` (UMAP's supervised mode will reject
> the raw string array directly).

:::{dropdown} Show full solution
```python
import umap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

y_enc = LabelEncoder().fit_transform(grades)

reducer_unsup = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                           random_state=42, n_jobs=1)
Z_unsup = reducer_unsup.fit_transform(X)

reducer_sup = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                         random_state=42, n_jobs=1)
Z_sup = reducer_sup.fit_transform(X, y=y_enc)

print(f'Unsupervised UMAP silhouette (true grades): {silhouette_score(Z_unsup, grades):.4f}')
print(f'Supervised UMAP silhouette (true grades):   {silhouette_score(Z_sup, grades):.4f}')
```
Output: unsupervised 0.9163 vs supervised 0.9135 — essentially the same,
which is itself an informative (if slightly anticlimactic) result: on data
this cleanly separated, unsupervised UMAP already recovers the label
structure almost perfectly on its own, so telling it the labels in advance
buys nothing. **Supervised UMAP earns its keep on harder problems** — noisy
or overlapping classes where the unsupervised embedding blends groups
together — where nudging the embedding toward known class structure can
recover a much cleaner separation than the unsupervised version, at the
cost of the embedding now depending on labels you'd need to already have
(e.g. for visualising a *training* set before building a classifier, not
for exploring genuinely unlabelled data).
:::

## Notebook 17: Gaussian Process Regression (Kriging)

**Exercise 1 — Add a new training point**

> **Hint:** Append `(Al=0.24, T=1100)` and its measured (noisy)
> conductivity to the 9 existing training rows, re-fit the scaler and GPR
> exactly as in Section 17.3, and compare `predict(..., return_std=True)`
> at the new point's location — before vs. after — not just the grid-wide
> optimum.

:::{dropdown} Show full solution
```python
Al_new, T_new = 0.24, 1100.0
y_new = true_conductivity(Al_new, T_new)   # continues the notebook's rng sequence

Al_train2 = np.append(Al_train, Al_new)
T_train2  = np.append(T_train, T_new)
y_train2  = np.append(y_train, y_new)

X_doe2 = np.column_stack([Al_train2, T_train2])
scaler_doe2 = StandardScaler()
X_doe2_sc = scaler_doe2.fit_transform(X_doe2)

gpr_doe2 = GaussianProcessRegressor(kernel=kernel_doe, n_restarts_optimizer=15,
                                     normalize_y=True, random_state=0)
gpr_doe2.fit(X_doe2_sc, y_train2)

# Uncertainty specifically AT the new point, before vs after
_, std_before = gpr_doe.predict(scaler_doe.transform([[Al_new, T_new]]), return_std=True)
_, std_after  = gpr_doe2.predict(scaler_doe2.transform([[Al_new, T_new]]), return_std=True)
print(f'Std at (0.24, 1100) before: {std_before[0]:.4f}')
print(f'Std at (0.24, 1100) after:  {std_after[0]:.4f}')
```
Output: the new observation reads `σ = 1.4143 mS/cm` (close to the
noise-free peak value there, ≈1.43). Uncertainty right at that point
collapses from **0.0771 to 0.0013** — almost two orders of magnitude,
exactly what a GP should do when you add data precisely where it was most
uncertain. The predicted optimum barely moves (Al 0.228→0.238 mol%, T
stays at 1103 °C) but its predicted value rises from μ=1.317 to μ=1.414,
pulled toward the new, higher observation. Interestingly, the *grid-averaged*
std over the whole domain ticks up slightly (0.1234→0.1329) rather than
down — a single new point sharpens the kernel's optimised length scales
(it now has to explain a bit more local curvature near the peak), and a
shorter length scale means the GP trusts extrapolation *away* from data
less than before. Local uncertainty collapsing while the domain-wide
average barely changes (or even rises slightly) is a normal, non-alarming
outcome of adding one point to a 9-point design — it is not a bug.
:::

**Exercise 2 — Leave-one-out cross-validation**

> **Hint:** For each of the 9 points, refit both a GPR (Section 17.3's exact
> kernel) and a quadratic RSM (`PolynomialFeatures(degree=2)` +
> `LinearRegression`, the same recipe as Part V's RSM notebook) on the
> other 8, predict the held-out point, and compare `sqrt(mean_squared_error(...))`
> across both sets of 9 leave-one-out predictions.

:::{dropdown} Show full solution
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

n_pts = len(Al_train)
loo_gpr, loo_rsm = np.zeros(n_pts), np.zeros(n_pts)

for i in range(n_pts):
    mask = np.arange(n_pts) != i
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_doe[mask])
    X_te_sc = sc.transform(X_doe[i:i+1])

    gpr_loo = GaussianProcessRegressor(kernel=kernel_doe, n_restarts_optimizer=10,
                                        normalize_y=True, random_state=0)
    gpr_loo.fit(X_tr_sc, y_train[mask])
    loo_gpr[i] = gpr_loo.predict(X_te_sc)[0]

    rsm = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    rsm.fit(X_doe[mask], y_train[mask])
    loo_rsm[i] = rsm.predict(X_doe[i:i+1])[0]

print(f'GPR LOO RMSE:            {np.sqrt(mean_squared_error(y_train, loo_gpr)):.4f} mS/cm')
print(f'Quadratic RSM LOO RMSE:  {np.sqrt(mean_squared_error(y_train, loo_rsm)):.4f} mS/cm')
```
Output: **GPR LOO RMSE = 0.2805**, **quadratic RSM LOO RMSE = 0.2161** — the
RSM actually generalises slightly *better* here, the reverse of the
"GPR is more flexible so it should win" intuition. With only 9 points in a
3×3 grid, leaving one out gives the GPR just 8 points to re-optimise a
2-length-scale Matérn kernel from scratch each time, and small-sample
kernel optimisation is noisy; the quadratic RSM, by contrast, has a fixed,
low-dimensional functional form (6 coefficients) that is a reasonable match
for this particular Gaussian-hill-shaped true surface even with only 8
points to fit it. This is a concrete illustration of Section 17.4's
comparison table: GPR's flexibility is an advantage when you have enough
data to support it, but a fixed, low-order polynomial can be the more
data-efficient choice at the very small end of the sample-size range —
neither method is unconditionally better.
:::

**Exercise 3 — Kernel sensitivity**

> **Hint:** Swap `Matern(..., nu=2.5)` for a plain `RBF(...)` with the same
> bounds in `kernel_doe`, refit, and compare both the predicted optimum
> location/value *and* the optimised kernel string (watch for a
> `noise_level` that's been driven to its lower bound — Section 17.2's
> `ConvergenceWarning` diagnostic applies here too).

:::{dropdown} Show full solution
```python
from sklearn.gaussian_process.kernels import RBF

kernel_rbf = ConstantKernel(1.0, (0.1, 10)) * RBF(
    length_scale=[1.0, 1.0], length_scale_bounds=(0.1, 10)
) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 0.1))

gpr_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=15,
                                    normalize_y=True, random_state=0)
gpr_rbf.fit(X_doe_sc, y_train)
print(f'Optimised RBF kernel: {gpr_rbf.kernel_}')

mu_rbf, _ = gpr_rbf.predict(X_pred, return_std=True)
best_idx_rbf = np.argmax(mu_rbf)
print(f'RBF optimum:    Al={X_pred[best_idx_rbf,0]*scaler_doe.scale_[0]+scaler_doe.mean_[0]:.3f}, '
      f'mu={mu_rbf[best_idx_rbf]:.4f}')
print(f'Matern optimum (Section 17.3): Al=0.228, T=1103, mu=1.3173')
```
Output: the RBF optimum lands at essentially the same place —
**Al=0.233 mol%, T=1103 °C, μ=1.336** — versus Matérn 5/2's Al=0.228 mol%,
T=1103 °C, μ=1.317. The two kernels agree closely on *where* the optimum
is; the more telling difference is the optimised `WhiteKernel` term, which
collapses to its lower bound (`noise_level≈1e-5`) for RBF instead of
settling near the true simulated noise (σ²≈0.025²). Exactly as Section
17.2 flagged, that's RBF's infinite-differentiability assumption absorbing
this small, somewhat noisy 9-point dataset almost as pure signal rather
than signal-plus-noise — a mild version of the same
"ConvergenceWarning at a bound" pathology, even without an explicit
warning firing here. Leave-one-out RMSE for RBF (0.2718) comes out
marginally *better* than Matérn's (0.2805) on this run, but **Matérn 5/2 is
still the kernel to trust more** for a real sintering/doping response: a
noise term pinned at its numerical floor is a sign of overfitting risk that
a single LOO comparison on 9 points isn't powerful enough to reliably catch.
:::

## Notebook 18: Bayesian Optimisation for Experiment Design

**Exercise 1 — Batch BO**

> **Hint:** "Kriging believer" means: fit the GP, find the next EI-maximising
> point, add it back into the training set with its *predicted* mean (not a
> real measurement) so the next-in-batch query accounts for it, repeat 3
> times, then finally run the 3 real (noisy) experiments and refit — mirror
> Section 18.3's loop structure but nest an inner 3-iteration "hallucination"
> loop inside each outer iteration.

:::{dropdown} Show full solution
```python
X_batch, y_batch = X_init.copy(), y_init.copy()
n_outer = 5   # 5 outer iterations x batch of 3 = 15 new queries, same budget as Section 18.3

for outer in range(n_outer):
    X_hallu, y_hallu, batch_points = X_batch.copy(), y_batch.copy(), []
    for b in range(3):
        gpr_h = GaussianProcessRegressor(kernel=kernel_bo, n_restarts_optimizer=3,
                                          normalize_y=True, random_state=outer)
        gpr_h.fit(X_hallu, y_hallu)
        x_next = next_query(gpr_h, y_hallu.max(), bounds, acq_fn='EI')
        mu_hallu = gpr_h.predict(x_next.reshape(1, -1))[0]   # "believed" value, not real
        X_hallu = np.vstack([X_hallu, x_next])
        y_hallu = np.append(y_hallu, mu_hallu)
        batch_points.append(x_next.copy())

    # Now run the *real* (noisy) experiments for the whole batch at once
    for x_next in batch_points:
        y_real = black_box(x_next[0], x_next[1])
        X_batch = np.vstack([X_batch, x_next])
        y_batch = np.append(y_batch, y_real)

print(f'Batch BO best after {4 + n_outer*3} evaluations: {y_batch.max():.4f}')
print(f'Sequential BO best (Section 18.3):                {y_obs.max():.4f}')
```
Output: batch BO's final best is **0.8878 — identical to sequential BO's**.
That's not a coincidence specific to batching; it's the same result Section
18.3's own take-home already flags: both runs share the same 4 initial
random points, and the winning point (rate=0.8878) was one of *those*, not
a point either the sequential or the batch acquisition loop went on to
find. Neither run's `best_so_far` trajectory moves past its starting value.
The real benefit of batching isn't visible in this single-seed comparison
at all — it's *wall-clock time*: batch BO reaches the same 19-evaluation
budget in only 5 sequential decision rounds instead of 15, running 3
furnace/synthesis experiments in parallel each round. The kriging-believer
heuristic's cost is that the 2nd and 3rd points in each batch are chosen
using a hallucinated (not real) value for the 1st, which is why batch BO
can occasionally under-explore relative to fully sequential BO on a harder
landscape than this one.
:::

**Exercise 2 — Noisy objective**

> **Hint:** Set `noise_std=0.10` in `black_box` (5× Section 18.1's default),
> rerun the Section 18.3 loop, and compare not just the reported best value
> but that value against the true, *noise-free* surface at the same (x1, x2)
> — the gap between the two is the point of this exercise.

:::{dropdown} Show full solution
```python
def black_box_noisy(x1, x2, noise_std=0.10):
    x1n, x2n = (x1 - 0.055)/0.045, (x2 - 500)/200
    return (0.85*np.exp(-0.5*(x1n**2+x2n**2)/0.25)
            + 0.3*np.exp(-0.5*((x1n+0.6)**2+(x2n+0.5)**2)/0.1)
            + rng.normal(0, noise_std))

def true_surface(x1, x2):   # noise-free, for auditing the reported optimum
    x1n, x2n = (x1 - 0.055)/0.045, (x2 - 500)/200
    return (0.85*np.exp(-0.5*(x1n**2+x2n**2)/0.25)
            + 0.3*np.exp(-0.5*((x1n+0.6)**2+(x2n+0.5)**2)/0.1))

X_init_n = np.column_stack([rng.uniform(0.01,0.10,4), rng.uniform(300,700,4)])
y_init_n = np.array([black_box_noisy(x[0], x[1]) for x in X_init_n])
X_obs_n, y_obs_n = X_init_n.copy(), y_init_n.copy()

kernel_bo_noisy = ConstantKernel(1.0) * Matern(
    length_scale=[0.05, 200], length_scale_bounds=[(0.001,0.5),(10,1000)], nu=2.5
) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))

for iteration in range(15):
    gpr_n = GaussianProcessRegressor(kernel=kernel_bo_noisy, n_restarts_optimizer=5,
                                      normalize_y=True, random_state=iteration)
    gpr_n.fit(X_obs_n, y_obs_n)
    x_next = next_query(gpr_n, y_obs_n.max(), bounds, acq_fn='EI')
    y_next = black_box_noisy(x_next[0], x_next[1])
    X_obs_n = np.vstack([X_obs_n, x_next])
    y_obs_n = np.append(y_obs_n, y_next)

best_idx_n = np.argmax(y_obs_n)
print(f'Noisy BO reported best: {y_obs_n[best_idx_n]:.4f}  '
      f'vs true (noise-free) surface there: {true_surface(*X_obs_n[best_idx_n]):.4f}')
```
Output: with `noise_std=0.10`, the reported best is **0.9986**, but the true
noise-free response at that exact (x1, x2) is only **0.8559** — the noisy
"best" overstates the real optimum by **+0.14**. For comparison, the
original `noise_std=0.02` run reported 0.8878 against a true value of
0.8527 — overstated by only +0.035, roughly 4× less. **BO handles noisy
objectives more gracefully than steepest ascent because the GP posterior
mean is a *regression* across every nearby observation, not a single
gradient estimate from one (possibly unlucky) noisy reading** — steepest
ascent has no mechanism to average away a noisy sample the way the GP does.
But this result also shows what the GP *can't* fully fix: the bookkeeping
step that reports "the best observed `y` so far" is still just picking the
single most optimistic noisy reading, and at higher noise that reading
increasingly overstates the truth. This is exactly why Section 18.7's
workflow (step 8) insists on confirming any candidate optimum with 2–3
replicate experiments before trusting it — the higher the noise, the more
that confirmation step matters.
:::

**Exercise 3 — Multi-objective BO**

> **Hint:** You don't need a new acquisition function to *identify* a Pareto
> front — fit a second GPR surrogate for particle size on a grid, then keep
> only the grid points where no other point has both higher rate *and* lower
> particle size (a "non-dominated" point). `scipy` isn't strictly required;
> a nested loop or vectorised boolean mask over the grid works fine.

:::{dropdown} Show full solution
```python
# Second (illustrative) objective: particle size grows with calcination T,
# shrinks slightly with higher precursor concentration.
def particle_size(x1, x2):
    return 15 + 0.04*(x2 - 300) - 40*x1 + rng.normal(0, 1.0, np.shape(x1))

size_obs = particle_size(X_obs[:, 0], X_obs[:, 1])   # reuse Section 18.3's X_obs

gpr_rate = GaussianProcessRegressor(kernel=kernel_bo, normalize_y=True, random_state=0)
gpr_rate.fit(X_obs, y_obs)
gpr_size = GaussianProcessRegressor(kernel=kernel_bo, normalize_y=True, random_state=0)
gpr_size.fit(X_obs, size_obs)

grid = np.column_stack([X1G.ravel(), X2G.ravel()])
mu_rate = gpr_rate.predict(grid)
mu_size = gpr_size.predict(grid)

# A point is Pareto-optimal if no other point has BOTH higher rate AND lower size
is_dominated = np.zeros(len(grid), dtype=bool)
for i in range(len(grid)):
    is_dominated[i] = np.any((mu_rate > mu_rate[i]) & (mu_size < mu_size[i]))
pareto_mask = ~is_dominated

size_threshold = 12.0   # e.g. a downstream process spec on particle size
feasible = mu_size < size_threshold
best_feasible_idx = np.argmax(np.where(feasible, mu_rate, -np.inf))
print(f'Pareto-optimal grid points: {pareto_mask.sum()} / {len(grid)}')
print(f'Best rate with size < {size_threshold}: '
      f'x1={grid[best_feasible_idx,0]:.4f}, x2={grid[best_feasible_idx,1]:.0f}, '
      f'rate={mu_rate[best_feasible_idx]:.4f}, size={mu_size[best_feasible_idx]:.2f}')
```
This is a template, not a fixed numeric answer — `particle_size` here is a
made-up illustrative second response, so the specific Pareto front and
"best feasible point" will depend on whatever second objective you plug
in. The general pattern is the one that matters: fit one GPR surrogate per
objective, evaluate both on a shared grid, and either (a) filter to a hard
constraint (`mu_size < threshold`) and maximise the other objective within
it, or (b) keep every *non-dominated* point as the actual Pareto front, as
computed above. Section 18.7's table lists the more principled alternative
— **EHVI** (Expected Hypervolume Improvement) — for when you want the
acquisition function itself, not just post-hoc grid filtering, to
explicitly balance exploring the full Pareto front against exploiting a
known-good region of it.
:::

## Notebook 19: Ensemble Methods: Random Forest and Gradient Boosting

**Exercise 1 — Hyperparameter tuning**

> **Hint:** `GridSearchCV(GradientBoostingRegressor(learning_rate=0.05, subsample=0.8, random_state=0), param_grid, cv=kf, scoring='r2')`
> with `param_grid = {'n_estimators': [...], 'max_depth': [...]}` — keep
> `learning_rate` and `subsample` fixed at Section 19.4's values so you're
> isolating the two hyperparameters the exercise asks about, then re-run
> `cross_val_predict` with `grid.best_params_` to get a genuinely
> out-of-sample Q² for the comparison (`grid.best_score_` alone is an
> internal CV score, not the same RMSECV metric used elsewhere in the notebook).

:::{dropdown} Show full solution
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 200, 300, 500], 'max_depth': [2, 3, 4, 6]}
gbm_base = GradientBoostingRegressor(learning_rate=0.05, subsample=0.8, random_state=0)
grid = GridSearchCV(gbm_base, param_grid, cv=kf, scoring='r2', n_jobs=-1)
grid.fit(X_sc, y)
print('Best params:', grid.best_params_)

gbm_tuned = GradientBoostingRegressor(learning_rate=0.05, subsample=0.8,
                                       random_state=0, **grid.best_params_)
y_tuned_cv = cross_val_predict(gbm_tuned, X_sc, y, cv=kf)
q2_tuned = r2_score(y, y_tuned_cv)
rmse_tuned = np.sqrt(mean_squared_error(y, y_tuned_cv))
print(f'Tuned GBM: Q2={q2_tuned:.4f}, RMSECV={rmse_tuned:.2f} mAh/g')
print(f'Default GBM (Section 19.4): Q2=0.7397, RMSECV=4.47 mAh/g')
```
Output: the search picks **`max_depth=2, n_estimators=300`** — *shallower*
trees than the notebook's default `max_depth=4`, not more of them. Tuned
performance is **Q²=0.7903, RMSECV=4.01 mAh/g**, an improvement of
**+0.051 Q²** and **−0.46 mAh/g RMSECV** over the default. This result
lines up with Section 19.4's own training-curve admonition about GBM
overfitting: with `max_depth=4` and 300 boosting rounds, the default model
was already fitting the training set almost exactly (training RMSE≈0.3
mAh/g) while its CV score lagged — the grid search's fix is exactly to
shrink each individual tree's capacity (`max_depth` 4→2) so 300 rounds of
boosting can no longer memorise the training noise as easily, trading a bit
of training-set fit for meaningfully better generalisation.
:::

**Exercise 2 — OOB score**

> **Hint:** Set `oob_score=True` on the same `RandomForestRegressor(...)` call
> from Section 19.3, fit it once on the *full* dataset (no `cross_val_predict`
> needed for the OOB number — it's built into `.fit()`), and read
> `rf.oob_score_` directly.

:::{dropdown} Show full solution
```python
rf_oob = RandomForestRegressor(n_estimators=200, max_features='sqrt',
                                min_samples_leaf=3, random_state=0, oob_score=True)
rf_oob.fit(X_sc, y)
print(f'RF OOB score (R2): {rf_oob.oob_score_:.4f}')
print(f'RF 10-fold CV Q2 (Section 19.3): {q2_rf:.4f}')
```
Output: **OOB score = 0.6048** vs. **10-fold CV Q² = 0.6047** — a difference
of only +0.0001, essentially exact agreement. This is the expected result:
each tree's out-of-bag samples (the ~1/3 of rows not drawn into that tree's
bootstrap sample) are, by construction, never seen by that tree during
training, so averaging predictions from only the trees that didn't see a
given row is already a form of built-in cross-validation — no separate
`KFold` loop needed. The practical payoff is `oob_score=True` gives you an
honest generalisation estimate from a *single* `.fit()` call, useful for
quick iteration, though a full `cross_val_predict` is still worth running
once at the end for the residual/predicted-vs-actual diagnostics OOB alone
doesn't give you.
:::

**Exercise 3 — SHAP values**

> **Hint:** `pip install shap`, then `shap.TreeExplainer(rf).shap_values(X_sc)`
> gives per-sample, per-feature attributions directly (works natively on
> tree ensembles, no background dataset sampling needed); `np.abs(shap_values).mean(axis=0)`
> is the SHAP analogue of `rf.feature_importances_`, and
> `explainer.shap_interaction_values(X_sc)` gives the pairwise interaction
> strengths the exercise also asks about.

:::{dropdown} Show full solution
```python
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sc)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
order_shap = np.argsort(mean_abs_shap)[::-1]
print('SHAP ranking:', [features[i] for i in order_shap])
print('Built-in ranking (Section 19.3):', [features[i] for i in np.argsort(rf.feature_importances_)[::-1]])

interact = explainer.shap_interaction_values(X_sc)
mean_abs_interact = np.abs(interact).mean(axis=0)
pairs = sorted(
    [(features[i], features[j], mean_abs_interact[i, j])
     for i in range(len(features)) for j in range(i+1, len(features))],
    key=lambda t: -t[2])
for f1, f2, val in pairs[:4]:
    print(f'  {f1} x {f2}: {val:.4f}')
```
Output: SHAP's mean-|value| ranking is **identical in order** to the
built-in impurity importance — `cool_rate > time_h > T_sinter > O2_pO2 >
Ni_frac > Li_excess` — reassuring agreement between the two methods on this
dataset, even though SHAP is theoretically the more rigorous, direction-aware
measure. The interaction values are the more interesting (and more
honest) result: the two *largest* pairwise interactions are **time_h ×
cool_rate (0.171)** and **Ni_frac × cool_rate (0.159)** — not the
`Ni_frac × O2_pO2` term that was actually built into the simulated
`capacity` formula in Section 19.1, which ranks only **8th of 15 pairs**
(0.062). This is a genuinely useful caveat, not a failure of SHAP: mean
absolute interaction strength is inflated by each feature's own overall
importance as well as by genuine interaction structure, so the two
globally most important features (`cool_rate` and `time_h`) show up with
large *apparent* interaction values in part simply because both terms
individually have large SHAP magnitudes to begin with. **Don't read the
single largest SHAP interaction pair as automatically "the" true
interaction** — cross-check it against domain knowledge or, as here,
against the actual generating model when you have it.
:::

## Notebook 20: Neural Networks and Deep Learning Basics

**Exercise 1 — Architecture search**

> **Hint:** Loop `hidden_layer_sizes` over a handful of 1-, 2-, and
> 3-layer shapes in `MLPRegressor` (same settings as Section 20.3
> otherwise), score each with the same `cross_val_predict` + `r2_score`
> recipe, and count parameters by hand: a `Linear` layer of shape
> `(n_in, n_out)` has `n_in*n_out + n_out` weights+biases — sum that across
> every layer including the final single-output one.

:::{dropdown} Show full solution
```python
architectures = [(8,), (16,), (32,), (64,),
                  (16,8), (32,16), (64,32),
                  (32,16,8), (64,32,16)]

def count_params(n_in, hidden, n_out=1):
    sizes = [n_in] + list(hidden) + [n_out]
    return sum(sizes[i]*sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))

for arch in architectures:
    mlp = MLPRegressor(hidden_layer_sizes=arch, activation='relu', solver='adam',
                        max_iter=500, learning_rate_init=1e-3, alpha=1e-4,
                        random_state=0, early_stopping=True, validation_fraction=0.1)
    yp = cross_val_predict(mlp, X_sc, Tg, cv=kf)
    q2 = r2_score(Tg, yp)
    print(f'hidden={arch!s:<14} params={count_params(n_waves, arch):>5}  Q2={q2:.4f}')
```
Output ranges from Q²=0.283 (a single 16-neuron layer — too narrow) up to
the best result, **Q²=0.605 at `hidden=(32,16)`, 2177 parameters** — and
**no architecture in this search reaches Q² > 0.95**. That's not a failed
search; it's the expected answer given what Section 20.3 already
established: with ~90 training samples per CV fold and only 50 input
channels, every one of these architectures still has far more parameters
than samples, so more layers or more neurons doesn't fix the fundamental
data-size mismatch — it mostly just reshuffles which noise gets
memorised. PLS's Q²=0.9895 on the identical data (Section 20.5) remains
completely out of reach for any plain MLP tried here; getting an MLP to
Q²>0.95 on this problem would require either far more training samples or
building in the same linear/collinear structure PLS exploits by
construction (e.g. via a PLS or PCA preprocessing step before the network).
:::

**Exercise 2 — Dropout regularisation**

> **Hint:** Change only `dropout=0.2` → `dropout=0.5` in the `SpectralMLP(...)`
> constructor in Section 20.4's cell (leave the optimiser, `weight_decay`,
> batch size, and early-stopping patience untouched so the comparison
> isolates dropout alone), rerun the training loop, and compare both the
> final train/val MSE curves and the held-out R².

:::{dropdown} Show full solution
```python
def run_with_dropout(dropout, seed=0):
    torch.manual_seed(seed)
    model = SpectralMLP(n_waves, hidden=(64, 32), dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
    best_val, best_state, patience_ctr = np.inf, None, 0
    for epoch in range(300):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, patience_ctr = val_loss, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= 30:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_va).cpu().numpy()
    return r2_score(Tg_va_true, pred)

for dp in [0.2, 0.5]:
    print(f'dropout={dp}: val R2 = {run_with_dropout(dp):.4f}')
```
Output: **R² drops from 0.947 (dropout=0.2, Section 20.4's own result) to
0.903 (dropout=0.5)** — heavier regularisation makes this network *worse*,
not better. Looking at the training curves explains why: at dropout=0.2,
Section 20.4's own admonition already noted the train and validation MSE
curves track almost identically the whole way through, meaning there was
no overfitting gap for stronger regularisation to close. Raising dropout to
0.5 instead just randomly zeroes half of every hidden layer's activations
on every forward pass — train MSE at convergence roughly quadruples
(≈40 → ≈160 °C²) because the network can no longer reliably use most of its
own capacity, and validation performance follows it down rather than
improving. **The lesson: dropout is a fix for a diagnosed overfitting
problem, not a knob to crank up "for safety"** — applied to a network that
wasn't overfitting in the first place, it only removes useful capacity.
:::

**Exercise 3 — Transfer learning concept**

> **Hint:** No numeric answer is required here — describe the two named
> strategies concretely for `SpectralMLP` specifically: what
> `model.load_state_dict(...)` and setting `param.requires_grad = False`
> on the first hidden layer's parameters would each look like in code, and
> reason about *why* freezing early layers makes sense for spectra
> specifically (what those early layers are likely learning).

:::{dropdown} Show full solution
```python
# (a) Fine-tuning: load pretrained weights, then keep training all of them
#     on the new, smaller 500-sample dataset with a lower learning rate.
model_new = SpectralMLP(n_waves, hidden=(64, 32), dropout=0.2).to(device)
model_new.load_state_dict(torch.load('spectral_mlp_pretrained.pt'))
optimizer = optim.Adam(model_new.parameters(), lr=1e-4, weight_decay=1e-4)  # smaller lr than training from scratch
# ... then train_dl / epoch loop exactly as in Section 20.4, on the new data

# (b) Feature extraction: freeze the first hidden block, only train the last layer
for name, param in model_new.named_parameters():
    if name.startswith('net.0') or name.startswith('net.1'):   # first Linear+ReLU block
        param.requires_grad = False
optimizer = optim.Adam(
    [p for p in model_new.parameters() if p.requires_grad], lr=1e-3
)
```
Both strategies assume the new 500-spectrum dataset shares the same 50
wavenumber channels and a *related* underlying chemistry (different
polymer system, same NIR absorption physics) — that's what transfer
learning requires to make sense at all. **(a) Fine-tuning** lets every
weight keep adapting, which is the better choice when the new dataset,
while smaller than you'd like from scratch, is still large enough (hundreds
of samples) to safely update the whole network without immediately
overfitting; a smaller learning rate than training-from-scratch protects
the already-useful pretrained weights from being overwritten too fast
early in training. **(b) Freezing the first hidden layer** treats it as a
fixed, general-purpose feature extractor — early layers in a spectral MLP
tend to learn broad, transferable patterns (something close to "does this
region of the spectrum have a peak, and how sharp") that are probably
similar across different polymer chemistries, while only the last layer
needs to relearn the *specific* mapping from those features to this new
system's Tg values. Freezing is the safer, lower-variance option when the
new dataset is smaller still (tens to low hundreds of samples) — training
fewer parameters means less opportunity to overfit, at the cost of the
model being unable to adapt if the new system's spectra genuinely need
different early-layer features than the pretrained ones learned.
:::
