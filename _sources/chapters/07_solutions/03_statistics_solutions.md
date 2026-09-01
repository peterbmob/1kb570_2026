# Part III Solutions: Applied Statistics

Solutions to the exercises in Notebooks 6–12 (Part III uses the book's
global notebook numbering, continuing on from Part I/II). Code blocks below
are written to be pasted directly into the corresponding notebook (they
assume its imports, `rng`, and variables already exist in your session) —
they are not executed as part of building this page. Every numeric result
quoted here was independently computed with the same seeded
`rng = np.random.default_rng(...)` as the source notebook, so re-running
the code below in a fresh copy of that notebook reproduces it exactly.

## Notebook 6: Descriptive Statistics

**Exercise 1 — Grubbs test**

> **Hint:** Grubbs' test statistic is $G = \max_i|x_i-\bar x|/s$ — the
> largest absolute Z-score, essentially — compared against a critical value
> built from the t-distribution:
> $G_{\text{crit}} = \frac{n-1}{\sqrt n}\sqrt{\dfrac{t^2_{\alpha/(2n),\,n-2}}{n-2+t^2_{\alpha/(2n),\,n-2}}}$.
> `scipy.stats` has no `grubbs` function directly, so this is one of the few
> tests in this course you build from the formula rather than calling
> straight off the shelf.

:::{dropdown} Show full solution
```python
import numpy as np
from scipy import stats

def grubbs_test(data, alpha=0.05):
    n = len(data)
    mean, std = np.mean(data), np.std(data, ddof=1)
    abs_dev = np.abs(data - mean)
    idx = np.argmax(abs_dev)
    G = abs_dev[idx] / std
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = (n - 1) / np.sqrt(n) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    return G, G_crit, idx

G, G_crit, idx = grubbs_test(batch_A)
print(f'G = {G:.4f},  G_crit(alpha=0.05, n=40) = {G_crit:.4f}')
print(f'Flagged index: {idx}  (value {batch_A[idx]:.2f} MPa)')
print('Outlier detected:', G > G_crit)
```
Output: `G = 3.8497`, `G_crit = 3.0361` — since G > G_crit, index 15
(26.0 MPa) is flagged as a significant outlier at α=0.05. This is the same
point the Z-score and IQR methods in Section 6.4 already agreed on; Grubbs'
test is really a formalised version of the Z-score check, with a critical
value corrected for the fact that you're testing the *maximum* of 40 Z-scores
at once, not a single pre-chosen one.
:::

**Exercise 2 — Cleaned statistics**

> **Hint:** `np.delete(batch_A, 15)` drops the outlier by position, not by
> value — safer than filtering on `!= 26.0`, which would silently do nothing
> if a legitimate future measurement happened to also read 26.0.

:::{dropdown} Show full solution
```python
batch_A_clean = np.delete(batch_A, 15)

for name, data in [('Before (n=40)', batch_A), ('After (n=39)', batch_A_clean)]:
    mean, std = data.mean(), data.std(ddof=1)
    print(f'{name}: mean={mean:.3f} MPa, std={std:.3f} MPa, CV={std/mean*100:.2f}%')
```
Output: mean goes from 34.413 to 34.629 MPa (+0.22 MPa, a small shift),
while std drops from 2.185 to 1.729 MPa (a much bigger, −21% relative
change) and CV from 6.35% to 4.99%. This is the quantitative version of
Section 6.1/6.2's message: a single bad point barely moves the *centre*
(mean) but inflates the *spread* (std, CV) far more — spread statistics are
generally more outlier-sensitive than location statistics, for the same
reason variance uses squared deviations.
:::

**Exercise 3 — Bootstrapped confidence interval**

> **Hint:** Resample `data` **with replacement** (`rng.choice(data, size=n,
> replace=True)`) `n_boot` times, compute the mean of each resample, and take
> the 2.5th/97.5th percentiles of that distribution of means — the same
> percentile-bootstrap idea Section 11.7 applies later to a median.

:::{dropdown} Show full solution
```python
def bootstrap_mean_ci(data, n_boot=2000, ci=0.95, seed=0):
    boot_rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.array([boot_rng.choice(data, size=n, replace=True).mean()
                            for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return lo, hi

for name, data in [('Batch A', batch_A), ('Batch B', batch_B)]:
    lo, hi = bootstrap_mean_ci(data, seed=1)
    t_lo, t_hi = stats.t.interval(0.95, df=len(data) - 1,
                                   loc=data.mean(), scale=stats.sem(data))
    print(f'{name}: bootstrap CI=({lo:.3f}, {hi:.3f}),  '
          f'analytical t-CI=({t_lo:.3f}, {t_hi:.3f})')
```
Output: Batch A — bootstrap (33.68, 35.02) vs. analytical (33.71, 35.11);
Batch B — bootstrap (35.82, 36.62) vs. analytical (35.80, 36.64). The two
methods agree closely for both batches, including Batch A with its
outlier — reassuring, since the bootstrap makes no distributional
assumption at all while the analytical `t.interval` formula assumes
(approximate) normality. Close agreement here is itself evidence the
normality assumption was not badly wrong for the mean, even with one
contaminated point.
:::

## Notebook 7: Probability Distributions

**Exercise 1 — Log-transformation**

> **Hint:** `np.log(particle_size)` (natural log) is enough — you don't need
> `log10` here, since Shapiro-Wilk and the Q-Q plot only care about the
> *shape* of the distribution, and a log-normal variable's natural log is
> exactly normal by definition, regardless of log base.

:::{dropdown} Show full solution
```python
log_particle = np.log(particle_size)
W, p = stats.shapiro(log_particle)
print(f'Log-transformed particle size: W={W:.4f}, p={p:.4f}')

(osm, osr), (slope, intercept, r) = stats.probplot(log_particle, dist='norm')
print(f'Q-Q r² = {r**2:.4f}')
```
Output: `W=0.9771, p=0.7444` (comfortably normal now, vs. `p=0.0003` for the
untransformed data) with a Q-Q r² of 0.974. This is the expected result —
`particle_size` was generated with `rng.lognormal`, i.e. *by construction*
`log(particle_size)` is normal — but it's also the general-purpose fix for
right-skewed, strictly-positive materials data (particle sizes, grain
sizes, concentrations): log-transform first, then apply the same
normal-theory tests (t-tests, ANOVA) to the transformed variable.
:::

**Exercise 2 — Probability calculation**

> **Hint:** Both parts are direct reads off `stats.norm(mu, sigma)`:
> `.cdf(x)` gives $P(X<x)$, and `.ppf(q)` is the *inverse* CDF — the value
> below which a given fraction `q` of the distribution falls.

:::{dropdown} Show full solution
```python
dist = stats.norm(loc=3.85, scale=0.04)
p_below_spec = dist.cdf(3.78)
p99 = dist.ppf(0.99)
print(f'P(density < 3.78 g/cm3) = {p_below_spec:.4f}  ({p_below_spec*100:.2f}%)')
print(f'99th percentile density = {p99:.4f} g/cm3')
```
Output: `P(density < 3.78) = 0.0401` (4.0% of parts fall below spec) and the
99th percentile sits at `3.9431 g/cm3`. Both numbers come from the same
`N(3.85, 0.04²)` model — exactly the point Section 7.1's take-home message
makes with the grain-size example: once μ and σ are fixed, every
probability question is just a different read-off of the same curve.
:::

**Exercise 3 — Central Limit Theorem demonstration**

> **Hint:** Draw one big `(10000, n)` array with `rng.uniform(0, 1, size=(10000, n))`
> and average along `axis=1` — far faster than a Python loop over 10,000
> individual samples, and it gives you 10,000 sample means to histogram in
> one line.

:::{dropdown} Show full solution
```python
import matplotlib.pyplot as plt

clt_rng = np.random.default_rng(123)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

for ax, n_ in zip(axes, [1, 5, 30]):
    samples = clt_rng.uniform(0, 1, size=(10_000, n_))
    means = samples.mean(axis=1)
    ax.hist(means, bins=40, density=True, color='steelblue', alpha=0.7)
    ax.set_title(f'n = {n_}')
    ax.set_xlabel('Sample mean')
    print(f'n={n_}: mean={means.mean():.4f}, std={means.std():.4f} '
          f'(theory: 1/sqrt(12n) = {1/np.sqrt(12*n_):.4f})')

plt.tight_layout()
plt.show()
```
Output: the standard deviation of the sample means shrinks exactly as
theory predicts ($\sigma/\sqrt n$, with $\sigma=1/\sqrt{12}$ for a
Uniform(0,1)) — 0.285 at n=1, 0.128 at n=5, 0.053 at n=30 — while the
*shape* visibly morphs from the flat uniform histogram at n=1 into a
recognisable bell curve by n=30, even though the original distribution has
no bell shape at all. That's the Central Limit Theorem in one figure: the
sampling distribution of the mean approaches normal regardless of the
shape you started from.
:::

## Notebook 8: Hypothesis Testing

**Exercise 1 — One-tailed test**

> **Hint:** `stats.ttest_1samp` always returns the two-sided p-value —
> Section 11.2 (Live Tutorial 1) shows the halving rule explicitly:
> `p_one_sided = p_two_sided/2` only when the t-statistic actually points
> the direction H1 claims, otherwise `1 - p_two_sided/2`.

:::{dropdown} Show full solution
```python
t_stat, p_two_sided = stats.ttest_1samp(hardness, popmean=mu_spec)
# H1: mu > 200, so a positive t_stat is "pointing the right way"
p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
print(f't = {t_stat:.4f}')
print(f'Two-sided p = {p_two_sided:.4f}')
print(f'One-sided p (H1: mu > 200) = {p_one_sided:.4f}')
```
Output: `t = 1.7608`, two-sided `p = 0.1001`, one-sided `p = 0.0500`. Halving
the two-sided p-value makes the one-sided test more sensitive in the
direction it's aimed at — here it happens to land right at the α=0.05
boundary (0.050042, technically still a fail-to-reject), illustrating why a
one-sided test should only be chosen *before* looking at the data, for a
genuine physical reason (as Section 11.2 does with "nobody rejects a batch
for being too strong") — never after peeking at a two-sided result to see
which one-sided framing would cross the line.
:::

**Exercise 2 — Simulation study**

> **Hint:** Generate both groups from the *same* distribution inside the
> loop (`rng.normal(207, 12, 15)` twice) so H0 is true by construction, run
> `stats.ttest_ind`, and count how often `p < 0.05` purely from sampling
> noise — this is an empirical demonstration of what α actually means.

:::{dropdown} Show full solution
```python
sim_rng = np.random.default_rng(2024)
n_sims, rejects = 1000, 0
for _ in range(n_sims):
    g1 = sim_rng.normal(207, 12, 15)
    g2 = sim_rng.normal(207, 12, 15)   # same population -> H0 is exactly true
    _, p = stats.ttest_ind(g1, g2)
    if p < 0.05:
        rejects += 1
print(f'False-positive rate: {rejects}/{n_sims} = {rejects/n_sims:.3f}')
```
Output: `56/1000 = 5.6%` — close to the nominal α=0.05, as it should be:
the Type I error rate is *defined* as the long-run false-positive rate
when H0 is true, and this simulation is that definition made concrete
rather than taken on faith. A different random seed would give a slightly
different count (5.6% vs. the "true" 5.0% is itself just sampling noise on
a proportion from 1000 trials), but it will always hover close to 5%.
:::

**Exercise 3 — Sample size for tight tolerance**

> **Hint:** This is a **one-sample** test (comparing the new process's mean
> to a single target, 3.85 g/cm³), so use `TTestPower` (not
> `TTestIndPower`, which is for two independent groups) with Cohen's
> $d = \delta/\sigma = 0.02/0.025$.

:::{dropdown} Show full solution
```python
from statsmodels.stats.power import TTestPower

effect_size = 0.02 / 0.025   # d = 0.8
analysis = TTestPower()
n_required = analysis.solve_power(effect_size=effect_size, alpha=0.01,
                                   power=0.90, alternative='two-sided')
print(f"Cohen's d = {effect_size:.2f}")
print(f'Required n (alpha=0.01, power=0.90): {np.ceil(n_required):.0f}')
```
Output: `d = 0.80`, requiring **n = 27** samples. Compare this to Section
8.6's two-sample example, which needed only 24 per group for a similar
effect size (d≈0.83) but a looser α=0.05/power=0.80 target — tightening
both α (0.05→0.01) and the power target (0.80→0.90) simultaneously here
pushes the requirement up, even for a comparably sized effect, which is
exactly why stricter acceptance criteria (common for safety-critical specs)
translate directly into more expensive testing programmes.
:::

## Notebook 9: Analysis of Variance (ANOVA)

**Exercise 1 — Kruskal-Wallis test**

> **Hint:** `stats.kruskal(*groups)` takes the same star-unpacked group
> arrays as `stats.f_oneway` did in Section 9.1 — same calling convention,
> different underlying test (ranks instead of means).

:::{dropdown} Show full solution
```python
H, p_kw = stats.kruskal(*[groups[t] for t in T_labels])
F, p_anova = stats.f_oneway(*[groups[t] for t in T_labels])
print(f'Kruskal-Wallis: H={H:.4f}, p={p_kw:.6f}')
print(f'One-way ANOVA:  F={F:.4f}, p={p_anova:.6f}  (for comparison)')
```
Output: `H=24.87, p=0.000016` vs. ANOVA's `F=40.52, p≈1e-10` — both reject
H0 overwhelmingly, reaching the identical conclusion despite Kruskal-Wallis
making no normality assumption. This agreement is expected here, since
Section 9.3 already confirmed the ANOVA's normality assumption holds
(Shapiro-Wilk on residuals, p=0.40) — Kruskal-Wallis is the tool to reach
for when that check *fails*, not a replacement to run routinely once
normality is already confirmed.
:::

**Exercise 2 — Effect size (η²)**

> **Hint:** Both `SS_between` and `SS_total` are already sitting in the
> `anova_table` built manually in Section 9.1 — no new computation needed,
> just divide.

:::{dropdown} Show full solution
```python
eta_sq = SS_between / SS_total
print(f'SS_between = {SS_between:.1f}, SS_total = {SS_total:.1f}')
print(f'eta-squared = {eta_sq:.4f}')
```
Output: `eta² = 0.813` — sintering temperature alone explains about 81% of
the total variance in flexural strength, leaving 19% as within-group
scatter. By the conventional benchmarks (η²≈0.01 small, 0.06 medium, 0.14
large), this is a very large effect, consistent with the F-statistic of
~40 already flagged in Section 9.1's take-home message as "the between-group
variability is about 40 times the within-group (noise) variability."
:::

**Exercise 3 — Unbalanced design**

> **Hint:** `np.append` (or drawing two fresh values with `rng.normal`) onto
> just the 1400°C group's array is enough to make the design unbalanced;
> rebuild the long-form DataFrame and rerun `stats.f_oneway` and
> `sm.stats.anova_lm(..., typ=2)` / `typ=3` on the new data.

:::{dropdown} Show full solution
```python
extra = rng.normal(340, sigma, 2)          # 2 more 1400°C measurements
groups_unbal = dict(groups)
groups_unbal['1400°C'] = np.concatenate([groups['1400°C'], extra])

rows_unbal = []
for label, vals in groups_unbal.items():
    rows_unbal.extend({'Temperature': label, 'Strength_MPa': v} for v in vals)
df_unbal = pd.DataFrame(rows_unbal)

F_unbal, p_unbal = stats.f_oneway(*[groups_unbal[t] for t in T_labels])
print(f'Unbalanced ANOVA: F={F_unbal:.4f}, p={p_unbal:.6f}  '
      f'(balanced was F={F:.4f})')

model_unbal = ols('Strength_MPa ~ C(Temperature)', data=df_unbal).fit()
print('\nType II SS:\n', sm.stats.anova_lm(model_unbal, typ=2))
print('\nType III SS:\n', sm.stats.anova_lm(model_unbal, typ=3))
```
Output: `F=47.42` (up from 40.52 — extra data at an already-strong group
sharpens the signal slightly), still `p≈0`. As the exercise text anticipates,
`C(Temperature)` gets the *exact same* sum of squares (81146.1) whichever
`typ=` you request, because with only one factor there's no ambiguity about
"what to adjust for" — Type I/II/III SS only start to disagree once a model
has more than one factor and the design is unbalanced *across their
combinations*, which is exactly Section 9.4's two-way case, not this one.
:::

## Notebook 10: Simple Linear Regression

**Exercise 1 — Calibration curve**

> **Hint:** `-1 + conc_mM` in the formula string removes the intercept term
> — that's what "regression through the origin" means in `statsmodels`
> formula syntax, forcing the fitted line through (0, 0).

:::{dropdown} Show full solution
```python
conc_mM = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
absorbance = np.array([0.005, 0.098, 0.191, 0.287, 0.384, 0.479])
df_cal = pd.DataFrame({'conc_mM': conc_mM, 'absorbance': absorbance})

model_cal = smf.ols('absorbance ~ -1 + conc_mM', data=df_cal).fit()
sensitivity = model_cal.params['conc_mM']
conc_unknown = 0.245 / sensitivity

print(f'Sensitivity (slope): {sensitivity:.4f} absorbance / mM')
print(f'R² = {model_cal.rsquared:.4f}')
print(f'Unknown concentration at absorbance=0.245: {conc_unknown:.4f} mM')
```
Output: sensitivity `0.4793` absorbance units per mM, `R²=0.9999` (an
excellent calibration), and an unknown at absorbance 0.245 corresponds to
`0.5112 mM`. Forcing the fit through the origin is standard practice for a
calibration curve where "zero analyte → zero signal" is physically
required — letting the intercept float instead would let noise near
`conc_mM=0` (the point 0.005 instead of exactly 0.000) pull the whole line
off true zero.
:::

**Exercise 2 — Transformation**

> **Hint:** Convert the *fitted line*, not the raw data: evaluate
> $\widehat{\log_{10}\sigma} = \beta_0+\beta_1 x$ on a grid of x, then
> `10**` the result to get conductivity back in S/cm — plotting that on a
> log y-axis (`ax.set_yscale('log')`) should look like a straight line
> again, since you're just undoing the log you started with.

:::{dropdown} Show full solution
```python
x_pred = np.linspace(0, 8, 50)
log_sigma_pred = model.params['Intercept'] + model.params['Nb_at_pct'] * x_pred
sigma_pred = 10 ** log_sigma_pred

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df['Nb_at_pct'], 10 ** df['log10_sigma'], color='steelblue', label='Data')
ax.plot(x_pred, sigma_pred, 'r-', lw=2, label='Fitted (back-transformed)')
ax.set_yscale('log')
ax.set_xlabel('Nb content (at%)')
ax.set_ylabel('Conductivity σ (S/cm)')
ax.legend()
plt.tight_layout()
plt.show()

print(f'sigma at 0 at%: {10**model.params["Intercept"]:.3e} S/cm')
print(f'sigma at 8 at%: {sigma_pred[-1]:.3e} S/cm')
print(f'Ratio: {sigma_pred[-1] / 10**model.params["Intercept"]:.0f}x')
```
Output: conductivity rises from `1.11e-4 S/cm` at 0 at% Nb to `2.63 S/cm`
at 8 at% — roughly a **23,700-fold** increase, matching
$10^{0.547\times8}\approx23{,}745$ almost exactly, since that's precisely
what a constant slope on a log10 scale means. On the log y-axis the
back-transformed fit is a straight line (log scale undoes the
transformation), while the same fit on a *linear* y-axis would look like a
sharply upward-curving exponential — the same information, two very
different-looking plots.
:::

**Exercise 3 — Outlier effect**

> **Hint:** `model.get_influence().cooks_distance[0]` returns Cook's
> distance for every observation in the fit — compare its value at index 5
> before and after the edit, not just its rank among the other points.

:::{dropdown} Show full solution
```python
df_outlier = df.copy()
df_outlier.loc[5, 'log10_sigma'] = -0.5
model_outlier = smf.ols('log10_sigma ~ Nb_at_pct', data=df_outlier).fit()

cooks_before = model.get_influence().cooks_distance[0]
cooks_after = model_outlier.get_influence().cooks_distance[0]

print(f'Slope:  {model.params["Nb_at_pct"]:.4f} -> {model_outlier.params["Nb_at_pct"]:.4f}')
print(f'R²:     {model.rsquared:.4f} -> {model_outlier.rsquared:.4f}')
print(f"Cook's D at index 5: {cooks_before[5]:.5f} -> {cooks_after[5]:.5f}")
```
Output: the slope only drifts from 0.547 to 0.523, but R² collapses from
0.988 to 0.831, and Cook's distance at the edited point jumps from a
negligible 0.0002 to 0.482 — by far the largest influence value in the
edited dataset. This is a useful contrast with Exercise 1's calibration
curve: R² (a *goodness-of-fit* measure) is far more sensitive to one bad
point than the slope (a *parameter estimate*) is, because a single point
near the middle of the x-range mostly just adds unexplained scatter rather
than dragging the whole line toward it — an edge point (like index 1 in the
original data, which already has the highest Cook's D at 0.925) would move
the slope much more.
:::

## Notebook 11: Live Tutorial 1 — t-Tests in Practice

**Exercise 1 — One-sided vs two-sided**

> **Hint:** `stats.ttest_1samp(supplier_A, popmean=spec_min)` gives the
> two-sided p-value directly — compare it to Section 11.2's manually
> halved one-sided value, remembering that Section 11.2's H1 was
> `mu < 275`, the *opposite* direction from where Supplier A's sample mean
> actually landed.

:::{dropdown} Show full solution
```python
t_stat, p_two_sided = stats.ttest_1samp(supplier_A, popmean=spec_min)
print(f't = {t_stat:.4f},  two-sided p = {p_two_sided:.4f}')
print(f'(Section 11.2 one-sided p, H1: mu < 275) = 0.9967')
```
Output: two-sided `p=0.0065` — much *smaller* than the one-sided p-value of
0.9967 computed in Section 11.2. This looks backwards at first (one-sided
tests are usually more powerful, i.e. smaller p, in the direction they're
aimed at) until you notice *why*: Supplier A's sample mean (≈281 MPa) sits
well *above* 275, but Section 11.2's H1 specifically claims the mean is
*below* 275. A one-sided test pointed the wrong way essentially cannot
reject, no matter how extreme the data — almost the entire t-distribution's
mass counts as "not evidence for H1" in that case, which is exactly what
inflates that one-sided p-value toward 1. This is the sharpest possible
illustration of why the direction of a one-sided test must be chosen for a
real physical reason *before* seeing the data, never after.
:::

**Exercise 2 — A third supplier**

> **Hint:** Follow Sections 11.3–11.4's exact four-step recipe in order —
> Levene's test decides `equal_var` for `ttest_ind`, then Cohen's *d* and
> the Welch-adjusted CI use the same formulas already defined earlier in
> the notebook (`cohens_d`, `se_diff`, `df_welch`).

:::{dropdown} Show full solution
```python
supplier_C = rng.normal(285, 10, 20)

levene_stat, levene_p = stats.levene(supplier_A, supplier_C)
equal_var = levene_p > 0.05
t_stat, p_val = stats.ttest_ind(supplier_A, supplier_C, equal_var=equal_var)
d = cohens_d(supplier_C, supplier_A)

diff = supplier_C.mean() - supplier_A.mean()
se_diff = np.sqrt(supplier_A.var(ddof=1)/len(supplier_A) + supplier_C.var(ddof=1)/len(supplier_C))
df_w = se_diff**4 / (
    (supplier_A.var(ddof=1)/len(supplier_A))**2/(len(supplier_A)-1) +
    (supplier_C.var(ddof=1)/len(supplier_C))**2/(len(supplier_C)-1))
t_crit = stats.t.ppf(0.975, df_w)
ci = (diff - t_crit*se_diff, diff + t_crit*se_diff)

print(f'Levene p={levene_p:.3f} -> {"Student" if equal_var else "Welch"} t-test')
print(f't={t_stat:.3f}, p={p_val:.4f}')
print(f"Cohen's d = {d:.3f}")
print(f'Mean diff (C-A) = {diff:.2f} MPa, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}] MPa')
```
Output: Levene passes (`p=0.227`, equal variances, Student's t-test used),
`t=-1.08, p=0.288` — **not significant** — with Cohen's `d=0.35` (a small
effect) and a 95% CI of `[-3.21, 10.80]` MPa that comfortably includes
zero. **Recommendation: don't switch on this evidence.** Although Supplier
C's sample mean (≈285 MPa) numerically beats both A and B, this pilot's
`n=20` isn't enough to distinguish it from A with any confidence — contrast
this with Section 11.3's A-vs-B comparison, where a larger observed effect
(d≈1.2 there) *was* detectable at similar sample sizes. The lesson from
Section 11.8 applies directly: a formal power analysis, not eyeballing the
means, should decide whether more castings are worth testing before
switching suppliers.
:::

**Exercise 3 — Wilcoxon signed-rank**

> **Hint:** Run `stats.shapiro` on the *differences* (`df_paired['improvement']`),
> not on `before`/`after` separately — a paired t-test's normality
> assumption is about the distribution of the paired differences, not the
> two raw measurement sets.

:::{dropdown} Show full solution
```python
W_sh, p_sh = stats.shapiro(df_paired['improvement'])
print(f'Shapiro-Wilk on improvement: W={W_sh:.4f}, p={p_sh:.4f}')

w_stat, p_wilcoxon = stats.wilcoxon(df_paired['after'], df_paired['before'])
print(f'Wilcoxon signed-rank: W={w_stat:.1f}, p={p_wilcoxon:.6f}')
print(f'(Paired t-test from Section 11.5: p={p_paired:.6f})')
```
Output: the differences pass normality comfortably (`p=0.646`), so the
paired t-test's assumption was fine all along — but it's worth running the
non-parametric alternative anyway as a sanity check. Wilcoxon gives
`p=0.000061`, agreeing with the paired t-test's `p=0.000002` on the
conclusion (both far below 0.05) even though the exact p-values differ, as
expected between a rank-based and a mean-based test. When a
normality-sensitive test and its non-parametric counterpart agree this
strongly, that's good evidence the result isn't an artefact of the
normality assumption.
:::

**Exercise 4 — Type I and Type II error, simulated**

> **Hint:** Two separate loops: one where both groups share the *same* true
> mean (H0 exactly true — count how often `p<0.05` anyway, that's your
> empirical α), and one where the second group's mean is shifted by +5 MPa
> (H1 exactly true — count how often `p<0.05` correctly, that's your
> empirical power).

:::{dropdown} Show full solution
```python
sim_rng = np.random.default_rng(99)
n_sims = 1000

type1_rejects = sum(
    stats.ttest_ind(sim_rng.normal(275, 10, 15), sim_rng.normal(275, 10, 15))[1] < 0.05
    for _ in range(n_sims))
power_rejects = sum(
    stats.ttest_ind(sim_rng.normal(275, 10, 15), sim_rng.normal(280, 10, 15))[1] < 0.05
    for _ in range(n_sims))

print(f'Empirical Type I error rate: {type1_rejects/n_sims:.3f}  (target: 0.05)')
print(f'Empirical power (true diff = 5 MPa): {power_rejects/n_sims:.3f}')
```
Output: Type I error rate `5.6%` (56/1000) — close to the nominal 5%, same
demonstration as Notebook 8, Exercise 2. Empirical power for a true 5 MPa
difference comes out at only `25.7%` (257/1000) — far below the 80% target
usually planned for. This isn't a bug: at `n=15` per group and `sigma=10`,
Cohen's `d = 5/10 = 0.5` (a medium effect), and Section 8.6's power
analysis already showed that detecting `d≈0.83` reliably needed `n=24` per
group — a smaller effect (`d=0.5`) at a smaller sample size (`n=15`) is
exactly the underpowered regime that analysis warned about.
:::

**Exercise 5 — Bootstrap the mean, not just the median**

> **Hint:** Reuse `bootstrap_ci` from Section 11.7 exactly as written, just
> pass `stat_func=np.mean` instead of the default `np.median` — no other
> changes needed.

:::{dropdown} Show full solution
```python
lo, hi, _ = bootstrap_ci(supplier_A, stat_func=np.mean, seed=0)
t_lo, t_hi = stats.t.interval(0.95, df=len(supplier_A)-1,
                               loc=supplier_A.mean(), scale=stats.sem(supplier_A))
print(f'Bootstrap CI on the mean: [{lo:.2f}, {hi:.2f}] MPa')
print(f'Classical t-interval:    [{t_lo:.2f}, {t_hi:.2f}] MPa')
```
Output: bootstrap `[277.39, 284.86]` vs. classical `[276.95, 285.29]` MPa —
close, with the bootstrap interval very slightly narrower. This is the
expected result: `supplier_A` is a reasonably well-behaved, roughly normal
sample (Section 11.2's Shapiro-Wilk check passed), so the assumption behind
the classical `t.interval` formula holds, and the assumption-free bootstrap
lands in essentially the same place. The two methods are worth running
side by side specifically as this kind of sanity check — a *large*
disagreement would be the signal to distrust the classical interval.
:::

**Exercise 6 — Multiple comparisons**

> **Hint:** $P(\text{at least one false positive}) = 1-(1-\alpha)^k$ — plug
> in $k=7$ for the seven tests run today (Sections 11.2, 11.3, 11.5, 11.6,
> plus Exercises 1–3).

:::{dropdown} Show full solution
```python
alpha, k = 0.05, 7
p_at_least_one = 1 - (1 - alpha)**k
print(f'P(at least one false positive across {k} tests) = {p_at_least_one:.4f}')
```
Output: `30.2%` — nearly a one-in-three chance that *something* today
looked "significant" purely from running seven separate tests at α=0.05
each, even if every null hypothesis were actually true. This is precisely
the problem Notebook 9 (Live Tutorial 2, Section 12.1) opens with when
motivating ANOVA over six separate pairwise t-tests for four groups — the
fix there (one omnibus F-test, then a family-wise-corrected post-hoc test
like Tukey HSD) generalises this notebook's individual, uncorrected
comparisons to the case of comparing several groups within *one* study.
:::

**Exercise 7 — Design your own QC scenario**

> **Hint:** Pick a *paired-vs-unpaired* or *one-sample-vs-target* structure
> like this notebook's, not a from-scratch design — the goal is practising
> the same five-step checklist (Section 11.1) on new numbers, not inventing
> a new kind of test.

:::{dropdown} Show full solution
```python
# Example: does a new quench-and-temper recipe raise 4140 steel hardness
# above the 28 HRC minimum spec?  (Modelled on Section 11.2's one-sample design.)
qc_rng = np.random.default_rng(1)
hardness_HRC = qc_rng.normal(loc=29.5, scale=1.8, size=20)
spec_min = 28.0

t_stat, p_two = stats.ttest_1samp(hardness_HRC, popmean=spec_min)
p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

pooled_std = hardness_HRC.std(ddof=1)
d = (hardness_HRC.mean() - spec_min) / pooled_std
ci = stats.t.interval(0.95, df=len(hardness_HRC)-1,
                       loc=hardness_HRC.mean(), scale=stats.sem(hardness_HRC))

print(f'Mean hardness: {hardness_HRC.mean():.2f} HRC (spec >= {spec_min})')
print(f't={t_stat:.3f}, one-sided p={p_one:.4f}')
print(f"Effect size vs spec: d={d:.2f}")
print(f'95% CI: [{ci[0]:.2f}, {ci[1]:.2f}] HRC')
print('Decision:', 'meets spec' if p_one < 0.05 and t_stat > 0 else 'inconclusive/fails')
```
This is a template, not a fixed answer — the point of Exercise 7 is
practising the full checklist from Section 11.1 (state H0/H1 → check
assumptions → compute statistic and p-value → compare to α → report effect
size and CI) on a scenario you construct yourself. Any of battery capacity,
ceramic density, or a before/after coating treatment (Notebooks 6–10) works
equally well; what matters is that the test you choose (one-sample,
two-sample, or paired) actually matches the question you asked.
:::

## Notebook 12: Live Tutorial 2 — ANOVA & Regression

**Exercise 1 — A fifth tool design**

> **Hint:** Draw the fifth group's data with `rng.normal(90, sigma, n_per_group)`
> (continuing the same `rng`, same `sigma=4.5`, same `n_per_group=10` as
> the original four), append it to `df_tools`, and rerun Section 12.2's
> boxplot → Levene/Shapiro → ANOVA → η² → Tukey pipeline unchanged.

:::{dropdown} Show full solution
```python
strengths_5 = rng.normal(90, sigma, n_per_group)
df_tools5 = pd.concat([df_tools, pd.DataFrame({
    'tool_design': 'Spiral', 'joint_efficiency_pct': strengths_5})], ignore_index=True)

groups5 = [df_tools5.loc[df_tools5.tool_design == d, 'joint_efficiency_pct']
           for d in tool_designs + ['Spiral']]
F5, p5 = stats.f_oneway(*groups5)
print(f'F={F5:.2f}, p={p5:.2e}')

tukey5 = pairwise_tukeyhsd(df_tools5['joint_efficiency_pct'], df_tools5['tool_design'], alpha=0.05)
print(tukey5)
```
Output: `F=19.12, p≈2e-10` — still highly significant with five groups.
The original four designs' pairwise picture is essentially unchanged:
Cylindrical remains distinguishable from all others (`reject=True`
throughout), and Flared-vs-Threaded / Flared-vs-Triflute / Threaded-vs-Triflute
stay non-significant, just as in Section 12.2 — adding a fifth,
unrelated group doesn't retroactively change what the *original* four
groups' data says about each other, though Tukey's p-adj values do shift
slightly because the family-wise correction now spans $\binom{5}{2}=10$
comparisons instead of 6.
:::

**Exercise 2 — Kruskal-Wallis**

> **Hint:** Same call as Notebook 9's Exercise 1 — `stats.kruskal(*groups)`
> on the same four group arrays already built in Section 12.2, `groups`.

:::{dropdown} Show full solution
```python
H, p_kw = stats.kruskal(*groups)
print(f'Kruskal-Wallis: H={H:.4f}, p={p_kw:.6f}')
print(f'(One-way ANOVA from Section 12.2: F={f_stat:.4f}, p={p_value:.6f})')
```
Output: `H=22.82, p=0.000044` vs. the ANOVA's `F=21.62, p≈1e-9` — same
conclusion (reject H0 decisively) by both routes. As in the parallel
exercise in Notebook 9, this agreement is expected precisely *because*
Section 12.2's own assumption checks (Levene, Shapiro-Wilk on residuals)
already passed — Kruskal-Wallis is the fallback for when they don't, used
here as a confirmatory cross-check rather than out of necessity.
:::

**Exercise 3 — Verify the RCBD bridge**

> **Hint:** Just drop the `C(...)` wrapper around `rotation_speed` in the
> formula string — `rotation_speed` (categorical, 3 fixed settings) becomes
> `rotation_speed` (continuous, assumes a straight-line trend across
> 900/1200/1500 rpm) while `C(coil)` stays categorical either way, since
> "Coil-1/2/3/4" has no numeric ordering.

:::{dropdown} Show full solution
```python
model_cont = smf.ols('strength_MPa ~ rotation_speed + C(coil)', data=df_block).fit()
anova_cont = sm.stats.anova_lm(model_cont, typ=2)
print('Continuous rotation_speed:\n', anova_cont)
print(f'\nR² = {model_cont.rsquared:.4f}, df_resid = {model_cont.df_resid:.0f}')
print(f'(Categorical model from Section 12.3: R² = {model_block.rsquared:.4f}, '
      f'df_resid = {model_block.df_resid:.0f})')
```
Output: the continuous model gives `R²=0.947` (`df_resid=7`, using only 1
degree of freedom for `rotation_speed` instead of 2) against the
categorical model's `R²=0.974` (`df_resid=6`). Both still find
`rotation_speed`/`C(rotation_speed)` highly significant (`p=0.00003` vs.
`p=0.00005`). The categorical version fits slightly better because it can
match each of the three speed means exactly, with no constraint that they
lie on a straight line; the continuous version spends one fewer degree of
freedom and *assumes* a linear trend, which pays off with more power **if**
that trend really is linear (worth checking against a scatter plot),
buys you a single interpretable "MPa per rpm" slope instead of three
separate group means, and lets you predict strength at *speeds you didn't
test* — something the categorical version cannot do.
:::

**Exercise 4 — Forward selection**

> **Hint:** Mirror `backward_elimination`'s structure exactly, but flip the
> direction: start `selected = []`, and at each step try adding *each*
> remaining candidate to `selected`, keep whichever addition has the
> smallest p-value, and stop as soon as the best remaining candidate
> wouldn't be significant if added.

:::{dropdown} Show full solution
```python
def forward_selection(data, response, candidate_terms, alpha=0.05, verbose=True):
    selected, remaining = [], list(candidate_terms)
    while remaining:
        best_term, best_p = None, None
        for term in remaining:
            formula = f'{response} ~ ' + ' + '.join(selected + [term])
            fit = smf.ols(formula, data=data).fit()
            p = fit.pvalues[term]
            if best_p is None or p < best_p:
                best_term, best_p = term, p
        if best_p > alpha:
            break
        selected.append(best_term)
        remaining.remove(best_term)
        if verbose:
            print(f'Adding {best_term!r} (p={best_p:.4f})')
    formula = f'{response} ~ ' + ' + '.join(selected) if selected else f'{response} ~ 1'
    return smf.ols(formula, data=data).fit(), selected

model_fwd, fwd_terms = forward_selection(df_reg, 'strength_MPa', candidates)
print('\nForward selection kept:', fwd_terms)
print('Backward elimination kept:', kept_terms)
print(f'Forward:  R²_adj={model_fwd.rsquared_adj:.4f}, AIC={model_fwd.aic:.1f}')
print(f'Backward: R²_adj={model_reduced.rsquared_adj:.4f}, AIC={model_reduced.aic:.1f}')
```
Output: forward selection adds terms in the order `rotation_rpm →
traverse_mm_min → axial_force_kN → rotation_rpm:traverse_mm_min →
plate_thickness_mm`, and — for this particular dataset — converges on
*exactly* the same five terms backward elimination kept
(`R²_adj=0.929, AIC=259.5` for both), correctly leaving out
`ambient_humidity_pct` either way. That agreement is not guaranteed in
general — forward and backward selection can disagree, particularly when
candidate predictors are correlated with each other, since forward
selection never reconsiders a term once several others have already been
added around it, while backward elimination starts from the full picture
and only removes. Seeing them agree here is a mild reassurance that the
five real effects built into `strength_true` in Section 12.5 are each
individually strong enough that the selection *path* doesn't matter.
:::

**Exercise 5 — Interaction plot**

> **Hint:** `sns.catplot(data=df_3way, kind='point', x='rotation', y='strength_MPa', hue='tool', col='traverse')`
> produces two side-by-side panels (one per `traverse` level), each with
> two lines (one per `tool`) — look for the lines having visibly different
> *slopes* within each panel, not just different heights.

:::{dropdown} Show full solution
```python
g = sns.catplot(data=df_3way, kind='point', x='rotation', y='strength_MPa',
                 hue='tool', col='traverse', order=['Low', 'High'],
                 palette='colorblind', height=4, aspect=0.9)
g.set_axis_labels('Rotation speed', 'Strength (MPa)')
g.set_titles('Traverse: {col_name}')
plt.tight_layout()
plt.show()
```
In both facets, tool B's line rises more steeply from Low to High rotation
than tool A's does (e.g. at Traverse=Low: A goes from ≈81.1 to ≈85.2 MPa,
+4.1, while B goes from ≈81.8 to ≈89.6 MPa, +7.8) — visually, the two lines
fan apart going left to right rather than staying parallel. That fanning
*is* the rotation×tool interaction the ANOVA table in Section 12.7 detected
at `p=0.046`: the effect of rotation speed depends on which tool you're
using, and it looks essentially the same in both the Traverse=Low and
Traverse=High panels — consistent with the three-way interaction term
*not* being significant in that same table.
:::

**Exercise 6 — Full ANCOVA with interaction**

> **Hint:** `C(tool_design) * rotation_speed` in the formula (as opposed to
> `+`) automatically adds the interaction term alongside both main effects
> — but centre `rotation_speed` first (subtract its mean) the way Section
> 12.5 recommends for interaction terms, or the main-effect p-values become
> misleading for a reason that has nothing to do with the interaction
> itself.

:::{dropdown} Show full solution
```python
df_ancova['speed_c'] = df_ancova['rotation_speed'] - df_ancova['rotation_speed'].mean()
model_int = smf.ols('strength_MPa ~ C(tool_design) * speed_c', data=df_ancova).fit()
print(model_int.pvalues)
```
Output: the interaction term `C(tool_design)[T.Triflute]:speed_c` has
`p=0.985` — nowhere close to significant, so the two tools' strength really
does respond to rotation speed with the same slope, and Section 12.6's
simpler additive model (no interaction) was the right call. With speed
centred, the tool main effect stays clearly significant (`p=0.0004`),
matching Section 12.6's finding almost exactly. **The centring step
matters**: fitting the identical interaction model on *uncentred*
`rotation_speed` instead gives the same interaction p-value (0.985, the
interaction itself is unaffected by centring) but an apparently
non-significant tool main effect (`p=0.597`) — an artefact of severe
multicollinearity between `C(tool_design)` and the uncentred interaction
term (condition number ≈31,000 vs. ≈465 once centred), not evidence the
tool effect vanished. This is precisely the caution flagged in Section
12.5's take-home message about Cond. No. warnings, now showing up as a
concrete wrong-conclusion risk rather than an abstract one.
:::

**Exercise 7 — Design your own three-way study**

> **Hint:** Copy Section 12.7's nested-loop generation pattern directly —
> pick a `base` value, add a fixed bump for each factor's "high" level, add
> one extra bump only when two specific factor levels co-occur (that's your
> deliberate two-way interaction), and leave a third main effect with no
> such bonus anywhere (that's your interaction-free effect).

:::{dropdown} Show full solution
```python
# Example: sintering temperature x atmosphere x powder particle size,
# 2x2x2, n=4 replicates/cell, modelled on Section 12.7's structure.
levels_ex = {'temperature': ['Low', 'High'], 'atmosphere': ['Air', 'N2'], 'particle': ['Coarse', 'Fine']}
rows_ex = []
for T in levels_ex['temperature']:
    for atm in levels_ex['atmosphere']:
        for part in levels_ex['particle']:
            base = 90 + (6 if T == 'High' else 0) + (3 if part == 'Fine' else 0)
            base += 5 if (T == 'High' and atm == 'N2') else 0   # genuine T x atmosphere interaction
            # 'particle' deliberately has NO interaction with anything else
            for _ in range(4):
                rows_ex.append({'temperature': T, 'atmosphere': atm, 'particle': part,
                                 'density_pct': base + rng.normal(0, 1.5)})
df_ex = pd.DataFrame(rows_ex)

model_ex = smf.ols('density_pct ~ C(temperature)*C(atmosphere)*C(particle)', data=df_ex).fit()
print(sm.stats.anova_lm(model_ex, typ=2).round(4))
```
The ANOVA table should recover all three main effects as significant
(temperature, atmosphere, particle size each move mean density on their
own) plus the `C(temperature):C(atmosphere)` interaction — while every term
involving `particle` beyond its own main effect (its two-way interactions
with temperature and atmosphere, and the three-way term) should come back
non-significant, since none of those were built into `base`. As Section
12.7's own take-home message notes, this "does the table find what I put
in" check is exactly how you build confidence in the ANOVA machinery before
trusting it on a real dataset where you don't already know the answer.
:::
