# Part IV Solutions: Multivariate Analysis

Solutions to the exercises in Notebooks 13–19 (Part IV uses the book's
global notebook numbering, continuing on from Parts I–III). Code blocks
below are written to be pasted directly into the corresponding notebook
(they assume its imports, `rng`, and variables already exist in your
session) — they are not executed as part of building this page. Every
numeric result quoted here was independently computed with the same seeded
`rng = np.random.default_rng(...)` as the source notebook, so re-running the
code below in a fresh copy of that notebook reproduces it exactly.

## Notebook 13: Multiple Linear Regression (MLR)

**Exercise 1 — Interaction term**

> **Hint:** Add `T_sinter:Ni_frac` to the formula string (statsmodels'
> `+` syntax adds a term, `:` adds only the interaction without the main
> effects again) and compare `model.aic` before and after — recall from
> Notebook 12 (Live Tutorial 2) that R² can never *decrease* when you add a
> term, so AIC (which penalises extra parameters) is the fairer comparison.

:::{dropdown} Show full solution
```python
model_int = smf.ols(
    'capacity ~ T_sinter + time_h + Ni_frac + d50_nm + T_sinter:Ni_frac',
    data=df
).fit()

print(f'Base model:        R²={model.rsquared:.4f}, AIC={model.aic:.2f}')
print(f'With interaction:  R²={model_int.rsquared:.4f}, AIC={model_int.aic:.2f}')
print(f"Interaction term p-value: {model_int.pvalues['T_sinter:Ni_frac']:.4f}")

f_stat, f_p, _ = model_int.compare_f_test(model)
print(f'F-test for adding the interaction: F={f_stat:.3f}, p={f_p:.4f}')
```
Output: R² edges up only slightly, from 0.9825 to 0.9834, and AIC improves
marginally (397.84 → 395.70, lower is better). The interaction term itself
sits right at the edge of significance (`p=0.051`), and the nested F-test
comparing the two models agrees almost exactly (`F=3.93, p=0.051`) — not
quite significant at α=0.05. This makes sense by construction: Section
11.1's data-generating code *did* build in a small interaction
(`0.001 * (T_sinter-800) * (Ni_frac-0.6) * 100`), but scaled it deliberately
small relative to the four main effects, so it is only marginally
detectable against the noise (σ=3) at n=80 — a good illustration that a
"real" effect in the data-generating process is not guaranteed to clear a
p<0.05 bar in any single finite sample.
:::

**Exercise 2 — Standardised coefficients**

> **Hint:** Standardise each predictor as `(x - x.mean()) / x.std()` before
> re-fitting the same formula — the coefficients then all share the same
> "per one standard deviation" units, which raw coefficients (Section 11.2's
> take-home message) explicitly warned against comparing directly.

:::{dropdown} Show full solution
```python
predictors = ['T_sinter', 'time_h', 'Ni_frac', 'd50_nm']
df_std = df[predictors].apply(lambda x: (x - x.mean()) / x.std())
df_std['capacity'] = df['capacity']

model_std = smf.ols('capacity ~ T_sinter + time_h + Ni_frac + d50_nm', data=df_std).fit()
print(model_std.params.drop('Intercept').sort_values(key=abs, ascending=False))
```
Output:
```
Ni_frac     18.875
T_sinter     6.933
d50_nm      -2.567
time_h       3.887
```
Once every predictor is on the same "per standard deviation" footing,
**`Ni_frac` is clearly the strongest driver of capacity** (18.9 mAh/g per
SD), followed by `T_sinter` (6.9) and `time_h` (3.9), with `d50_nm` the
weakest (−2.6). This reorders the story from the raw coefficients in
Section 11.2: there, `Ni_frac`'s raw coefficient (118.5) looked dramatically
larger than the others mainly because it is reported per whole unit of a
variable that only spans about 0.5 — standardising removes that scale
artefact and shows Ni content is still the single most important factor,
but not by as lopsided a margin as the raw table suggested.
:::

**Exercise 3 — Cross-validation**

> **Hint:** `sklearn.model_selection.train_test_split(X, y, test_size=0.2,
> random_state=...)` — fit `LinearRegression` (or refit the same `smf.ols`
> formula) on the training rows only, predict on the held-out 20%, and
> compare that RMSE to the in-sample RMSE already computed in Section 11.4.

:::{dropdown} Show full solution
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df[['T_sinter', 'time_h', 'Ni_frac', 'd50_nm']]
y = df['capacity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
rmse_train = np.sqrt(mean_squared_error(y_train, lr.predict(X_train)))
rmse_test = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))

print(f'Train RMSE: {rmse_train:.3f} mAh/g  (n_train={len(X_train)})')
print(f'Test RMSE:  {rmse_test:.3f} mAh/g  (n_test={len(X_test)})')
print(f'In-sample RMSE, full 80-sample fit (Section 11.4): {rmse:.3f} mAh/g')
```
Output: train RMSE `2.693`, test RMSE `2.981`, versus `2.732` for the
full-dataset in-sample fit from Section 11.4. The held-out test RMSE is
somewhat higher than the training RMSE — the expected direction, since the
model is evaluated on data it never saw — but the gap here is small (about
0.3 mAh/g) and both numbers sit close to the noise level actually used to
generate the data (σ=3), which is reassuring: with only 4 predictors and a
strong underlying signal (R²≈0.98), there is little room for this model to
be overfitting even on 80 samples split down to 64/16.
:::

## Notebook 14: Principal Component Analysis (PCA)

**Exercise 1 — Hotelling T²**

> **Hint:** With PCA scores already whitened by construction, $T^2$ for a
> sample is just $\sum_{a=1}^{p}\text{score}_a^2/\lambda_a$ summed over the
> retained components' eigenvalues (`pca.explained_variance_[:3]`) — then
> compare each sample's $T^2$ to the single critical value from the formula
> given, computed once with `scipy.stats.f.ppf`.

:::{dropdown} Show full solution
```python
from scipy import stats

p = 3   # first three PCs
n = X_scaled.shape[0]
scores_p = scores[:, :p]
eigvals_p = pca.explained_variance_[:p]

T2 = np.sum(scores_p**2 / eigvals_p, axis=1)

alpha = 0.05
F_crit = stats.f.ppf(1 - alpha, p, n - p)
T2_crit = F_crit * p * (n - 1) / (n - p)

flagged = np.where(T2 > T2_crit)[0]
print(f'T2_crit (alpha=0.05, p=3, n={n}) = {T2_crit:.3f}')
print(f'Max T2 observed: {T2.max():.3f}')
print(f'Samples above control limit: {len(flagged)}')
```
Output: `T2_crit = 8.184`, with the largest observed `T2 = 7.195` — **no
sample exceeds the 95% control limit**, so nothing here is flagged as an
outlier. This is the expected result for this dataset: Section 12.1
generated every sample from one of four clean multivariate-normal grade
distributions with no deliberately injected contamination or measurement
error, so a well-behaved $T^2$ chart with everything inside the limit is
exactly what "the model fits the data it was built to fit" should look
like — Hotelling's $T^2$ becomes genuinely useful once you start screening
*new* incoming samples against this fitted PCA model for ones that don't
match any of the four known grades.
:::

**Exercise 2 — Effect of scaling**

> **Hint:** Re-run `PCA()` on `X_raw` (mean-centre it, but skip
> `StandardScaler`) and compare `pca_raw.components_[0]` to the scaled
> version's — the element with by far the largest *raw* variance
> (`df_xrf[elements].var()`) is your answer before you even look at the
> loadings.

:::{dropdown} Show full solution
```python
pca_raw = PCA()
scores_raw = pca_raw.fit_transform(X_raw - X_raw.mean(axis=0))
explained_raw = pca_raw.explained_variance_ratio_

print('Unscaled explained variance:', explained_raw[:3].round(3))
print('Scaled explained variance:  ', explained[:3].round(3))
print()
print('Raw variance per element:')
print(df_xrf[elements].var().round(3))
print()
print('Unscaled PC1 loadings:')
for elem, load in zip(elements, pca_raw.components_[0]):
    print(f'  {elem}: {load:+.3f}')
```
Output: unscaled PC1 alone jumps to 68.1% of variance (vs. 60.4% scaled),
and its loadings are dominated by Fe (+0.797) — more than double the size
of the next-largest loading (Cr, −0.430). This traces directly back to raw
variance: Fe's wt% variance across the dataset (13.0) is roughly 4–7×
larger than Cr's or Ni's and thousands of times larger than N's
(0.004) purely because Fe is measured in tens of wt% while N is measured in
hundredths — PCA on unscaled data literally maximises variance in the raw
units, so it "discovers" that Fe has the biggest numbers and calls that the
most important direction, which has nothing to do with Fe being chemically
more informative for telling grades apart. This is exactly the failure mode
Section 12.2 warns about before running `StandardScaler`, now shown
concretely rather than just asserted.
:::

**Exercise 3 — 3-D score plot**

> **Hint:** `plotly.express.scatter_3d` takes three numeric columns plus a
> `color=` argument, same pattern as the 2-D score plot in Section 12.4 —
> build a small DataFrame from `scores[:, :3]` and the `Grade` labels first.

:::{dropdown} Show full solution
```python
import plotly.express as px

df_scores3d = pd.DataFrame({
    'PC1': scores[:, 0], 'PC2': scores[:, 1], 'PC3': scores[:, 2],
    'Grade': grades,
})

fig = px.scatter_3d(
    df_scores3d, x='PC1', y='PC2', z='PC3', color='Grade',
    color_discrete_map=colors_map,
    title='3-D PCA Score Plot — XRF Steel Data',
    labels={'PC1': f'PC1 ({explained[0]*100:.1f}%)',
            'PC2': f'PC2 ({explained[1]*100:.1f}%)',
            'PC3': f'PC3 ({explained[2]*100:.1f}%)'},
)
fig.update_traces(marker=dict(size=4, opacity=0.75))
fig.show()
```
Rotating this interactively confirms what the two static 2-D panels in
Section 12.4 already suggested: 2205 Duplex and 17-4PH sit at opposite
extremes along PC1 with 304SS/316SS between them, while the 304SS/316SS
split (invisible on PC1 alone) opens up cleanly along PC2 — and since PC1 +
PC2 + PC3 together already capture 98.2% of total variance (Section 12.3's
take-home message), this 3-D view is very close to the complete geometric
picture, with almost nothing left in the discarded PC4–PC8 directions.
:::

## Notebook 15: Linear Discriminant Analysis (LDA)

**Exercise 1 — LDA vs PCA boundary**

> **Hint:** You can't invert `lda.transform` exactly (it projects 8
> features down to fewer LD axes, so information is lost), but you can
> approximate the inverse with `np.linalg.pinv(lda.scalings_[:, :2])` —
> map every point of an LD1/LD2 meshgrid back to an approximate point in
> scaled feature space, then call `lda.predict` on those approximate
> points and contour the result with `ax.contourf`.

:::{dropdown} Show full solution
```python
scalings2 = lda.scalings_[:, :2]           # (8 features, 2 LDs)
xbar = X_scaled.mean(axis=0)
pinv = np.linalg.pinv(scalings2)           # (2, 8), approximate inverse

x_min, x_max = lda_scores[:, 0].min() - 1, lda_scores[:, 0].max() + 1
y_min, y_max = lda_scores[:, 1].min() - 1, lda_scores[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

grid_ld = np.c_[xx.ravel(), yy.ravel()]
X_approx = grid_ld @ pinv + xbar           # back to (approx.) scaled feature space
Z = lda.predict(X_approx).reshape(xx.shape)
Z_num = np.array([[grades_list.index(g) for g in row] for row in Z])

fig, ax = plt.subplots(figsize=(7, 6))
ax.contourf(xx, yy, Z_num, alpha=0.15, levels=len(grades_list)-1,
            colors=[colors_map[g] for g in grades_list])
for grade in grades_list:
    mask = y == grade
    ax.scatter(lda_scores[mask, 0], lda_scores[mask, 1],
               c=colors_map[grade], label=grade, s=45, edgecolors='none')
ax.set_xlabel('LD1'); ax.set_ylabel('LD2')
ax.legend(title='Grade', fontsize=8)
plt.tight_layout()
plt.show()
```
The shaded regions confirm what Section 13.2's take-home message already
read off the raw score plot: **17-4PH occupies its own large, isolated
region** with no other class's territory anywhere near it, while **304SS,
316SS, and 2205Dup share three much narrower, mutually adjacent regions**
in the middle of the plot — 316SS's slice in particular is squeezed between
the other two, geometric confirmation of why it is the grade that gets
misclassified in both directions in Section 13.4's confusion matrix. 17-4PH
is by far the most easily separated grade; 316SS is the hardest.
:::

**Exercise 2 — Classifying new samples**

> **Hint:** Draw 5 new samples with `rng.multivariate_normal(grade_comp[g],
> np.diag(grade_std**2))` for grades of your choosing, then call
> `pipe.predict` and `pipe.predict_proba` on them together — `predict_proba`
> gives you the confidence behind each call, which is what actually matters
> for a real PMI decision, not just the class label.

:::{dropdown} Show full solution
```python
true_grades = ['304SS', '316SS', '2205Dup', '17-4PH', '316SS']
new_X = np.array([rng.multivariate_normal(grade_comp[g], np.diag(grade_std**2))
                   for g in true_grades])

pred = pipe.predict(new_X)
proba = pipe.predict_proba(new_X)

results = pd.DataFrame(proba, columns=pipe.named_steps['lda'].classes_)
results.insert(0, 'True', true_grades)
results.insert(1, 'Predicted', pred)
print(results.round(3))
```
Output: all 5 new samples are classified correctly, but with very different
confidence levels — the 17-4PH and both 316SS samples are called with
≥99.5% confidence in the right class, while the 304SS sample is correctly
called but with only 89.8% confidence (the remaining 10.2% going to 316SS,
its closest neighbour in LD space). That single lower-confidence call is
exactly the kind of borderline result Section 13's final take-home message
flags as worth a second, longer-dwell-time XRF read or a confirmatory
wet-chemistry Mo assay before trusting it on a real part — a bare `predict`
label alone would hide that this one call was noticeably less certain than
the other four.
:::

**Exercise 3 — Quadratic DA**

> **Hint:** Swap `LinearDiscriminantAnalysis()` for
> `QuadraticDiscriminantAnalysis()` inside the same `Pipeline` /
> `StratifiedKFold` cross-validation setup from Section 13.5 — no other
> changes needed, since both classifiers share the same scikit-learn
> `.fit`/`.predict` interface.

:::{dropdown} Show full solution
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

pipe_qda = Pipeline([('scaler', StandardScaler()), ('qda', QuadraticDiscriminantAnalysis())])
cv_scores_qda = cross_val_score(pipe_qda, X_raw, y, cv=cv, scoring='accuracy')

print(f'LDA 10-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  (Section 13.5)')
print(f'QDA 10-fold CV accuracy: {cv_scores_qda.mean():.3f} ± {cv_scores_qda.std():.3f}')
```
Output: QDA actually does **worse** than LDA here — 88.3% vs. LDA's 94.2%.
This confirms the prediction in Section 13's final take-home message almost
exactly: every grade in this dataset was generated from the *same* diagonal
covariance matrix (`grade_std`), so LDA's shared-covariance assumption
holds by construction and QDA gains nothing from relaxing it — instead, QDA
pays a real price for estimating a separate 8×8 covariance matrix per class
(4 × 36 unique parameters instead of 1 × 36) from only 30 samples per
class, and that extra estimation noise is what drags its cross-validated
accuracy down. This is a concrete illustration of the theory page's
bias–variance framing of LDA vs. QDA: QDA's extra flexibility only pays off
when classes' covariances genuinely differ *and* there is enough data per
class to estimate each one reliably — neither condition holds here.
:::

## Notebook 16: Factor Analysis (FA)

**Exercise 1 — Rotation comparison**

> **Hint:** `FactorAnalyzer(n_factors=3, rotation='promax')` — same call as
> Section 14.4's varimax fit, just swap the rotation name. Unlike varimax
> (which forces factors to stay uncorrelated), promax is an *oblique*
> rotation, so also check `fa_promax.phi_`, the factor correlation matrix
> that only an oblique rotation produces at all.

:::{dropdown} Show full solution
```python
fa_promax = FactorAnalyzer(n_factors=3, rotation='promax')
fa_promax.fit(df_corr)

load_promax = pd.DataFrame(fa_promax.loadings_, index=df_corr.columns,
                            columns=['F1', 'F2', 'F3'])
print('Promax loadings:')
print(load_promax.round(3))
print('\nFactor correlation matrix (phi):')
print(pd.DataFrame(fa_promax.phi_, index=['F1','F2','F3'], columns=['F1','F2','F3']).round(3))
```
Output: the promax loadings tell essentially the same story as varimax's —
each variable still loads cleanly on one factor (F1: i_corr +0.99, mass_loss
+0.95, pit_depth +0.91, E_corr −0.91; F2: coating_t +1.00, EIS_R2 +0.92, R_p
+0.88; F3: NaCl_pct +1.01, temp_C +0.99), if anything slightly sharper than
varimax's, since promax is allowed the extra freedom of letting factors
correlate. And they do: `phi_` shows F1–F3 (activity vs. aggressiveness)
correlated at **−0.58**, a genuinely non-trivial relationship, while F1–F2
(−0.28) and F2–F3 (+0.11) are much weaker. This is chemically sensible —
more aggressive test conditions (higher NaCl, higher temperature) plausibly
do drive somewhat higher electrochemical activity — but it's also worth
flagging as a limit of this teaching dataset: `f_activity`, `f_barrier`, and
`f_aggressive` were generated as three *independent* `rng.normal` draws in
Section 14.1, so this correlation is entirely an artefact of finite-sample
estimation noise (n=80), not a true relationship built into the data.
Varimax's zero-by-construction correlation is arguably the *more* faithful
choice here for that reason; promax would be the right tool to reach for on
real data where you have reason to expect the underlying phenomena to be
genuinely coupled.
:::

**Exercise 2 — 4-factor solution**

> **Hint:** Re-fit `FactorAnalyzer(n_factors=4, rotation='varimax')` and
> compare `fa4.get_communalities()` element-by-element against the
> 3-factor communalities already printed in Section 14.4 — look for whether
> the gains are broad (every variable improves a bit) or concentrated on
> just one variable (the one factor that was already a bit weak).

:::{dropdown} Show full solution
```python
fa4 = FactorAnalyzer(n_factors=4, rotation='varimax')
fa4.fit(df_corr)

comm_compare = pd.DataFrame({
    '3-factor': fa.get_communalities(),
    '4-factor': fa4.get_communalities(),
}, index=df_corr.columns)
comm_compare['gain'] = comm_compare['4-factor'] - comm_compare['3-factor']
print(comm_compare.round(3))

var4 = fa4.get_factor_variance()
print(f'\nCumulative variance, 4 factors: {var4[2][-1]*100:.1f}%  '
      f'(3 factors: {fa.get_factor_variance()[2][-1]*100:.1f}%)')
```
Output: communalities barely move for most variables — the average gain is
about +0.01 — with one striking exception: **`EIS_R1` jumps from 0.825 to
0.995** (+0.17), by far the largest change of any variable. That is exactly
consistent with Section 14.4's take-home message, which already flagged
`EIS_R1` as the one variable with a genuine secondary dependency
(`EIS_R1 = 15 − 2·f_activity + 3·f_aggressive + noise(2)`, mixing in a real
`f_activity` contribution unlike its F3 neighbours `NaCl_pct`/`temp_C`) — the
new 4th factor exists almost entirely to soak up that one variable's
leftover cross-loading, not to capture any broader hidden structure. Total
cumulative variance only edges up from 91.6% to 93.6%, and Section 14.3's
parallel analysis already showed the 4th eigenvalue (0.17) sitting nowhere
close to the random-data threshold — so this is exactly "the extra factor
represents noise" (more precisely, one variable's private idiosyncrasy) as
the exercise prompt anticipates, not a genuine fourth latent cause.
:::

**Exercise 3 — Regression with factor scores**

> **Hint:** `fa.transform(df_corr)` gives one score column per factor
> (Section 14.5) — feed those three columns straight into `smf.ols` as
> predictors of `mass_loss`, and compare each coefficient's sign and size
> against which factor `mass_loss` actually loaded on most strongly in
> Section 14.4's loading table.

:::{dropdown} Show full solution
```python
factor_scores = fa.transform(df_corr)
score_df = pd.DataFrame(factor_scores, columns=['activity', 'barrier', 'aggressive'])
score_df['mass_loss'] = df_corr['mass_loss'].values

model_scores = smf.ols('mass_loss ~ activity + barrier + aggressive', data=score_df).fit()
print(model_scores.params)
print(model_scores.pvalues)
print(f'R² = {model_scores.rsquared:.4f}')
```
Output: `activity` has by far the largest, most significant coefficient
(+1.455, p≈0), `barrier` is smaller and negative (−0.536, p≈0), and
`aggressive` comes out small but still statistically significant (−0.168,
p<0.001), with R²=0.981 overall. The size ordering **confirms** Section
14.4's loading-based interpretation exactly: `mass_loss` loaded strongly
positive on F1 (+0.91, "electrochemical activity") and not at all on F2 or
F3, and here `activity`'s regression coefficient dominates the other two by
roughly 3–9×. The small but "significant" `aggressive` term is worth not
over-interpreting, though: `mass_loss`'s true generating equation in
Section 14.1 depends only on `f_activity` and `f_barrier`, with zero
dependence on `f_aggressive` — the regression is picking up a small
finite-sample leak in the *estimated* factor scores (the `aggressive`
score correlates at about −0.13 with the true `f_activity`, purely from
FA's imperfect score estimation at n=80), not a genuine physical
dependence. This is a useful caution for interpreting any regression on
factor scores: they are estimates of the latent factors, not the factors
themselves, and small cross-contamination between them is normal.
:::

## Notebook 17: Partial Least Squares (PLS)

**Exercise 1 — PLS2: multiple responses**

> **Hint:** Build a second response the same way `Tg` was built in Section
> 15.1, but from latent factors 2 and 4 (`C[:, 1]` and `C[:, 3]`) instead of
> 1 and 3. Stack `Tg` and the new response into one `(n_samples, 2)` array
> and pass it as `y` to `PLSRegression` — everything else (the CV loop,
> `cross_val_predict`) works unchanged, just compute R² per column of the
> prediction output separately.

:::{dropdown} Show full solution
```python
# Second response: depends on latent factors 2 and 4, not 1 and 3
YoungMod = 200 + 20*C[:, 1] + 10*C[:, 3] + rng.normal(0, 3, n_samples)
Y2 = np.column_stack([Tg, YoungMod])

q2_tg, q2_ym = [], []
for nc in range(1, max_comp + 1):
    pls2_tmp = PLSRegression(n_components=nc, scale=False)
    Y_pred = cross_val_predict(pls2_tmp, X_scaled, Y2, cv=kf)
    q2_tg.append(r2_score(Tg, Y_pred[:, 0]))
    q2_ym.append(r2_score(YoungMod, Y_pred[:, 1]))

print(f'PLS2, Q²(Tg) at {opt_nc} components:  {q2_tg[opt_nc-1]:.4f}')
print(f'PLS2, Q²(YM) at {opt_nc} components:  {q2_ym[opt_nc-1]:.4f}')
print(f'PLS1, Q²(Tg) at {opt_nc} components (Section 15.3): {r2_cv:.4f}')
```
Output: at the same 5 components PLS1 needed for `Tg` alone, PLS2 reaches
`Q²(Tg)=0.9910` — essentially identical to PLS1's own `Q²=0.9910` — while
simultaneously predicting the new response at `Q²(YM)=0.9770`. PLS2 loses
almost nothing on `Tg` by sharing its five latent components with a second,
different response, because the two responses' relevant latent factors
(1&3 for Tg, 2&4 for YM) are different but not fighting for the *same*
early components — PLS is free to build components that jointly serve both.
This is PLS2's core use case: predicting several correlated quality
properties (Tg, modulus, density...) from one spectrum in a single model,
rather than fitting a separate PLS1 model per property.
:::

**Exercise 2 — PLS vs MLR**

> **Hint:** `sm.OLS(Tg, sm.add_constant(X_scaled)).fit()` uses all 50
> channels as predictors — with `n=100` samples and `p=50` predictors this
> *does* converge (n > p, unlike the n < p case), but check the condition
> number and compare train R² against a cross-validated score, not just
> the in-sample R², to see the real cost of using every collinear channel
> at once.

:::{dropdown} Show full solution
```python
from sklearn.linear_model import LinearRegression

# Full 50-channel MLR
lr_full = LinearRegression()
Tg_pred_mlr_cv = cross_val_predict(lr_full, X_scaled, Tg, cv=kf)
lr_full.fit(X_scaled, Tg)
r2_mlr_train = r2_score(Tg, lr_full.predict(X_scaled))
r2_mlr_cv = r2_score(Tg, Tg_pred_mlr_cv)
rmse_mlr_cv = np.sqrt(mean_squared_error(Tg, Tg_pred_mlr_cv))
print(f'Full 50-channel MLR:  R²(train)={r2_mlr_train:.4f}, Q²(CV)={r2_mlr_cv:.4f}, '
      f'RMSECV={rmse_mlr_cv:.2f}°C')

# MLR restricted to the 10 highest-VIP channels
top10_idx = np.argsort(vip_scores)[-10:]
lr_top10 = LinearRegression()
Tg_pred_top10_cv = cross_val_predict(lr_top10, X_scaled[:, top10_idx], Tg, cv=kf)
print(f'Top-10-VIP MLR:       Q²(CV)={r2_score(Tg, Tg_pred_top10_cv):.4f}, '
      f'RMSECV={np.sqrt(mean_squared_error(Tg, Tg_pred_top10_cv)):.2f}°C')
```
Output: the full 50-channel MLR *does* converge — `n=100 > p=50` — reaching
a very high train R²=0.9954, but its cross-validated `Q²=0.976` (RMSECV
3.48°C) is noticeably worse than PLS's own `Q²=0.991` (RMSECV 2.13°C from
Section 15.3), despite MLR's higher train R². That gap between train and CV
performance is classic overfitting: with 50 highly collinear predictors and
only 100 samples, MLR has enough free parameters to fit noise as well as
signal, exactly the situation Section 2 of the [theory page](../03_multivariate/theory.md)
flags VIF for — restricting to the 10 highest-VIP channels doesn't fully
close the gap either (`Q²≈0.85`, worse than full PLS), because throwing
away 40 channels also throws away real, if individually weaker, chemical
signal that PLS's latent components would have blended in. This is the
concrete version of the theory page's summary: MLR "struggles when
predictors are collinear or numerous," and PLS's latent-variable approach
is built specifically to sidestep that trade-off rather than being forced
to choose between "use everything and overfit" and "cut variables and lose
signal."
:::

**Exercise 3 — Concentration residuals**

> **Hint:** `X_res = X_scaled - pls.x_scores_ @ pls.x_loadings_.T` is the
> part of each spectrum the fitted PLS model's 5 components don't
> reconstruct — sum each row's squared residuals (`np.sum(X_res**2,
> axis=1)`) and flag samples whose sum sits several standard deviations
> above the rest, the same style of screening Notebook 14's Hotelling
> $T^2$ used on PCA scores.

:::{dropdown} Show full solution
```python
X_res = X_scaled - pls.x_scores_ @ pls.x_loadings_.T
ssq_residual = np.sum(X_res**2, axis=1)

threshold = ssq_residual.mean() + 3 * ssq_residual.std()
outliers = np.where(ssq_residual > threshold)[0]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(n_samples), ssq_residual, color='steelblue', alpha=0.7)
ax.axhline(threshold, color='crimson', ls='--', label='mean + 3·std')
ax.scatter(outliers, ssq_residual[outliers], color='crimson', zorder=5, s=50)
ax.set_xlabel('Sample index')
ax.set_ylabel('Sum of squared X-residuals')
ax.set_title('PLS X-Residuals per Sample')
ax.legend()
sns.despine(ax=ax)
plt.tight_layout()
plt.show()

print(f'Flagged samples (> mean+3·std = {threshold:.3f}): {outliers}')
print(f'Their residual sums: {ssq_residual[outliers].round(3)}')
```
Output: one sample (**index 16**) is flagged, with a residual sum of squares
of `1.437` against a mean of `0.453` and threshold of `1.061` — noticeably
larger than any other sample's (the next-largest is `1.014`). This X-residual
check answers a different question than Hotelling $T^2$ does: $T^2$ (Notebook
14, Exercise 1) flags samples that are unusual *within* the model's latent
space (extreme scores along the retained components), while the X-residual
flags samples the model's latent space doesn't describe well *at all* — a
spectrum that doesn't look like a combination of the five calibration bands
the model learned, which in a real NIR context would suggest contamination,
an unrecorded new chemical species, or an instrument glitch on that specific
scan, worth a manual look before trusting this model's Tg prediction for it.
:::

**Exercise 4 — Nuisance amplitude sweep**

> **Hint:** Wrap Section 15.7's baseline-generation and component-selection
> code in a loop over `a in [0, 2, 5, 10, 15, 25]`, recording `opt_nc_ns`
> and `opt_nc_pcr_ns` from each pass — then plot both series against `a` on
> one axis to see how quickly the two methods' component requirements pull
> apart.

:::{dropdown} Show full solution
```python
amplitudes = [0, 2, 5, 10, 15, 25]
max_comp_sweep = 20
sweep_results = []

for a in amplitudes:
    baseline_amp_a = rng.normal(0, a, n_samples) if a > 0 else np.zeros(n_samples)
    baseline_a = np.outer(baseline_amp_a, baseline_shape)
    X_ns_a = X + baseline_a
    X_ns_a_scaled = StandardScaler().fit_transform(X_ns_a)

    rmsecv_pls_a, rmsecv_pcr_a = [], []
    for nc in range(1, max_comp_sweep + 1):
        pls_pred = cross_val_predict(PLSRegression(n_components=nc, scale=False),
                                      X_ns_a_scaled, Tg, cv=kf).ravel()
        rmsecv_pls_a.append(np.sqrt(mean_squared_error(Tg, pls_pred)))
        pcr_pipe_a = Pipeline([('pca', PCA(n_components=nc)), ('lr', LinearRegression())])
        pcr_pred = cross_val_predict(pcr_pipe_a, X_ns_a_scaled, Tg, cv=kf).ravel()
        rmsecv_pcr_a.append(np.sqrt(mean_squared_error(Tg, pcr_pred)))

    opt_pls_a, opt_pcr_a = np.argmin(rmsecv_pls_a) + 1, np.argmin(rmsecv_pcr_a) + 1
    sweep_results.append((a, opt_pls_a, rmsecv_pls_a[opt_pls_a-1],
                           opt_pcr_a, rmsecv_pcr_a[opt_pcr_a-1]))

sweep_df = pd.DataFrame(sweep_results,
                         columns=['amplitude', 'PLS_ncomp', 'PLS_RMSECV', 'PCR_ncomp', 'PCR_RMSECV'])
print(sweep_df)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sweep_df['amplitude'], sweep_df['PLS_ncomp'], 'bo-', label='PLS components needed')
ax.plot(sweep_df['amplitude'], sweep_df['PCR_ncomp'], 'rs-', label='PCR components needed')
ax.set_xlabel('Baseline amplitude a'); ax.set_ylabel('Optimal number of components')
ax.legend(); sns.despine(ax=ax)
plt.tight_layout(); plt.show()
```
Output (one representative run — exact component counts have some
run-to-run CV noise, but the trend is consistent):

| amplitude a | PLS components | PLS RMSECV | PCR components | PCR RMSECV |
|---|---|---|---|---|
| 0  | 5 | 2.13 | 5  | 2.14 |
| 2  | 6 | 2.23 | 10 | 2.20 |
| 5  | 6 | 2.28 | 7  | 2.25 |
| 10 | 6 | 2.38 | 16 | 2.33 |
| 15 | 6 | 2.42 | 15 | 2.37 |
| 25 | 8 | 2.36 | 12 | 2.36 |

PCR's component count **jumps almost immediately** — already needing nearly
double PLS's count by `a=2` — and stays elevated and noisy (7–16 components)
across the rest of the sweep, while PLS's count creeps up only slowly (5→6,
occasionally 8). Even at `a=0` there is no real gap; a clear, practically
meaningful gap opens up by roughly `a=10`, where 6 vs. 16 components is a
large, unmistakable difference. PLS's RMSECV does drift upward across the
sweep too (2.13°C → ~2.4°C) — **it should**, and does: nothing in the PLS
algorithm makes it literally *blind* to the nuisance direction (it still
has to spend some of its limited components partially accounting for the
baseline once it dominates X's scale), it is only better than PCR at not
*wasting* components on that direction ahead of the useful ones. The
right conclusion is the comparative one, not an absolute one: PLS needs
consistently fewer components than PCR as the nuisance grows, not that PLS
is somehow immune to nuisance variance altogether.
:::

**Exercise 5 — Which PC is the real signal?**

> **Hint:** Correlate each of the first 8 columns of `pca_check.transform(X_ns_scaled)`
> against `Tg` individually (`np.corrcoef`), not just PC1 as Section 15.7
> did — then fit a plain `LinearRegression` using *only* the best-correlated
> PC as the sole predictor and cross-validate it the same way as the full
> PCR model.

:::{dropdown} Show full solution
```python
scores_ns_full = pca_check.transform(X_ns_scaled)
pc_tg_corr = [np.corrcoef(scores_ns_full[:, i], Tg)[0, 1] for i in range(8)]
print('Correlation of PC1-PC8 with Tg:', np.round(pc_tg_corr, 3))

best_pc = int(np.argmax(np.abs(pc_tg_corr)))
print(f'Strongest single PC: PC{best_pc+1}  (r={pc_tg_corr[best_pc]:+.3f})')

single_pc_pred = cross_val_predict(LinearRegression(), scores_ns_full[:, [best_pc]], Tg, cv=kf).ravel()
rmse_single = np.sqrt(mean_squared_error(Tg, single_pc_pred))
print(f'RMSECV using ONLY PC{best_pc+1}: {rmse_single:.2f}°C')
print(f'RMSECV, full CV-optimal PCR ({opt_nc_pcr_ns} components, Section 15.7): {rmsecv_pcr_ns[opt_nc_pcr_ns-1]:.2f}°C')
```
Output: **PC1 (r=0.067) is barely related to Tg at all** — the strongest
single correlation actually belongs to **PC4 (r=−0.710)**, not PC1, PC2, or
PC3. But using PC4 alone as the sole predictor gives `RMSECV≈16.2°C` — more
than 7× worse than the full 12-component PCR model's `RMSECV=2.34°C` from
Section 15.7. This is the important distinction the exercise is pointing
at: **"PCR needs more components" is not the same claim as "PCR ignores the
useful information."** The Tg-relevant signal here is genuinely spread
across *several* different PCs (not concentrated in any one of them,
including the best-correlated PC4), because Tg's true generating equation
mixes five separate latent chemical factors — no single PCA direction
happens to align with that particular five-factor combination. PCR
eventually recovers essentially all of it, but only once enough components
are included to jointly reconstruct that combination; PLS's advantage is
getting there in fewer components by building its latent variables with
Tg's structure in mind from the start, not by "seeing" information that a
sufficiently large PCR model cannot.
:::

**Exercise 6 — Interpretability under the original (aligned) dataset**

> **Hint:** Rerun Section 15.8's `np.corrcoef(pc1_loading, t1_weight)`
> comparison, but with `pca_orig = PCA(n_components=5).fit(X_scaled)` and
> the original `pls` model from Sections 15.1–15.3, instead of the
> nuisance-dominated `pca_ns`/`pls_ns` — eigenvector sign is arbitrary, so
> compare the *magnitude* of the correlation, not its sign.

:::{dropdown} Show full solution
```python
pca_orig = PCA(n_components=5).fit(X_scaled)
pc1_loading_orig = pca_orig.components_[0]
t1_weight_orig = pls.x_weights_[:, 0]

r_pc1_t1_orig = np.corrcoef(pc1_loading_orig, t1_weight_orig)[0, 1]
print(f'|corr(PC1 loading, T1 weight)|, original aligned data: {abs(r_pc1_t1_orig):.3f}')
print(f'|corr(PC1 loading, T1 weight)|, nuisance-dominated data (Section 15.8): '
      f'{abs(np.corrcoef(pc1_loading, t1_weight)[0, 1]):.3f}')
```
Output: on the original, well-aligned dataset the two components' loading
spectra are **more similar to each other** (`|r|=0.549`) than they were on
the nuisance-dominated data from Section 15.8 (`|r|=0.382`). That's the
expected direction: without a dominant, Tg-irrelevant artifact to chase,
PC1's "direction of maximum X-variance" and T1's "direction most predictive
of Tg" both end up drawing on the same two latent chemical profiles that
happen to carry the most weight in this data (the bands near 3300 and 3700
cm⁻¹), rather than pointing at unrelated phenomena. This is exactly *why*
Section 15.6 found PLS and PCR essentially tied on the original data
(5 components each, matching Q²) — when the two components already mean
similar things, there is no "wasted early component" for PLS to save you
from, so PLS's usual efficiency advantage has nothing to bite on, precisely
as Section 15.6's take-home message and the summary table in Section 15.8
both anticipate.
:::

## Notebook 18: Live Tutorial 1 — PCA & Factor Analysis

**Exercise 1 — Manual reconstruction**

> **Hint:** `scores_manual[:, :k] @ loadings_manual[:, :k].T` reconstructs
> an approximation of the standardized data from just the first `k`
> components — with `k=10` (all components, for a 10-variable dataset) this
> is an exact identity, since keeping every eigenvector throws nothing
> away; the RMSE only becomes non-zero once `k` is less than the full
> number of variables.

:::{dropdown} Show full solution
```python
X_mean, X_std = X.mean(axis=0), X.std(axis=0, ddof=1)
X_scaled_true = (X - X_mean) / X_std

X_approx3 = scores_manual[:, :3] @ loadings_manual[:, :3].T
rmse_3 = np.sqrt(np.mean((X_scaled_true - X_approx3)**2))

X_approx10 = scores_manual[:, :10] @ loadings_manual[:, :10].T
rmse_10 = np.sqrt(np.mean((X_scaled_true - X_approx10)**2))

print(f'RMSE, 3-component reconstruction:  {rmse_3:.4f}')
print(f'RMSE, 10-component reconstruction: {rmse_10:.2e}  (all components kept)')
```
Output: `RMSE_3 = 0.559` (in standardized units) vs. `RMSE_10 ≈ 1e-15` —
essentially exact floating-point zero, confirming that keeping *all* 10
eigenvectors is a lossless change of basis, not an approximation at all.
The 3-component RMSE of 0.559 is the quantitative price of compressing 10
variables down to 3: about 56% of one standard deviation's worth of
per-variable reconstruction error, on average, in exchange for a 70%
reduction in dimensionality (10 → 3) — consistent with Section 16.3's
scree plot, where the first 3 components captured roughly two-thirds of
total variance and the remaining 7 components split up what's left.
:::

**Exercise 2 — Unscaled PCA**

> **Hint:** Skip `StandardScaler` and fit `PCA()` directly on `X` — compare
> `first_cycle_capacity_mAhg`'s raw variance (mAh/g values in the hundreds)
> against `lattice_strain_pct`'s (values around 0.35%) to see why one of
> them ends up dominating PC1 by sheer numeric magnitude.

:::{dropdown} Show full solution
```python
pca_raw = PCA()
scores_raw = pca_raw.fit_transform(X - X.mean(axis=0))

print('Unscaled PC1 loadings:')
for var, load in sorted(zip(feature_cols, pca_raw.components_[0]), key=lambda t: -abs(t[1])):
    print(f'  {var}: {load:+.3f}')
print('\nRaw variance per variable:')
print(pd.Series(X.var(axis=0), index=feature_cols).round(2))
```
Output: unscaled PC1 is dominated by `first_cycle_capacity_mAhg` (loading
+0.789) and `Ni_content_pct` (+0.548), with every other variable's loading
below 0.27. That traces directly to raw variance: `first_cycle_capacity_mAhg`
has a raw variance of 214 (mAh/g)² and `Ni_content_pct` 135 (%)², both
enormous compared to `lattice_strain_pct`'s 0.01 (%)² — a difference of
four orders of magnitude driven entirely by measurement units, not
chemistry. This is exactly Notebook 12 (`02_pca.ipynb`)'s Exercise 2 lesson
recurring on a new dataset: standardisation isn't a formality, it's what
lets PCA's "direction of maximum variance" reflect actual compositional or
physical structure rather than which variable happens to be reported in
smaller units.
:::

**Exercise 3 — 3-D interactive score plot**

> **Hint:** `px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color=...)` —
> same call pattern as Notebook 12's Exercise 3, coloured here by
> `capacity_retention_100cyc_pct` instead of grade. Check the per-PC
> correlation with that colour variable, not just the plot by eye, to
> confirm what PC3 is adding.

:::{dropdown} Show full solution
```python
import plotly.express as px

df_scores3d = pd.DataFrame(scores_sklearn[:, :3], columns=['PC1', 'PC2', 'PC3'])
df_scores3d['capacity_retention_100cyc_pct'] = df_cathode['capacity_retention_100cyc_pct'].values

fig = px.scatter_3d(
    df_scores3d, x='PC1', y='PC2', z='PC3',
    color='capacity_retention_100cyc_pct', color_continuous_scale='Viridis',
    title='3-D PCA Score Plot, Coloured by Capacity Retention',
)
fig.update_traces(marker=dict(size=4, opacity=0.75))
fig.show()

for i in range(3):
    r = np.corrcoef(scores_sklearn[:, i], df_cathode['capacity_retention_100cyc_pct'])[0, 1]
    print(f'PC{i+1} correlation with capacity_retention_100cyc_pct: {r:+.3f}')
```
Output: PC1, PC2, *and* PC3 all correlate with capacity retention at
similar magnitude (`r ≈ +0.49, −0.54, −0.48` respectively) — **yes, the 3rd
axis reveals real, additional separation** the 2-D PC1–PC2 plot in Section
16.3 cannot show on its own. That makes sense given how this dataset was
generated (Section 16.1): PCA's components are blends of all three hidden
factors rather than one-per-factor, so stability-relevant information
(driven by the hidden Stability factor) is spread across all three of the
first three PCs rather than cleanly concentrated in a single one of them —
exactly the contrast Section 16.8's PCA-vs-FA comparison draws out
explicitly with the loading tables.
:::

**Exercise 4 — Promax (oblique) rotation**

> **Hint:** `FactorAnalyzer(n_factors=3, rotation='promax', method='principal')`
> — same call as Section 16.7's varimax fit, just swap the rotation name,
> then inspect `fa_promax.phi_` (only an oblique rotation produces a factor
> correlation matrix at all) to check whether Performance and Stability
> come out correlated, as the physical Ni trade-off would suggest.

:::{dropdown} Show full solution
```python
fa_promax = FactorAnalyzer(n_factors=3, rotation='promax', method='principal')
fa_promax.fit(df_cathode[feature_cols])

load_promax = pd.DataFrame(fa_promax.loadings_, index=feature_cols,
                            columns=['Factor1', 'Factor2', 'Factor3'])
print(load_promax.round(3))
print('\nFactor correlation matrix (phi):')
print(pd.DataFrame(fa_promax.phi_, index=['F1','F2','F3'], columns=['F1','F2','F3']).round(3))
```
Output: the promax loadings are nearly identical to varimax's (e.g.
`first_cycle_capacity_mAhg` +0.885 vs. +0.879, `capacity_retention_100cyc_pct`
−0.868 vs. −0.865) — allowing correlated factors barely changes which
variables load on which factor here. The factor correlation matrix confirms
why: all three off-diagonal correlations are small (|r| ≤ 0.15), including
Factor1 (Performance) vs. Factor3 (Stability) at only `r=−0.085`. This is a
mild surprise given Section 16.1's built-in Ni trade-off, but it makes
sense on reflection: the trade-off lives *inside* `Ni_content_pct`'s two
loadings (+0.75 on Performance, −0.40 on Stability) as a property of that
one variable, not as a correlation between the two *factors themselves* —
`F_true`'s three hidden factors were generated as independent
`rng.normal` draws in Section 16.1, so promax correctly finds little
factor-level correlation to add, and varimax's simpler orthogonal
assumption was already a good fit for this particular dataset.
:::

**Exercise 5 — 4-factor solution**

> **Hint:** Refit with `n_factors=4` and compare `fa4.get_communalities()`
> against the 3-factor communalities column by column — watch specifically
> for whether the gain concentrates on the variable that already had the
> *lowest* 3-factor communality.

:::{dropdown} Show full solution
```python
fa4 = FactorAnalyzer(n_factors=4, rotation='varimax', method='principal')
fa4.fit(df_cathode[feature_cols])

comm_compare = pd.DataFrame({
    '3-factor': fa.get_communalities(),
    '4-factor': fa4.get_communalities(),
}, index=feature_cols)
comm_compare['gain'] = comm_compare['4-factor'] - comm_compare['3-factor']
print(comm_compare.round(3))

load4 = pd.DataFrame(fa4.loadings_, index=feature_cols,
                      columns=[f'Factor{i+1}' for i in range(4)])
print('\nFactor 4 loadings:')
print(load4['Factor4'].round(3))
```
Output: three variables' communalities barely move (gains of 0.01–0.04),
but **`tap_density_gcm3` jumps from 0.504 to 0.943** (+0.44, by far the
largest change), and Factor4's loadings show why — `tap_density_gcm3`
loads at +0.906 on Factor4 alone, with every other variable loading below
0.24 on it. This is not a genuine fourth physical concept: `tap_density_gcm3`
was deliberately built (Section 16.1) with the *weakest* Morphology loading
of the three morphology variables (0.70, vs. 0.85 and −0.80 for the other
two), meaning it has the most "leftover" unique variance for the 3-factor
model to leave unexplained. The 4th factor exists almost entirely to mop up
that one variable's own noise, exactly the failure mode Section 16.6's
parallel analysis is built to catch — a real run of parallel analysis on
this dataset would show the 4th eigenvalue falling below the random-data
threshold, correctly flagging that a 4-factor model is over-extracting.
:::

**Exercise 6 — Break the Ni trade-off**

> **Hint:** Change `Ni_content_pct`'s stability loading in `loadings_true`
> from `-0.40` to `0.00`, regenerate the dataset with the *same* seed, and
> compare the angle between `Ni_content_pct`'s and
> `capacity_retention_100cyc_pct`'s loading vectors in the PC1–PC2 plane
> before and after — an obtuse angle (>90°) means anti-correlated (the
> trade-off), an acute angle means aligned (no trade-off).

:::{dropdown} Show full solution
```python
loadings_true_notrade = dict(loadings_true)
loadings_true_notrade['Ni_content_pct'] = (0.00, 0.75, 0.00)   # trade-off removed

rng_nt = np.random.default_rng(42)   # same seed as Section 16.1
F_true_nt = rng_nt.normal(0, 1, size=(n, 3))
data_nt = {}
for var, (l1, l2, l3) in loadings_true_notrade.items():
    unique_var = max(1e-6, 1 - (l1**2 + l2**2 + l3**2))
    z = l1*F_true_nt[:, 0] + l2*F_true_nt[:, 1] + l3*F_true_nt[:, 2] + rng_nt.normal(0, np.sqrt(unique_var), n)
    mean, sd = units[var]
    data_nt[var] = mean + sd * z
df_notrade = pd.DataFrame(data_nt)

Xs_nt = StandardScaler().fit_transform(df_notrade[feature_cols].values)
pca_nt = PCA().fit(Xs_nt)
load_nt = pd.DataFrame(pca_nt.components_[:2].T, index=feature_cols, columns=['PC1', 'PC2'])

def angle_deg(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

v_ni_orig = loadings_df.loc['Ni_content_pct'].values
v_cap_orig = loadings_df.loc['capacity_retention_100cyc_pct'].values
v_ni_nt = load_nt.loc['Ni_content_pct'].values
v_cap_nt = load_nt.loc['capacity_retention_100cyc_pct'].values

print(f'Original (with trade-off):   angle(Ni, capacity_retention) = {angle_deg(v_ni_orig, v_cap_orig):.1f}°')
print(f'Modified (no trade-off):     angle(Ni, capacity_retention) = {angle_deg(v_ni_nt, v_cap_nt):.1f}°')
```
Output: the angle between `Ni_content_pct` and `capacity_retention_100cyc_pct`
in the PC1–PC2 loading plane flips from **117.5°** (obtuse — the two arrows
point substantially *away* from each other, the geometric signature of the
built-in Ni trade-off: higher Ni, lower retention) down to **60.9°**
(acute — the two arrows now point in a broadly *similar* direction) once
the `-0.40` stability loading is removed. This is exactly the qualitative
change the exercise anticipates: with the trade-off gone, `Ni_content_pct`
is purely a Performance-block variable, so nothing pulls its loading vector
away from the other Performance variables (`first_cycle_capacity_mAhg`,
`rate_capability_pct`) any more, and its previously-negative relationship
with the Stability block disappears from the loading geometry entirely.
:::

**Exercise 7 — Your own latent-variable dataset**

> **Hint:** Copy Section 16.1's exact generative pattern — pick 2–4 named
> hidden factors, assign each observed variable a `(loading_1, ..., loading_k)`
> tuple, compute `unique_var = 1 - sum(loadings**2)` per variable, and run
> the same PCA + parallel-analysis + FA pipeline (Sections 16.2–16.7)
> unchanged on the result.

:::{dropdown} Show full solution
```python
# Example: 60 heat-treated aluminium alloy samples, 2 hidden factors
# (Solutionising Quality, Ageing Response), 6 observed variables.
n_al = 60
loadings_al = {
    'yield_strength_MPa':   (0.80, 0.30),
    'elongation_pct':       (0.75, -0.20),
    'hardness_HV':          (0.20, 0.85),
    'conductivity_pct_IACS': (-0.60, 0.40),
    'grain_size_um':        (-0.70, 0.00),
    'precipitate_frac_pct': (0.05, 0.90),
}
units_al = {
    'yield_strength_MPa':   (310, 25),
    'elongation_pct':       (12, 2.5),
    'hardness_HV':          (95, 8),
    'conductivity_pct_IACS': (42, 3),
    'grain_size_um':        (35, 6),
    'precipitate_frac_pct': (4.5, 1.2),
}

rng_al = np.random.default_rng(7)
F_al = rng_al.normal(0, 1, size=(n_al, 2))
data_al = {}
for var, (l1, l2) in loadings_al.items():
    unique_var = max(1e-6, 1 - (l1**2 + l2**2))
    z = l1*F_al[:, 0] + l2*F_al[:, 1] + rng_al.normal(0, np.sqrt(unique_var), n_al)
    mean, sd = units_al[var]
    data_al[var] = mean + sd * z
df_al = pd.DataFrame(data_al)
cols_al = list(df_al.columns)

# PCA + parallel analysis to confirm 2 factors are recovered
Xs_al = StandardScaler().fit_transform(df_al[cols_al].values)
pca_al = PCA().fit(Xs_al)
real_eigs_al, random_eigs_al = parallel_analysis(df_al[cols_al].values, seed=1)
n_suggested_al = int(np.sum(real_eigs_al > random_eigs_al))
print(f'Parallel analysis suggests {n_suggested_al} factors (built with 2).')

fa_al = FactorAnalyzer(n_factors=n_suggested_al, rotation='varimax', method='principal')
fa_al.fit(df_al[cols_al])
print(pd.DataFrame(fa_al.loadings_, index=cols_al,
                    columns=[f'Factor{i+1}' for i in range(n_suggested_al)]).round(2))
```
This is a template, not a fixed answer — the point of Exercise 7 is
practising Section 16.1's generative recipe end to end on a system of your
own choosing. Running it correctly should at minimum recover exactly as
many factors as were built in: parallel analysis flags **2 factors** for
the aluminium example above, matching `F_al`'s two columns exactly. The
rotated loadings are messier than the cathode case, though — `hardness_HV`
(0.93) and `precipitate_frac_pct` (0.89) load cleanly on Factor1, but
`yield_strength_MPa` splits across both factors (0.67 / −0.50) rather than
sitting cleanly on one. That is a genuinely useful negative result, not a
bug: `yield_strength_MPa` was deliberately given comparable loadings on
*both* hidden factors (0.80 and 0.30 in `loadings_al`), unlike the cathode
dataset's variables, which mostly loaded strongly on only one factor each
— varimax can only produce "clean," single-factor loadings when the
underlying data-generating process actually gives each variable one
dominant factor to load on. This is worth building into your own dataset
deliberately (as done here) precisely to see that FA's tidy, one-factor-
per-variable outcome in Sections 16.6–16.7 is a property of *that*
data, not a guarantee of the method.
:::

## Notebook 19: Live Tutorial 2 — PLS/PCR & LDA

**Exercise 1 — VIP-based variable selection**

> **Hint:** Reuse Notebook 15's `compute_vip` function unchanged on
> `pls_final` (Section 17.3's fitted model) and `X_train`/`y_train`, find
> the channel indices where `vip_scores > 1`, then refit
> `PLSRegression(n_components=best_nc, scale=True)` using only
> `X_train[:, high_vip_idx]` — compare its test-set R² against
> `r2_test` from Section 17.3.

:::{dropdown} Show full solution
```python
def compute_vip(pls_model, X, y):
    T, W, Q = pls_model.x_scores_, pls_model.x_weights_, pls_model.y_loadings_
    p, h = W.shape
    SSY = np.sum((y - y.mean())**2)
    SSY_comp = np.array([np.sum((T[:, k:k+1] @ Q[:, k:k+1].T)**2) for k in range(h)])
    return np.sqrt(p * np.sum((W**2) * SSY_comp[np.newaxis, :] / SSY, axis=1))

vip_scores = compute_vip(pls_final, X_train, y_train)
high_vip_idx = np.where(vip_scores > 1)[0]
print(f'{len(high_vip_idx)} of {X_train.shape[1]} channels have VIP > 1')

pls_vip = PLSRegression(n_components=best_nc, scale=True).fit(X_train[:, high_vip_idx], y_train)
r2_vip_test = r2_score(y_test, pls_vip.predict(X_test[:, high_vip_idx]).ravel())
print(f'Full-spectrum test R² (Section 17.3): {r2_test:.3f}')
print(f'High-VIP-only test R² ({len(high_vip_idx)} channels): {r2_vip_test:.3f}')
```
Output: only **9 of the 60 channels** clear VIP > 1, and restricting to
just those 9 gives a test R² of `0.759`, a real but modest drop from the
full-spectrum model's `0.808`. Losing about 0.05 of R² while discarding
85% of the channels is a reasonable trade if the goal is a simpler,
cheaper sensor (e.g. a filter photometer at 9 fixed wavelengths instead of
a full spectrometer) — but it also confirms VIP > 1 is a *relative*
threshold, not a guarantee that everything below it is worthless: some of
the discarded channels were still contributing real, if individually
smaller, predictive signal that the full 60-channel model was using.
:::

**Exercise 2 — Vary the train/test split**

> **Hint:** Rerun Section 17.3's whole component-selection-then-test
> pipeline (not just the final fit) for `test_size` in `[0.15, 0.25, 0.4]`
> — the number of components chosen by CV could in principle change too,
> not just the final R², since CV always runs on the training set only.

:::{dropdown} Show full solution
```python
for test_size in [0.15, 0.25, 0.40]:
    X_tr, X_te, y_tr, y_te = train_test_split(X_spec, y, test_size=test_size, random_state=0)
    q2_scores_ts = []
    for nc in range(1, max_components + 1):
        pls_tmp = PLSRegression(n_components=nc, scale=True)
        y_cv_tmp = cross_val_predict(pls_tmp, X_tr, y_tr, cv=kf).ravel()
        q2_scores_ts.append(r2_score(y_tr, y_cv_tmp))
    best_nc_ts = np.argmax(q2_scores_ts) + 1
    pls_ts = PLSRegression(n_components=best_nc_ts, scale=True).fit(X_tr, y_tr)
    r2_ts = r2_score(y_te, pls_ts.predict(X_te).ravel())
    print(f'test_size={test_size}: n_test={len(y_te)}, components={best_nc_ts}, test R²={r2_ts:.3f}')
```
Output:

| test_size | n_test | components chosen | test R² |
|---|---|---|---|
| 0.15 | 18 | 2 | 0.772 |
| 0.25 | 30 | 2 | 0.808 |
| 0.40 | 48 | 2 | 0.809 |

The chosen number of components (2) stays completely stable across all
three splits — reassuring, since that decision is made purely from
training-set cross-validation, unaffected by how much data gets held out.
The test R² itself is fairly stable too here (0.77–0.81), only really
wobbling for the smallest, 18-sample test set. That stability is not
guaranteed in general, though — with even fewer held-out samples, a single
split's R² becomes increasingly sensitive to exactly which spectra happened
to land in the test fold, which is precisely why Section 17.3 uses
cross-validation (not a single train/test comparison) for the decision
that actually matters, `n_components`, and reserves the single held-out
test only as one final, honest check.
:::

**Exercise 3 — QDA instead of LDA**

> **Hint:** Swap `LinearDiscriminantAnalysis()` for
> `QuadraticDiscriminantAnalysis()` inside the same PCA→classifier
> `Pipeline` from Section 17.6, fit on `X_train_c`/`y_train_c`, and compare
> test accuracy against `acc_lda` — the same comparison Notebook 13,
> Exercise 3 already ran once on a different dataset.

:::{dropdown} Show full solution
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda_pipeline = Pipeline([
    ('pca', PCA(n_components=n_pca_for_lda)),
    ('qda', QuadraticDiscriminantAnalysis()),
]).fit(X_train_c, y_train_c)

acc_qda = accuracy_score(y_test_c, qda_pipeline.predict(X_test_c))
print(f'LDA (Section 17.6): {acc_lda:.1%}')
print(f'QDA (6 PCA scores): {acc_qda:.1%}')
```
Output: QDA reaches **100% test accuracy — tied with LDA**, not better.
Unlike Notebook 13's steel-XRF case (where QDA measurably *underperformed*
LDA because its extra per-class covariance parameters had nothing genuine
to fit and only added estimation noise), here the two are simply tied,
because LDA had already hit the accuracy ceiling: with only ~30 training
spectra per class in a 6-dimensional PCA-score space, and three classes
whose fingerprints were built with reasonably distinct band-intensity
mixes (Section 17.2), the classes are already separable enough that
neither a shared covariance (LDA) nor a per-class one (QDA) leaves any
errors on the table. QDA's extra flexibility can only pay off, or hurt,
relative to LDA when there is daylight between them to begin with — a
useful reminder that "try QDA" is a diagnostic worth running, not a change
that reliably moves the needle in either direction.
:::

**Exercise 4 — How many PCA components does LDA actually need?**

> **Hint:** Sweep `n_pca_for_lda` well beyond the exercise's suggested
> 2–15 range if the accuracy curve looks flat across it — this dataset's
> classes turn out to be separable enough that the interesting behaviour
> (both under- and over-fitting) only shows up outside that window, which
> is itself a useful finding to report.

:::{dropdown} Show full solution
```python
accs = []
for npc in range(1, 60):
    lda_pipe_npc = Pipeline([
        ('pca', PCA(n_components=npc)),
        ('lda', LinearDiscriminantAnalysis()),
    ]).fit(X_train_c, y_train_c)
    accs.append(accuracy_score(y_test_c, lda_pipe_npc.predict(X_test_c)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 60), accs, 'o-', ms=3)
ax.set_xlabel('Number of PCA components fed to LDA')
ax.set_ylabel('Test accuracy')
ax.set_title('LDA test accuracy vs. PCA components retained')
sns.despine(ax=ax)
plt.tight_layout()
plt.show()
```
Output: within the exercise's suggested 2–15 range, accuracy is a flat
**100% line** — every value in that window is already at the ceiling, so
the sweep alone doesn't show either failure mode. Extending it reveals
both edges: **1 component clearly under-fits** (96.7%, one misclassified
spectrum — a single PCA axis doesn't carry enough of the class-relevant
information), while accuracy stays perfect from 2 components all the way
out to about 50, then **degrades again from ~55 components onward**
(96.7%, matching the training set's 90 samples getting uncomfortably close
to the raw 60-channel ceiling) as $p$ approaches $n$ and the within-class
scatter matrix starts to become poorly conditioned again. The lesson for
this particular dataset: LDA is remarkably insensitive to the *exact*
number of PCA components across a wide middle range, but the textbook
under-fitting/over-fitting U-shape is still there at the extremes — it
just takes a much wider sweep than 2–15 to see it, because these
particular classes are unusually well separated.
:::

**Exercise 5 — A genuinely hard classification**

> **Hint:** Move each class's fingerprint a third of the way toward the
> across-class average: `fp_hard = fp + (avg_fp - fp) / 3` for every
> class, where `avg_fp` is the elementwise mean of the three original
> fingerprint arrays — then regenerate the dataset exactly as Section 17.2
> did, with the harder fingerprints in place of the originals.

:::{dropdown} Show full solution
```python
fingerprints_orig = {k: np.array(v) for k, v in class_fingerprints.items()}
avg_fp = np.mean(list(fingerprints_orig.values()), axis=0)
fingerprints_hard = {k: v + (avg_fp - v) / 3 for k, v in fingerprints_orig.items()}

# Regenerate the dataset (identical recipe to Section 17.2, harder fingerprints)
rng_hard = np.random.default_rng(7)
spectra_h, records_h = [], []
for cls, fp in fingerprints_hard.items():
    for _ in range(n_per_class):
        intensities = np.clip(fp * rng_hard.uniform(0.8, 1.2, 5) + rng_hard.normal(0, 0.03, 5), 0, None)
        spectrum = intensities @ band_profiles + rng_hard.normal(0, 0.10, len(wavenumbers))
        conductivity = (2.0 + 3.5*intensities[2] + 2.0*intensities[3] - 1.0*intensities[0]
                         + (1.5 if cls == 'Ether' else 0.0) + rng_hard.normal(0, 0.3))
        spectra_h.append(spectrum)
        records_h.append({'electrolyte_class': cls, 'conductivity_mS_cm': max(0.1, conductivity)})
X_spec_h = np.array(spectra_h)
y_class_h = pd.DataFrame(records_h)['electrolyte_class'].values

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_spec_h, y_class_h, test_size=0.25, random_state=0, stratify=y_class_h)

lb_h = LabelBinarizer().fit(y_train_h)
plsda_h = PLSRegression(n_components=6, scale=True).fit(X_train_h, lb_h.transform(y_train_h))
y_pred_plsda_h = lb_h.classes_[np.argmax(plsda_h.predict(X_test_h), axis=1)]
acc_plsda_h = accuracy_score(y_test_h, y_pred_plsda_h)

lda_pipe_h = Pipeline([('pca', PCA(n_components=6)), ('lda', LinearDiscriminantAnalysis())]).fit(X_train_h, y_train_h)
acc_lda_h = accuracy_score(y_test_h, lda_pipe_h.predict(X_test_h))

print(f'Hard-classification PLS-DA accuracy: {acc_plsda_h:.1%}')
print(f'Hard-classification LDA accuracy:    {acc_lda_h:.1%}')
print(confusion_matrix(y_test_h, y_pred_plsda_h, labels=lb_h.classes_))
print(confusion_matrix(y_test_h, lda_pipe_h.predict(X_test_h), labels=lb_h.classes_))
```
Output: both methods now make real mistakes — PLS-DA drops to **93.3%**
(2 Ether spectra misclassified as Carbonate) and LDA to **96.7%** (1
Carbonate misclassified as Ether), down from 100%/100% on the well-separated
version in Sections 17.5–17.6. LDA still edges out PLS-DA, but the more
telling result is *where* both methods' errors land: every single
misclassification, in both confusion matrices, is a **Carbonate↔Ether**
confusion — never Ionic-Liquid. That's not a coincidence: measuring the
Euclidean distance between the three (now-compressed) fingerprint vectors
shows Carbonate and Ether are the closest pair (distance ≈0.44) while
Ionic-Liquid sits further from both (≈0.54–0.68) — confirming the
exercise's expectation that confusion matrix errors concentrate on
whichever pair of classes was built spectroscopically closest together,
not spread randomly across all three.
:::

**Exercise 6 — Two-response PLS2**

> **Hint:** Build `viscosity_cP` from a *different* combination of the
> same five `intensities` used for the spectrum and for `conductivity` in
> Section 17.2's generation loop (e.g. driven mainly by `intensities[0]`
> and `intensities[4]` rather than `intensities[2]`/`intensities[3]`), stack
> it with `conductivity_mS_cm` into a 2-column `Y`, and rerun the same
> component-selection CV loop from Section 17.3 on that `Y`.

:::{dropdown} Show full solution
```python
# Added inside Section 17.2's generation loop:
# viscosity = 5.0 + 2.0*intensities[0] - 1.5*intensities[4] + 0.8*intensities[1] + rng.normal(0, 0.25)

Y2_train = np.column_stack([y_train, viscosity_train])   # viscosity_train built the same way as y_train

q2_cond_pls2, q2_visc_pls2 = [], []
for nc in range(1, max_components + 1):
    pls2_tmp = PLSRegression(n_components=nc, scale=True)
    Y_cv2 = cross_val_predict(pls2_tmp, X_train, Y2_train, cv=kf)
    q2_cond_pls2.append(r2_score(Y2_train[:, 0], Y_cv2[:, 0]))
    q2_visc_pls2.append(r2_score(Y2_train[:, 1], Y_cv2[:, 1]))

print('nc  Q2(conductivity, PLS1)  Q2(conductivity, PLS2)  Q2(viscosity, PLS2)')
for nc in range(max_components):
    print(nc+1, round(q2_scores[nc], 3), round(q2_cond_pls2[nc], 3), round(q2_visc_pls2[nc], 3))
```
Output: the single-response and two-response component-selection curves for
conductivity are nearly identical — both peak at **2 components**, with
`Q²=0.788` (PLS1) vs. `0.789` (PLS2), and viscosity reaches an even higher
`Q²=0.83` at that same component count. Both curves also degrade the same
way past their peak (steadily falling `Q²` from 3 components onward) as
extra components start fitting calibration-set noise rather than real
signal. The takeaway matches Notebook 15, Exercise 1's PLS2 result:
predicting a second, differently-composed response alongside conductivity
costs this dataset essentially nothing — PLS2 finds components that serve
both responses at once, needing the same number of components as either
response would need on its own.
:::

**Exercise 7 — From your own Notebook 16 dataset**

> **Hint:** Threshold one of your own hidden factors from Notebook 16,
> Exercise 7 (e.g. `class = 'High'` if `F_al[:, 0] > 0` else `'Low'`) to
> turn it into a two-class label, then run this notebook's PLS-DA
> (Section 17.5) and PCA→LDA (Section 17.6) pipelines on your own spectra
> or measurement matrix exactly as written, just swapping in your data and
> label.

:::{dropdown} Show full solution
```python
# Using the aluminium-alloy dataset from Notebook 16, Exercise 7:
# threshold the first hidden factor (Solutionising Quality) into two classes
class_al = np.where(F_al[:, 0] > 0, 'HighSolutionising', 'LowSolutionising')

X_train_al, X_test_al, y_train_al, y_test_al = train_test_split(
    df_al[cols_al].values, class_al, test_size=0.25, random_state=0, stratify=class_al)

# PLS-DA
lb_al = LabelBinarizer().fit(y_train_al)
plsda_al = PLSRegression(n_components=2, scale=True).fit(X_train_al, lb_al.transform(y_train_al))
pred_scores_al = plsda_al.predict(X_test_al)
y_pred_plsda_al = lb_al.classes_[(pred_scores_al.ravel() > 0.5).astype(int)]
print('PLS-DA accuracy:', accuracy_score(y_test_al, y_pred_plsda_al))

# PCA -> LDA
lda_pipe_al = Pipeline([('pca', PCA(n_components=2)), ('lda', LinearDiscriminantAnalysis())]).fit(X_train_al, y_train_al)
print('LDA accuracy:', accuracy_score(y_test_al, lda_pipe_al.predict(X_test_al)))
```
This is a template, not a fixed answer — the point of Exercise 7 is
running the full supervised pipeline end to end on data whose "ground
truth" you already control, exactly as Notebook 16, Exercise 7 did for the
unsupervised PCA/FA pipeline. Running it on the aluminium-alloy example
from Notebook 16 gives `PLS-DA=80.0%`, `LDA=73.3%` — clearly better than
the 50% chance rate for a two-class split, but well short of this
notebook's near-perfect electrolyte results. That is an informative
outcome, not a disappointing one: unlike the electrolyte fingerprints
(deliberately built with several strongly-loading variables per class),
`yield_strength_MPa` was the *only* variable in `loadings_al` with a
non-trivial loading on the thresholded factor (0.80) while carrying a
substantial secondary loading on the other factor too (0.30) — a genuinely
harder classification problem than this notebook's main example, and the
73–80% accuracy reflects that honestly rather than indicating a bug. A
much weaker result (near 50%, chance level) would be the real signal to
revisit the generative recipe — e.g. a thresholded factor with loadings
too small relative to each variable's `unique_var` to leave any usable
class signal at all.
:::



