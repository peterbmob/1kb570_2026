# Theoretical Background: Applied Statistics

This page explains the *ideas* behind Part III's statistical methods before
showing the equations. Statistics can look intimidating on paper, but almost
every method here is answering one of two everyday lab questions: *"Is this
difference I'm seeing real, or could it just be random noise?"* and *"How
sure am I about this number?"* Read each section's intuition first; the
formulas that follow are there so you can check exactly what the software is
computing, not something you need to memorise or derive.

---

## 1. Probability Distributions

### Why distributions matter

Every measurement you take has some random scatter — repeat a hardness test
five times and you get five slightly different numbers, even on nominally
identical samples. A **probability distribution** is a mathematical
description of that scatter: it tells you which values are common, which are
rare, and how far a measurement typically lands from the "true" value.
Choosing the right statistical test almost always starts with asking "what
distribution does this kind of data tend to follow?"

### Normal Distribution — the "bell curve"

Most measurement noise piles up symmetrically around a central value, with
values far from the centre becoming rarer and rarer — the familiar bell
shape. This is the **normal (Gaussian) distribution**, described by just two
numbers: the mean $\mu$ (where the peak sits) and the standard deviation
$\sigma$ (how wide the bell is):

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

You will rarely type this formula yourself — `scipy.stats.norm` computes it —
but it's worth knowing what it encodes: a handy rule of thumb is that about
68% of measurements fall within one $\sigma$ of the mean, 95% within two
$\sigma$, and 99.7% within three $\sigma$ ("the 68–95–99.7 rule"). This is why
a measurement more than about 2 standard deviations from the mean starts to
look unusual, and more than 3 starts to look like an outlier (Notebook 6).

**Why the normal distribution shows up so often** is not a coincidence: the
**Central Limit Theorem** says that if you *average* many independent random
quantities — even ones that individually don't look bell-shaped at all — the
average itself tends toward a normal distribution as you include more
values. Since most lab measurements are themselves an average of many tiny
random influences (thermal noise, minor impurities, instrument jitter), this
is why bell-shaped data turns up everywhere in practice, and why so many
statistical tests are built assuming it.

### Student's t-Distribution — normal distribution's cautious cousin

The normal distribution above assumes you *know* the true spread $\sigma$
exactly. In real experiments you never do — you only have an *estimate* of
it, $s$, computed from a limited sample. This extra layer of uncertainty
makes the sampling distribution of the mean slightly wider-tailed than a
true normal distribution, especially with few observations. The **Student's
t-distribution** accounts for this:

$$t = \frac{\bar{X} - \mu}{s/\sqrt{n}}$$

with $\nu = n-1$ ("degrees of freedom" — roughly, how many independent
pieces of information went into estimating $s$). With few samples ($\nu$
small), the t-distribution has noticeably heavier tails than the normal
distribution — extreme values are more plausible than the normal curve would
suggest, which is exactly the point: it's compensating for the fact that
your estimate of $s$ itself might be off. As $n$ grows, $s$ becomes a more
reliable stand-in for $\sigma$, and the t-distribution converges to the
normal curve. This is the distribution behind every t-test and every
"$\bar{x} \pm t \cdot s/\sqrt{n}$" confidence interval in this part.

### F-Distribution — comparing two amounts of "spread"

Sometimes the question isn't about a mean, but about *variability itself*:
"is group A more variable than group B?" or (in ANOVA) "is the variability
*between* group averages large compared to the variability *within* each
group?" The **F-distribution** describes the ratio of two independent
variance estimates:

$$F = \frac{\chi^2_{\nu_1}/\nu_1}{\chi^2_{\nu_2}/\nu_2}$$

A large $F$ means the numerator's variability is much bigger than the
denominator's — in ANOVA (Section 3 below) this is exactly the signal that
group means differ more than random noise alone would explain.

### Chi-Squared Distribution — the building block for variability

The **chi-squared distribution** describes what happens when you square and
add up several independent, standard-normal quantities:

$$\chi^2 = \sum_{i=1}^k Z_i^2 \sim \chi^2(k)$$

You will meet it mainly as a background piece used to build the F-test and
to construct a confidence interval for a variance — you rarely reason about
it directly, but it explains where "degrees of freedom" bookkeeping in later
sections comes from.

---

## 2. Hypothesis Testing Framework

### The idea: a courtroom, not a calculation

Hypothesis testing borrows its logic from a courtroom: you start by assuming
"innocent until proven guilty" — here, that there is **no real effect**
(the **null hypothesis**, $H_0$) — and only conclude there *is* an effect
(the **alternative hypothesis**, $H_1$) if the data provide strong enough
evidence against $H_0$. You never "prove $H_0$ true"; you only ever fail to
find enough evidence to reject it, exactly like a "not guilty" verdict is
not the same as "proven innocent."

### Two ways to be wrong

Because you're making a decision under uncertainty, two kinds of mistakes
are possible:

| Decision \ Truth | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject $H_0$ | **Type I error** (false alarm) | Correct (you caught a real effect) |
| Fail to reject $H_0$ | Correct | **Type II error** (you missed a real effect) |

- **$\alpha$** (significance level) is the false-alarm rate you're willing to
  accept — conventionally 5%. It's a dial *you* set before looking at the
  data, based on how costly a false alarm would be.
- **Power** ($1-\beta$) is the chance of catching a real effect when one
  truly exists. Power improves with a larger sample size, a bigger true
  effect, or a looser $\alpha$ — Section "Power Analysis" below turns this
  relationship around to answer "how many samples do I need?"

### The p-value — "how surprising is this data, if nothing were going on?"

The **p-value** answers a specific, narrow question: *if $H_0$ were really
true (no effect), how likely would we be to see data at least this extreme,
just by chance?*

$$p = P(|T| \geq |t_\text{obs}| \mid H_0)$$

A small p-value means "this data would be a pretty strange coincidence if
there were really nothing going on" — which is evidence (not proof) that
something *is* going on. The common threshold is to reject $H_0$ when
$p < \alpha$ (usually 0.05). Two things the p-value does **not** tell you:
it is not the probability that $H_0$ is true, and a very small p-value does
not necessarily mean the effect is *large* or *practically* important —
with enough data, even a tiny, unimportant difference can produce a tiny
p-value. Always look at the effect size (e.g. **Cohen's *d*** in the
[t-test section below](#effect-size-cohens-d)) alongside the p-value.

### Confidence intervals — the number *and* its uncertainty

A p-value gives a yes/no verdict; a **confidence interval** gives you a
range of plausible values for the thing you're actually trying to measure,
which is usually more useful:

$$\bar{X} \pm t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}}$$

A 95% confidence interval means: if you repeated this whole experiment many
times, 95% of the intervals you'd compute this way would contain the true
population mean. It directly answers "how precisely do I know this value?"
— a narrow interval is a precise estimate, a wide one says "don't over-trust
this number yet."

### Power analysis — planning before you run the experiment

Before spending time and materials on an experiment, it's worth asking:
*"if there really is an effect of a certain size, will this many
experiments be enough to detect it?"* Rearranging the ingredients of a
hypothesis test (effect size, noise level, $\alpha$, desired power) gives an
estimate of the sample size needed:

$$n \approx \frac{2\sigma^2(z_{\alpha/2} + z_\beta)^2}{\delta^2}$$

In words: you need *more* samples when the noise $\sigma$ is large, and
*fewer* samples when the effect you're looking for, $\delta$, is large — a
small, subtle effect hiding in noisy data is intrinsically harder to detect
and needs a bigger experiment to catch reliably. A common shortcut for a
5%-significance, 80%-power two-sample comparison is $n \approx
16\sigma^2/\delta^2$ per group.

---

## 2b. The t-Test Family — Variants and When to Use Each

### One-sample, two-sample, and paired — what changes?

All three t-tests share the same core logic: build a t-statistic, compare
it to the t-distribution, and get a p-value. What changes is the *question*
being asked — and therefore the way $H_0$ is formulated and the standard
error is calculated.

**One-sample t-test** — compares one group's mean against a single known
target $\mu_0$ (a specification, a reference value, a historical standard):

$$H_0: \mu = \mu_0 \qquad t = \frac{\bar{X} - \mu_0}{s/\sqrt{n}}, \quad \nu = n-1$$

Interpretation: the t-statistic counts how many "standard errors" the sample
mean sits away from the target. If it's large enough, we conclude the
process has drifted from $\mu_0$.

**Two-sample independent t-test** — compares two *separate* groups. $H_0$
now says the two population means are equal:

$$H_0: \mu_1 = \mu_2 \quad (\text{equivalently, } \mu_1 - \mu_2 = 0)$$

The test statistic is the difference between sample means, scaled by the
**pooled standard error** (valid when both groups have similar spread —
checked by Levene's test below):

$$t = \frac{\bar{X}_1 - \bar{X}_2}{s_p\,\sqrt{1/n_1 + 1/n_2}}, \quad
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}, \quad
\nu = n_1+n_2-2$$

**Paired t-test** — used when the *same unit* is measured twice
(before/after a treatment, or matched samples). You compute the
per-unit difference $d_i = X_{i,2} - X_{i,1}$ and run a one-sample
t-test on those differences:

$$H_0: \mu_d = 0 \qquad t = \frac{\bar{d}}{s_d/\sqrt{n}}, \quad \nu = n-1$$

When the two measurements are **positively correlated** — the usual case
for before/after or matched-sample data, since a specimen's own baseline
tends to carry over — subtracting them cancels that shared baseline
variation out, isolating the *treatment effect* from the *between-specimen*
variability and making the paired test more sensitive than treating
"before" and "after" as two independent groups. That gain depends on the
correlation, though: pairing also costs degrees of freedom ($n-1$ instead
of $n_1+n_2-2$), so if the pairing variable turns out to be unrelated to
the outcome, that cost isn't offset by any variance reduction, and the
paired test can end up *less* powerful than the independent-samples
version.

---

### Checking equal variances: Levene's test

The standard two-sample t-test (and one-way ANOVA) rests on an important
assumption: both groups share the same underlying variance. If one process
is much noisier than the other, the pooled standard error is misleading and
the resulting p-value can be unreliable. **Levene's test** checks this
before you commit to a test:

$$H_0: \sigma_1^2 = \sigma_2^2$$

$$W = \frac{N - k}{k - 1} \cdot \frac{\displaystyle\sum_{i=1}^k n_i\,(\bar{Z}_i - \bar{Z})^2}{\displaystyle\sum_{i=1}^k \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_i)^2}$$

where $Z_{ij} = |X_{ij} - \bar{X}_i|$ are the **absolute deviations** from
each group's mean, $k$ is the number of groups, and $N$ is the total
number of observations. By working with absolute deviations rather than
the observations themselves, Levene's test is much less sensitive to
departures from normality than the older Bartlett test.

Practical rule:
- $p > 0.05$: no evidence of unequal variances → proceed with the
  standard (pooled) t-test.
- $p \leq 0.05$: the variances differ significantly → switch to
  **Welch's t-test** (described next).

When in doubt, Welch's test is the safer default because it performs
well whether or not the variances are truly equal.

---

### Welch's t-test — when variances differ

When Levene's test signals unequal variances, the pooled standard
error is no longer valid. **Welch's t-test** relaxes the equal-variance
assumption by estimating each group's variance separately and combining
them only in the denominator:

$$t_W = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{s_1^2/n_1 +\, s_2^2/n_2}}$$

The formula looks almost identical to the standard t-test, but there
is one crucial difference: the **degrees of freedom** can no longer be
the simple $n_1+n_2-2$. Welch uses the *Satterthwaite approximation*
to compute effective degrees of freedom that reflect the imbalance
between the two variance estimates:

$$\nu_W = \frac{\left(s_1^2/n_1 +\, s_2^2/n_2\right)^2}{\dfrac{(s_1^2/n_1)^2}{n_1-1} + \dfrac{(s_2^2/n_2)^2}{n_2-1}}$$

$\nu_W$ is generally not a whole number and will be *smaller* than the
equal-variance degrees of freedom — the test is "paying a precision
penalty" for not assuming equal spread. A smaller $\nu_W$ means a
wider t-distribution, so the critical value is higher and it takes a
larger t-statistic to reach significance. This is the correct price to
pay when your groups genuinely differ in variability.

> **Bottom line**: use Levene's test first. If the p-value from Levene
> is above 0.05, the standard t-test is fine; if it is below 0.05,
> switch to Welch. In `scipy`, this is as simple as setting
> `equal_var=False` in `stats.ttest_ind()`.

---

(effect-size-cohens-d)=
### Effect size: Cohen's *d* — "how big is the difference, really?"

A p-value tells you whether a difference is *detectable* given your
sample size. It does **not** tell you whether the difference is
*practically important*. With enough data, even a trivially tiny
difference eventually becomes "statistically significant." **Cohen's
*d*** solves this by expressing the difference in units of natural
variability — completely independent of sample size:

$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_p}$$

where $s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$
is the same pooled standard deviation used in the two-sample t-test.
For a one-sample test comparing against a known target $\mu_0$:

$$d = \frac{\bar{X} - \mu_0}{s}$$

Cohen (1988) proposed rough benchmarks that have become widely adopted:

| $|d|$ | Informal label |
|---|---|
| < 0.2 | Negligible |
| 0.2 – 0.5 | Small |
| 0.5 – 0.8 | Medium |
| > 0.8 | Large |

These are conventions, not laws. In precision manufacturing, even
$d = 0.3$ might be commercially decisive; in fundamental research with
high scatter, only $d > 1$ may be worth acting on. Treat the
benchmarks as a *first reading*, then apply domain knowledge to judge
whether the effect matters in your specific context.

**Always report Cohen's *d* alongside the p-value**: the p-value
answers *"is this difference real?"* and Cohen's *d* answers *"does
this actually matter?"* — you need both to give a complete and honest
summary of an experiment's result.

---

## 3. Analysis of Variance (ANOVA)

### The idea: is the "between-group" variation more than noise?

A two-sample t-test compares two groups. What if you have three, four, or
more sintering temperatures to compare at once? You could run t-tests on
every pair, but doing many tests inflates your false-alarm rate (testing
enough pairs, *something* will look "significant" by chance alone). ANOVA
instead asks one combined question: *"is the variation between the group
averages bigger than the variation you'd expect from noise within each
group?"*

### Splitting the variation into pieces

ANOVA works by decomposing the total spread of all the data into two
sources: differences *between* group means, and leftover, unexplained
scatter *within* each group:

$$\underbrace{\sum_i \sum_j (y_{ij} - \bar{y})^2}_{\text{SS}_{\text{Total}}} = \underbrace{\sum_i n_i(\bar{y}_i - \bar{y})^2}_{\text{SS}_{\text{Between}}} + \underbrace{\sum_i \sum_j (y_{ij} - \bar{y}_i)^2}_{\text{SS}_{\text{Within}}}$$

("SS" stands for "sum of squares" — a running total of squared deviations,
the same building block used everywhere in statistics to measure spread.)
If the groups really do differ, $\text{SS}_{\text{Between}}$ should be large
relative to $\text{SS}_{\text{Within}}$. Dividing each by its degrees of
freedom turns them into "average variation per source" (mean squares), and
their ratio is exactly an F-statistic:

$$F = \frac{\text{SS}_{\text{Between}}/(k-1)}{\text{SS}_{\text{Within}}/(N-k)} = \frac{\text{MS}_{\text{Between}}}{\text{MS}_{\text{Within}}}$$

A large $F$ means "the groups differ by more than noise alone would
explain" — exactly the F-distribution question from Section 1.

### Two-way ANOVA — when a second factor is also varying

If you change *two* factors at once (say, sintering temperature and
atmosphere), the same idea extends to ask three questions simultaneously:
does temperature matter, does atmosphere matter, and does their *combination*
matter beyond what each does separately (the **interaction**)?

$$y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}$$

The interaction term $(\alpha\beta)_{ij}$ is precisely the same idea as the
interaction effects in Part V's factorial designs — ANOVA is, in fact, the
statistical machinery behind analysing a factorial experiment's results.

---

## 4. Linear Regression: Theory

### The idea

Where ANOVA compares discrete groups, regression describes how a response
changes *continuously* with a factor — a straight line through a scatter of
points. The model says the response is a straight-line trend plus random
scatter:

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$$

### Least squares — why *this* particular line?

Infinitely many lines could be drawn through a scatter of points. **Least
squares** picks the one specific line that makes the total squared vertical
distance between the data and the line as small as possible:

$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

Squaring the distances (rather than, say, just adding them up) does two
useful things: it makes every deviation positive so they don't cancel out,
and it penalises large deviations more heavily than small ones — a single
badly-fit point costs more than several mildly-off points, which nudges the
fitted line to avoid leaving any one observation far away. Under reasonable
conditions (the **Gauss–Markov theorem**), this least-squares line is the
*most precise* unbiased straight-line estimate you can construct from the
data — which is why it's the default choice.

### R² — how much of the story does the line explain?

$$R^2 = 1 - \frac{\text{SS}_{\text{Residual}}}{\text{SS}_{\text{Total}}} = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

Think of $\text{SS}_{\text{Total}}$ as "how much the data varies if you knew
nothing except the overall average," and $\text{SS}_{\text{Residual}}$ as
"how much variation is *still* left after the line does its best." $R^2$ is
the *fraction* of the original variation that the line successfully
accounts for: $R^2=1$ means the line passes through every point exactly;
$R^2=0$ means the line does no better than just guessing the average every
time. **Adjusted R²** applies a small penalty for every extra predictor
added to the model, so it doesn't automatically reward you for throwing in
more variables that don't really help:

$$\bar{R}^2 = 1 - (1 - R^2)\frac{n-1}{n-p-1}$$

### Regression diagnostics — checking the model is trustworthy

A high $R^2$ alone doesn't guarantee the fit is valid — the model's
assumptions need checking too, and this is done visually:

1. **Residuals vs Fitted** — the leftover errors should look like
   patternless noise scattered evenly around zero. A curve or funnel shape
   here means the straight-line assumption or constant-noise assumption is
   being violated.
2. **Q-Q plot** — checks that the residuals are approximately normally
   distributed (points should hug the diagonal reference line).
3. **Scale-Location** — checks that the *size* of the scatter stays
   constant across the fitted range, rather than fanning out.
4. **Cook's Distance** — flags individual points that, if removed, would
   noticeably swing the fitted line — worth a second look, since a single
   unusual measurement can otherwise dominate the whole result.
