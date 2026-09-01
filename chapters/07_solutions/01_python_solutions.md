# Part I Solutions: Python for Data Analysis

Solutions to the exercises in Notebooks 1–5. Code blocks below are written
to be pasted directly into the corresponding notebook (they assume its
imports and variables already exist in your session) — they are not
executed as part of building this page.

## Notebook 1: Python Basics

**Exercise 1 — Molar mass calculator**

> **Hint:** Loop over the dictionary's `(element, count)` pairs and look
> each element up in `ATOMIC_MASS` — this is exactly the sum-of-products
> pattern already used for `MM_LiFePO4` in Section 1.1, just generalised
> to any formula instead of one hard-coded example.

:::{dropdown} Show full solution
```python
ATOMIC_MASS = {'Li': 6.94, 'Ni': 58.69, 'Mn': 54.94, 'Co': 58.93,
               'O': 16.00, 'Fe': 55.85, 'P': 30.97}

def molar_mass(formula_dict):
    """Sum atomic_mass * count over every element in the formula."""
    return sum(ATOMIC_MASS[element] * count for element, count in formula_dict.items())

nmc811 = {'Li': 1, 'Ni': 0.8, 'Mn': 0.1, 'Co': 0.1, 'O': 2}
print(f'NMC811 molar mass: {molar_mass(nmc811):.3f} g/mol')
```
Output: `NMC811 molar mass: 97.279 g/mol`. This generalises the pattern from
Section 1.1's `MM_LiFePO4` calculation — a `for` loop (or, equivalently, the
generator expression inside `sum()`) is what lets the function handle *any*
formula dict, not just the one hard-coded case.
:::

**Exercise 2 — Top-N filter**

> **Hint:** The list comprehension needs two conditions joined with `and`
> inside the `if` clause — `[x for x in seq if cond1 and cond2]`.

:::{dropdown} Show full solution
```python
samples = [
    {'id': 'S01', 'T_C': 700, 'time_h': 6,  'capacity': 148},
    {'id': 'S02', 'T_C': 750, 'time_h': 6,  'capacity': 161},
    {'id': 'S03', 'T_C': 800, 'time_h': 6,  'capacity': 169},
    {'id': 'S04', 'T_C': 800, 'time_h': 12, 'capacity': 172},
]
top = [s['id'] for s in samples if s['T_C'] >= 750 and s['capacity'] > 160]
print('Filtered:', top)
```
Output: `Filtered: ['S02', 'S03', 'S04']` — S01 fails on both conditions
(700°C and 148 mAh/g), the other three pass both.
:::

**Exercise 3 — Arrhenius**

> **Hint:** Convert `T_C` to Kelvin and `Ea_kJ` to J/mol before plugging into
> $k = A\exp(-E_a/RT)$ — mixing kJ and J (or °C and K) is the single most
> common bug in this kind of formula.

:::{dropdown} Show full solution
```python
import math

def k_arrhenius(A, Ea_kJ, T_C):
    R = 8.314          # J mol^-1 K^-1
    T_K = T_C + 273.15
    Ea_J = Ea_kJ * 1000
    return A * math.exp(-Ea_J / (R * T_K))

for T in [25, 100, 200, 400]:
    print(f'T={T}°C: k={k_arrhenius(1e13, 80, T):.4e}')
```
Output:
```
T=25°C:  k=9.6344e-02
T=100°C: k=6.3235e+01
T=200°C: k=1.4719e+04
T=400°C: k=6.1943e+06
```
The rate constant spans **8 orders of magnitude** across this temperature
range — the whole reason Arrhenius plots use a log scale, and a preview of
why sintering temperature (Notebook 16 onward) is almost always the
dominant factor in a materials-synthesis DoE study.
:::

## Notebook 2: NumPy Arrays

**Exercise 1 — Crystal density**

> **Hint:** Density = (mass of Z formula units) / (unit cell volume).
> Convert Å to cm ($1\,\text{Å}=10^{-8}\,\text{cm}$) before cubing —
> forgetting this conversion is the classic mistake here.

:::{dropdown} Show full solution
```python
import numpy as np

a_cm = 5.64e-8            # Å -> cm
Z = 4
M_NaCl = 22.99 + 35.45     # g/mol
N_A = 6.022e23

V_cell = a_cm**3
density = Z * M_NaCl / (N_A * V_cell)
print(f'NaCl density: {density:.3f} g/cm3')
```
Output: `NaCl density: 2.164 g/cm3` — close to the accepted literature value
(2.17 g/cm³); the small gap is from the rounded atomic masses and lattice
parameter used here.
:::

**Exercise 2 — Rolling mean**

> **Hint:** `np.convolve(capacity, kernel, mode='valid')` slides a length-10
> averaging kernel (`np.ones(10)/10`) across the array — `mode='valid'`
> only keeps positions where the kernel fully overlaps the data, so the
> output is 9 elements shorter than the input.

:::{dropdown} Show full solution
```python
import numpy as np

rng = np.random.default_rng(7)
cycle_numbers = np.arange(1, 101)
capacity = 170.0 * np.exp(-0.0015 * cycle_numbers) + rng.normal(0, 0.5, 100)

kernel = np.ones(10) / 10
rolling_mean = np.convolve(capacity, kernel, mode='valid')
print('Length:', len(rolling_mean), '(100 - 10 + 1 = 91)')
print('First 3:', rolling_mean[:3].round(3))
print('Last 3: ', rolling_mean[-3:].round(3))
```
Output: 91 values, from `[168.50, 168.28, 168.03]` near cycle 1 down to
`[147.90, 147.53, 147.18]` near cycle 100 — a visibly smoother version of
the noisy raw `capacity` array, tracking the same exponential fade with
the cycle-to-cycle noise averaged out.
:::

**Exercise 3 — Correlation matrix**

> **Hint:** `np.corrcoef` expects variables as *rows* by default — since
> `steel_data`'s columns are the properties, pass `rowvar=False` or you'll
> get the correlation between *samples* instead of between *properties*.

:::{dropdown} Show full solution
```python
import numpy as np

steel_data = np.array([
    [7.85, 200, 250, 180],
    [7.80, 210, 550, 320],
    [7.75, 193, 520, 300],
    [7.85, 205, 620, 380],
])
names = ['density', 'E_mod', 'yield_str', 'hardness']
corr = np.corrcoef(steel_data, rowvar=False)
print(corr.round(3))
```
Output:
```
[[ 1.     0.48  -0.29  -0.145]
 [ 0.48   1.     0.306  0.351]
 [-0.29   0.306  1.     0.988]
 [-0.145  0.351  0.988  1.   ]]
```
**Yield strength and hardness are by far the most correlated pair**
(r = 0.988) — physically sensible, since both properties respond to the
same underlying microstructural strengthening mechanisms (this small,
4-sample toy dataset happens to make that relationship almost perfectly
linear; real data would show more scatter around it).
:::

## Notebook 3: Pandas DataFrames

**Exercise 1 — CSV round-trip**

> **Hint:** `to_csv`/`read_csv` round-trips values correctly, but not
> necessarily *dtypes* — an integer column with no missing values usually
> survives, but check with `df.dtypes` before comparing with `.equals()`.

:::{dropdown} Show full solution
```python
df.to_csv('alloys.csv', index=False)
df_read = pd.read_csv('alloys.csv')

print('Equal (raw):', df.equals(df_read))
print(df.dtypes.equals(df_read.dtypes))  # check if dtypes actually match

# If dtypes drifted (e.g. int64 -> int64 is usually fine, but mixed columns
# can differ), align them explicitly before comparing:
df_read = df_read.astype(df.dtypes.to_dict())
print('Equal (after dtype alignment):', df.equals(df_read))
```
For this particular dataset (no missing values, no mixed types) the raw
round-trip already matches — the dtype-alignment step is the one to reach
for when it doesn't, which becomes common once a column has any `NaN`
(forcing an int column to become float on read-back).
:::

**Exercise 2 — Specific strength**

> **Hint:** `Series.map(dict)` looks up each row's `alloy` value in the
> dictionary and returns the matching density — a vectorised alternative to
> looping row by row.

:::{dropdown} Show full solution
```python
density_dict = {'304SS': 8.0, '316SS': 8.0, 'Mild': 7.85, 'ToolS': 7.85}
df['specific_strength'] = df['tensile_MPa'] / df['alloy'].map(density_dict)
print(df[['sample_id', 'alloy', 'tensile_MPa', 'specific_strength']].round(2))
```
Output ranges from 50.96 (A05, Mild, annealed) up to 152.87
(A08, ToolS, Q&T) — tool steel's combination of high tensile strength and
unremarkable density gives it by far the best strength-to-weight ratio of
the four alloys here.
:::

**Exercise 3 — Q-factor**

> **Hint:** Build the ratio column, then combine `alloy` and
> `heat_treatment` into one label (e.g. string concatenation) before
> sorting, so each combination is identifiable in the output.

:::{dropdown} Show full solution
```python
df['Q'] = df['tensile_MPa'] / df['hardness_HV']
df['combo'] = df['alloy'] + ' ' + df['heat_treatment']
print(df[['combo', 'Q']].sort_values('Q', ascending=False).round(3))
```
**Highest Q: ToolS Annealed (3.455).** **Lowest Q: 304SS Annealed (2.861).**
Q is highest for annealed tool steel and lowest for annealed 304 stainless —
a materially meaningful result: Q ≈ tensile/hardness roughly tracks how much
tensile strength each hardness point "buys" you, and it drops noticeably
after quench-and-temper for every alloy here except 316SS, where the two
treatments land almost the same (3.031 vs 3.304).
:::

## Notebook 4: Matplotlib & Seaborn

**Exercise 1 — XRD pattern**

> **Hint:** Reuse Notebook 1 §1.4's `interplanar_spacing_cubic` function to
> get each peak's 2θ position, then build a fine 2θ grid and sum a
> Gaussian (via `scipy.stats.norm.pdf`) centred at each peak — `ax.vlines`
> alone only gives sharp lines, not the broadened peaks the exercise asks for.

:::{dropdown} Show full solution
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

def interplanar_spacing_cubic(h, k, l, a):
    d = a / math.sqrt(h**2 + k**2 + l**2)
    lambda_CuKa = 1.5406
    sin_theta = lambda_CuKa / (2 * d)
    two_theta = math.degrees(2 * math.asin(sin_theta))
    return d, two_theta

a_Au = 4.078
planes = [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2)]
peaks_2theta = [interplanar_spacing_cubic(*p, a=a_Au)[1] for p in planes]
print([f'{p:.2f}' for p in peaks_2theta])
# ['38.19', '44.39', '64.59', '77.58', '81.74']

two_theta_grid = np.linspace(30, 90, 2000)
fwhm = 0.3
sigma = fwhm / 2.3548   # FWHM -> Gaussian sigma

pattern = np.zeros_like(two_theta_grid)
for peak in peaks_2theta:
    pattern += norm.pdf(two_theta_grid, loc=peak, scale=sigma)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(two_theta_grid, pattern, color='steelblue', lw=1.2)
ax.vlines(peaks_2theta, 0, pattern.max()*1.05, color='crimson', lw=0.8, ls=':', alpha=0.6)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensity (a.u.)')
ax.set_title('Simulated XRD Pattern — FCC Gold (Cu Kα)')
plt.tight_layout()
plt.show()
```
The five peaks land at 38.19°, 44.39°, 64.59°, 77.58°, and 81.74° — matching
gold's well-known FCC powder pattern order (111 strongest and lowest-angle,
higher-index planes progressively weaker and further apart in this
simplified equal-intensity version).
:::

**Exercise 2 — Seaborn violin plot**

> **Hint:** `sns.violinplot` takes the same long-form `x=`/`y=` arguments as
> `sns.boxplot` — build a small two-column DataFrame (`size_nm`, `route`)
> exactly like the KDE example in Section 4.3 already does.

:::{dropdown} Show full solution
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_ps = pd.DataFrame({
    'size_nm': np.concatenate([route_A, route_B]),
    'route': ['A'] * len(route_A) + ['B'] * len(route_B),
})

fig, ax = plt.subplots(figsize=(6, 4))
sns.violinplot(data=df_ps, x='route', y='size_nm', hue='route',
               palette='colorblind', legend=False, ax=ax)
ax.set_xlabel('Synthesis route')
ax.set_ylabel('Particle size (nm)')
ax.set_title('Particle Size Distribution — Violin Plot')
sns.despine(ax=ax)
plt.tight_layout()
plt.show()
```
A violin plot shows the same information as the histogram/KDE pair in
Section 4.3 (Route A wider and lower, Route B narrower and higher) but adds
an explicit box-and-whisker summary inside each violin — useful when you
want both the distribution shape *and* the quartiles in one glance.
:::

**Exercise 3 — Save figure**

> **Hint:** Call `fig.savefig(...)` twice, once per format, right after
> `plt.tight_layout()` and before `plt.show()` — calling `savefig` after
> `show()` sometimes saves a blank figure depending on backend.

:::{dropdown} Show full solution
```python
# ... build the 3-panel figure exactly as in Section 4.7, then:
plt.tight_layout()
fig.savefig('sintering_study.png', dpi=300, bbox_inches='tight')
fig.savefig('sintering_study.pdf', bbox_inches='tight')
plt.show()
```
`dpi=300` only matters for the raster PNG — PDF is a vector format and
renders at whatever resolution the viewer requests, so `dpi` has no effect
there. `bbox_inches='tight'` on both calls trims the extra whitespace
`tight_layout()` doesn't always fully remove around the saved edges.
:::

## Notebook 5: Interactive Visualization with Plotly

**Exercise 1 — Parallel coordinates plot**

> **Hint:** `px.parallel_coordinates` needs numeric columns only (drop or
> encode `Grade`) — pass `color='Tensile_MPa'` separately to still see the
> grade-related structure via colour, without a non-numeric axis.

:::{dropdown} Show full solution
```python
import plotly.express as px

fig = px.parallel_coordinates(
    df_poly,
    dimensions=['Tensile_MPa', 'E_Modulus_MPa', 'Density_g_cm3', 'Tg_C'],
    color='Tensile_MPa',
    color_continuous_scale='Viridis',
    title='Parallel Coordinates — Polymer Properties',
)
fig.show()
```
Each polymer grade traces its own path across the four axes; PEEK's much
higher tensile strength and modulus show up as lines consistently at the
top of those two axes, visually separating it from the commodity polymers
(HDPE/LDPE/PP) clustered lower down — the kind of multi-variable-at-once
view a 2-D scatter plot can't give you.
:::

**Exercise 2 — Animated scatter**

> **Hint:** `animation_frame` needs one categorical column to step through
> — use `Chemistry` directly, no `reset_index` needed here since `df_cyc`
> is already tidy (one row per cycle per chemistry).

:::{dropdown} Show full solution
```python
import plotly.express as px

fig = px.scatter(
    df_cyc, x='Cycle', y='Capacity_mAh_g',
    animation_frame='Chemistry',
    range_y=[df_cyc['Capacity_mAh_g'].min() - 2, df_cyc['Capacity_mAh_g'].max() + 2],
    title='Capacity vs Cycle — Animated by Chemistry',
    template='plotly_white',
)
fig.show()
```
Note the fixed `range_y` — without it, Plotly rescales the y-axis for every
animation frame, which makes it look like every chemistry has similar fade
behaviour even though their decay rates (`k`) differ by 7×. A shared,
fixed axis range is what actually lets you compare frames fairly.
:::

**Exercise 3 — Contour plot**

> **Hint:** `go.Contour` wants the same `x`, `y`, `z` shapes as
> `go.Surface` already built in Section 5.4 — reuse `T_grid`, `P_grid`,
> `Y_surface` directly, just swap the trace type.

:::{dropdown} Show full solution
```python
import plotly.graph_objects as go

fig = go.Figure(data=go.Contour(
    x=T_grid, y=P_grid, z=Y_surface,
    colorscale='Viridis',
    contours=dict(showlabels=True),
    colorbar=dict(title='Yield (%)'),
))
fig.update_layout(
    title='Reaction Yield — Contour Map',
    xaxis_title='Temperature (°C)',
    yaxis_title='Pressure (bar)',
    template='plotly_white',
)
fig.show()
```
A contour plot is the "looking straight down" version of Section 5.4's 3-D
surface — harder to build spatial intuition from at a glance, but far
easier to read off a specific optimum (T, P) pair precisely, which is
exactly the trade-off DoE response-surface plots in Part V make
repeatedly (compare to Notebook 19's contour plots).
:::
