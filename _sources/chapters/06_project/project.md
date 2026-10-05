---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Course Project: Independent Data Analysis

## Overview

The course project gives you the opportunity to integrate everything you have
learned — from descriptive statistics and hypothesis testing through multivariate
analysis and experimental design — in a realistic materials science context of
your own choosing.

You will:

1. **Select** a publicly available materials science dataset
2. **Explore** it with statistical and multivariate methods
3. **Model** one or more responses using regression, PLS, or DoE methods
4. **Document** your analysis in a reproducible Jupyter Notebook
5. **Present** your findings in a 10-minute talk — this presentation, not a
   written hand-in, is the graded deliverable (see "Presentation Format"
   below)

Optionally, you may also **compare** classical methods with one or more
AI/ML approaches from Part VI of this course (clustering, GPR, ensemble methods,
neural networks).

---

## Learning Outcomes

By completing this project you will have demonstrated the ability to:

- Import, clean, and describe a real dataset (Parts I–III)
- Document that dataset's origin, licence, and variable definitions well
  enough for someone else to reuse it (Part II)
- Apply multivariate analysis and dimensionality reduction (Part IV)
- Design or interpret an existing experimental design and optimise a response
  (Part V)
- Interpret all results in their physical/chemical context
- (Optional) Apply at least one AI/ML method from Part VI and compare its
  performance with a classical approach

---

## Deliverables

There is no written report to hand in. The graded deliverable is a
**10-minute presentation** of your results (see "Presentation Format"
below). That talk is backed by a single **Jupyter Notebook** (`.ipynb`) that a marker can run
top-to-bottom without errors — it is where your analysis actually lives and
is checked for reproducibility, but you will not submit it as a stand-alone
report. Structure the notebook the same way you will structure the talk, so
that preparing one prepares the other:

| Section | Content | Approx. length |
|---|---|---|
| **1. Introduction** | Physical context, research question, why the dataset is relevant | ~0.5 page |
| **2. Data description** | Source, licence, variables, units, sample size (i.e. minimal FAIR metadata — Part II); descriptive statistics and visualisation | ~1 page |
| **3. Hypothesis / question** | Clearly stated hypothesis or optimisation goal | 2–4 sentences |
| **4. Methods** | Methods chosen from the course with justification | ~0.5 page |
| **5. Results** | Figures and tables with captions; all code that produces them | ~2–3 pages |
| **6. Interpretation** | Physical/chemical meaning of findings | ~1 page |
| **7. (Optional) ML comparison** | Apply ≥ 1 ML method, compare metrics to classical approach | ~0.5–1 page |
| **8. Conclusions** | Key findings, limitations, suggested next steps | ~0.5 page |
| **References** | Dataset citation + key methodological references | APA or numeric style |

---

## Presentation Format (10 Minutes)

Ten minutes is short — about the length of one full lap of the notebook
sections above, at roughly **one slide per minute**. The single biggest risk
is running out of time before reaching your results, so the timing below
front-loads the parts a marker can assess in seconds (what did you do, and
why) and protects a solid block for the parts that actually demonstrate your
analysis (what did you find, and what does it mean).

| # | Section | Suggested time | Cumulative | Slides |
|---|---|---|---|---|
| 1 | Introduction & research question | 1:00 | 1:00 | 1 |
| 2 | Data description (source, size, cleaning) | 1:30 | 2:30 | 1–2 |
| 3 | Hypothesis / optimisation goal | 0:30 | 3:00 | 1 (can share a slide with §1) |
| 4 | Methods, with justification | 1:00 | 4:00 | 1 |
| 5 | **Results** — figures, tables, key numbers | 3:30 | 7:30 | 3–4 |
| 6 | Interpretation — physical/chemical meaning | 1:00 | 8:30 | 1 |
| 7 | (Optional) ML comparison | 0:45 | 9:15 | 1 |
| 8 | Conclusions, limitations, next steps | 0:45 | 10:00 | 1 |

If you skip the optional ML comparison (item 7), give its 45 seconds to
Results or Interpretation instead — do not compress the whole talk to leave
a gap. Aim for **8–10 slides total**; a title slide and a "questions/backup"
slide with material you expect to be asked about (a table you didn't have
room for, an alternative model you tried) do not count against the ten
minutes and are worth preparing.

**Delivery tips**:
- Rehearse with a timer at least once, out loud, before the real thing — the
  first read-through is always longer than you expect.
- Show finished plots (exported as static images), not a live-running
  notebook — re-running cells live burns time unpredictably and risks a
  crash mid-talk. Keep the notebook itself as a backup/reproducibility
  artifact, not your presentation surface.
- One clear message per results slide beats a dense grid of subplots; if a
  figure needs more than 15 seconds of explanation to make its point, it
  is probably two slides.
- Practise the transition into Results (end of §4) and out of it (start of
  §6) — that boundary is where talks most often lose the thread under time
  pressure.

---

## Suggested Databases

### 1. Materials Project
**URL**: [materialsproject.org](https://materialsproject.org)  
**Type**: DFT-computed properties of inorganic crystals  
**Size**: >150 000 materials  
**Good for**: band gap prediction, formation energy, elastic constants, PCA of
crystal structure features, clustering by composition  

```python
# Requires: pip install mp-api
from mp_api.client import MPRester

with MPRester("YOUR_API_KEY") as mpr:   # free API key at materialsproject.org
    docs = mpr.summary.search(
        elements=["Fe", "O"],
        fields=["material_id", "formula_pretty", "band_gap",
                "formation_energy_per_atom", "energy_above_hull",
                "density", "volume"]
    )
df_mp = pd.DataFrame([d.dict() for d in docs])
```

**Example questions**:
- Do different iron-oxide polymorphs form distinct clusters in PCA space?
- Can MLR or PLS predict the band gap from structural features?

---

### 2. MatBench — ready-to-use benchmark datasets
**URL**: [matbench.materialsproject.org](https://matbench.materialsproject.org/) /
         [matminer](https://hackingmaterials.lbl.gov/matminer/)  
**Type**: Curated property-prediction datasets (experimental + computed)  
**Access**: `pip install matminer`

Matbench is a suite of 13 benchmark tasks, from 312 steels to 132 752
DFT formation energies, with a public leaderboard of what published models
reach on each one — a useful reality check for your own error numbers.
The tasks, and about 30 further datasets that are not part of the
benchmark, all load through `matminer`:

```python
from matminer.datasets import load_dataset

# Steel strength dataset (312 alloys, 13 alloying elements in wt %)
df_steel = load_dataset("steel_strength")
print(df_steel.head())
print(df_steel.columns.tolist())

# Experimental band gaps (~4600 materials)
df_gap = load_dataset("matbench_expt_gap")

# Refractive index (4764 DFT-computed structures)
df_diel = load_dataset("matbench_dielectric")

# Superconductor critical temperature (~16 000 compounds; a matminer
# dataset, not one of the 13 Matbench tasks)
df_sc = load_dataset("superconductivity2018")
```

| Dataset | n | Features | Response |
|---|---|---|---|
| `steel_strength` | 312 | 13 alloying elements (wt %) | Yield strength, UTS (MPa), elongation (%) |
| `matbench_expt_gap` | 4604 | formula string | Eg (eV) |
| `matbench_dielectric` | 4764 | structure | Refractive index |
| `superconductivity2018` | 16 414 | formula string | Tc (K) |

**Example questions (steel dataset)**:
- Which alloying elements most strongly affect tensile strength? (MLR, PLS)
- Are there compositionally distinct clusters? (K-Means, PCA)
- Fit a surrogate and predict the optimal composition for maximum UTS (GPR, BO)

---

### 3. AFLOW — High-Throughput Computational Materials
**URL**: [aflow.org](http://aflow.org)  
**Access**: REST API or `aflow` Python package  

```python
# pip install aflow
import aflow

# Fetch binary oxides with band gap data
results = (aflow.search()
           .filter(aflow.K.nspecies == 2)
           .filter(aflow.K.Egap > 0)
           .select(aflow.K.compound,
                   aflow.K.Egap,
                   aflow.K.energy_atom,
                   aflow.K.agl_thermal_conductivity_300K,
                   aflow.K.density)
           .orderby(aflow.K.Egap, reverse=True)
           .limit(500))
df_aflow = pd.DataFrame(list(results))
```

---

### 4. UCI Machine Learning Repository — Materials Datasets
**URL**: [archive.ics.uci.edu](https://archive.ics.uci.edu)

Readily accessible via `pandas` or `ucimlrepo`:

```python
# pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

# Concrete compressive strength (1030 samples, 8 mix design + age → strength)
concrete = fetch_ucirepo(id=165)
df_concrete = concrete.data.features.join(concrete.data.targets)

# Steel plates faults (1941 samples, 27 features, 7 fault types)
steel_faults = fetch_ucirepo(id=198)

# Glass identification (214 samples, 9 oxide % → glass type)
glass = fetch_ucirepo(id=42)
df_glass = glass.data.features.join(glass.data.targets)
```

| ID | Dataset | n | Variables | Suggested analysis |
|---|---|---|---|---|
| 165 | Concrete compressive strength | 1030 | 8 mix + age | MLR, RSM, RF |
| 42 | Glass identification | 214 | 9 oxide % | PCA, LDA, clustering |
| 109 | Wine quality (proxy for ceramics QC) | 6497 | 11 physicochemical | PLS, RF, clustering |
| 462 | Corrosion of carbon steel | 1000 | 7 environmental | Regression, BO |

---

### 5. Zenodo & Figshare — Dataset from Published Papers
**URL**: [zenodo.org](https://zenodo.org) | [figshare.com](https://figshare.com)

Search with keywords like *"materials data"*, *"perovskite dataset"*,
*"ionic conductivity dataset"*, etc. Most entries include a DOI-linked CSV.

```python
import pandas as pd

# Example: dataset from a published paper (replace URL with the actual Zenodo link)
url = "https://zenodo.org/record/XXXXXXX/files/dataset.csv"
df_pub = pd.read_csv(url)
```

---

### 6. NIST Standard Reference Database
**URL**: [nist.gov/srd](https://www.nist.gov/srd)

NIST provides thermophysical, mechanical, and spectroscopic data for a
wide range of materials and compounds. Data can be downloaded as CSV or accessed
via the NIST Chemistry WebBook API.

---

### 7. NOMAD — the repository from Part II
**URL**: [nomad-lab.eu](https://nomad-lab.eu)  
**Type**: Open repository of computational (mostly DFT) and experimental
materials data, uploaded by research groups worldwide  
**Access**: REST API with `requests` — no API key, no extra package

You already queried NOMAD in Part II
([Live Tutorial 2](../02_data_handling/07_nomad_api_live_tutorial.ipynb)).
For a project, the same `POST /entries/query` call becomes your data
source: filter on composition and on which properties are available, ask
only for the fields you need, and page through the results.

```python
import requests

NOMAD_URL = "https://nomad-lab.eu/prod/v1/api/v1"

query = {
    "query": {
        "results.material.elements": ["O"],
        "results.material.n_elements": 2,          # strictly binary oxides
        "results.properties.available_properties":
            "electronic.band_structure_electronic.band_gap",
    },
    "pagination": {"page_size": 100},
    "required": {"include": [
        "entry_id",
        "results.material.chemical_formula_reduced",
        "results.material.symmetry.crystal_system",
        "results.properties.structures.structure_original.cell_volume",
        "results.properties.electronic.band_structure_electronic.band_gap.value",
        "results.method.simulation.dft.xc_functional_type",
    ]},
}
page = requests.post(f"{NOMAD_URL}/entries/query", json=query, timeout=60).json()
print(page["pagination"]["total"])     # about 21 000 entries (October 2026)
```

The [Data Access Guide](data_access.ipynb) turns the nested JSON answer
into a flat DataFrame. Three things to handle before you model anything:

- **Units are SI.** Band gaps arrive in joule and volumes in m³ — convert
  to eV and Å³ first.
- **Rows are not independent.** NOMAD stores every upload, so the same
  compound appears many times, computed by different groups with
  different codes and functionals. In a 300-entry sample of the query
  above there were only 25 distinct compounds. Split train and test sets
  **by formula**, exactly as described under "Tips" below.
- **The method is a variable.** LDA and GGA functionals underestimate
  band gaps, so keep `xc_functional_type` as a column: either filter on
  it or use it as a factor.

**Example questions**:
- For the same oxide, how much does the computed band gap differ between
  LDA, GGA, and hybrid functionals? (ANOVA or paired *t*-test)
- Can crystal system, volume per atom, and elemental descriptors predict
  the band gap? (PLS vs Random Forest)

---

### 8. Kaggle
**URL**: [kaggle.com/datasets](https://www.kaggle.com/datasets)  
**Type**: Community-uploaded datasets on every subject; quality varies from
excellent to unusable  
**Access**: `pip install kagglehub` — public datasets download without an
account

```python
import os
import kagglehub
import pandas as pd

# The identifier is the last two parts of the dataset's URL:
# kaggle.com/datasets/munumbutt/superconductor-dataset
path = kagglehub.dataset_download("munumbutt/superconductor-dataset")
df_sc = pd.read_csv(os.path.join(path, "train.csv"))
```

Only datasets that ask you to accept terms first, and all competition
data, need a Kaggle API token
([how to authenticate](https://github.com/Kaggle/kagglehub#authenticate)).

| Identifier | n | Content | Response |
|---|---|---|---|
| `munumbutt/superconductor-dataset` | 21 263 | 81 ready-made elemental features | Tc (K) |
| `vinven7/comprehensive-database-of-minerals` | 3112 | Element counts, crystal structure, molar mass | Mohs hardness, specific gravity, refractive index |
| `patrickfleith/nasa-battery-dataset` | 7565 | Li-ion cell test cycles, one time-series CSV each (228 MB) | Capacity, resistances |
| `uciml/glass` | 214 | 9 oxide % (same data as UCI ID 42) | Glass type |

Anyone can upload to Kaggle, so treat every dataset as unverified until
you have checked it yourself:

- **Find the original source.** Many datasets are re-uploads. The
  superconductor table above is the UCI "Superconductivty Data" set
  (Hamidieh, 2018); cite that paper, not the Kaggle page.
- **Check the licence** on the dataset page before you reuse the data.
- **Look for placeholder values.** In the minerals database about 80 % of
  the Mohs hardness values are `0`, which means *not recorded*, not
  *infinitely soft*. `df.describe()` will not warn you; a histogram will.

**Example questions**:
- The Kaggle superconductor table has ready-made features, and
  `superconductivity2018` from matminer has only formulae. Do your own
  CBFV features (see "Tips" below) predict Tc as well as the published
  ones?
- Does battery capacity fade faster at low ambient temperature? (ANOVA,
  regression on cycle number)

---

### 9. Systembolaget — a product catalogue as a dataset
**URL**: [github.com/C4illin/systembolaget-data](https://github.com/C4illin/systembolaget-data)  
**Type**: Unofficial, community-maintained copy of Systembolaget's product
catalogue, updated daily  
**Size**: about 27 000 products, 81 fields each (October 2026)  
**Access**: one JSON file, no key

Not a materials dataset — but relating the composition of a formulated
product to its properties is the same statistical problem, and this
catalogue is real, large, and far from clean.

```python
import requests
import pandas as pd

# One request returns everything: about 100 MB. Download once, keep the
# columns you need, and save them as a CSV.
r = requests.get("https://susbolaget.emrik.org/v1/products", timeout=300)
df_sb = pd.DataFrame(r.json())
```

| Column | Meaning |
|---|---|
| `categoryLevel1` … `categoryLevel3` | Product type, in Swedish (`Vin`, `Öl`, `Sprit`, …) down to style |
| `alcoholPercentage`, `sugarContentGramPer100ml` | Composition |
| `volume`, `price` | ml and SEK — compute price per litre before comparing products |
| `tasteClockBitter`, `tasteClockBody`, `tasteClockSweetness`, … | Systembolaget's seven taste ratings, scale 1–12 |
| `country`, `vintage`, `packagingLevel1`, `isOrganic` | Origin and packaging |

Two things to handle first:

- **A taste rating of `0` means "not rated".** 60 % of the products,
  mostly wines and spirits that are only sold to order, have no taste
  ratings at all. Replace `0` with `NaN` in those seven columns before
  any PCA or regression.
- **Price is extremely right-skewed** (from 23 to almost 400 000 SEK per
  litre). Model `log(price per litre)`, not the price itself.

**Example questions**:
- Do the taste ratings separate beer styles in a PCA score plot, and can
  LDA classify the style from them?
- How much of the price per litre is explained by alcohol content, sugar,
  country, and packaging? (MLR on the log price)
- Is organic wine more expensive than conventional wine of the same type?
  (*t*-test)

---

### 10. More Places to Look

| Repository | What you find there |
|---|---|
| [JARVIS](https://jarvis.nist.gov) | NIST's database of DFT-computed materials properties; Python access with `jarvis-tools` |
| [OQMD](https://oqmd.org) | DFT formation energies and stabilities for a very large set of inorganic compounds |
| [Crystallography Open Database](https://www.crystallography.net/cod/) | Open-access experimental crystal structures as CIF files |
| [Materials Cloud Archive](https://archive.materialscloud.org) | Datasets published alongside computational-materials papers, each with a DOI |
| [Battery Archive](https://www.batteryarchive.org) | Battery cycling data collected from several laboratories |
| [OpenML](https://www.openml.org) | Tabular machine-learning datasets; loads with `sklearn.datasets.fetch_openml` |
| [Hugging Face Datasets](https://huggingface.co/datasets) | General dataset hub; loads with the `datasets` package |
| [Mendeley Data](https://data.mendeley.com) | General research-data repository, similar to Zenodo |

Whichever source you pick, Section 2 of your notebook has to answer the
same questions: who produced the data, under which licence, and what each
column means.

---

## Suggested Project Topics

Below are five concrete project suggestions to help you get started. You are
welcome to choose your own topic as long as it has a materials science context
and is sufficiently complex (see requirements above).

---

### Project A — Alloy Strength Prediction (steel_strength dataset)
**Dataset**: `matminer` → `steel_strength` (n = 312)  
**Analysis pathway**:
1. Descriptive statistics and pairwise correlation plot
2. PCA to visualise compositional space
3. MLR and PLS to predict UTS; compare Q² values
4. Feature importance: PLS VIP scores vs RF feature importances
5. (Optional) GPR surrogate + BO to find the composition that maximises UTS

---

### Project B — Ceramic Compressive Strength (Concrete dataset)
**Dataset**: UCI ID 165 — concrete compressive strength  
**Analysis pathway**:
1. Exploratory data analysis; identify non-linear relationships
2. Full-factorial subset from the dataset at 2 levels of water/cement and age
3. Fit RSM quadratic model; estimate optimum w/c ratio
4. Compare MLR vs GBM vs RF predictions (cross-validated RMSECV)
5. (Optional) Bayesian optimisation to maximise strength over the design space

---

### Project C — Glass Composition Classification (Glass dataset)
**Dataset**: UCI ID 42 — glass identification  
**Analysis pathway**:
1. Descriptive stats per glass type; box plots of oxide %
2. PCA — do different glass types separate in the score plot?
3. LDA — discriminant functions for classification; confusion matrix
4. Clustering (K-Means, hierarchical) — do clusters match known types?
5. (Optional) Random Forest classification; compare accuracy with LDA

---

### Project D — Materials Project Band Gap Prediction
**Dataset**: Materials Project REST API (band gaps of binary or ternary oxides)  
**Analysis pathway**:
1. Fetch 200–500 compounds; compute simple descriptors (electronegativity
   difference, cation charge, anion radius, …)
2. MLR and PLS on the descriptors vs $E_g$
3. PCA to identify chemical groups
4. RF and GBM regression; compare RMSE with PLS
5. (Optional) GPR for uncertainty-aware prediction

---

### Project E — Superconductor Critical Temperature
**Dataset**: `superconductivity2018` (Tc ≥ 0, ~16 k compounds)  
**Analysis pathway**:
1. Exploratory analysis: distribution of Tc; identify chemical families
2. Compute Magpie-style elemental features with `matminer`
3. PLS regression; select optimal components with cross-validation
4. Gradient Boosting and Random Forest; feature importance
5. (Optional) Clustering by Tc range and elemental family; t-SNE visualisation

---

## Minimum Analysis Requirements

Your project must include **at least four** of the following:

- [ ] Descriptive statistics (mean, std, IQR, pairwise correlation)
- [ ] At least one statistical test (t-test, ANOVA, or $\chi^2$)
- [ ] A regression model (MLR or PLS) with cross-validated $Q^2$
- [ ] PCA or LDA with a properly labelled score plot
- [ ] An experimental design interpretation or optimization (DoE, RSM, or BO)
- [ ] A comparison between at least two modelling approaches with a common metric

If you include an AI/ML method (Part VI), this replaces the optional item and
counts toward the four minimum requirements.

---

## Grading Criteria

| Criterion | Weight | Excellent (A) | Satisfactory (C) |
|---|---|---|---|
| **Reproducibility** | 20 % | Notebook runs without errors; all cells output correct results | Minor issues, all fixable in < 5 min |
| **Depth of analysis** | 25 % | ≥ 4 methods applied, all with cross-validation/significance tests | 3 methods, some without validation |
| **Physical interpretation** | 25 % | Every result is interpreted in terms of materials science | Results presented without context |
| **Visualisation** | 15 % | Publication-quality figures with proper labels, captions | Adequate plots, missing labels |
| **Presentation & structure** | 15 % | All 8 sections covered, on time (±30 s), clear delivery | Most sections present; noticeably over/under time |

---

## Tips: Fast Exploration and Featurization

Two techniques from the course's **Advanced Regression** exercise
([open in Colab](https://colab.research.google.com/github/peterbmob/DHMVADoE/blob/main/Excercises/AdvancedRegression_new.ipynb))
are worth reusing directly in your project — they turn two of the more
time-consuming steps (Section 2: Data description, and turning raw chemical
formulae into model-ready numbers) into a few lines of code.

### Automated EDA with `ydata-profiling`

Instead of hand-writing `.describe()`, histograms, and a correlation
heatmap one at a time for Section 2, generate a full exploratory report in
one call:

```python
# pip install ydata-profiling
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title='Profiling Report', html={'style': {'full_width': True}})
profile.to_notebook_iframe()      # inline in Jupyter/Colab
# profile.to_file('report.html')  # or save as a standalone HTML file
```

In under a minute this gives you missing-value counts, per-variable
distributions, warnings for unrealistic values (negative temperatures,
implausible outliers), and a full pairwise correlation matrix — everything
Section 2 of the notebook template asks for, plus a "high cardinality"
warning that flags, for example, when a composition column has far fewer
unique formulae than rows — exactly the situation that makes the leakage-safe
split below necessary. Export or screenshot the two or three most
informative panels (missing-values overview, top correlations) for your
Data slide rather than the raw report — the report itself is for your own
exploration, not for the audience.

### Featurizing chemical formulae with CBFV

Several suggested datasets (`matbench_expt_gap`, `superconductivity2018`,
and the NIST-JANAF heat-capacity data used in the Advanced Regression
exercise) give you only a **chemical formula string** (e.g. `"Fe2O3"`) as
input — not usable directly by MLR, PLS, PCA, or any `scikit-learn` model,
all of which need numeric features. **CBFV** (composition-based feature
vectors) solves this by converting a formula into a numeric vector using
tables of elemental properties (electronegativity, atomic radius, valence
electron count, …), one of the standard featurization approaches described
in [Machine Learning for Materials Scientists: An Introductory Guide toward
Best Practices](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c01907),
which the Advanced Regression exercise follows step by step.

```bash
# Clone once per session (Colab/Jupyter shell syntax — drop the leading "!"
# for a plain terminal)
!git clone https://github.com/Kaaiian/CBFV.git
```

```python
from CBFV.cbfv.composition import generate_features

# generate_features expects a DataFrame with a 'formula' column and a
# 'target' column -- rename your response column first
df_train = df_train.rename(columns={'Cp': 'target'})

X_train, y_train, formulae_train, skipped_train = generate_features(
    df_train,
    elem_prop='oliynyk',      # or 'magpie', 'jarvis', 'mat2vec', ...
    drop_duplicates=False,    # keep repeated formulae measured under different conditions
    extend_features=True,     # carry extra columns (e.g. temperature) through as features
    sum_feat=True,            # add sum-based, not just mean-based, elemental features
)
```

Two gotchas the exercise notebook demonstrates, both worth avoiding from
the start:

- **Split by formula, not by row, before featurizing.** The same compound
  measured under several conditions (e.g. heat capacity at different
  temperatures) produces multiple rows with the same formula. A plain
  `train_test_split` on rows lets the same compound leak into both the
  training and test sets, so the model can "memorise" it instead of
  generalising — inflating your test score. Get the list of unique
  formulae first, split *that* list into train/val/test, then filter the
  row-level DataFrame by which set each formula landed in.
- **Scale (and optionally normalise) *after* featurizing, fit only on the
  training set.** `StandardScaler().fit_transform()` on `X_train`, then
  `.transform()` (never `.fit_transform()` again) on `X_val`/`X_test` —
  exactly as in Part IV's PCA/PLS notebooks — since fitting the scaler on
  data that includes the validation or test set is another, subtler form
  of leakage.

The full worked example — profiling, leakage-safe splitting, CBFV
featurization, scaling, and comparing Linear/Ridge/Kernel-Ridge regression
on the same heat-capacity dataset — is the Advanced Regression exercise
notebook linked above; it is a reasonable template to adapt directly if
your chosen dataset gives you formulae rather than ready-made descriptors.

---

## Quick-Start Template

The following cell structure is a minimal starting template. Copy it into a new
notebook and adapt to your chosen dataset:

```python
# ── 0. Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_squared_error

sns.set_theme(style='ticks', palette='colorblind')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 11})
rng = np.random.default_rng(2024)

# ── 1. Load data ───────────────────────────────────────────────────────────────
# df = pd.read_csv("your_data.csv")
# OR
# from matminer.datasets import load_dataset
# df = load_dataset("steel_strength")

# ── 2. Descriptive statistics ─────────────────────────────────────────────────
print(df.describe().round(3))
# df.hist(figsize=(12, 8))

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
# X = df[feature_cols].values
# y = df[target_col].values
# scaler = StandardScaler()
# X_sc = scaler.fit_transform(X)

# ── 4. PCA ────────────────────────────────────────────────────────────────────
# pca = PCA()
# T = pca.fit_transform(X_sc)
# ...

# ── 5. PLS regression ─────────────────────────────────────────────────────────
# kf = KFold(n_splits=10, shuffle=True, random_state=42)
# pls = PLSRegression(n_components=nc, scale=False)
# y_cv = cross_val_predict(pls, X_sc, y, cv=kf)
# Q2 = r2_score(y, y_cv)
# RMSECV = np.sqrt(mean_squared_error(y, y_cv))
```
