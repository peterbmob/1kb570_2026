# Data Handling, Statistics & Design of Experiments for Material Scientists

**Course code:** 1KB570 | **Credits:** 5 hp | **Level:** Second cycle  
**Department of Chemistry–Ångström, Uppsala University**

---

## Welcome

This book is the practical companion for course **1KB570**.  
All concepts are introduced through executable Python notebooks with
materials-science examples — batteries, polymers, alloys, ceramics, and catalysts.

By the end of the course you will be able to:

- Load, clean, and visualise experimental data in Python
- Structure and describe a dataset with FAIR metadata and a shared vocabulary
  of materials-science concepts (EMMO), using NOMAD/NOMAD Oasis as a worked
  example of a FAIR research data infrastructure
- Select and apply the right statistical test for a given dataset
- Build and validate multivariate regression and classification models
- Design efficient experiments (full factorial, fractional factorial, RSM)
- Optimise multi-response processes using DoE and simplex methods

---

## How to use this book

Every chapter contains **executable code cells**.  
You can run them locally after installing the requirements:

```bash
pip install -r requirements.txt
```

Or launch an individual notebook in JupyterLab / Google Colab.

The chapters build on each other, but each notebook is also self-contained —
intermediate learners can jump directly to the topic they need.

---

## Book structure

| Part | Topic | Key tools |
|------|-------|-----------|
| I | Prerequisites — Python for Data Analysis | `numpy`, `pandas`, `matplotlib`, `plotly` |
| II | Data Handling & Materials Ontologies | `json`, `pandas`, `owlready2`, `rdflib` |
| III | Applied Statistics | `scipy.stats`, `statsmodels` |
| IV | Multivariate Analysis | `scikit-learn`, `factor_analyzer` |
| V | Design of Experiments | `pyDOE3`, `scipy.optimize` |
| VI | Introduction to AI and Machine Learning (Extra Material) | `scikit-learn` |

---

## Prerequisites

- Basic programming experience (variables, loops, functions)
- Undergraduate-level mathematics (linear algebra, basic calculus)
- Some familiarity with experimental chemistry or materials science

---

*The course is examined by written tests (3 hp) and laboratory exercises (2 hp).*
