# Part I: Prerequisites — Python for Data Analysis

This part is a **refresher**: it exists so that everyone starts Part II on
equal footing regardless of how recently they last wrote Python. The entry
requirements for 1KB570 already assume prior
programming experience; if you are comfortable with variables, loops,
functions, and basic array/table manipulation, skim the notebooks below and
move straight to [Part II: Data Handling & Materials Ontologies](../02_data_handling/overview.md).
If Python (or the scientific stack specifically) is genuinely new to you,
work through Notebooks 1–5 first — everything from Part II onward assumes
that material. Notebooks 6–7 are practical, not conceptual: how to get an
environment of your own running, and how to keep your work under version
control. They stand alone and are worth reading whenever you first need
them, whether that is today or partway through the course project.

Each tool here solves a specific practical problem: NumPy makes calculations on
whole columns of numbers as easy as calculations on a single value; pandas gives
those columns names so a dataset reads like a labelled table instead of an
anonymous grid; Matplotlib/Seaborn and Plotly turn tables of numbers into
figures you can actually interpret at a glance. Every later part of the course —
data handling, statistics, multivariate analysis, DoE, and machine learning —
is written using exactly these four tools, so time spent here pays off
throughout.

## Notebooks in this chapter

| Notebook | What you learn |
|----------|---------------|
| 1. Python Basics | Variables, control flow, functions, data structures |
| 2. NumPy Arrays | Fast numerical arrays, broadcasting, linear algebra |
| 3. Pandas DataFrames | Tabular data — loading, cleaning, filtering, grouping |
| 4. Matplotlib & Seaborn | Static publication-quality plots |
| 5. Plotly | Interactive charts and 3-D surface plots |
| 6. Environment Setup | Anaconda/Miniconda locally, Google Colab, and UPPMAX — running Python your own way |
| 7. Git & GitHub | Version control, branches, and collaborating on code with others |

## Why Python?

Python is the dominant language for scientific data analysis.
The same code you write in a notebook can be packaged into an automated
analysis pipeline, a web dashboard, or a shared library — with no changes.

Key advantages for materials scientists:
- **numpy / scipy** give you MATLAB-level numerical power
- **pandas** makes spreadsheet-style workflows fully reproducible
- **scikit-learn** provides state-of-the-art machine learning with consistent APIs
- **plotly** generates interactive figures suitable for reports and presentations
