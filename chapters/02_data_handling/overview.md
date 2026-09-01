# Part II: Data Handling & Materials Ontologies

Every method in the rest of this course — a t-test, a PCA, a factorial design —
assumes you already have a clean table of numbers to feed it. In real
materials science, getting to that table is often the hard part: a
composition might live in a colleague's spreadsheet, a synthesis temperature
in a lab notebook, and an XRD pattern in an instrument's proprietary export
format, each using slightly different names and units for the same thing.
This part is about closing that gap — not just cleaning individual files (Part I
already covered that with pandas), but *structuring and describing* data
well enough that it remains usable by other people, other software, and
your own future self.

That is exactly what the **FAIR data principles** — Findable, Accessible,
Interoperable, Reusable — are a checklist for, and exactly the problem an
**ontology** like **EMMO** (the European Materials Modelling Ontology) is
built to solve: it gives every quantity, process, and material a precise,
shared meaning, so that "yield strength" in your dataset means the same
thing as "yield strength" in someone else's. We will use **NOMAD** and
**NOMAD Oasis** — a real, EMMO-aware materials data infrastructure used
across European materials science — as a concrete, worked example of what a
FAIR, ontology-backed database actually looks like on disk and in code,
without requiring you to have your own server to follow along.

Every notebook here answers a question you will eventually ask for real:
*"how do I describe this sample so a colleague — or a piece of software —
interprets it exactly the way I intended?"*, *"what does an 'ontology'
actually look like in code, and why not just use a spreadsheet header?"*,
*"if my data lived in a shared database instead of my laptop, what would
that even look like — and which kind of database?"*, and finally *"how do
I structure my own small dataset the same way, starting today?"*

## Notebooks in this chapter

| Notebook | What you learn |
|----------|---------------|
| 1. FAIR Data & Metadata | FAIR principles; metadata as Python dicts/JSON; saving self-describing data |
| 2. Ontologies & EMMO | What an ontology is; EMMO's structure; browsing it with `owlready2`/`rdflib` |
| 3. Databases: SQL & NoSQL | Relational schemas with `sqlite3`/SQLAlchemy vs. document stores with MongoDB |
| 4. NOMAD & NOMAD Oasis | Anatomy of a research-data repository: uploads, entries, schemas, search |
| 5. Building a FAIR Dataset | Capstone: design an EMMO-informed schema and structure a real dataset with it |
| 6. Live Tutorial 1: SQL & NoSQL | Hands-on session: a live database investigation with `sqlite3`/SQLAlchemy and MongoDB, at lab scale |
| 7. Live Tutorial 2: The NOMAD API | Hands-on session: querying, paginating, and plotting real NOMAD data; API keys |

Notebooks 6 and 7 are both written as self-contained hands-on tutorial
sessions, meant to be run in that order. Notebook 6 stays
entirely local — no network, no account — and puts Notebook 3's SQL/NoSQL
concepts to work on a realistic, lab-scale investigation. Notebook 7 then
connects to the real, public NOMAD repository (no registration needed for
the reading and plotting done there — an account is only required for the
optional API-key section at the end). Both combine Part I's
`numpy`/`pandas`/`matplotlib`/`seaborn`/`plotly` skills with everything
Notebooks 1–5 introduced conceptually.

## Why bother with ontologies and databases at all?

Materials science increasingly runs on *reused* data: a machine-learning
model trained on someone else's alloy dataset, a screening study built on
a public repository of DFT calculations, a meta-analysis across five labs'
battery-cycling data. None of that is possible if every dataset describes
"annealing temperature" differently, or if a dataset simply disappears when
a PhD student graduates. The tools in this part exist to prevent both
failure modes:

| Problem in the lab | Concept that addresses it | Notebook |
|---|---|---|
| "I don't remember what column `T2` meant in this old CSV" | Metadata | 1 |
| "Is 'tensile strength' the same thing as 'ultimate tensile stress'?" | Ontology (EMMO) | 2 |
| "I need my group's data to be queryable and reliable, not just files in a folder" | Database (SQL / NoSQL) | 3 |
| "Where do I actually put data so others (and I) can find it again?" | Repository (NOMAD/Oasis) | 4 |
| "How do I apply all of this to *my* dataset, not just examples?" | Schema design | 5 |

This part is deliberately hands-on and code-first: by the end you will have
written Python that builds, validates, and queries a small FAIR,
ontology-tagged dataset of your own — the same skills used later when you
select and clean data for the [course project](../06_project/project.md).
