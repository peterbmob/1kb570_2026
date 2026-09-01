# Part II Solutions: Data Handling & Materials Ontologies

## Notebook 1: FAIR Data & Metadata

**Exercise 1 — Fill the gaps**

> **Hint:** `fair_check`'s six required fields are `identifier`, `title`,
> `licence`, `creator`, `created`, `method` — `measurement` (Section 1.3)
> is missing most of them. Add each as a top-level key, following
> `cathode_record`'s style for the values.

:::{dropdown} Show full solution
```python
import uuid
from datetime import date

measurement['identifier'] = f'urn:uuid:{uuid.uuid4()}'
measurement['title']      = 'Vickers hardness measurement, sample STL-2024-014'
measurement['licence']    = 'CC-BY-4.0'
measurement['method']     = 'Vickers hardness testing, HV10'
# 'creator' and 'created' both already exist under different names
# ('operator', 'date') -- fair_check looks for the exact keys, so add them:
measurement['creator'] = measurement['operator']
measurement['created'] = measurement['date']

print_fair_report(measurement, 'measurement')
```
`5/6 -> 6/6`. Notice the two near-misses this exposes: `operator`/`date`
already held the *right information*, just under field names `fair_check`
wasn't looking for — a good reminder that "the data is there" and "the
data is Findable/Reusable by a generic checker" are not automatically the
same thing.
:::

**Exercise 2 — Round trip with a list**

> **Hint:** `json.dump` handles a list of dicts exactly like a single
> dict — no special syntax needed, just pass the list as the object to save.

:::{dropdown} Show full solution
```python
import json

three_measurements = [
    {**measurement, 'sample_id': 'STL-2024-014'},
    {**measurement, 'sample_id': 'STL-2024-015', 'value': 191},
    {**measurement, 'sample_id': 'STL-2024-016', 'value': 179},
]

with open('measurements.json', 'w', encoding='utf-8') as f:
    json.dump(three_measurements, f, indent=2)

with open('measurements.json', encoding='utf-8') as f:
    reloaded = json.load(f)

print(reloaded == three_measurements)   # True
```
The dictionary-unpacking (`{**measurement, ...}`) trick builds three
variants sharing most fields but overriding `sample_id`/`value` — the same
"don't repeat yourself" instinct behind Notebook 1's own `make_entry`-style
helper functions later in this part.
:::

**Exercise 3 — Extend the checklist**

> **Hint:** The existing dict comprehension only checks "is the field
> present and non-empty" — a keyword-count check needs a small helper
> function instead, since it's a genuinely different kind of test.

:::{dropdown} Show full solution
```python
def fair_check_extended(record, required_fields=('identifier', 'title', 'licence',
                                                   'creator', 'created', 'method')):
    report = {field: (field in record and record[field] not in (None, '', []))
              for field in required_fields}
    # Special-case check: keywords needs *at least 2* entries, not just "present"
    keywords = record.get('keywords', [])
    report['keywords (>=2)'] = len(keywords) >= 2
    return report

print_fair_report_ext = lambda record, name: [
    print(f'FAIR self-check for {name}:'),
    *[print(f"  {'✓' if ok else '✗ MISSING'}  {field}")
      for field, ok in fair_check_extended(record).items()]
]
print_fair_report_ext(cathode_record, 'cathode_record')
```
`cathode_record` passes (`keywords` has 4 entries); re-run on the
`measurement` dict from Exercise 1 and it fails this new check (no
`keywords` field at all) even though it now passes the original six —
exactly the kind of gap a real FAIR audit is meant to surface.
:::

## Notebook 2: Ontologies & EMMO

**Exercise 1 — Extend the toy ontology**

> **Hint:** `Alloy` and `TensileStrength` are just two more `class ... (Parent):`
> blocks inside the same `with onto:` block, and `hasTensileStrength`
> follows `hasMeasuredCapacity`'s exact pattern — a relation class with
> `is_a = [hasQuantitativeProperty]`.

:::{dropdown} Show full solution
```python
with onto:
    class Alloy(MaterialObject):
        pass

    class TensileStrength(Quantity):
        pass

    class hasTensileStrength(MaterialObject >> TensileStrength):
        is_a = [hasQuantitativeProperty]

steel_sample = Alloy('steel_batch_12')
uts = TensileStrength('UTS_batch_12')
steel_sample.hasTensileStrength.append(uts)

print(f'{steel_sample.name} is a {[c.name for c in steel_sample.is_a]}')
print(f'{steel_sample.name} hasTensileStrength -> '
      f'{[t.name for t in steel_sample.hasTensileStrength]}')
```
This mirrors `CathodeMaterial`/`hasMeasuredCapacity` exactly — the whole
point of the exercise is noticing that adding a new material/quantity pair
to this toy ontology is a completely mechanical, repeatable pattern once
you've done it once.
:::

**Exercise 2 — Ancestors of your own choice**

> **Hint:** Call `find_class_by_label(emmo, "Alloy")` first and check
> whether it returns `None` before calling `ancestor_chain` on it — not
> every everyday materials-science word necessarily has an exact-label
> match in EMMO's *foundational* vocabulary.

:::{dropdown} Show full solution
```python
for label in ['Alloy', 'Density']:
    iri = find_class_by_label(emmo, label)
    print(f'{label} -> {iri}')
    if iri:
        print('  ancestors:', ancestor_chain(emmo, iri))
    else:
        print('  NOT FOUND (no exact label match in core EMMO)')
```
Running this for real against the live ontology: **`"Alloy"` returns no
match at all** — not even a partial one (searching every label in EMMO for
the substring "alloy" turns up nothing). This is a genuinely useful,
non-obvious result, and it is exactly what Section 2.4's own text warned
you to expect: EMMO is a *foundational* ontology, and domain-specific
materials-science vocabulary like "alloy" belongs in a domain ontology
built on top of it (the same role BattINFO plays for batteries) — at the
time of writing, no such term exists in EMMO core.

**`"Density"`, by contrast, resolves cleanly** and climbs through
`Quantity → ... → Sign → SemioticEntity → Semiotics` — the **same
semiotic branch as `SpecificHeatCapacity`** in Section 2.6's worked
example, not the `PhysicalObject` branch `Crystal` climbed. That's the
expected pattern: any physical *quantity* (a number-with-a-unit) lands in
the Semiotics branch as a "sign," while physical *objects* and *materials*
land in the PhysicalObject/Material branch instead — density is a
quantity, so it follows density's cousins, not `Crystal`'s.
:::

**Exercise 3 — Compare vocabularies**

> **Hint:** Go through the FAIR table letter by letter and ask what a
> plain string like `'temperature_C'` *can't* do that an EMMO IRI like
> `'emmo:ThermodynamicTemperature'` can.

:::{dropdown} Show full solution
- **Findable**: An EMMO IRI is a globally unique, resolvable identifier —
  a search across many datasets for "everything tagged
  `emmo:ThermodynamicTemperature`" works regardless of what each dataset's
  own column happened to be named; a plain string like `'temperature_C'`
  only matches other datasets using that exact same string. **Easier.**
- **Accessible**: Largely unaffected either way — accessibility is about
  the retrieval protocol (HTTP, licence terms), not the vocabulary used
  for individual field names. **No real change.**
- **Interoperable**: This is the letter EMMO IRIs exist for — two datasets
  using `emmo:ThermodynamicTemperature` under two completely different
  field names (`temperature_C` vs. `T_anneal`, exactly Notebook 4 §4.4's
  example) can still be combined automatically by software; two datasets
  each using their own free-text string cannot, without a human manually
  reconciling the names first. **Much easier.**
- **Reusable**: An EMMO IRI documents *exactly* what a field means,
  disambiguating it from every homonym/synonym problem Section 2.1 opened
  with — a future reader (or your own code, six months later) doesn't have
  to guess whether "conductivity" meant electrical or thermal. **Easier**,
  though it does add the (one-time) cost of learning enough EMMO to pick
  the right concept in the first place — the honest trade-off the "Requires
  more upfront modelling effort" row in Section 2.3's table already named.
:::

## Notebook 3: Databases for FAIR Data — SQL and Document Stores

**Exercise 1 — Extend the relational schema**

> **Hint:** This is the exact `instruments` table Notebook 6 (the live
> tutorial) later builds "for real" — a fourth table plus one new
> `instrument_id` foreign-key column on `synthesis_steps`, then a `JOIN`
> across all three.

:::{dropdown} Show full solution
```python
conn.executescript('''
CREATE TABLE instruments (
    instrument_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    serial_number TEXT
);
ALTER TABLE synthesis_steps ADD COLUMN instrument_id INTEGER REFERENCES instruments(instrument_id);
''')

cur.execute("INSERT INTO instruments (instrument_id, name, serial_number) VALUES (1, 'Furnace-A', 'FA-2201')")
cur.execute("UPDATE synthesis_steps SET instrument_id = 1 WHERE sample_id = 'batch_05' AND step_name = 'calcination_2'")
cur.execute("UPDATE synthesis_steps SET instrument_id = 1 WHERE sample_id = 'batch_06' AND step_name = 'calcination_2'")
conn.commit()

df_instr = pd.read_sql_query('''
    SELECT st.sample_id, st.step_name, i.name AS instrument
    FROM synthesis_steps st
    JOIN instruments i ON i.instrument_id = st.instrument_id
''', conn)
print(df_instr)
```
Note `ALTER TABLE ... ADD COLUMN` (not a full `CREATE TABLE`) — SQLite
supports adding a single nullable column to an existing table without
recreating it, which is exactly what you need here since `synthesis_steps`
already has 15 rows in it from Section 3.3.
:::

**Exercise 2 — Break the document schema on purpose**

> **Hint:** Nothing stops the insert — the real question is what happens
> to the *query*, not the insert.

:::{dropdown} Show full solution
```python
samples_collection.insert_one({
    'sample_id': 'batch_typo',
    'material': 'LiFePO4',
    'metod': 'sol-gel',    # typo: should be 'method'
})

hits = list(samples_collection.find({'method': 'sol-gel'}, {'_id': 0, 'sample_id': 1}))
print(hits)   # batch_typo is NOT in this list
```
The insert succeeds silently — no schema to violate — but `batch_typo`
**never shows up** in the `find({'method': 'sol-gel'})` query, because
MongoDB is matching the literal field name `method`, and this document has
`metod` instead. The database has no way to know these were meant to be
the same field. Catching this requires *application-level* validation
before every insert — exactly the `validate_entry`-style structural
check Notebook 4 introduces, checking that only an allowed set of field
names appears in each document.
:::

**Exercise 3 — Design question**

> **Hint:** Which of Section 3.7's three "Best for" rows matches
> "same event type, but a genuinely different, chemistry-dependent set of
> fields per record"?

:::{dropdown} Show full solution
This is a **document-database** problem, not a relational one. The whole
premise — "different battery chemistries report different diagnostic
fields" — is precisely the "heterogeneous, nested, evolving data" case
Section 3.7's table assigns to document stores: a rigid SQL table would
either need a separate `internal_resistance` *and* `na_ion_diagnostic`
column with one of them always `NULL` (wasteful, and brittle every time a
new chemistry is added), or a much more awkward key-value side table. A
document collection instead lets a Li-ion cycle record and a Na-ion cycle
record share the common fields (cycle number, voltage, capacity) while
each carrying its own chemistry-specific fields naturally, with no schema
migration needed when a third chemistry shows up next year. It would only
graduate to a NOMAD Oasis if this cycler log needed to be FAIR, EMMO-tagged,
citable research data shared beyond the lab's own database — a genuine
possibility eventually, but not what the question as posed is actually
asking about (a working internal log, not a publication-ready archive).
:::

## Notebook 4: NOMAD & NOMAD Oasis

**Exercise 1 — Extend the local repository**

> **Hint:** `query_entries` needs a new parameter (e.g. `min_efficiency`)
> following the exact pattern its existing `min_capacity` parameter
> already uses.

:::{dropdown} Show full solution
```python
local_repository += [
    make_entry('batch_10', 'solid-state reaction', 710, 150.5, 0.935),
    make_entry('batch_11', 'sol-gel',              660, 174.0, 0.96),
    make_entry('batch_12', 'solid-state reaction', 740, 158.0, 0.90),
]

def query_entries(entries, method=None, min_capacity=None, min_efficiency=None):
    results = entries
    if method is not None:
        results = [e for e in results if e['archive']['data']['synthesis']['method'] == method]
    if min_capacity is not None:
        results = [e for e in results
                   if e['archive']['data']['results']['capacity_mAh_g'] >= min_capacity]
    if min_efficiency is not None:
        results = [e for e in results
                   if e['archive']['data']['results']['coulombic_efficiency'] >= min_efficiency]
    return results

hits = query_entries(local_repository, min_efficiency=0.93)
print(f'{len(hits)} entries with coulombic_efficiency >= 0.93 (any method)')
for e in hits:
    d = e['archive']['data']
    print(f"  {d['name']:<10} eff={d['results']['coulombic_efficiency']}  method={d['synthesis']['method']}")
```
Of the 8 total entries (5 original + 3 new), batches 06, 07, 08, 09, and 11
clear the 0.93 efficiency bar — spanning both synthesis methods, which is
exactly the kind of cross-cutting question a flat file-per-batch folder
can't answer without opening every file by hand.
:::

**Exercise 2 — Schema evolution**

> **Hint:** Add `'operator': str` to `cathode_schema` at the same nesting
> level as `'name'`, then re-run `validate_entry` on the *original*
> `nomad_style_entry` (which was never given an `operator` field).

:::{dropdown} Show full solution
```python
cathode_schema_v2 = {**cathode_schema, 'operator': str}
errors = validate_entry(nomad_style_entry['archive']['data'], cathode_schema_v2)
print(errors)   # ['MISSING: operator']
```
Output: `['MISSING: operator']`. This is exactly what happens in a real
repository after a schema changes: **entries uploaded under the old schema
version don't retroactively gain the new field** — they simply start
failing validation against the new schema (or, more commonly in a real
system, get flagged for migration/re-processing). This is why schema
changes in production data infrastructure are usually versioned rather
than made in place — old entries keep validating against the schema
version they were actually uploaded under, rather than silently breaking.
:::

**Exercise 3 — Design question**

> **Hint:** The current schema format is `{key: type}` — a range
> constraint needs the value to be something *callable* or otherwise
> checkable, not just a type.

:::{dropdown} Show full solution
One clean approach: let a schema value be either a type (as now) **or** a
`(type, validator_function)` tuple, and branch on which one it is:

```python
def validate_entry_v2(entry, schema):
    errors = []
    def _check(data, schema, path=''):
        for key, expected in schema.items():
            full_path = f'{path}.{key}' if path else key
            if key not in data:
                errors.append(f'MISSING: {full_path}')
                continue
            if isinstance(expected, dict):
                _check(data[key], expected, full_path)
            elif isinstance(expected, tuple) and callable(expected[1]):
                typ, validator = expected
                if not isinstance(data[key], typ):
                    errors.append(f'WRONG TYPE at {full_path}')
                elif not validator(data[key]):
                    errors.append(f'OUT OF RANGE at {full_path}: {data[key]!r}')
            elif not isinstance(data[key], expected):
                errors.append(f'WRONG TYPE at {full_path}')
    _check(entry, schema)
    return errors

cathode_schema_range = {**cathode_schema,
    'results': {**cathode_schema['results'],
                'coulombic_efficiency': ((int, float), lambda v: 0 <= v <= 1)}}

print(validate_entry_v2(nomad_style_entry['archive']['data'], cathode_schema_range))  # []
broken = json.loads(json.dumps(nomad_style_entry['archive']['data']))
broken['results']['coulombic_efficiency'] = 1.4
print(validate_entry_v2(broken, cathode_schema_range))  # OUT OF RANGE flagged
```
This is a small taste of what a real schema system (NOMAD's Metainfo
included) actually does — `validate_entry` here only ever checked
*structure*; real schemas also encode *semantic* constraints like valid
ranges, units, and enumerated allowed values.
:::

## Notebook 5: Capstone — Building a FAIR Dataset

**Exercise 1 — Break it on purpose**

> **Hint:** Build the dict by hand, omitting `results.hardness_HV`
> entirely (not just setting it to `None` — the validator checks for key
> *presence*, not truthiness, for nested sections).

:::{dropdown} Show full solution
```python
broken_record = make_sample_record('TS-013', 850, 1.0, 'oil', hardness_HV=500)
del broken_record['results']['hardness_HV']
dataset.append(broken_record)

all_errors = {}
for record in dataset:
    errors = validate_entry(record, sample_schema)
    if errors:
        all_errors[record['sample_id']] = errors

print(all_errors)   # {'TS-013': ['MISSING: results.hardness_HV']}
```
Exactly one record fails, with a precise, actionable error message
pointing at the missing nested field — the whole reason Step 4 validates
in a loop over every record instead of just checking `len(dataset) == 13`.
:::

**Exercise 2 — New quantity**

> **Hint:** Add the field in three places: `sample_schema`,
> `make_sample_record`'s signature and body, and the `quantity_kind_iri`
> sub-dict — miss one and the schema and the record builder will disagree.

:::{dropdown} Show full solution
```python
sample_schema['process']['tempering_temperature_C'] = (int, float)

def make_sample_record_v2(sample_id, temperature_C, time_h, quench_medium,
                           tempering_temperature_C, hardness_HV, creator='A. Lindqvist'):
    record = make_sample_record(sample_id, temperature_C, time_h, quench_medium,
                                 hardness_HV, creator)
    record['process']['tempering_temperature_C'] = tempering_temperature_C
    record['process']['quantity_kind_iri']['tempering_temperature_C'] = 'emmo:ThermodynamicTemperature'
    return record

dataset_v2 = [make_sample_record_v2(f'TS-{i:03d}', T, t, medium, 200, hv)
              for i, ((T, t, medium), hv) in enumerate(
                  zip(conditions, [simulate_hardness(*c) for c in conditions]), start=1)]

errors = {r['sample_id']: validate_entry(r, sample_schema)
          for r in dataset_v2 if validate_entry(r, sample_schema)}
print('Validation errors:', errors or 'none')
```
The same EMMO concept (`emmo:ThermodynamicTemperature`) tags two different
fields (`temperature_C` and `tempering_temperature_C`) here — a direct,
concrete instance of Notebook 2 §2.4's point that ontology concepts are
reusable across as many fields as genuinely share that physical meaning.
:::

**Exercise 3 — Your own dataset**

> **Hint:** Pick something with a clear "what I controlled" vs. "what I
> measured" split — that split *is* the `process`/`results` schema
> structure, almost for free.

:::{dropdown} Show full solution
Example sketch for a sol-gel ZnO thin-film dataset:

```python
zno_schema = {
    'identifier': str, 'sample_id': str, 'creator': str, 'created': str,
    'process': {
        'precursor_concentration_M': (int, float),
        'spin_speed_rpm':            (int, float),
        'anneal_temperature_C':      (int, float),
    },
    'results': {
        'film_thickness_nm':  (int, float),
        'transmittance_pct':  (int, float),
    },
}

def make_zno_record(sample_id, conc_M, spin_rpm, anneal_C, thickness_nm, transmittance_pct,
                     creator='A. Lindqvist'):
    return {
        'identifier':  f'urn:uuid:{uuid.uuid4()}',
        'sample_id':   sample_id, 'creator': creator, 'created': date.today().isoformat(),
        'process': {
            'precursor_concentration_M': conc_M,
            'spin_speed_rpm': spin_rpm,
            'anneal_temperature_C': anneal_C,
            'quantity_kind_iri': {'anneal_temperature_C': 'emmo:ThermodynamicTemperature'},
        },
        'results': {
            'film_thickness_nm': thickness_nm,
            'transmittance_pct': transmittance_pct,
        },
    }
```
This follows Steps 1–2's exact pattern — the point of this exercise isn't
a "correct" answer, it's practising the design step itself before you do
it for real with the course project's actual dataset.
:::

## Notebook 6: Live Tutorial 1 — SQL and NoSQL in Practice

**Exercise 1 — Close the typo gap in SQL**

> **Hint:** SQLite's `CHECK` constraint has to be declared as part of
> `CREATE TABLE` — you can't `ALTER TABLE` one in later, so this means
> recreating `synthesis_steps` with the constraint in place, not patching
> the existing table.

:::{dropdown} Show full solution
```python
conn.executescript('''
CREATE TABLE synthesis_steps_v2 (
    step_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id     TEXT NOT NULL REFERENCES samples(sample_id),
    step_order    INTEGER NOT NULL,
    step_name     TEXT NOT NULL CHECK (step_name IN ('mixing', 'calcination_1', 'calcination_2')),
    temperature_C REAL,
    duration_h    REAL,
    instrument_id INTEGER REFERENCES instruments(instrument_id)
);
INSERT INTO synthesis_steps_v2 SELECT * FROM synthesis_steps;
DROP TABLE synthesis_steps;
ALTER TABLE synthesis_steps_v2 RENAME TO synthesis_steps;
''')

try:
    cur.execute(
        "INSERT INTO synthesis_steps (sample_id, step_order, step_name, temperature_C, duration_h, instrument_id) "
        "VALUES ('batch_09', 5, 'clacination_2', 655, 6.0, 4)"
    )
    conn.commit()
except sqlite3.IntegrityError as e:
    print('Rejected:', e)
```
Verified output: `Rejected: CHECK constraint failed: step_name IN
('mixing','calcination_1','calcination_2')` — the exact typo that slipped
through silently in Section 6.6 is now caught immediately at insert time.
:::

**Exercise 2 — Close the typo gap in MongoDB**

> **Hint:** Loop over `doc['steps']`, check each step's `'name'` against
> the same three-value allowed set, and call this function *before*
> `insert_one` — MongoDB itself won't call it for you.

:::{dropdown} Show full solution
```python
ALLOWED_STEP_NAMES = {'mixing', 'calcination_1', 'calcination_2'}

def validate_step(doc):
    errors = []
    for i, step in enumerate(doc.get('steps', [])):
        if step.get('name') not in ALLOWED_STEP_NAMES:
            errors.append(f"steps[{i}]: invalid name {step.get('name')!r}")
    return errors

errs = validate_step(sloppy_doc)
print(errs)   # ["steps[2]: invalid name 'clacination_2'"]
if not errs:
    coll.insert_one(sloppy_doc)
else:
    print('Insert blocked -- fix the document before trying again.')
```
This catches exactly `batch_50`'s typo, and it is precisely the
application-level pattern Section 6.6's take-home message pointed to:
MongoDB itself enforces nothing here, so the check has to live in your
own code, called every time, the same discipline `validate_entry`
(Notebook 4) established for NOMAD-style archives.
:::

**Exercise 3 — Two-level breakdown, written up**

> **Hint:** Look at Furnace-C's row within *each* method separately, not
> just its overall mean — is it still the lowest of the three furnaces in
> both sub-tables?

:::{dropdown} Show full solution
Running `df_furnace.groupby(['method', 'furnace'])['capacity_mAh_g'].agg(['count', 'mean'])`
gives (with this notebook's seed):

| method | furnace | count | mean |
|---|---|---|---|
| sol-gel | Furnace-A | 7 | 158.99 |
| sol-gel | Furnace-B | 9 | 161.71 |
| sol-gel | Furnace-C | 7 | 157.73 |
| solid-state reaction | Furnace-A | 7 | 156.29 |
| solid-state reaction | Furnace-B | 10 | 159.37 |
| solid-state reaction | Furnace-C | 5 | 155.58 |

**Write-up for the process engineer:** "Furnace-C has the lowest mean
capacity within *both* synthesis methods separately (157.73 vs. 158.99/
161.71 for sol-gel; 155.58 vs. 156.29/159.37 for solid-state), not just in
the pooled average — so this isn't an artefact of one method happening to
be over-represented on Furnace-C. That consistency across both methods is
reassuring, but the sub-group sample sizes are now down to 5–10 batches
each, so this is still suggestive rather than statistically confirmed;
recommend a proper two-way test (Part III, Notebook 9) before treating it
as settled."
:::

**Exercise 4 — Selectivity matters**

> **Hint:** `instrument_id` only has 4 distinct values across 200,000
> rows, so any single value matches roughly a quarter of the table — an
> index can't narrow the search down nearly as much as it did for the
> (nearly unique) `sample_id`.

:::{dropdown} Show full solution
```python
t0 = time.perf_counter()
for _ in range(20):
    cur2.execute('SELECT COUNT(*) FROM big_steps WHERE instrument_id = ?', (4,)).fetchall()
t_no_index = (time.perf_counter() - t0) / 20

cur2.execute('CREATE INDEX idx_instrument_id ON big_steps(instrument_id)')

t0 = time.perf_counter()
for _ in range(20):
    cur2.execute('SELECT COUNT(*) FROM big_steps WHERE instrument_id = ?', (4,)).fetchall()
t_with_index = (time.perf_counter() - t0) / 20

print(f'Speedup: {t_no_index/t_with_index:.1f}x')
```
**Verified result: instrument_id=4 matches 49,699 of 200,000 rows (24.8%
— roughly the expected quarter for 4 near-equally-likely values), and
indexing it gives only a ~6.3× speedup — dramatically less than
`sample_id`'s ~1,769× speedup from Section 6.7.** This is exactly what
selectivity predicts: an index helps most when it can immediately narrow
"scan everything" down to "look up almost nothing" (a near-unique column
like `sample_id`); when a quarter of the table matches regardless, the
database still has to read and return a quarter of the table either way,
index or not — there just isn't as much to skip.
:::

**Exercise 5 — From this notebook to a real, shared lab database**

> **Hint:** Notebook 3 §3.5 already answered the SQL half explicitly;
> the MongoDB half follows the same "only the client constructor changes"
> pattern Notebook 3 §3.6 set up.

:::{dropdown} Show full solution
**SQL → PostgreSQL:** Exactly as Notebook 3 §3.5 demonstrated — swap the
`sqlite3.connect(':memory:')` connection for a SQLAlchemy engine pointing
at a real server (`create_engine('postgresql+psycopg2://user:pass@labserver:5432/db')`),
and any query written through SQLAlchemy's own API runs unchanged. The one
real catch (flagged in this book's own accuracy pass on Notebook 3): the
raw `CREATE TABLE` DDL used directly in `sqlite3.executescript` here — with
SQLite-specific syntax like `AUTOINCREMENT` and `PRAGMA foreign_keys` —
would need rewriting for PostgreSQL (`SERIAL`/`GENERATED ... AS IDENTITY`
instead of `AUTOINCREMENT`, and no `PRAGMA` statements at all), not just a
connection-string swap.

**MongoDB → a real server:** Only the client constructor changes —
`mongomock.MongoClient()` becomes `pymongo.MongoClient('mongodb://labserver:27017')`
— every `insert_many`, `find`, and `aggregate` call in Sections 6.5–6.6 is
already the exact real `pymongo` API; `mongomock` exists purely so this
notebook doesn't require a running server to follow along.
:::

## Notebook 7: Live Tutorial 2 — Querying and Plotting Real NOMAD Data

**Exercise 1 — A different binary system**

> **Hint:** Just change the element list and re-run — but treat any
> specific totals/counts as a live snapshot, not a fixed answer, since
> NOMAD's database grows continuously.

:::{dropdown} Show full solution
```python
for elems in [['Ti', 'O'], ['Zn', 'O']]:
    query = {'query': {'results.material.elements': elems}, 'pagination': {'page_size': 50},
             'required': {'include': ['entry_id', 'results.material.symmetry.crystal_system']}}
    r = requests.post(f'{BASE_URL}/entries/query', json=query)
    result = r.json()
    from collections import Counter
    cs = Counter(e.get('results', {}).get('material', {}).get('symmetry', {}).get('crystal_system')
                 for e in result['data'])
    print(elems, 'total:', f"{result['pagination']['total']:,}", 'sample distribution:', dict(cs))
```
**Live-verified snapshot at time of writing:** Ti–O has 70,439 total
entries in NOMAD, Zn–O has 61,620. In a 50-entry sample of each, both
systems actually show similar structural diversity — 6 distinct crystal
systems apiece — though which systems appear differs (Ti–O's sample
includes tetragonal, absent from Zn–O's; Zn–O's includes hexagonal,
absent from Ti–O's, consistent with ZnO's well-known wurtzite structure).
Because NOMAD's holdings keep growing, re-running this later will give
different exact counts — the comparison method matters more here than
memorising a specific number.
:::

**Exercise 2 — Paginate to at least 300 entries**

> **Hint:** Swap the `for _ in range(n_pages)` loop for a `while
> len(all_entries) < min_entries:` loop, and check the stopping condition
> at the top of each iteration, same as Part I §1.3's titration example.

:::{dropdown} Show full solution
```python
def fetch_at_least(query_body, min_entries=300, page_size=50, sleep_s=0.2):
    all_entries = []
    body = json.loads(json.dumps(query_body))
    body['pagination'] = {'page_size': page_size}
    while len(all_entries) < min_entries:
        r = requests.post(f'{BASE_URL}/entries/query', json=body)
        r.raise_for_status()
        page = r.json()
        all_entries.extend(page['data'])
        after = page['pagination'].get('next_page_after_value')
        if after is None:
            break   # fewer total matches than min_entries -- stop, don't loop forever
        body['pagination']['page_after_value'] = after
        time.sleep(sleep_s)
    return all_entries

entries_300plus = fetch_at_least(query_binary_feo, min_entries=300)
print(f'Collected {len(entries_300plus)} entries (requested >= 300).')
```
The `if after is None: break` guard matters as much as the `while`
condition itself — without it, a query with fewer than 300 total matches
would loop forever asking for a next page that will never come.
:::

**Exercise 3 — A different oxide system with experimental comparison**

> **Hint:** Repeat Section 7.7's two-step search-then-fetch pattern for
> `['Ti', 'O']`; rutile TiO₂'s well-known experimental gap is ≈3.0 eV.

:::{dropdown} Show full solution
```python
query_has_gap_tio2 = {
    'query': {
        'results.material.elements': ['Ti', 'O'], 'results.material.n_elements': 2,
        'results.properties.available_properties': 'electronic.band_structure_electronic.band_gap',
    },
    'pagination': {'page_size': 20},
    'required': {'include': ['entry_id', 'results.material.chemical_formula_reduced']},
}
r = requests.post(f'{BASE_URL}/entries/query', json=query_has_gap_tio2)
candidates_tio2 = r.json()['data']

rows = [{'entry_id': e['entry_id'],
         'formula': e['results']['material'].get('chemical_formula_reduced'),
         'band_gap_eV': fetch_band_gap_eV(e['entry_id'])}
        for e in candidates_tio2]
df_gap_tio2 = pd.DataFrame(rows)
gaps_tio2 = df_gap_tio2['band_gap_eV'].dropna()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(gaps_tio2, bins=10, color=sns.color_palette('colorblind')[3], edgecolor='white')
ax.axvline(3.0, color='k', ls='--', lw=1.5, label='Experimental rutile TiO2 gap ~= 3.0 eV')
ax.set_xlabel('Computed band gap (eV)'); ax.legend()
plt.show()
```
Expect the same qualitative pattern Section 7.7 found for ZnO: standard
DFT functionals (GGA/LDA) systematically **underestimate** band gaps
relative to experiment, so the computed distribution typically sits below
the 3.0 eV reference line, not centred on it.
:::

**Exercise 4 — Visible failures instead of silent NaN**

> **Hint:** Add one `print` call right where the function currently
> returns `np.nan` on a bad response — the fix is one line, not a
> restructure.

:::{dropdown} Show full solution
```python
def fetch_band_gap_eV_verbose(entry_id):
    body = {'required': {'results': {'properties': {'electronic': {'band_gap': '*'}}}}}
    r = requests.post(f'{BASE_URL}/entries/{entry_id}/archive/query', json=body)
    if r.status_code != 200:
        print(f'WARNING: {entry_id} returned status {r.status_code} -- skipping.')
        return np.nan
    band_gaps = (r.json().get('data', {}).get('archive', {})
                 .get('results', {}).get('properties', {})
                 .get('electronic', {}).get('band_gap', []))
    if not band_gaps:
        print(f'WARNING: {entry_id} returned 200 but no band_gap data -- skipping.')
        return np.nan
    return band_gaps[0]['value'] * J_TO_EV
```
Two separate failure modes are now distinguishable: a bad HTTP status
(server/network problem) vs. a successful response that simply doesn't
contain the field you asked for (the "available doesn't mean present"
gotcha Section 7.7 already flagged) — exactly the kind of distinction
`validate_entry` (Notebook 4) makes between "missing" and "wrong type."
:::

**Exercise 5 — New element system, full pipeline**

> **Hint:** This is Sections 7.3, 7.5, and 7.7 run back-to-back for one
> system you pick — reuse every function already defined
> (`fetch_pages`, `fetch_band_gap_eV`) rather than rewriting them.

:::{dropdown} Show full solution
```python
elems = ['Ga', 'N']   # GaN, experimental gap ~3.4 eV (wurtzite)

# 7.3-style crystal-system bar chart
q1 = {'query': {'results.material.elements': elems}, 'pagination': {'page_size': 50},
      'required': {'include': ['entry_id', 'results.material.symmetry.crystal_system']}}
r1 = requests.post(f'{BASE_URL}/entries/query', json=q1)
cs = pd.Series([e.get('results', {}).get('material', {}).get('symmetry', {}).get('crystal_system')
                for e in r1.json()['data']]).value_counts()
cs.plot(kind='bar'); plt.title(f'{"-".join(elems)} crystal systems'); plt.show()

# 7.5-style paginated space-group histogram (>=150 entries)
q2 = {'query': {'results.material.elements': elems, 'results.material.n_elements': 2},
      'required': {'include': ['entry_id', 'results.material.symmetry.space_group_number']}}
entries = fetch_at_least(q2, min_entries=150)
sg = pd.Series([e['results']['material'].get('symmetry', {}).get('space_group_number')
                for e in entries]).dropna()
sg.plot(kind='hist', bins=30); plt.title(f'{"-".join(elems)} space groups (n={len(entries)})'); plt.show()

# 7.7-style band-gap histogram with experimental reference
q3 = {'query': {'results.material.elements': elems, 'results.material.n_elements': 2,
                'results.properties.available_properties': 'electronic.band_structure_electronic.band_gap'},
      'pagination': {'page_size': 20}, 'required': {'include': ['entry_id']}}
candidates = requests.post(f'{BASE_URL}/entries/query', json=q3).json()['data']
gaps = pd.Series([fetch_band_gap_eV(e['entry_id']) for e in candidates]).dropna()
plt.hist(gaps, bins=10); plt.axvline(3.4, color='k', ls='--', label='Experimental GaN gap ~3.4 eV')
plt.legend(); plt.show()
```
GaN is a useful contrast case precisely *because* it isn't an oxide — the
query pattern (search on `elements` + `n_elements`, then fetch archives for
the numeric property) is identical regardless of chemistry, which is the
whole point of building it as reusable functions in the first place.
:::

**Exercise 6 — Scatter plot across two properties**

> **Hint:** Add `'results.material.n_elements'` to the `required.include`
> list when re-collecting `entries_500`, then colour by a boolean
> (`n_elements == 2`) rather than the raw integer, for a clean two-colour plot.

:::{dropdown} Show full solution
```python
query_binary_feo_v2 = {
    'query': {'results.material.elements': ['Fe', 'O']},   # note: no n_elements filter this time
    'required': {'include': ['entry_id', 'results.material.n_elements',
                              'results.material.symmetry.space_group_number']},
}
entries_v2 = fetch_pages(query_binary_feo_v2, n_pages=10, page_size=50)

rows = [{'n_elements': e['results']['material'].get('n_elements'),
         'space_group_number': e['results']['material'].get('symmetry', {}).get('space_group_number'),
         'is_binary': e['results']['material'].get('n_elements') == 2}
        for e in entries_v2]
df_scatter = pd.DataFrame(rows).dropna()

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df_scatter, x='n_elements', y='space_group_number',
                 hue='is_binary', alpha=0.5, ax=ax)
ax.set_xlabel('Number of elements'); ax.set_ylabel('Space group number')
plt.show()
```
With `n_elements` taking only a handful of integer values, expect vertical
"stripes" of points rather than a smooth trend — space group doesn't have
an obvious monotonic relationship with compositional complexity, which is
itself a real, useful finding: symmetry and stoichiometric complexity are
largely independent properties of a crystal structure.
:::

**Exercise 7 — Robust pagination**

> **Hint:** Wrap the single request inside the loop in a `try`/`except
> requests.exceptions.RequestException`, retry exactly once, and `break`
> out of the whole function only if the retry also fails.

:::{dropdown} Show full solution
```python
def fetch_pages_robust(query_body, n_pages=5, page_size=50, sleep_s=0.2):
    all_entries = []
    body = json.loads(json.dumps(query_body))
    body['pagination'] = {'page_size': page_size}
    for page_num in range(n_pages):
        for attempt in (1, 2):
            try:
                r = requests.post(f'{BASE_URL}/entries/query', json=body, timeout=30)
                r.raise_for_status()
                page = r.json()
                break
            except requests.exceptions.RequestException as e:
                print(f'WARNING: page {page_num} attempt {attempt} failed ({e}).')
                if attempt == 2:
                    print(f'Skipping page {page_num} after 2 failed attempts.')
                    page = None
        if page is None:
            continue   # skip this page, try the next one anyway
        all_entries.extend(page['data'])
        after = page['pagination'].get('next_page_after_value')
        if after is None:
            break
        body['pagination']['page_after_value'] = after
        time.sleep(sleep_s)
    return all_entries
```
The key design choice: a failed page is *skipped*, not fatal — the
function keeps trying subsequent pages rather than losing everything
already collected, exactly the "defensive but transparent" principle
Exercise 4 and Notebook 4's `validate_entry` both apply elsewhere in
this part.
:::

**Exercise 8 — Connect it to Notebook 1**

> **Hint:** This is Notebook 5's capstone pattern (CSV + JSON FAIR record,
> Steps 5–6) applied to `df_binary` instead of the synthetic tool-steel
> dataset — same two files, same round-trip check.

:::{dropdown} Show full solution
```python
import uuid
from datetime import date

df_binary.to_csv('nomad_feo_binary.csv', index=False)

fair_record = {
    'identifier': f'urn:uuid:{uuid.uuid4()}',
    'title':      'Binary Fe-O compounds from NOMAD (space group survey)',
    'creator':    'A. Lindqvist, Dept. of Chemistry, Uppsala University',
    'created':    date.today().isoformat(),
    'licence':    'CC-BY-4.0',   # matches NOMAD's own open-access entries
    'keywords':   ['Fe-O', 'NOMAD', 'space group', 'binary oxide'],
    'source':     'NOMAD API, POST /entries/query, results.material.elements=[Fe,O], n_elements=2',
    'n_records':  len(df_binary),
}
with open('nomad_feo_binary_record.json', 'w', encoding='utf-8') as f:
    json.dump(fair_record, f, indent=2)

# Round-trip check, same discipline as every other export in this course
df_reloaded = pd.read_csv('nomad_feo_binary.csv')
with open('nomad_feo_binary_record.json', encoding='utf-8') as f:
    record_reloaded = json.load(f)
print('CSV round trip OK:', df_reloaded.equals(df_binary))
print('JSON round trip OK:', record_reloaded == fair_record)
```
The `'source'` field here is doing real work: unlike Notebook 5's
synthetic dataset (where "how was this generated" meant "which Python
function"), this dataset's provenance is a *live external API call* —
recording the exact query is what makes this archive reproducible later,
even after NOMAD's own holdings have grown and a fresh query would no
longer return exactly the same 500 entries.
:::
