# Design notes

## One relational source of truth, two consumers
Raw files → SQLite → **SQL views** (`v_drug_condition`, `v_drug_org`, `v_drug_sensitive_cell_line`,
`v_drug_pharm_class`, `v_drug_ae`, `v_drug_extracted`). The R2RML mappings and the graph-ML exporter both read these
views, and `tests/test_pipeline.py::test_ml_edges_equal_kg_triples` asserts that the two edge sets are identical for
every relation. So the RDF graph and the ML graph cannot silently drift apart.

## Enterprise analogy
| Enterprise | Here |
|---|---|
| Business partner / supplier | Sponsor organisation (`bp:Organization`) |
| Product | Drug (`bp:Drug`), marketed product (`bp:DrugProduct`) |
| Transaction | Clinical trial (`bp:ClinicalTrial`), reaction summary (`bp:ReactionSummary`) |
| Master-data management | Entity resolution (names → canonical drugs/orgs) |
| Data quality rules | SHACL shapes |
| Business rules / inference | OWL-RL (type hierarchy closure, sub-property lifting) |

## Entity resolution (deterministic, auditable)
* Drugs: lower-case, strip accents, remove salts/doses/dosage-form words (`Erlotinib Hydrochloride 150 mg Tablets` →
  `erlotinib`). Brand names resolve only when
  the openFDA NDC records list them (brand/generic name) for a seed drug; alias sources are the seed list, GDSC2 and
  NDC. Trial interventions are linked to drugs by whole-phrase match against this alias index
  (so `Erlotinib + Bevacizumab` links to both).
* Organisations: strip legal suffixes (`Inc.`, `AG`, `Ltd`…), `&`→`and`. **Subsidiaries and parents stay separate**
  (e.g. Genentech vs. Roche) because merging them needs corporate-hierarchy data this project does not have.
* Conditions → cancer types: curated hierarchy (`resources/cancer_types.csv`) + keyword rules
  (`resources/condition_keywords.csv`). Unmapped conditions remain valid nodes without a cancer-type link.
* Matching is longest-phrase-wins; ambiguous phrases (one phrase → several entities) are skipped, not guessed.
  `reports/resolution_report.json` lists match rates and overlaps so you can judge coverage.
* Known limits: no fuzzy/embedding matching, no typo repair, no synonyms beyond what the sources state.

## Literature extraction
Two backends write the same table (`extracted_relation`) with provenance (PMID, evidence sentence, extractor, model):
* `rules-cooccurrence-v1`: offline baseline; sentence must contain a known drug, a known object and a trigger word;
  sentences with negation cues are skipped; `confidence` is NULL.
* `llm-anthropic-v1`: Anthropic API, strict JSON; every item is **dropped** unless the predicate is allowed, the drug
  is a known seed drug, the evidence sentence occurs verbatim in the abstract, and the object occurs in the evidence.
  Responses are cached on disk. `confidence` is the model's self-report and is **not calibrated**.
* `auto` uses the LLM backend when `ANTHROPIC_API_KEY` is set and the SDK is installed, otherwise rules with a loud warning.
* Precision is **not measured** by this repo. `biopharma-kg audit-extraction` writes a sample CSV for manual review.

## Graph ML: leakage controls
* Trial nodes are **not** in the ML graph (drug ← trial → condition would reveal test edges in 2 hops).
* No inverse relations in the training graph.
* `drug_condition` uses a **temporal split** on the first trial start date; dated context edges (sponsors, literature)
  at or after the validation cutoff are dropped. Undated edges go to training.
* Validation/test triples with an entity unseen in training are dropped (counts reported).
* Evaluation is **filtered** and **type-constrained** (only entities that occur as tails of the target relation are
  ranked); ties use the "realistic" rank. `tests/test_ml.py` checks it against a brute-force implementation and
  against PyKEEN's own `RankBasedEvaluator`.
* `reports/ml_results.json` includes a shortcut report: how many test pairs are directly linked by another relation
  (e.g. a literature edge) in training.
* Hyper-parameters (lr 0.01, epochs) were chosen from a mechanics test on a *synthetic* planted-structure graph
  (PyKEEN's default lr 1e-3 under-trains at these step counts) and were **not tuned on real data**.
