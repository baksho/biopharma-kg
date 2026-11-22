# Test fixtures

Everything in this directory is **hand-written, minimal and exists only for the unit/integration tests**.
It follows the *shapes* of the real sources (ClinicalTrials.gov v2 JSON, openFDA JSON, PubMed XML,
GDSC fitted-dose-response columns) so parsers can be exercised offline. It is **not data**: the numbers,
trials, sponsors and abstracts are invented, are never used by the pipeline, and must not be cited.
The pipeline itself only ever uses data fetched live from the real sources.
