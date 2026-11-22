# Data sources

All data is real, public, and fetched at run time. **Nothing is bundled** (`data/` is git-ignored), so this
repository does not redistribute any third-party data. I have not independently verified each provider's current
licence text, so check the linked terms before reusing outputs outside personal/educational work.

| Source | What is used | Endpoint / file | Terms (check yourself) |
|---|---|---|---|
| ClinicalTrials.gov | Registered studies per seed drug: sponsor, phase, start date, conditions, interventions | REST API v2, `GET /api/v2/studies?query.intr=<drug>` (paginated via `nextPageToken`) | https://clinicaltrials.gov/about-site/terms-conditions |
| openFDA – NDC | Marketed products: labeler, dosage form, route, pharmacologic class | `GET https://api.fda.gov/drug/ndc.json` | https://open.fda.gov/terms/ |
| openFDA – FAERS | Reaction **report counts** per drug (`count=patient.reaction.reactionmeddrapt.exact`) | `GET https://api.fda.gov/drug/event.json` | https://open.fda.gov/terms/ |
| PubMed (NCBI E-utilities) | Titles/abstracts for the seed drugs (cancer-restricted query) | `esearch.fcgi` + `efetch.fcgi` | https://www.ncbi.nlm.nih.gov/home/about/policies/ . Abstracts may carry publisher copyright: they stay in your local `data/` and are not committed. |
| GDSC2 (Sanger) | Fitted dose-response (LN_IC50, AUC) per (drug, cell line) | `GDSC2_fitted_dose_response_*.xlsx/.csv` from https://www.cancerrxgene.org/downloads/bulk_download | https://www.cancerrxgene.org/legal : **download it yourself**; do not commit it. |

## GDSC2 manual download (the reliable path)

The auto-download URL in `configs/default.yaml` is a best-effort guess: the release-8.5 directory exists on
`cog.sanger.ac.uk`, but the exact file name/date was **not verified**. If `biopharma-kg fetch` cannot get it, it stops
(GDSC2 is `required` by default) and prints these steps:

1. Open https://www.cancerrxgene.org/downloads/bulk_download and download the **GDSC2 fitted dose response** file
   (`.xlsx` or `.csv`).
2. Put it in `data/raw/gdsc/` (or set `gdsc.file` in your config).
3. Re-run. Column names are matched flexibly; a missing required column gives an explicit error.

To run *without* sensitivity data (not recommended), set `gdsc: {required: false}`.

## What the numbers mean (and don't)

* **FAERS counts are spontaneous reports**, not incidence or risk. They are biased by reporting behaviour and drug
  popularity. The graph stores them as `bp:ReactionSummary.reportCount`; queries rank by count, nothing more.
* **"Sensitive" in GDSC2** is *relative per drug*: a cell line is flagged sensitive when its LN_IC50 is in the lowest
  `gdsc.sensitive_quantile` (default 25 %) of that drug's responses (drugs need ≥ 10 responses). It is a screening
  statistic, not a clinical response.
* **Trial ≠ efficacy.** A registered trial says a drug was *investigated* for a condition.
