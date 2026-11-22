"""GDSC2 fitted dose-response reader.

Columns relied on (from the Sanger "GDSC fitted data description": "Possible columns ... not
every listed column is present in every file"):

  required: COSMIC_ID, CELL_LINE_NAME, DRUG_ID, DRUG_NAME, LN_IC50, AUC
  optional: PUTATIVE_TARGET, PATHWAY_NAME, TCGA_DESC, Z_SCORE

Missing required columns raise ``GdscFormatError`` listing what was found. Licence: GDSC data
is licensed for internal research/educational use; do not redistribute the raw files (this
repository never commits them; ``data/`` is git-ignored).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .http import FetchError, HttpClient

log = logging.getLogger(__name__)

REQUIRED = ["COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME", "LN_IC50", "AUC"]
OPTIONAL = ["PUTATIVE_TARGET", "PATHWAY_NAME", "TCGA_DESC", "Z_SCORE"]
_ALIASES = {
    "LN IC50": "LN_IC50", "LNIC50": "LN_IC50", "LN-IC50": "LN_IC50",
    "TARGET": "PUTATIVE_TARGET", "TARGET_PATHWAY": "PATHWAY_NAME", "TARGET PATHWAY": "PATHWAY_NAME",
    "CANCER TYPE (MATCHING TCGA LABEL)": "TCGA_DESC", "TCGA DESC": "TCGA_DESC", "TCGA-DESC": "TCGA_DESC",
    "Z-SCORE": "Z_SCORE", "ZSCORE": "Z_SCORE",
    "COSMIC ID": "COSMIC_ID", "CELL LINE NAME": "CELL_LINE_NAME", "DRUG ID": "DRUG_ID", "DRUG NAME": "DRUG_NAME",
}

MANUAL_DOWNLOAD_HELP = (
    "GDSC2 file not found. Download 'GDSC2 fitted dose response' (an .xlsx file) from "
    "https://www.cancerrxgene.org/downloads/bulk_download (section 'Drug Screening - IC50s' -> GDSC2-dataset) "
    "and place it in data/raw/gdsc/ (or set gdsc.file in your config). The GDSC data terms apply "
    "(https://www.cancerrxgene.org/legal)."
)


class GdscFormatError(RuntimeError):
    pass


def find_gdsc_file(configured: str | Path | None, raw_dir: Path) -> Path | None:
    if configured:
        p = Path(configured)
        if not p.exists():
            raise FileNotFoundError(f"gdsc.file is set to {p} but it does not exist")
        return p
    gdsc_dir = raw_dir / "gdsc"
    if gdsc_dir.exists():
        cands = sorted(list(gdsc_dir.glob("*.xlsx")) + list(gdsc_dir.glob("*.csv")))
        if cands:
            return cands[0]
    return None


def download_gdsc(client: HttpClient, url: str, raw_dir: Path) -> Path:
    dest = raw_dir / "gdsc" / Path(url.split("?")[0]).name
    try:
        client.download(url, dest)
    except FetchError as e:
        raise FetchError(f"{e}\n{MANUAL_DOWNLOAD_HELP}") from e
    return dest


def _norm_col(c: str) -> str:
    c = str(c).strip()
    up = c.upper()
    if up in _ALIASES:
        return _ALIASES[up]
    return re.sub(r"[\s\-]+", "_", up)


def read_gdsc(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif p.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
    else:
        raise GdscFormatError(f"Unsupported GDSC file type {p.suffix!r}; expected .xlsx or .csv")
    df.columns = [_norm_col(c) for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise GdscFormatError(
            f"GDSC file {p.name} is missing required column(s) {missing}. Found columns: {list(df.columns)[:30]}. "
            "Is this the 'fitted dose response' file?"
        )
    keep = REQUIRED + [c for c in OPTIONAL if c in df.columns]
    df = df[keep].copy()
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = None
    df["LN_IC50"] = pd.to_numeric(df["LN_IC50"], errors="coerce")
    df["AUC"] = pd.to_numeric(df["AUC"], errors="coerce")
    df["Z_SCORE"] = pd.to_numeric(df["Z_SCORE"], errors="coerce")
    df["COSMIC_ID"] = pd.to_numeric(df["COSMIC_ID"], errors="coerce")
    df["DRUG_ID"] = pd.to_numeric(df["DRUG_ID"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["DRUG_NAME", "LN_IC50", "CELL_LINE_NAME"])
    if len(df) < before:
        log.info("Dropped %d GDSC rows with missing drug name / cell line / LN_IC50", before - len(df))
    if df.empty:
        raise GdscFormatError(f"GDSC file {p.name} has no usable rows after cleaning")
    return df.reset_index(drop=True)
