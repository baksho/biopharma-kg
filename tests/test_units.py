"""Unit tests: name normalisation, HTTP client, parsers. No network."""
import json
import socket

import pytest

from biopharma_kg.ingest import ctgov, openfda, pubmed
from biopharma_kg.ingest.gdsc import GdscFormatError, read_gdsc
from biopharma_kg.ingest.http import FetchError, HttpClient
from biopharma_kg.resolve.index import PhraseIndex
from biopharma_kg.resolve.names import contains_whole_phrase, drug_key, normalize_drug, normalize_org, org_key
from conftest import FIX


# ------------------------------------------------------------------ names
@pytest.mark.parametrize("raw,expected", [
    ("Erlotinib Hydrochloride 150 mg Tablets", "erlotinib"),
    ("5-Fluorouracil", "5 fluorouracil"),
    ("Tarceva (erlotinib)", "tarceva"),
    ("CISPLATIN", "cisplatin"),
    ("Imatinib Mesylate", "imatinib"),
    ("Café-drug", "cafe drug"),
    ("", ""),
    (None, ""),
])
def test_normalize_drug(raw, expected):
    assert normalize_drug(raw) == expected


def test_drug_key_is_iri_safe():
    assert drug_key("5-Fluorouracil") == "5_fluorouracil"
    assert " " not in drug_key("a b c")


@pytest.mark.parametrize("raw,expected", [
    ("Pfizer Inc.", "pfizer"), ("Pfizer", "pfizer"), ("Genentech, Inc.", "genentech"),
    ("F. Hoffmann-La Roche AG", "f hoffmann la roche"), ("AstraZeneca AB", "astrazeneca"),
    ("Johnson & Johnson", "johnson and johnson"),
])
def test_normalize_org(raw, expected):
    assert normalize_org(raw) == expected


def test_org_variants_merge():
    assert org_key("Genentech, Inc.") == org_key("Genentech")
    assert org_key("AstraZeneca") == org_key("AstraZeneca AB")


def test_whole_phrase():
    assert contains_whole_phrase("erlotinib bevacizumab", "erlotinib")
    assert not contains_whole_phrase("nonerlotinibx", "erlotinib")


def test_phrase_index_longest_match_and_ambiguity():
    idx = PhraseIndex()
    idx.add("lung cancer", "lc")
    idx.add("non small cell lung cancer", "nsclc")
    idx.add("ambig", "a1")
    idx.add("ambig", "a2")
    hits = idx.find("treated non small cell lung cancer and ambig")
    assert [h[0] for h in hits] == ["nsclc"]  # longest match wins; ambiguous phrase skipped
    assert not idx.add("placebo", "x")        # stoplisted


# ------------------------------------------------------------------ http
def test_http_retries_then_succeeds_and_caches(tmp_path, mock_server):
    c = HttpClient(tmp_path / "cache", max_retries=5, backoff_s=0.01)
    assert c.get_json(mock_server + "/flaky") == {"ok": True}
    # second call is served from cache (no new request would change the answer either way)
    before = len(__import__("conftest")._Handler.calls)
    assert c.get_json(mock_server + "/flaky") == {"ok": True}
    assert len(__import__("conftest")._Handler.calls) == before


def test_http_404_handling(tmp_path, mock_server):
    c = HttpClient(tmp_path / "cache", max_retries=1, backoff_s=0.01)
    with pytest.raises(FetchError):
        c.get_json(mock_server + "/nope")
    assert c.get_json(mock_server + "/nope", accept_404=True) is None


def test_http_unreachable_gives_actionable_error(tmp_path):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()  # a port nobody listens on
    c = HttpClient(tmp_path / "cache", max_retries=2, backoff_s=0.01, timeout_s=1)
    with pytest.raises(FetchError, match="outbound HTTPS"):
        c.get_json(f"http://127.0.0.1:{port}/x")


# ------------------------------------------------------------------ ClinicalTrials.gov
def test_ctgov_dates():
    assert ctgov.normalize_date("2019") == "2019-01-01"
    assert ctgov.normalize_date("2019-03") == "2019-03-01"
    assert ctgov.normalize_date("2019-03-15") == "2019-03-15"
    assert ctgov.normalize_date("March 2019") is None
    assert ctgov.normalize_date(None) is None


def test_ctgov_parse_fixture():
    data = json.loads((FIX / "ctgov.json").read_text())
    p = ctgov.parse_study(data["erlotinib_p1"]["studies"][1])
    assert p["nct_id"] == "NCT00000002" and p["start_date"] == "2011-05-14"
    assert p["phase"] == "PHASE2" and p["lead_sponsor"] == "Genentech"
    assert p["conditions"] == ["Non-Small Cell Lung Cancer", "Lung Adenocarcinoma"]
    assert [i["name"] for i in p["interventions"]] == ["Erlotinib", "Bevacizumab"]
    nodate = ctgov.parse_study(data["erlotinib_p2"]["studies"][1])
    assert nodate["start_date"] is None and nodate["phase"] is None


def test_ctgov_schema_errors_are_loud():
    with pytest.raises(ctgov.SchemaError):
        ctgov.parse_study({"foo": 1})
    with pytest.raises(ctgov.SchemaError):
        ctgov.parse_study({"protocolSection": {"identificationModule": {}}})


def test_ctgov_pagination(tmp_path, mock_server):
    c = HttpClient(tmp_path / "cache", backoff_s=0.01)
    studies = ctgov.fetch_studies_for_drug(c, mock_server + "/ctgov", "erlotinib", page_size=3, max_studies=50)
    assert [s["protocolSection"]["identificationModule"]["nctId"] for s in studies] == [f"NCT0000000{i}" for i in range(1, 6)]
    assert len(ctgov.fetch_studies_for_drug(c, mock_server + "/ctgov", "erlotinib", 3, 4)) == 4  # cap respected


def test_ctgov_missing_studies_key_raises(tmp_path, mock_server):
    c = HttpClient(tmp_path / "cache", backoff_s=0.01)
    with pytest.raises(ctgov.SchemaError):
        ctgov.fetch_studies_for_drug(c, mock_server + "/badctgov", "x", 3, 3)  # 200 without "studies"


# ------------------------------------------------------------------ openFDA
def test_ndc_parse_labeler_fallbacks():
    recs = json.loads((FIX / "ndc.json").read_text())
    erl = openfda.parse_ndc(recs["erlotinib"][0])
    assert erl["labeler"] == "Genentech, Inc." and erl["pharm_class"][0].startswith("Kinase Inhibitor")
    assert erl["route"] == "ORAL" and erl["active_ingredients"][0]["name"].startswith("ERLOTINIB")
    gef = openfda.parse_ndc(recs["gefitinib"][0])
    assert gef["labeler"] == "AstraZeneca Pharmaceuticals LP"       # labeler_name preferred
    bare = openfda.parse_ndc({"product_ndc": "1-1"})
    assert bare["labeler"] is None and bare["pharm_class"] == []
    assert openfda.parse_ndc({"brand_name": "x"}) is None            # no product_ndc


def test_faers_parse_drops_malformed():
    assert openfda.parse_faers_counts([{"term": "RASH", "count": 3}, {"term": "X"}, {"count": 2}, "junk"]) == [("RASH", 3)]


def test_openfda_404_is_empty(tmp_path, mock_server):
    c = HttpClient(tmp_path / "cache", backoff_s=0.01)
    assert openfda.fetch_ndc(c, mock_server + "/fda", "cisplatin", 5) == []
    assert len(openfda.fetch_ndc(c, mock_server + "/fda", "erlotinib", 5)) == 2


# ------------------------------------------------------------------ PubMed
def test_pubmed_parse():
    arts = pubmed.parse_pubmed_xml((FIX / "pubmed.xml").read_text())
    by = {a["pmid"]: a for a in arts}
    assert set(by) == {"111", "222", "333"}
    assert by["111"]["year"] == 2010 and by["222"]["year"] == 2019   # Year element and MedlineDate fallback
    assert "in vitro" in by["111"]["abstract"] and "<i>" not in by["111"]["abstract"]  # inline markup flattened
    assert by["111"]["abstract"].startswith("RESULTS:")
    assert by["333"]["abstract"] == "" and by["333"]["year"] is None
    assert pubmed.parse_pubmed_xml("") == []
    with pytest.raises(FetchError):
        pubmed.parse_pubmed_xml("<not xml")


# ------------------------------------------------------------------ GDSC
def test_gdsc_read_fixture():
    df = read_gdsc(FIX / "gdsc_mini.csv")
    assert len(df) == 36 and df["DRUG_NAME"].nunique() == 3
    assert {"LN_IC50", "AUC", "TCGA_DESC", "PUTATIVE_TARGET"} <= set(df.columns)


def test_gdsc_missing_required_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("COSMIC_ID,DRUG_NAME\n1,x\n")
    with pytest.raises(GdscFormatError, match="missing required column"):
        read_gdsc(p)


def test_gdsc_column_aliases_and_optional_missing(tmp_path):
    p = tmp_path / "alias.csv"
    p.write_text("COSMIC ID,CELL LINE NAME,DRUG ID,DRUG NAME,LN IC50,AUC\n1,A,5,Foo,0.5,0.9\n")
    df = read_gdsc(p)
    assert df.loc[0, "LN_IC50"] == 0.5 and df["TCGA_DESC"].isna().all()


# ------------------------------------------------------------------ GDSC required / optional
def test_fetch_requires_gdsc_by_default_and_can_opt_out(cfg, tmp_path):
    from biopharma_kg.ingest.fetch import fetch_all
    cfg.raw["gdsc"].update(file=None, url=None)          # no file, no URL -> unobtainable
    with pytest.raises(FileNotFoundError, match="cancerrxgene.org/downloads"):
        fetch_all(cfg, ["gdsc"])
    cfg.raw["gdsc"]["required"] = False
    out = fetch_all(cfg, ["gdsc"])
    assert out["sources"]["gdsc"]["file"] is None and "warning" in out["sources"]["gdsc"]
