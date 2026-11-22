"""Shared test fixtures: a local mock of the four public APIs and a config pointing at it.

The mock serves the *hand-written* files in tests/fixtures (see the README there). It exercises our
client/parsers/pagination/404 handling; it cannot prove the live services still look like this, which is
what `biopharma-kg smoke` is for.
"""
from __future__ import annotations

import json
import re
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from biopharma_kg.config import load_config

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"


def _drug_from(q: str) -> str:
    m = re.search(r'"([^"]+)"', q)
    return (m.group(1) if m else q).lower()


class _Handler(BaseHTTPRequestHandler):
    calls: list[str] = []

    def log_message(self, *a):  # silence
        pass

    def _send(self, code: int, body: str, ctype="application/json"):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        _Handler.calls.append(self.path)
        ct = json.loads((FIX / "ctgov.json").read_text())
        if u.path == "/ctgov/studies":
            drug = q.get("query.intr", "").lower()
            if drug == "erlotinib":
                page = ct["erlotinib_p2"] if q.get("pageToken") == "tok2" else ct["erlotinib_p1"]
            else:
                page = ct.get(drug, {"studies": []})
            return self._send(200, json.dumps(page))
        if u.path == "/badctgov/studies":  # 200 but wrong shape: simulates schema drift
            return self._send(200, json.dumps({"unexpected": []}))
        if u.path == "/fda/drug/ndc.json":
            res = json.loads((FIX / "ndc.json").read_text()).get(_drug_from(q.get("search", "")))
            if not res:
                return self._send(404, json.dumps({"error": {"code": "NOT_FOUND", "message": "No matches found!"}}))
            return self._send(200, json.dumps({"results": res}))
        if u.path == "/fda/drug/event.json":
            res = json.loads((FIX / "faers.json").read_text()).get(_drug_from(q.get("search", "")))
            if not res:
                return self._send(404, json.dumps({"error": {"code": "NOT_FOUND"}}))
            return self._send(200, json.dumps({"results": res}))
        if u.path == "/eutils/esearch.fcgi":
            ids = {"erlotinib": ["111"], "gefitinib": ["222", "333"]}.get(_drug_from(q.get("term", "")), [])
            return self._send(200, json.dumps({"esearchresult": {"idlist": ids}}))
        if u.path == "/eutils/efetch.fcgi":
            want = set(q.get("id", "").split(","))
            root = ET.fromstring((FIX / "pubmed.xml").read_text())
            for art in list(root):
                if art.find("./MedlineCitation/PMID").text not in want:
                    root.remove(art)
            return self._send(200, ET.tostring(root, encoding="unicode"), "text/xml")
        if u.path == "/flaky":
            n = sum(1 for c in _Handler.calls if c.startswith("/flaky"))
            return self._send(503 if n < 3 else 200, json.dumps({"ok": n >= 3}))
        return self._send(404, "{}")


@pytest.fixture(scope="session")
def mock_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture()
def cfg(tmp_path, mock_server):
    """Config rooted at the repo (so mappings/ontology/shapes/queries resolve), data in tmp, APIs -> mock."""
    c = load_config(root=REPO)
    raw = c.raw
    raw["paths"] = {"data_dir": str(tmp_path / "data"), "reports_dir": str(tmp_path / "reports")}
    raw["seed"] = {"drug_list": str(tmp_path / "seeds.txt"), "max_drugs": None, "use_gdsc_drugs": True}
    (tmp_path / "seeds.txt").write_text("erlotinib\ngefitinib\ncisplatin\n")
    raw["gdsc"]["file"] = str(FIX / "gdsc_mini.csv")
    raw["ctgov"].update(base_url=mock_server + "/ctgov", page_size=3, max_studies_per_drug=50)
    raw["openfda"].update(base_url=mock_server + "/fda")
    raw["pubmed"].update(base_url=mock_server + "/eutils")
    raw["http"].update(max_retries=3, backoff_s=0.01, timeout_s=5)
    raw["extraction"].update(backend="rules", min_condition_frequency=1)
    return c
