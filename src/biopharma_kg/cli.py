"""Command-line interface: ``biopharma-kg <command>`` (or ``python -m biopharma_kg <command>``)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import BASE_IRI, __version__
from .config import load_config

SOURCES = ["gdsc", "ctgov", "openfda", "pubmed"]


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def rebase(root: Path, new_iri: str) -> int:
    """Replace the placeholder base IRI in ontology, shapes, mappings, queries and config."""
    if not new_iri.endswith("/"):
        raise SystemExit("--iri must end with '/' (e.g. https://example.org/biopharma/)")
    n = 0
    for pattern in ("ontology/*.ttl", "shapes/*.ttl", "mappings/*.ttl", "queries/*.rq", "configs/*.yaml"):
        for f in root.glob(pattern):
            s = f.read_text()
            if BASE_IRI in s:
                f.write_text(s.replace(BASE_IRI, new_iri))
                n += 1
    py = root / "src" / "biopharma_kg"
    for name in ("__init__.py", "kg/build.py", "kg/query.py"):
        f = py / name
        if f.exists():
            s = f.read_text()
            if BASE_IRI in s:
                f.write_text(s.replace(BASE_IRI, new_iri))
                n += 1
    print(f"Rewrote {n} file(s). Re-run `biopharma-kg build-kg` to regenerate the graph.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biopharma-kg", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", help="extra YAML config merged over configs/default.yaml")
    p.add_argument("--root", help="project root (default: current directory)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download source data for the seed drugs into data/raw")
    f.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES)
    f.add_argument("--force", action="store_true", help="re-fetch even if raw files exist")
    f.add_argument("--limit-drugs", type=int)
    sub.add_parser("load", help="raw files -> SQLite tables")
    sub.add_parser("resolve", help="entity resolution; writes reports/resolution_report.json")
    e = sub.add_parser("extract", help="relation extraction from PubMed abstracts (LLM or rules)")
    e.add_argument("--backend", choices=["auto", "llm", "rules", "both"])
    e.add_argument("--max-articles", type=int)
    a = sub.add_parser("audit-extraction", help="sample extracted relations to CSV for manual precision review")
    a.add_argument("--n", type=int, default=50)
    a.add_argument("--seed", type=int, default=0)
    sub.add_parser("build-kg", help="SQLite -> RDF (R2RML via morph-kgc) -> data/processed/kg.nq")
    v = sub.add_parser("validate", help="SHACL validation")
    v.add_argument("--strict", action="store_true", help="exit 1 if there are sh:Violation results")
    sub.add_parser("reason", help="OWL 2 RL materialisation (owlrl)")
    q = sub.add_parser("query", help="run SPARQL competency questions")
    q.add_argument("--only", nargs="+")
    m = sub.add_parser("ml", help="link-prediction experiments (needs the [ml] extra)")
    m.add_argument("--quick", action="store_true", help="5 epochs, 1 seed (for a fast check)")
    m.add_argument("--tasks", nargs="+")
    m.add_argument("--models", nargs="+")
    r = sub.add_parser("run-all", help="fetch -> load -> resolve -> extract -> build-kg -> validate -> reason -> query -> ml")
    r.add_argument("--skip-fetch", action="store_true")
    r.add_argument("--skip-ml", action="store_true")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--backend", choices=["auto", "llm", "rules", "both"])
    sub.add_parser("smoke", help="check live APIs + run a small end-to-end pipeline in data/smoke")
    sub.add_parser("status", help="show what exists on disk")
    rb = sub.add_parser("rebase", help="replace the placeholder base IRI everywhere")
    rb.add_argument("--iri", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    root = Path(args.root) if args.root else Path.cwd()
    if args.cmd == "rebase":
        return rebase(root, args.iri)
    cfg = load_config(args.config, root=root)

    try:
        if args.cmd == "fetch":
            from .ingest.fetch import fetch_all
            _print(fetch_all(cfg, args.sources, force=args.force, limit_drugs=args.limit_drugs))
        elif args.cmd == "load":
            from .load import load_all
            _print(load_all(cfg))
        elif args.cmd == "resolve":
            from .resolve.entities import resolve_all
            _print(resolve_all(cfg))
        elif args.cmd == "extract":
            from .extract.run import run_extraction
            _print(run_extraction(cfg, args.backend, max_articles=args.max_articles))
        elif args.cmd == "audit-extraction":
            from .extract.run import audit_sample
            _print(audit_sample(cfg, args.n, args.seed))
        elif args.cmd == "build-kg":
            from .kg.build import build_kg
            _print(build_kg(cfg))
        elif args.cmd == "validate":
            from .kg.validate import validate_kg
            res = validate_kg(cfg)
            _print(res)
            if args.strict and res["violations"]:
                return 1
        elif args.cmd == "reason":
            from .kg.reason import reason_kg
            _print(reason_kg(cfg))
        elif args.cmd == "query":
            from .kg.query import run_queries
            _print(run_queries(cfg, args.only))
        elif args.cmd == "ml":
            from .ml.run import run_ml
            _print(run_ml(cfg, quick=args.quick, tasks=args.tasks, models=args.models))
        elif args.cmd == "run-all":
            from .pipeline import run_all
            _print(run_all(cfg, quick=args.quick, skip_fetch=args.skip_fetch, skip_ml=args.skip_ml, backend=args.backend))
        elif args.cmd == "smoke":
            from .smoke import run_smoke
            rep = run_smoke(cfg)
            _print(rep)
            return 0 if rep.get("all_sources_ok") else 1
        elif args.cmd == "status":
            from .pipeline import status
            _print(status(cfg))
    except ImportError as e:
        if args.cmd in ("ml", "run-all", "smoke") and any(x in str(e) for x in ("torch", "pykeen")):
            print(f"Missing optional dependency: {e}\nInstall the ML extra:  pip install -e '.[ml]'", file=sys.stderr)
            return 2
        raise
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
