"""Measure KG build + OWL-RL reasoning cost on synthetic databases of increasing size (uses tests/synth_db.py).

    python scripts/scale_check.py 100 400 1200

Prints one line per size: drugs, KG triples, triples fed to the reasoner, build seconds, reasoning seconds, peak RSS (MB).
Numbers depend on the machine; use them to choose `kg.reasoning_max_triples` for yours."""
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from biopharma_kg.config import load_config  # noqa: E402
from biopharma_kg.kg.build import build_kg, quad_stats  # noqa: E402
from biopharma_kg.kg.reason import reason_kg  # noqa: E402
from synth_db import make_synthetic_db  # noqa: E402


def main(sizes):
    for n in sizes:
        tmp = Path(tempfile.mkdtemp())
        cfg = load_config(root=ROOT)
        cfg.raw["paths"] = {"data_dir": str(tmp / "data"), "reports_dir": str(tmp / "reports")}
        cfg.raw["kg"]["reasoning_max_triples"] = 10**9  # never skip: we are measuring
        make_synthetic_db(cfg.db_path, n_drugs=n, n_cond=max(60, n // 2), n_cl=max(120, n), n_ae=max(50, n // 2), trials_per_drug=10)
        t0 = time.time(); build_kg(cfg); tb = time.time() - t0
        total = sum(quad_stats(cfg.kg_nquads).get("by_graph", {}).values()) if isinstance(quad_stats(cfg.kg_nquads), dict) else -1
        t0 = time.time(); rep = reason_kg(cfg); tr = time.time() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"drugs={n:5d} kg_triples={rep['input_triples_total']:8d} reasoner_input={rep['input_triples_for_reasoning']:8d} "
              f"inferred_kept={rep['inferred_triples_kept']:8d} build_s={tb:6.1f} reason_s={tr:7.1f} peak_rss_mb={rss:7.0f}", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [100, 400])
