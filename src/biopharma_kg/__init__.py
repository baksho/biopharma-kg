"""biopharma-kg: ontology-driven biomedical knowledge graph pipeline."""

__version__ = "0.1.0"

# Namespaces. The base IRI is a *placeholder* in the reserved ``.example`` TLD.
# Change it in configs/default.yaml (``kg.base_iri``) AND in
# mappings/*.ttl, ontology/biopharma.ttl and shapes/biopharma-shapes.ttl
# (``make rebase IRI=...`` does all four) before publishing the graph.
BASE_IRI = "https://biopharma-kg.example/"
NS = BASE_IRI + "ns#"
