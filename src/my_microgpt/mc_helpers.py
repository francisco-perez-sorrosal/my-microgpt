"""Mechanistic interpretability helpers: concept vector data model and persistence."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConceptVectorData:
    """A concept vector with its extraction metadata."""

    concept: str
    layer: int
    hook_point: str
    alpha: float
    num_positive: int
    num_negative: int
    vector: list[float]


def save_concept_vector(data: ConceptVectorData, path: str) -> str:
    """Save a concept vector to JSON. Returns the path."""
    payload = {
        "concept": data.concept,
        "layer": data.layer,
        "hook_point": data.hook_point,
        "alpha": data.alpha,
        "num_positive": data.num_positive,
        "num_negative": data.num_negative,
        "vector": data.vector,
    }
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f"concept vector saved to {path} ({Path(path).stat().st_size:,} bytes)")
    return path


def load_concept_vector(path: str) -> ConceptVectorData:
    """Load a concept vector from JSON."""
    raw = json.loads(Path(path).read_text())
    data = ConceptVectorData(
        concept=raw["concept"],
        layer=raw["layer"],
        hook_point=raw["hook_point"],
        alpha=raw["alpha"],
        num_positive=raw["num_positive"],
        num_negative=raw["num_negative"],
        vector=raw["vector"],
    )
    print(f"concept vector loaded from {path} (concept='{data.concept}', layer={data.layer}, dim={len(data.vector)})")
    return data
