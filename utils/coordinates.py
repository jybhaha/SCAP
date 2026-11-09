"""Shared Stage-2 coordinate files for SCAP and patch-baseline comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1


def canonical_coords(coords):
    return [[list(map(int, box)) for box in sample] for sample in coords]


def coords_sha256(coords):
    payload = json.dumps(canonical_coords(coords), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def save_stage2_coords(path, coords_by_sample, image_size=None):
    """Save coordinates without silently changing their order or values."""
    records = []
    for sample_id, coords in coords_by_sample.items():
        canonical = canonical_coords(coords)
        records.append({"sample_id": str(sample_id), "coords": canonical, "sha256": coords_sha256(coords)})
    document = {
        "schema_version": SCHEMA_VERSION,
        "convention": "(x1,y1,x2,y2), zero-based, end-exclusive",
        "image_size": list(image_size) if image_size is not None else None,
        "samples": records,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_stage2_coords(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported coordinate schema: {document.get('schema_version')}")
    result = {}
    for record in document["samples"]:
        coords = canonical_coords(record["coords"])
        if coords_sha256(coords) != record["sha256"]:
            raise ValueError(f"Coordinate checksum mismatch for sample {record['sample_id']}")
        result[str(record["sample_id"])] = coords
    return result
