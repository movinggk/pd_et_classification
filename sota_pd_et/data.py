"""Read the existing official PADS binaries without changing the source dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "pads" / "1.0.0"
LABELS = {"Parkinson's": 0, "Essential Tremor": 1}
MOVEMENT_SHAPE = (132, 976)
RETAINED_SEGMENTS = (
    "Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold",
    "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1",
    "Entrainment2",
)
CHANNEL_NAMES = tuple(
    f"{task}_{wrist}_{sensor}_{axis}"
    for task in RETAINED_SEGMENTS
    for wrist in ("LeftWrist", "RightWrist")
    for sensor in ("Accelerometer", "Gyroscope")
    for axis in ("X", "Y", "Z")
)


@dataclass
class Cohort:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    rows: list[dict]
    provenance: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cohort(root: str | Path = DEFAULT_DATA_ROOT) -> Cohort:
    """Select ET by its condition string; original label 2 is NOT an ET target.

    X[i] contains one participant's entire (132, 976) movement representation.
    The 132 channels are task-segment/wrist/sensor/axis combinations, rather than
    132 simultaneously recorded physical axes. No sliding windows are created.
    """
    root = Path(root).expanduser().resolve()
    checksum_path = root / "SHA256SUMS.txt"
    checksums = {}
    for line in checksum_path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            checksums[name.removeprefix("*").removeprefix("./")] = digest
    source_hashes = {}

    def checked(relative: str) -> Path:
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Path escapes the data directory: {relative}")
        if relative not in checksums:
            raise ValueError(f"File missing from the official checksum list: {relative}")
        actual = sha256_file(path)
        if actual != checksums[relative]:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        source_hashes[relative] = actual
        return path

    with checked("preprocessed/file_list.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    if not table or not {"id", "condition"}.issubset(table[0]):
        raise ValueError("The official participant table needs id and condition columns.")
    if len({r["id"] for r in table}) != len(table):
        raise ValueError("Duplicate participant IDs in the official table.")
    selected = sorted((r for r in table if r["condition"] in LABELS), key=lambda r: r["id"])
    if not selected or {r["condition"] for r in selected} != set(LABELS):
        raise ValueError("Both published PD and ET conditions must be present.")
    # These two files define the shape and ordering of the supplied binary inputs.
    checked("scripts/run_preprocessing.py")
    checked("scripts/utils/data_handling.py")
    X = np.empty((len(selected), *MOVEMENT_SHAPE), dtype=np.float32)
    rows = []
    for index, participant in enumerate(selected):
        pid = participant["id"]
        if re.fullmatch(r"[0-9]{3}", pid) is None:
            raise ValueError(f"Unexpected participant ID: {pid!r}")
        patient = json.loads(checked(f"patients/patient_{pid}.json").read_text())
        if str(patient["id"]) != pid or patient["condition"] != participant["condition"]:
            raise ValueError(f"Participant table/JSON disagreement: {pid}")
        path = checked(f"preprocessed/movement/{pid}_ml.bin")
        if path.stat().st_size != int(np.prod(MOVEMENT_SHAPE)) * 4:
            raise ValueError(f"Expected a little-endian float32 {MOVEMENT_SHAPE} file: {path}")
        values = np.fromfile(path, dtype="<f4").reshape(MOVEMENT_SHAPE)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite movement samples: {pid}")
        X[index] = values
        condition = participant["condition"]
        rows.append({
            "participant_id": pid,
            "condition": condition,
            "target_et": LABELS[condition],
            # Audit flag only. This is never included in the model input.
            "pd_with_et_comment": condition == "Parkinson's" and
            "essential tremor" in participant.get("disease_comment", "").lower(),
        })
    return Cohort(
        X=X,
        y=np.array([r["target_et"] for r in rows], dtype=np.int64),
        ids=np.array([r["participant_id"] for r in rows]),
        rows=rows,
        provenance={
            "data_root": str(root),
            "dataset": "PADS v1.0.0",
            "dataset_doi": "https://doi.org/10.13026/m0w9-zx22",
            "checksum_list_sha256": sha256_file(checksum_path),
            "verified_source_sha256": source_hashes,
            "label_mapping": LABELS,
            "participant_count": len(rows),
            "condition_counts": {condition: sum(r["condition"] == condition for r in rows)
                                 for condition in LABELS},
            "movement_shape": list(X.shape),
            "channel_names": list(CHANNEL_NAMES),
            "input": "Author-supplied movement binaries only; no questionnaire or demographics.",
            "preprocessing": "Published gravity correction, segment/task selection and 48-sample crop.",
            "timestamp_handling": "Binaries use sample indices. Raw timestamps are not resampled or gap-filtered.",
            "pd_with_et_comment_ids": [r["participant_id"] for r in rows if r["pd_with_et_comment"]],
        },
    )
