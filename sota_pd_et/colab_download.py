"""Download the exact official PADS v1.0.0 PD/ET input subset, using stdlib only.

The checksum list is pinned independently of the downloaded data. Every cached
or newly downloaded file must match that list. Only the participant table, two
preprocessing source files (for provenance), and 304 participants' JSON/movement
files are acquired. This module never downloads the complete dataset archive or
executes the official preprocessing scripts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from urllib.error import URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


BASE_URL = "https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/"
CHECKSUM_LIST_SHA256 = "514cd95405a12afcdfb126d47d1f559e2e8a744f03e586c3088d5e4fd7b02c46"
EXPECTED_CONDITION_COUNTS = {"Parkinson's": 276, "Essential Tremor": 28}
EXPECTED_TABLE_SIZE = 469
MOVEMENT_BYTES = 132 * 976 * 4  # Published little-endian float32 representation.
TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 4
BLOCK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(relative: str) -> str:
    """Accept only canonical relative POSIX paths, without traversal or URLs."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Unsafe dataset path: {relative!r}")
    path = PurePosixPath(relative)
    if (path.is_absolute() or any(p in {"", ".", ".."} for p in relative.split("/"))
            or ":" in relative or any(ord(c) < 32 for c in relative)
            or str(path) != relative):
        raise ValueError(f"Unsafe dataset path: {relative!r}")
    return relative


def _destination(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = root / _safe_relative(relative)
    # Also reject existing symlinks that would direct writes outside the root.
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Dataset path escapes the output directory: {relative!r}")
    return path


def _read_checksums(path: Path) -> dict[str, str]:
    checksums = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]) is None:
            raise ValueError(f"Malformed checksum-list line {number}.")
        relative = fields[1].removeprefix("*").removeprefix("./")
        relative = _safe_relative(relative)
        if relative in checksums:
            raise ValueError(f"Duplicate checksum-list path: {relative!r}")
        checksums[relative] = fields[0].lower()
    if not checksums:
        raise ValueError("The official checksum list is empty.")
    return checksums


def _download_verified(root: Path, relative: str, expected_sha256: str) -> dict:
    """Reuse a matching local file or atomically replace it after verification."""
    destination = _destination(root, relative)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return {"path": relative, "sha256": expected_sha256,
                "bytes": destination.stat().st_size, "reused": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + quote(relative, safe="/")
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        temporary = None
        try:
            request = Request(url, headers={
                "User-Agent": "PADS-PD-ET-Colab/1.0 (selective research download)",
                "Accept-Encoding": "identity",
            })
            digest = hashlib.sha256()
            size = 0
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                resolved = urlsplit(response.geturl())
                origin = urlsplit(BASE_URL)
                if (resolved.scheme != "https" or resolved.netloc != origin.netloc
                        or not resolved.path.startswith(origin.path)):
                    raise ValueError(f"Unexpected download redirect: {response.geturl()}")
                with tempfile.NamedTemporaryFile(
                        mode="wb", dir=destination.parent,
                        prefix=f".{destination.name}.", suffix=".part", delete=False) as output:
                    temporary = Path(output.name)
                    while True:
                        block = response.read(BLOCK_BYTES)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                    output.flush()
                    os.fsync(output.fileno())
            if digest.hexdigest() != expected_sha256:
                raise ValueError(f"SHA-256 mismatch for {relative}")
            os.replace(temporary, destination)
            return {"path": relative, "sha256": expected_sha256,
                    "bytes": size, "reused": False}
        except (OSError, URLError, HTTPException, ValueError) as error:
            last_error = error
            if attempt < MAX_ATTEMPTS:
                print(f"Retry {attempt}/{MAX_ATTEMPTS - 1}: {relative}: {error}", flush=True)
                time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not acquire verified file after {MAX_ATTEMPTS} attempts: {relative}"
    ) from last_error


def _select_participants(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"id", "condition"}.issubset(reader.fieldnames):
            raise ValueError("The participant table needs id and condition columns.")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if any(not isinstance(pid, str) or re.fullmatch(r"[0-9]{3}", pid) is None for pid in ids):
        raise ValueError("Every participant ID must contain exactly three decimal digits.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate participant IDs in the official table.")
    if len(rows) != EXPECTED_TABLE_SIZE:
        raise ValueError(f"Expected {EXPECTED_TABLE_SIZE} official participants, found {len(rows)}.")
    selected = sorted((row for row in rows if row["condition"] in EXPECTED_CONDITION_COUNTS),
                      key=lambda row: row["id"])
    counts = {condition: sum(row["condition"] == condition for row in selected)
              for condition in EXPECTED_CONDITION_COUNTS}
    if counts != EXPECTED_CONDITION_COUNTS:
        raise ValueError(f"Unexpected PD/ET condition counts: {counts}")
    return selected


def download_dataset(root: str | Path, workers: int = 4) -> dict:
    """Acquire and validate the 304-participant PD/ET subset of PADS v1.0.0.

    ``root`` is the version directory (for example ``/content/pads/1.0.0``).
    Up to six workers are allowed to keep requests to PhysioNet modest. The
    returned JSON-serializable provenance dictionary can be saved by the caller
    alongside experiment results. No acquisition manifest is written into the
    source dataset directory. Re-running verifies hashes before reusing files.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 6:
        raise ValueError("workers must be an integer between 1 and 6.")
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print(f"Preparing verified PADS v1.0.0 PD/ET inputs in {root}", flush=True)
    results = [_download_verified(root, "SHA256SUMS.txt", CHECKSUM_LIST_SHA256)]
    checksums = _read_checksums(root / "SHA256SUMS.txt")
    table_name = "preprocessed/file_list.csv"
    if table_name not in checksums:
        raise ValueError(f"File missing from official checksum list: {table_name}")
    results.append(_download_verified(root, table_name, checksums[table_name]))
    participants = _select_participants(root / table_name)
    relative_files = ["scripts/run_preprocessing.py", "scripts/utils/data_handling.py"]
    for row in participants:
        relative_files.extend((f"patients/patient_{row['id']}.json",
                               f"preprocessed/movement/{row['id']}_ml.bin"))
    missing = [name for name in relative_files if name not in checksums]
    if missing:
        raise ValueError(f"Files missing from official checksum list: {missing}")
    print(f"Selected {len(participants)} participants: {EXPECTED_CONDITION_COUNTS}. "
          f"Checking {len(relative_files) + 2} files with {workers} workers.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_verified, root, name, checksums[name]): name
                   for name in relative_files}
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if completed % 25 == 0 or completed == len(relative_files):
                    reused = sum(result["reused"] for result in results)
                    print(f"Verified {completed + 2}/{len(relative_files) + 2} files "
                          f"({reused} reused).", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    # Cross-check semantic identity as well as file hashes before model loading.
    for row in participants:
        pid = row["id"]
        patient_path = _destination(root, f"patients/patient_{pid}.json")
        patient = json.loads(patient_path.read_text(encoding="utf-8"))
        if patient.get("id") != pid or patient.get("condition") != row["condition"]:
            raise ValueError(f"Participant table/JSON disagreement: {pid}")
        movement_path = _destination(root, f"preprocessed/movement/{pid}_ml.bin")
        if movement_path.stat().st_size != MOVEMENT_BYTES:
            raise ValueError(f"Unexpected movement binary size for participant {pid}.")
    reused = sum(result["reused"] for result in results)
    metadata = {
        "dataset": "PADS v1.0.0", "dataset_doi": "https://doi.org/10.13026/m0w9-zx22",
        "base_url": BASE_URL, "data_root": str(root), "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checksum_list_sha256": CHECKSUM_LIST_SHA256,
        "participant_count": len(participants),
        "condition_counts": dict(EXPECTED_CONDITION_COUNTS),
        "participant_ids": [row["id"] for row in participants],
        "verified_file_count": len(results), "downloaded_files": len(results) - reused,
        "reused_files": reused, "verified_bytes": sum(result["bytes"] for result in results),
        "verified_source_sha256": {result["path"]: result["sha256"]
                                   for result in sorted(results, key=lambda item: item["path"])},
    }
    print(f"Ready: {len(participants)} participants; {len(results)} verified files; "
          f"{len(results) - reused} downloaded and {reused} reused.", flush=True)
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="PADS version directory to populate")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests (1-6)")
    arguments = parser.parse_args()
    record = download_dataset(arguments.root, arguments.workers)
    print(json.dumps({key: value for key, value in record.items()
                      if key not in {"participant_ids", "verified_source_sha256"}}, indent=2))
