#!/usr/bin/env python3
"""Download the complete PADS 1.0.0 release and verify official SHA-256 hashes.

Uses only Python's standard library. Existing files are reused only if their
SHA-256 digest matches PhysioNet's manifest. Source files are never transformed.
"""

import argparse
import hashlib
import http.client
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "data" / "pads"
SOURCE = "https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/"
MIRROR = "https://physionet-open.s3.amazonaws.com/parkinsons-disease-smartwatch/1.0.0/"
CONNECTIONS = threading.local()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(url):
    return urlopen(Request(url, headers={"User-Agent": "PADS-public-data-acquisition/1.0"}), timeout=90)


def mirror_response(filename):
    # Reuse one HTTPS connection per worker to avoid 12,689 TLS handshakes.
    if not hasattr(CONNECTIONS, "connection"):
        CONNECTIONS.connection = http.client.HTTPSConnection("physionet-open.s3.amazonaws.com", timeout=90)
    connection = CONNECTIONS.connection
    connection.request("GET", "/parkinsons-disease-smartwatch/1.0.0/" + quote(filename, safe="/"),
                       headers={"User-Agent": "PADS-public-data-acquisition/1.0"})
    response = connection.getresponse()
    if response.status != 200:
        response.read()
        response.close()
        raise OSError(f"Official S3 mirror returned HTTP {response.status} for {filename}")
    return response


def entries(manifest):
    seen = set()
    for line in manifest.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        relative = PurePosixPath(filename)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid digest: {line!r}")
        if relative.is_absolute() or ".." in relative.parts or "\\" in filename or filename in seen:
            raise ValueError(f"Invalid or duplicate manifest path: {filename!r}")
        seen.add(filename)
        yield digest, filename


def acquire(entry, destination, verify_only=False):
    expected, filename = entry
    target = destination / filename
    if target.is_file() and sha256(target) == expected:
        return target.stat().st_size, "reused"
    if verify_only:
        raise ValueError(f"Missing file or checksum mismatch: {filename}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(5):
        try:
            digest = hashlib.sha256()
            with mirror_response(filename) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {filename}")
            os.replace(partial, target)
            return target.stat().st_size, "downloaded"
        except Exception:
            if hasattr(CONNECTIONS, "connection"):
                CONNECTIONS.connection.close()
                del CONNECTIONS.connection
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    provenance = BASE / "acquisition"
    destination = BASE / "1.0.0"
    provenance.mkdir(parents=True, exist_ok=True)
    manifest = provenance / "SHA256SUMS.remote.txt"
    if not manifest.exists():
        with request(SOURCE + "SHA256SUMS.txt") as response:
            manifest.write_bytes(response.read())
    release_entries = list(entries(manifest))
    print(f"Verifying {len(release_entries):,} official files; destination: {destination}", flush=True)
    start = time.monotonic()
    total_bytes = 0
    counts = {"downloaded": 0, "reused": 0}
    failures = []
    last_report = start
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire, e, destination, args.verify_only): e[1] for e in release_entries}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                size, status = future.result()
                total_bytes += size
                counts[status] += 1
            except Exception as error:
                failures.append({"path": futures[future], "error": str(error)})
                print(f"FAILED {futures[future]}: {error}", flush=True)
            now = time.monotonic()
            if now - last_report >= 20 or number == len(futures):
                print(f"{number:,}/{len(futures):,} files completed; {total_bytes / 1e6:.1f} MB verified; "
                      f"{len(failures)} failures; {now - start:.0f}s elapsed", flush=True)
                last_report = now
    report = {
        "dataset": "PADS - Parkinsons Disease Smartwatch dataset",
        "version": "1.0.0",
        "dataset_doi": "10.13026/m0w9-zx22",
        "dataset_page": "https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/",
        "official_file_source": SOURCE,
        "download_mirror": MIRROR,
        "checksum_manifest_url": SOURCE + "SHA256SUMS.txt",
        "checksum_manifest_sha256": sha256(manifest),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_release_directory": str(destination),
        "expected_files": len(release_entries),
        "verified_files": sum(counts.values()),
        "verified_bytes": total_bytes,
        "counts": counts,
        "all_official_checksums_passed": not failures,
        "failures": failures,
        "license": "CC BY-NC-SA 4.0",
        "license_page": "https://physionet.org/content/parkinsons-disease-smartwatch/view-license/1.0.0/",
        "source_files_transformed": False,
    }
    name = "integrity_verification.json" if args.verify_only else "download_manifest.json"
    (provenance / name).write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} files failed. Rerun to retry; valid files will be reused.")
    # The checksum list does not checksum itself. Preserve its downloaded bytes
    # beside the release so the local directory also contains this upstream file.
    (destination / "SHA256SUMS.txt").write_bytes(manifest.read_bytes())
    print(f"SUCCESS: All {len(release_entries):,} files match the official SHA-256 manifest.", flush=True)


if __name__ == "__main__":
    main()
