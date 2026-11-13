#!/usr/bin/env python3
"""Validate local PADS structure and write participant/recording indexes.

Does not change any upstream file, generate model inputs, or assign data splits.
"""

import csv
import json
import math
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "data" / "pads"
RELEASE = BASE / "1.0.0"
OUTPUT = BASE / "acquisition"
CHANNELS = ["Time", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z",
            "Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z"]
UNITS = ["s", "g", "g", "g", "rad/s", "rad/s", "rad/s"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    acquisition = json.loads((OUTPUT / "download_manifest.json").read_text())
    require(acquisition["all_official_checksums_passed"], "Download integrity verification did not pass.")
    with (RELEASE / "preprocessed/file_list.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        participants = list(reader)
    by_id = {row["id"]: row for row in participants}
    expected_ids = {f"{i:03d}" for i in range(1, 470)}
    require(len(participants) == 469 and set(by_id) == expected_ids, "Unexpected participant IDs.")
    conditions = Counter(row["condition"] for row in participants)
    binary = [row for row in participants if row["condition"] in ("Parkinson's", "Essential Tremor")]
    write_csv(OUTPUT / "pd_et_participants.csv", fields, binary)
    # These labels reproduce the published condition strings, with no comment-based relabeling.
    numeric_examples = {next(row["id"] for row in participants if row["condition"] == condition)
                        for condition in ("Parkinson's", "Essential Tremor", "Healthy")}
    inventory = []
    referenced = set()
    sessions = 0
    raw_rows = 0
    numeric_sample_files = 0
    questionnaire_items = Counter()
    task_counts = Counter()
    row_counts = Counter()
    overlapping_comments = []
    for number, row in enumerate(participants, 1):
        subject = row["id"]
        patient = json.loads((RELEASE / f"patients/patient_{subject}.json").read_text())
        require(str(patient["id"]) == subject and patient["condition"] == row["condition"],
                f"Patient/table mismatch for {subject}.")
        if row["condition"] == "Parkinson's" and "essential tremor" in row["disease_comment"].lower():
            overlapping_comments.append(subject)
        questionnaire = json.loads((RELEASE / f"questionnaire/questionnaire_response_{subject}.json").read_text())
        require(questionnaire["subject_id"] == subject, f"Questionnaire ID mismatch for {subject}.")
        questionnaire_items[len(questionnaire["item"])] += 1
        require(len(questionnaire["item"]) == 30, f"Unexpected questionnaire size for {subject}.")
        observation = json.loads((RELEASE / f"movement/observation_{subject}.json").read_text())
        require(observation["subject_id"] == subject, f"Observation ID mismatch for {subject}.")
        require(observation["sampling_rate"] == 100, f"Unexpected nominal sample rate for {subject}.")
        require(len(observation["session"]) == 11, f"Unexpected session count for {subject}.")
        for session in observation["session"]:
            sessions += 1
            task_counts[session["record_name"]] += 1
            require(len(session["records"]) == 2, f"Unexpected wrist count for {subject}.")
            require({r["device_location"] for r in session["records"]} == {"LeftWrist", "RightWrist"},
                    f"Unexpected wrists for {subject}.")
            for record in session["records"]:
                require(record["channels"] == CHANNELS and record["units"] == UNITS,
                        f"Unexpected columns or units for {subject}.")
                relative = Path("movement") / record["file_name"]
                path = RELEASE / relative
                require(path.resolve().is_relative_to(RELEASE.resolve()), "Invalid recording path.")
                require(path.is_file(), f"Missing raw recording: {relative}.")
                require(relative.as_posix() not in referenced, f"Duplicated raw reference: {relative}.")
                referenced.add(relative.as_posix())
                count = 0
                with path.open("rb") as stream:
                    for line in stream:
                        require(line.count(b",") == 6, f"Unexpected column count: {relative}, row {count + 1}.")
                        if subject in numeric_examples:
                            values = [float(value) for value in line.split(b",")]
                            require(all(math.isfinite(value) for value in values),
                                    f"Nonfinite sample value: {relative}, row {count + 1}.")
                        count += 1
                require(count == session["rows"], f"Raw row count mismatch: {relative}.")
                raw_rows += count
                row_counts[count] += 1
                numeric_sample_files += subject in numeric_examples
                inventory.append({"participant_id": subject, "condition": row["condition"],
                                  "task": session["record_name"], "wrist": record["device_location"],
                                  "sampling_rate_hz": observation["sampling_rate"],
                                  "rows": count, "columns": len(CHANNELS), "raw_path": relative.as_posix()})
        # Shapes and float32 serialization follow the released preprocessing script.
        require((RELEASE / f"preprocessed/movement/{subject}_ml.bin").stat().st_size == 132 * 976 * 4,
                f"Unexpected preprocessed movement size for {subject}.")
        require((RELEASE / f"preprocessed/questionnaire/{subject}_ml.bin").stat().st_size == 30 * 4,
                f"Unexpected preprocessed questionnaire size for {subject}.")
        questionnaire_values = struct.unpack("<30f", (RELEASE / f"preprocessed/questionnaire/{subject}_ml.bin").read_bytes())
        require(questionnaire_values == tuple(float(item["answer"]) for item in questionnaire["item"]),
                f"Questionnaire binary/JSON mismatch for {subject}.")
        if subject in numeric_examples:
            movement_values = struct.iter_unpack("<f", (RELEASE / f"preprocessed/movement/{subject}_ml.bin").read_bytes())
            require(all(math.isfinite(value[0]) for value in movement_values),
                    f"Nonfinite preprocessed movement sample for {subject}.")
        if number % 100 == 0:
            print(f"Structurally validated {number}/469 participants.", flush=True)
    actual = {p.relative_to(RELEASE).as_posix() for p in (RELEASE / "movement/timeseries").glob("*.txt")}
    require(actual == referenced, "Unreferenced or missing raw time-series files.")
    for folder, pattern in [("patients", "patient_*.json"), ("movement", "observation_*.json"),
                            ("questionnaire", "questionnaire_response_*.json")]:
        require(len(list((RELEASE / folder).glob(pattern))) == 469, f"Unexpected file count in {folder}.")
    write_csv(OUTPUT / "recordings.csv", list(inventory[0]), inventory)
    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "participant_count": len(participants),
        "condition_counts": dict(sorted(conditions.items())),
        "pd_et_participant_count": len(binary),
        "pd_et_condition_counts": dict(Counter(row["condition"] for row in binary)),
        "pd_et_selection": "Published condition is exactly Parkinson's or Essential Tremor.",
        "upstream_numeric_label_warning": "Label 2 combines all differential diagnoses; it is not an ET-only label.",
        "pd_ids_with_essential_tremor_comment": overlapping_comments,
        "comment_policy": "Flag only; no exclusions or relabeling applied.",
        "movement_sessions": sessions,
        "raw_recording_files": len(inventory),
        "raw_row_count": raw_rows,
        "raw_rows_per_file_counts": dict(sorted(row_counts.items())),
        "raw_columns": CHANNELS,
        "raw_units": UNITS,
        "task_session_counts": dict(task_counts),
        "all_raw_row_counts_and_column_widths_checked": True,
        "all_raw_paths_match_observation_references": True,
        "numeric_finiteness_sample_participant_ids": sorted(numeric_examples),
        "numeric_finiteness_sample_raw_files": numeric_sample_files,
        "numeric_finiteness_scope": "All values in all raw recordings of the three listed participants only.",
        "questionnaire_items_per_participant": dict(questionnaire_items),
        "preprocessed_movement_shape": [132, 976],
        "preprocessed_movement_dtype": "little-endian float32",
        "preprocessed_shape_verification": "Byte size checked for all 469 movement and 469 questionnaire files; not a full numeric audit.",
        "all_preprocessed_questionnaires_match_raw_json": True,
        "preprocessed_movement_numeric_finiteness_sample_participant_ids": sorted(numeric_examples),
        "source_files_transformed": False,
        "training_performed": False,
    }
    (OUTPUT / "validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
