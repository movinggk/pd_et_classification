# PADS dataset acquisition

Acquired on **7 October 2026** for this PD-versus-ET classification project.

- Dataset: **PADS - Parkinsons Disease Smartwatch dataset**, version **1.0.0**.
- Official source: <https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/>.
- Dataset DOI: <https://doi.org/10.13026/m0w9-zx22>.
- Files were downloaded through PhysioNet's public `physionet-open` S3 mirror, which the official dataset page lists as a download option.
- All **12,689 files** in PhysioNet's published SHA-256 list were downloaded and checked against their respective hashes. The checksum list itself is also preserved in the local release directory; it is not included in its own hash list.
- The checksummed files total **1,516,287,599 bytes**; including the checksum list, the local release totals **1,517,701,375 bytes** (about **1.52 GB**, or **1.41 GiB**).
- Upstream files retain their original names, directories, bytes, diagnosis assignments, and preprocessing. No model was trained.

## Local contents

The complete release is in [`1.0.0/`](1.0.0/).

| Location | Contents |
| --- | --- |
| `1.0.0/movement/timeseries/` | 10,318 raw wrist recordings as comma-separated text files |
| `1.0.0/movement/observation_*.json` | 469 observation files linking participants, tasks, wrists, channels, units, and expected row counts |
| `1.0.0/patients/` | 469 participant JSON files with diagnoses and demographics |
| `1.0.0/questionnaire/` | 469 questionnaire JSON files, each containing 30 items |
| `1.0.0/preprocessed/movement/` | 469 movement binary files supplied by the dataset authors |
| `1.0.0/preprocessed/questionnaire/` | 469 questionnaire binary files supplied by the dataset authors |
| [`1.0.0/preprocessed/file_list.csv`](1.0.0/preprocessed/file_list.csv) | Official participant table, including explicit `condition` and combined numeric `label` |
| `1.0.0/scripts/` | The authors' loader, preprocessing, and example scripts |
| [`acquisition/download_manifest.json`](acquisition/download_manifest.json) | Sources, version, timestamps, verified file count, exact byte count, and checksum results |
| [`acquisition/validation_summary.json`](acquisition/validation_summary.json) | Local structural checks and their explicit scope |
| [`acquisition/recordings.csv`](acquisition/recordings.csv) | One row per raw recording, with participant ID, diagnosis, task, wrist, sampling rate, shape, and relative path |
| [`acquisition/pd_et_participants.csv`](acquisition/pd_et_participants.csv) | 304 official participant-table rows selected by the PD/ET condition strings |

## Diagnosis counts and PD/ET selection

Counts are independent participants, not recording or window counts.

| Published `condition` | Participants |
| --- | ---: |
| `Parkinson's` | 276 |
| `Essential Tremor` | 28 |
| `Healthy` | 79 |
| `Atypical Parkinsonism` | 15 |
| `Multiple Sclerosis` | 11 |
| `Other Movement Disorders` | 60 |
| **Total** | **469** |

The PD/ET index selects rows with `condition` exactly equal to `Parkinson's` or `Essential Tremor`, yielding **276 PD + 28 ET = 304 participants**. It preserves every original table column, including the original `label`; it does not define a new numeric binary target.

The original numeric label is `0` for healthy, `1` for PD, and `2` for the combined differential-diagnosis group. **Selecting all rows with `label == 2` does not select ET alone.** Use the named `condition` to select ET.

PD records whose `disease_comment` mentions essential tremor are flagged in the validation summary. They retain their published diagnosis and are included in the PD/ET index; no comment-based exclusions or relabeling were applied.

## Signal formats

Raw files have no header and contain seven columns in this order:

1. Time, in seconds.
2. Accelerometer X, in g.
3. Accelerometer Y, in g.
4. Accelerometer Z, in g.
5. Gyroscope X, in radians per second.
6. Gyroscope Y, in radians per second.
7. Gyroscope Z, in radians per second.

The observation metadata specifies a nominal sampling rate of **100 Hz**. Each participant has 11 tasks and two wrists per task, giving 22 raw files per participant. Each file has 1,024 or 2,048 rows; the observation JSON supplies the expected row count for each task.

The supplied preprocessed movement files are headerless float32 arrays with shape **(132, 976)** in channel-major order. Their channel order follows the released `scripts/run_preprocessing.py`: 11 retained task segments, then left/right wrist, then accelerometer/gyroscope, then X/Y/Z. The script splits the longer recordings, removes the `LiftHold`, `PointFinger`, and `TouchIndex` tasks and time channels, removes an estimated gravitational offset from acceleration, and crops the first 48 samples. These operations were already applied by the authors; acquisition did not rerun them.

Each preprocessed questionnaire file contains 30 float32 values. Follow the raw questionnaire JSON for question text and ordering.

For work with raw recordings, the authors recommend removing approximately the first 0.5 seconds because the watch vibrates at recording onset. Both wrists, all tasks, and all windows from one participant should remain in the same evaluation fold.

## Verification and reproduction

Checksum verification covered every file in the official manifest. Structural checks covered all participant, observation, and questionnaire JSON files; every raw recording's row count and seven-column width; every recording reference; and all preprocessed binary file sizes. All 469 preprocessed questionnaires were decoded as little-endian float32 and compared with their raw JSON answers. Numeric finiteness was checked for every value in the raw recordings and preprocessed movement arrays of three explicitly listed participants, one each from PD, ET, and healthy controls. This is an acquisition/structure check, not a full numeric or clinical audit. See the validation summary for exact scope and results.

From the project root, using Python's standard library:

```sh
python3 scripts/acquire_pads.py --workers 24
python3 scripts/acquire_pads.py --verify-only
python3 scripts/inspect_pads.py
```

The acquisition command reuses existing files only when their hashes match the official manifest. The verification-only command performs no network requests when the saved checksum list is present.

## License and citation

The official license is **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. See the [official license page](https://physionet.org/content/parkinsons-disease-smartwatch/view-license/1.0.0/) and the saved [`acquisition/license-page.html`](acquisition/license-page.html). The upstream `LICENSE.txt` is a zero-byte file, and its empty-file SHA-256 hash matches the published manifest; the empty file should not be interpreted as an absence of license terms.

The dataset page requests citation of the dataset, its original study, and PhysioNet. Use its current [citation instructions](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) when preparing a publication.
