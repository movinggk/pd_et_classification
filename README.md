# PD and ET data demonstration

This workspace contains the acquired **PADS v1.0.0** dataset and a completed demonstration of the recordings with Fourier-based analysis.

For a fresh clone on another machine, follow [Running on another machine](#running-on-another-machine), including the verified download workaround for `SHA256SUMS.txt` errors.

- **Pretrained-model comparison code (not a completed benchmark):** [Seven encoders, frozen features and supervised fine-tuning, tests, and execution guide](sota_pd_et/FOUNDATION_COMPARISON_README.md). Includes TSPulse, UTICA, TiRex, MOMENT, Chronos-2, TiViT, VQShape, and optional MantisV2 with shared participant-level nested splits.

- **Completed MantisV2 head-only/last-block comparison:** [Korean LaTeX report and saved-result audit](MantisV2_FineTune_Report/README.md), [report PDF](output/pdf/mantis_v2_finetune_report_ko.pdf), and [execution guide](sota_pd_et/MANTIS_FINETUNE_README.md). On the same 132-channel inputs and participant-level nested splits, head-only gives BA 81.57%, ET recall 26/28, PD specificity 70.29%, PR-AUC 0.6391 and MCC 0.3815; last-block gives BA 76.76%, ET recall 16/28, PD specificity 96.38%, PR-AUC 0.6463 and MCC 0.5534. Last-block does not improve BA over head-only; the metrics show a sensitivity/specificity tradeoff. Its same-seed rerun exactly reproduces the earlier saved predictions and model tensors. The report audits saved results without training or inference.

- **Completed temporal/cross-task experiment:** [Korean LaTeX report and reproduction files](Temporal_CrossTask_Report/README.md), [report PDF](output/pdf/temporal_cross_task_report_ko.pdf), and [model/execution guide](sota_pd_et/TASK_INTERACTION_README_KO.md). The completed seven-model trial uses identical three-condition, 60-channel inputs and participant-level 5-outer/3-inner-fold validation. Full-model balanced accuracy is 56.16%, compared with 61.57% for the variant with neither module and 61.05% for MantisV2. This run does not demonstrate an improvement from the proposed structure; the report independently recomputes saved evaluation metrics without retraining.

- **Pretrained transformer experiment:** [Frozen MantisV2 features and reproduction instructions](sota_pd_et/TRANSFORMER_README.md), with a [completed Apple MPS GPU evaluation](artifacts/pads_mantis_v2/REPORT.md). The same participant-level nested splits give 70.89% balanced accuracy and 12/28 ET recall, compared with 67.13% and 10/28 for MultiROCKET–HYDRA; pooled AUROC is lower (0.8876 versus 0.9000). This is an exploratory comparison.

- **Runnable classification code:** [`sota_pd_et/README.md`](sota_pd_et/README.md), a MultiROCKET–HYDRA model with participant-level nested cross-validation and a separate prediction script. The completed user-run experiment is summarized with evaluation figures in the [Korean article](output/pdf/model_explanation_ko.pdf); [editable source and reproduction instructions](sota_pd_et/docs/README.md) are included.

- **Standalone Google Colab notebook:** [`PD_ET_Classification_Colab.ipynb`](PD_ET_Classification_Colab.ipynb). Upload this single file to Colab, select a GPU runtime for optional CUDA acceleration of HYDRA, and run all cells. CPU runtimes also work. It embeds the classification code, installs an isolated environment, downloads and verifies only the required PADS inputs, runs nested evaluation and the final fit, displays figures, and downloads a result ZIP. MultiROCKET and ridge remain on the CPU. No repository or companion files are required.

- **Open the demonstration:** [`demonstration/outputs/index.html`](demonstration/outputs/index.html), a self-contained offline explorer.
- **Executed notebook:** [`demonstration/PADS_demonstration.ipynb`](demonstration/PADS_demonstration.ipynb).
- **Presentation sequence, exact methods, timing limitations, and rebuilding:** [`demonstration/README.md`](demonstration/README.md).
- **Dataset layout, integrity checks, and PD/ET participant selection:** [`data/pads/README.md`](data/pads/README.md).
- **Original public-data search:** [`PUBLIC_DATASETS.md`](PUBLIC_DATASETS.md) and [`PUBLIC_DATA_SEARCH.md`](PUBLIC_DATA_SEARCH.md).

The demonstration compares PD and ET through descriptive participant-level spectra and explicitly reports sampling gaps. It does not report classifier performance.

## Running on another machine

Git includes the project source and setup scripts. Datasets, downloaded model weights, local Python environments, experiment results, and generated exports are excluded by `.gitignore`. A fresh clone therefore needs a separate dataset download before classification can run. The comparison runner does not download the dataset automatically.

Run the commands below from the **repository root**, the directory containing this `README.md`, `scripts/`, and `sota_pd_et/`. Use **Python 3.12** and activate the environment in which you installed the foundation comparison requirements. In that environment, install the portable requirements if needed:

```sh
python -m pip install -r sota_pd_et/requirements-foundations.txt
```

Use that requirements file on the target machine; `requirements-foundations-lock-macos.txt` records the original macOS environment and is not a universal cross-platform lock. See the [foundation comparison guide](sota_pd_et/FOUNDATION_COMPARISON_README.md) for environment and device setup.

### Download the verified classification inputs

If the loader reports that `data/pads/1.0.0/SHA256SUMS.txt` does not exist, the expected dataset directory is missing its official checksum list. The dataset may not have been downloaded, may be incomplete, or may be stored at a different path. Adding the checksum list alone to Git would not supply the required participant files.

The standard download command is:

```sh
python -m sota_pd_et.colab_download data/pads/1.0.0 --workers 4
```

However, on **14 November 2026**, this command was observed to fail with:

```text
Retry 1/3: SHA256SUMS.txt: SHA-256 mismatch for SHA256SUMS.txt
...
RuntimeError: Could not acquire verified file after 4 attempts: SHA256SUMS.txt
```

The failure occurs while verifying the checksum list itself, before downloading participant inputs. A direct comparison of the [PhysioNet checksum list](https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/SHA256SUMS.txt) and the [official S3 mirror checksum list](https://physionet-open.s3.amazonaws.com/parkinsons-disease-smartwatch/1.0.0/SHA256SUMS.txt) found that only the `LICENSE.txt` entry differed; all **611 classification input file hashes** were identical. The S3 checksum list still matched the downloader's pinned SHA-256:

```text
514cd95405a12afcdfb126d47d1f559e2e8a744f03e586c3088d5e4fd7b02c46
```

Some PhysioNet input requests also redirected to S3, which the current downloader's cross-host redirect check rejected. To handle both issues, use the official S3 mirror for the entire download. **Run the following instead of the standard download command if you encounter either error:**

```sh
python -c "from sota_pd_et import colab_download as d; d.BASE_URL='https://physionet-open.s3.amazonaws.com/parkinsons-disease-smartwatch/1.0.0/'; d.download_dataset('data/pads/1.0.0', workers=4)"
```

This command changes the source URL only inside that Python process. It does not edit the downloader or `.gitignore`, and it preserves the pinned checksum-list verification, per-file SHA-256 verification, and redirect checks. Do not disable checksum validation or replace the pinned hash merely to accept a mismatching response.

The downloader acquires **612 verified files**: the checksum list, the official participant table, two preprocessing source files used for provenance, and the patient JSON and movement binary for each of **304 participants (276 PD and 28 ET)**. Despite the module's name, it works on a regular computer as well as in Colab. Rerunning it reuses files whose hashes match and retries missing or invalid files.

The expected directory layout is:

```text
data/pads/1.0.0/
├── SHA256SUMS.txt
├── patients/
│   └── patient_*.json
├── preprocessed/
│   ├── file_list.csv
│   └── movement/
│       └── *_ml.bin
└── scripts/
    ├── run_preprocessing.py
    └── utils/
        └── data_handling.py
```

This subset supplies the classification inputs. The raw-recording demonstration requires the full dataset described in the [dataset guide](data/pads/README.md); the selective downloader does not acquire the raw wrist recordings.

### Validate the dataset before training

After the download finishes successfully, run:

```sh
python -m sota_pd_et.compare_foundations --data-root data/pads/1.0.0 --mode both --check-data
```

The check verifies input checksums and constructs the participant-level evaluation splits for frozen and fine-tuning modes. It exits without downloading model weights or training. Expected output includes:

- `participants`: `304`.
- `counts`: `{"Parkinson's": 276, "Essential Tremor": 28}`.
- `shape`: `[304, 132, 976]`.
- `training_executed`: `false`.

Omit `--baseline` on a fresh clone unless you separately provide the previous experiment artifacts; they are excluded from Git. After the check passes, run your intended comparison command from the [comparison guide](sota_pd_et/FOUNDATION_COMPARISON_README.md).

If the dataset is stored elsewhere, pass the directory that **directly contains `SHA256SUMS.txt`** to `--data-root`:

```sh
python -m sota_pd_et.compare_foundations --data-root "/path/to/pads/1.0.0" --mode both --check-data
```
