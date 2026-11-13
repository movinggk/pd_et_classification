# PD and ET data demonstration

This workspace contains the acquired **PADS v1.0.0** dataset and a completed demonstration of the recordings with Fourier-based analysis.

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
