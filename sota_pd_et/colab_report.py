"""Verify completed nested-CV outputs and create portable figures and a report.

This module never fits a model. Run with the same numerical environment as the
evaluation, plus matplotlib::

    python -m sota_pd_et.colab_report --run /path/to/results --output /path/to/report
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .evaluate import PERFORMANCE_METRICS, describe_values, metrics


CAVEATS = (
    "The positive class is Essential Tremor (ET=1); Parkinson's disease is PD=0. "
    "ET is predicted only when the ridge margin is strictly greater than zero.",
    "Reported performance uses outer held-out participant predictions. "
    "Inner alpha-selection results and the full-cohort final model's training predictions "
    "are not test performance.",
    "Decision scores are signed, uncalibrated ridge margins, not probabilities. "
    "Pooled AUROC and average precision compare margins from separately fitted models, "
    "whose numerical scales can differ; inspect the gray per-fold curves and fold metrics too.",
    "Fold means and standard deviations are descriptive, not confidence intervals, "
    "because training sets overlap. Repeating cross-validation does not create "
    "additional independent participants.",
    "This evaluation uses the published PADS movement representation, including its "
    "upstream preprocessing and task choices. It establishes neither an external "
    "clinical validation nor a state-of-the-art ranking.",
)
METRIC_LABELS = {
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
    "roc_auc": "AUROC (ET positive)",
    "average_precision_et": "ET average precision",
    "et_recall": "ET recall",
    "pd_recall": "PD recall",
    "et_precision": "ET precision",
    "f1_et": "ET F1",
    "macro_f1": "Macro F1",
    "matthews_correlation": "Matthews correlation",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def compare_metrics(actual, saved, context):
    for key, value in actual.items():
        require(key in saved, f"{context}: missing metric {key}.")
        if isinstance(value, list):
            require(saved[key] == value, f"{context}: {key} disagrees.")
        else:
            expected = float(saved[key])
            require(np.isfinite(expected) and abs(expected - value) <= 1e-12,
                    f"{context}: {key} disagrees.")


def compare_descriptive(actual, saved, context):
    for key, value in actual.items():
        require(key in saved, f"{context}: missing metric {key}.")
        for statistic in ("mean", "std"):
            a, b = value[statistic], saved[key][statistic]
            require((a is None and b is None) or
                    (a is not None and b is not None and np.isfinite(float(b))
                     and abs(a - float(b)) <= 1e-12),
                    f"{context}: {key} {statistic} disagrees.")
        require(np.allclose(value["values"], saved[key]["values"], rtol=0, atol=1e-12),
                f"{context}: {key} values disagree.")


def verify_run(run):
    """Recompute metrics and verify participant coverage, folds, and stored labels."""
    run = Path(run).expanduser().resolve()
    names = ("manifest.json", "summary.json", "cohort.csv", "splits.json",
             "oof_predictions.csv", "fold_metrics.csv")
    manifest = read_json(run / "manifest.json")
    require(manifest.get("status") == "complete", "The evaluation manifest is not complete.")
    summary = read_json(run / "summary.json")
    config = manifest["config"]
    repeats, outer_folds = int(config["repeats"]), int(config["outer_folds"])
    require(repeats >= 1 and outer_folds >= 2, "Invalid repeat or outer fold count.")
    cohort = read_csv(run / "cohort.csv")
    targets = {row["participant_id"]: int(row["target_et"]) for row in cohort}
    require(len(targets) == len(cohort) > 0, "Cohort IDs must be unique and nonempty.")
    require(set(targets.values()) == {0, 1}, "Require exactly PD=0 and ET=1.")
    counts = Counter(targets.values())
    require(counts[0] == summary["cohort_counts"]["Parkinson's"] and
            counts[1] == summary["cohort_counts"]["Essential Tremor"],
            "Summary class counts disagree with the cohort.")
    predictions = read_csv(run / "oof_predictions.csv")
    require(len(predictions) == len(targets) * repeats,
            "Wrong total number of outer predictions.")
    for row in predictions:
        repeat, fold = int(row["repeat"]), int(row["fold"])
        require(1 <= repeat <= repeats and 1 <= fold <= outer_folds,
                "Prediction repeat/fold is outside the configured range.")
        pid, target, score = row["participant_id"], int(row["target_et"]), float(row["decision_score_et"])
        require(pid in targets and target == targets[pid], "Outer target disagrees with cohort.")
        require(np.isfinite(score), "Nonfinite outer decision score.")
        require(int(row["predicted_et"]) == int(score > 0.0),
                "Stored prediction disagrees with the fixed strict score>0 rule.")
    plans = read_json(run / "splits.json")
    expected_keys = {(r, f) for r in range(1, repeats + 1) for f in range(1, outer_folds + 1)}
    require(Counter((int(p["repeat"]), int(p["fold"])) for p in plans)
            == Counter({key: 1 for key in expected_keys}), "Outer split plan is incomplete or duplicated.")
    saved_folds = summary["outer_fold_metrics"]
    fold_csv = read_csv(run / "fold_metrics.csv")
    for rows, context in ((saved_folds, "Summary"), (fold_csv, "Metric CSV")):
        require(Counter((int(p["repeat"]), int(p["fold"])) for p in rows)
                == Counter({key: 1 for key in expected_keys}), f"{context}: incomplete or duplicate folds.")
    require(Counter(int(row["repeat"]) for row in summary["repeat_metrics"])
            == Counter({r: 1 for r in range(1, repeats + 1)}),
            "Summary repeat metrics are incomplete or duplicated.")
    repeat_metrics, fold_metrics, groups = [], [], {}
    for repeat in range(1, repeats + 1):
        group = [p for p in predictions if int(p["repeat"]) == repeat]
        require(Counter(p["participant_id"] for p in group)
                == Counter({pid: 1 for pid in targets}),
                f"Repeat {repeat}: each participant must be tested exactly once.")
        y, scores = arrays(group)
        result = {"repeat": repeat, **metrics(y, scores)}
        saved = next(p for p in summary["repeat_metrics"] if int(p["repeat"]) == repeat)
        compare_metrics(metrics(y, scores), saved, f"Repeat {repeat}")
        repeat_metrics.append(result)
        groups[repeat] = group
        for fold in range(1, outer_folds + 1):
            key = (repeat, fold)
            plan = next(p for p in plans if (int(p["repeat"]), int(p["fold"])) == key)
            train, test = set(plan["train_ids"]), set(plan["test_ids"])
            require(len(train) == len(plan["train_ids"]) and len(test) == len(plan["test_ids"])
                    and not train & test and train | test == set(targets),
                    f"Repeat {repeat}, fold {fold}: invalid participant split.")
            selected = [p for p in group if int(p["fold"]) == fold]
            require(set(p["participant_id"] for p in selected) == test,
                    f"Repeat {repeat}, fold {fold}: outer predictions disagree with split membership.")
            inner_valid = Counter()
            require(len(plan["inner_splits"]) == int(config["inner_folds"]),
                    f"Repeat {repeat}, fold {fold}: wrong inner fold count.")
            for inner in plan["inner_splits"]:
                a, b = set(inner["train_ids"]), set(inner["validation_ids"])
                require(len(a) == len(inner["train_ids"]) and len(b) == len(inner["validation_ids"])
                        and not a & b and a | b == train,
                        f"Repeat {repeat}, fold {fold}: invalid inner participant split.")
                inner_valid.update(inner["validation_ids"])
            require(inner_valid == Counter({pid: 1 for pid in train}),
                    f"Repeat {repeat}, fold {fold}: incomplete inner validation coverage.")
            computed = metrics(*arrays(selected))
            saved = next(p for p in saved_folds if (int(p["repeat"]), int(p["fold"])) == key)
            flat = next(p for p in fold_csv if (int(p["repeat"]), int(p["fold"])) == key)
            compare_metrics(computed, saved, f"Repeat {repeat}, fold {fold} summary")
            compare_metrics({k: v for k, v in computed.items() if not isinstance(v, list)}, flat,
                            f"Repeat {repeat}, fold {fold} CSV")
            require(float(saved["selected_alpha"]) == float(flat["selected_alpha"]),
                    f"Repeat {repeat}, fold {fold}: selected alpha disagrees.")
            fold_metrics.append({"repeat": repeat, "fold": fold,
                                 "selected_alpha": float(saved["selected_alpha"]), **computed})
    across_repeats = {key: describe_values([row[key] for row in repeat_metrics]) for key in PERFORMANCE_METRICS}
    across_folds = {key: describe_values([row[key] for row in fold_metrics]) for key in PERFORMANCE_METRICS}
    compare_descriptive(across_repeats, summary["across_repeats"], "Across repeats")
    compare_descriptive(across_folds, summary["across_outer_folds_descriptive"], "Across outer folds")
    y = np.array(list(targets.values()), dtype=int)
    baseline = metrics(y, -np.ones(len(y)))
    compare_metrics(baseline, summary["always_predict_pd_baseline"], "Always-PD baseline")
    require(abs(float(summary["average_precision_no_skill_reference"]) - float(y.mean())) <= 1e-12,
            "Average-precision reference disagrees with ET prevalence.")
    report = {
        "run_directory": str(run),
        "verification": "Passed: completed manifest; cohort labels/counts; one outer prediction per participant "
        "per repeat; fixed score>0 rule; outer/inner split membership and coverage; recomputed "
        "repeat/fold metrics and descriptive statistics; always-PD baseline.",
        "input_sha256": {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in names},
        "cohort_counts": summary["cohort_counts"],
        "repeat_metrics": repeat_metrics,
        "outer_fold_metrics": fold_metrics,
        "across_repeats": across_repeats,
        "across_outer_folds_descriptive": across_folds,
        "always_predict_pd_baseline": baseline,
        "average_precision_no_skill_reference": float(y.mean()),
        "interpretation": list(CAVEATS),
    }
    return report, groups


def arrays(rows):
    return (np.array([int(p["target_et"]) for p in rows]),
            np.array([float(p["decision_score_et"]) for p in rows]))


def pr_steps(y, scores):
    """Return coordinates whose rectangle area equals noninterpolated AP."""
    precision, recall, _ = precision_recall_curve(y, scores)
    rx, py = recall[::-1], precision[::-1]
    x, p = [float(rx[0])], [float(py[0])]
    for index in range(1, len(rx)):
        x.extend((float(rx[index - 1]), float(rx[index])))
        p.extend((float(py[index]), float(py[index])))
    return x, p


def create_figures(output, report, groups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    for result in report["repeat_metrics"]:
        repeat = result["repeat"]
        group = groups[repeat]
        y, scores = arrays(group)
        prefix = f"repeat_{repeat:02d}"
        matrix = np.asarray(result["confusion_matrix_true_rows_predicted_columns_pd_et"])
        fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        for (row, column), count in np.ndenumerate(matrix):
            ax.text(column, row, str(count), ha="center", va="center", fontsize=20,
                    color="white" if count > matrix.max() / 2 else "#17324d")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["PD", "ET"], yticklabels=["PD", "ET"],
               xlabel="Predicted condition", ylabel="True condition",
               title=f"Outer held-out predictions — repeat {repeat}\n"
                     f"Balanced accuracy {result['balanced_accuracy']:.3f}")
        fig.colorbar(image, ax=ax, label="Participants")
        filename = f"{prefix}_confusion.png"
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)
        figures.append(filename)
        for kind in ("roc", "pr"):
            fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
            fold_ids = sorted({int(p["fold"]) for p in group})
            for index, fold in enumerate(fold_ids):
                fy, fs = arrays([p for p in group if int(p["fold"]) == fold])
                x, value = roc_curve(fy, fs)[:2] if kind == "roc" else pr_steps(fy, fs)
                ax.plot(x, value, color="#a6adb6", linewidth=1.1, alpha=0.8,
                        label="Individual outer folds" if index == 0 else None)
            if kind == "roc":
                x, value, _ = roc_curve(y, scores)
                ax.plot([0, 1], [0, 1], color="#7e8792", linestyle="--", linewidth=1,
                        label="No discrimination")
                label = f"Pooled outer predictions (AUROC={result['roc_auc']:.3f})"
                axis_labels = {"xlabel": "False positive rate (PD called ET)", "ylabel": "ET recall"}
            else:
                x, value = pr_steps(y, scores)
                prevalence = float(y.mean())
                ax.axhline(prevalence, color="#7e8792", linestyle="--", linewidth=1,
                           label=f"ET prevalence={prevalence:.3f}")
                label = f"Pooled outer predictions (AP={result['average_precision_et']:.3f})"
                axis_labels = {"xlabel": "ET recall", "ylabel": "ET precision"}
            ax.plot(x, value, color="#087f8c", linewidth=2.3, label=label)
            ax.set(**axis_labels, xlim=(0, 1), ylim=(0, 1.02),
                   title=f"Outer held-out {'ROC' if kind == 'roc' else 'precision–recall'} — repeat {repeat}")
            ax.grid(alpha=0.2)
            ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=8)
            fig.text(0.5, -0.035, "Pooled curves compare margins from different fitted models.",
                     ha="center", fontsize=9, color="#56616f")
            filename = f"{prefix}_{kind}.png"
            fig.savefig(output / filename, dpi=170, bbox_inches="tight")
            plt.close(fig)
            figures.append(filename)
    return figures


def write_report(output, report, figures):
    rows = []
    for scope, source in (("pooled_outer_repeat", report["repeat_metrics"]),
                          ("outer_fold", report["outer_fold_metrics"])):
        for row in source:
            rows.append({"scope": scope, "repeat": row["repeat"], "fold": row.get("fold", ""),
                         "selected_alpha": row.get("selected_alpha", ""),
                         **{key: value for key, value in row.items()
                            if key not in {"repeat", "fold", "selected_alpha"} and not isinstance(value, list)}})
    with (output / "report_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["figures"] = figures
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    esc = html.escape
    columns = "".join(f"<th>Repeat {row['repeat']}</th>" for row in report["repeat_metrics"])
    body = []
    for key in PERFORMANCE_METRICS:
        values = "".join(f"<td>{row[key]:.4f}</td>" for row in report["repeat_metrics"])
        stat = report["across_outer_folds_descriptive"][key]
        deviation = "n/a" if stat["std"] is None else f"{stat['std']:.4f}"
        baseline = report["always_predict_pd_baseline"][key]
        body.append(f"<tr><th>{esc(METRIC_LABELS[key])}</th>{values}"
                    f"<td>{stat['mean']:.4f} ± {deviation}</td><td>{baseline:.4f}</td></tr>")
    images = []
    for name in figures:
        data = base64.b64encode((output / name).read_bytes()).decode("ascii")
        images.append(f'<img alt="{esc(name)}" src="data:image/png;base64,{data}">')
    notes = "".join(f"<li>{esc(note)}</li>" for note in CAVEATS)
    fold_rows = "".join(
        f"<tr><td>{row['repeat']}</td><td>{row['fold']}</td><td>{row['selected_alpha']:g}</td>"
        f"<td>{row['balanced_accuracy']:.4f}</td><td>{row['et_recall']:.4f}</td>"
        f"<td>{row['pd_recall']:.4f}</td><td>{row['roc_auc']:.4f}</td>"
        f"<td>{row['average_precision_et']:.4f}</td><td>{row['n_et']}</td></tr>"
        for row in report["outer_fold_metrics"]
    )
    counts = report["cohort_counts"]
    text = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PD/ET participant-level nested-CV report</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#17324d}}
table{{border-collapse:collapse;display:block;overflow:auto}}td,th{{padding:9px 14px;border:1px solid #dce3e8;text-align:right}}
th:first-child{{text-align:left}}thead{{background:#edf5f6}}.figures{{display:flex;flex-wrap:wrap;gap:12px}}
img{{width:min(100%,560px);height:auto;object-fit:contain}}code{{overflow-wrap:anywhere}}</style>
<h1>PD/ET participant-level nested cross-validation</h1>
<p>Participants: PD={counts["Parkinson's"]}; ET={counts['Essential Tremor']}.</p>
<p><strong>Verification passed.</strong> {esc(report['verification'])}</p>
<table><thead><tr><th>Metric</th>{columns}<th>Outer folds, descriptive mean ± SD</th><th>Always-PD baseline</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p>Repeat columns pool one held-out prediction per participant in each repeat. The fold column describes all outer folds across all repeats; SD is not a confidence interval.</p>
<h2>Individual outer folds</h2>
<table><thead><tr><th>Repeat</th><th>Fold</th><th>Selected alpha</th><th>Balanced accuracy</th><th>ET recall</th><th>PD recall</th><th>AUROC</th><th>ET AP</th><th>ET participants</th></tr></thead>
<tbody>{fold_rows}</tbody></table>
<h2>Interpretation</h2><ul>{notes}</ul>
<h2>Outer held-out predictions</h2><div class="figures">{''.join(images)}</div>
<p>Verified run: <code>{esc(report['run_directory'])}</code>. This report contains its figures and requires no external image files.</p>
</html>
"""
    (output / "report.html").write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New report directory; default: RUN/colab_report.")
    args = parser.parse_args(argv)
    report, groups = verify_run(args.run)
    output = (args.output or args.run / "colab_report").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    figures = create_figures(output, report, groups)
    write_report(output, report, figures)
    print(f"Verified outer evaluation and saved report: {output}")
    for result in report["repeat_metrics"]:
        print(f"Repeat {result['repeat']}: BA={result['balanced_accuracy']:.4f}; "
              f"AUROC={result['roc_auc']:.4f}; ET recall={result['et_recall']:.4f}; "
              f"PD recall={result['pd_recall']:.4f}; AP={result['average_precision_et']:.4f}")
    return output


if __name__ == "__main__":
    main()
