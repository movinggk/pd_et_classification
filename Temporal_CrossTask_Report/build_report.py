"""Audit saved PADS results and author LaTeX. NEVER trains or predicts a model.

Run with the existing project environment; the generated report.tex is a
standalone XeLaTeX document with every table/plot embedded. No model forward
pass is called, including during final-model artifact verification.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, roc_auc_score, roc_curve)
import torch


PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT / "artifacts/pads_temporal_cross_task/20261108T071720522119Z"
MODELS = ("full", "no_temporal", "no_interaction", "neither", "concat_mlp", "mantis", "multirocket_hydra")
LABEL = dict(zip(MODELS, ("Full", "No-temporal", "No-interaction", "Neither", "Concat-MLP", "MantisV2", "MR--HYDRA")))


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, label):
    if not np.isclose(a, b, rtol=0, atol=1e-12):
        raise ValueError(f"Saved result mismatch: {label}: {a} vs {b}")


def calculate(rows):
    y = np.array([int(x["target_et"]) for x in rows])
    raw = np.array([float(x["raw_score_et"]) for x in rows])
    pred = np.array([int(x["predicted_et"]) for x in rows])
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)), "roc_auc": float(roc_auc_score(y, raw)),
            "average_precision_et": float(average_precision_score(y, raw)),
            "et_recall": tp / (tp + fn), "pd_recall": tn / (tn + fp),
            "et_precision": float(precision_score(y, pred, zero_division=0)),
            "f1_et": float(f1_score(y, pred, zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "matthews_correlation": float(matthews_corrcoef(y, pred)),
            "pd_correct": tn, "pd_called_et": fp, "et_called_pd": fn, "et_correct": tp,
            "n_pd": tn + fp, "n_et": fn + tp,
            "confusion_matrix_true_rows_predicted_columns_pd_et": [[tn, fp], [fn, tp]]}


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text())
    saved = json.loads((run / "summary.json").read_text())
    plan = json.loads((run / "splits.json").read_text())
    if manifest["status"] != "complete" or tuple(manifest["completed_models"]) != MODELS:
        raise ValueError("The report requires the completed seven-model run.")
    cfg = manifest["config"]
    expected = {"outer_folds": 5, "inner_folds": 3, "repeats": 1,
                "seed": 20261017, "hidden_dim": 32, "heads": 4, "temporal_layers": 1,
                "windows_per_segment": 4, "interaction_rank": 4, "dropout": 0.2,
                "epochs": [15, 30, 60], "weight_decays": [0.001, 0.01], "thresholds": [0.0],
                "learning_rate": 0.001, "batch_size": 16, "fit_final": True}
    if any(cfg[k] != value for k, value in expected.items()):
        raise ValueError("This report template documents the specified default protocol; adapt it for a different run.")
    cohort = read_csv(run / "cohort.csv")
    ids = [x["participant_id"] for x in cohort]
    target = {x["participant_id"]: int(x["target_et"]) for x in cohort}
    if len(ids) != len(set(ids)) or len(ids) != 304 or sum(target.values()) != 28:
        raise ValueError("Expected 304 unique participants, including 28 ET.")
    seen, split_counts, inner_et_counts = [], [], []
    for s in plan:
        tr, te = set(s["train_ids"]), set(s["test_ids"])
        assert not tr & te and tr | te == set(ids)
        assert [ids[i] for i in s["train_indices"]] == s["train_ids"]
        assert [ids[i] for i in s["test_indices"]] == s["test_ids"]
        validation = []
        for inner in s["inner_splits"]:
            a, b = set(inner["train_ids"]), set(inner["validation_ids"])
            assert not a & b and a | b == tr
            assert [s["train_ids"][i] for i in inner["train_positions"]] == inner["train_ids"]
            assert [s["train_ids"][i] for i in inner["validation_positions"]] == inner["validation_ids"]
            validation.extend(inner["validation_ids"])
            inner_et_counts.append(sum(target[i] for i in a))
        assert sorted(validation) == sorted(tr)
        seen.extend(s["test_ids"])
        split_counts.append({"fold": s["fold"], "train_pd": sum(target[i] == 0 for i in tr),
                             "train_et": sum(target[i] == 1 for i in tr),
                             "test_pd": sum(target[i] == 0 for i in te), "test_et": sum(target[i] == 1 for i in te)})
    assert sorted(seen) == sorted(ids)
    results, predictions, folds, choices = {}, {}, [], []
    for name in MODELS:
        rows = read_csv(run / name / "oof_predictions.csv")
        assert len(rows) == 304 and {r["participant_id"] for r in rows} == set(ids)
        for row in rows:
            pid = row["participant_id"]
            assert int(row["target_et"]) == target[pid] and int(row["repeat"]) == 1
            assert pid in plan[int(row["fold"]) - 1]["test_ids"]
            raw, threshold = float(row["raw_score_et"]), float(row["decision_threshold"])
            assert np.isfinite(raw) and threshold == 0
            close(float(row["decision_score_et"]), raw - threshold, "decision_score")
            assert int(row["predicted_et"]) == int(raw > threshold)
        result = calculate(rows)
        for key, value in result.items():
            recorded = saved[name]["repeat_metrics"][0][key]
            if isinstance(value, list):
                assert value == recorded
            else:
                close(value, recorded, f"{name}/{key}")
        local_folds = []
        for split in plan:
            fold = split["fold"]
            folder = run / name / f"repeat_01_fold_{fold:02d}"
            outer = read_csv(folder / "test_predictions.csv")
            assert {r["participant_id"]: r for r in outer} == {r["participant_id"]: r for r in rows if int(r["fold"]) == fold}
            value = calculate(outer)
            recorded = json.loads((folder / "metrics.json").read_text())
            for key, val in value.items():
                if not isinstance(val, list):
                    close(val, recorded[key], f"{name}/fold{fold}/{key}")
            local_folds.append(value)
            folds.append({"model": name, "fold": fold, **{k: v for k, v in value.items() if not isinstance(v, list)}})
            selected = json.loads((folder / "selected.json").read_text())
            search = read_csv(folder / "inner_search.csv")
            best = max(float(x["inner_mean_balanced_accuracy"]) for x in search)
            close(selected["inner_mean_balanced_accuracy"], best, "best inner candidate")
            candidates = [r for r in search if abs(float(r["inner_mean_balanced_accuracy"]) - best) <= 1e-12]
            if name in MODELS[:5]:
                chosen = min(candidates, key=lambda r: (-float(r["weight_decay"]), int(r["epochs"]), abs(float(r["threshold"])), -float(r["threshold"])))
                assert float(chosen["weight_decay"]) == selected["weight_decay"] and int(chosen["epochs"]) == selected["epochs"]
            else:
                chosen = min(candidates, key=lambda r: (-float(r["alpha"]), abs(float(r["threshold"])), -float(r["threshold"])))
                assert float(chosen["alpha"]) == selected["alpha"]
            inner = read_csv(folder / "inner_oof_predictions.csv")
            assert len(inner) == len(split["train_ids"]) and {r["participant_id"] for r in inner} == set(split["train_ids"])
            byid = {r["participant_id"]: r for r in inner}
            inner_ba = []
            for s in split["inner_splits"]:
                people = [byid[i] for i in s["validation_ids"]]
                assert all(int(x["target_et"]) == target[x["participant_id"]] for x in people)
                inner_ba.append(balanced_accuracy_score([int(x["target_et"]) for x in people], [int(x["predicted_et"]) for x in people]))
            close(np.mean(inner_ba), selected["inner_mean_balanced_accuracy"], "selected inner OOF BA")
            choices.append({"model": name, "fold": fold, **selected})
        result["mean_fold_roc_auc"] = float(np.mean([x["roc_auc"] for x in local_folds]))
        result["mean_fold_ap"] = float(np.mean([x["average_precision_et"] for x in local_folds]))
        close(result["mean_fold_roc_auc"], saved[name]["mean_outer_fold_roc_auc"], "mean fold AUC")
        result["nested_seconds"] = float(sum(x["elapsed_seconds"] for x in saved[name]["outer_fold_metrics"]))
        results[name] = result
        predictions[name] = {x["participant_id"]: x for x in rows}
    y = np.array([target[i] for i in ids])
    comparisons = json.loads((run / "paired_comparisons.json").read_text())
    full_correct = np.array([int(predictions["full"][i]["predicted_et"]) == target[i] for i in ids]).astype(float)
    for name, saved_comparison in comparisons.items():
        other_correct = np.array([int(predictions[name][i]["predicted_et"]) == target[i] for i in ids]).astype(float)
        delta = full_correct - other_correct
        rng = np.random.default_rng(cfg["seed"])
        draws = np.zeros(cfg["bootstrap_samples"])
        estimate = 0.0
        for c in (0, 1):
            index = np.flatnonzero(y == c)
            estimate += float(delta[index].mean()) / 2
            draws += delta[rng.choice(index, size=(cfg["bootstrap_samples"], len(index)), replace=True)].mean(1) / 2
        close(estimate, saved_comparison["reference_minus_other_balanced_accuracy"], "paired BA difference")
        for a, b in zip(np.quantile(draws, [0.025, 0.975]), saved_comparison["conditional_percentile_95_interval"]):
            close(a, b, "paired bootstrap interval")
    source_hashes = manifest["source_code_sha256"]
    with zipfile.ZipFile(run / "evaluated_sources.zip") as archive:
        for name, digest in source_hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    source_data = Path(cfg["data_root"])
    for name, digest in manifest["dataset"]["verified_source_sha256"].items():
        assert sha(source_data / name) == digest
    encoder = json.loads((run / "mantis_encoder_provenance.json").read_text())
    assert sha(run / "mantis_frozen_features.npy") == encoder["feature_sha256"]
    assert np.load(run / "mantis_frozen_features.npy", mmap_mode="r").shape == (304, 30720)
    final = torch.load(run / "final_model.pt", map_location="cpu", weights_only=True)
    assert set(final["metadata"]["training_ids"]) == set(ids)
    assert all(torch.isfinite(v).all() for v in final["state_dict"].values())
    assert torch.isfinite(final["mean"]).all() and torch.all(final["scale"] > 0) and torch.isfinite(final["scale"]).all()
    assert final["metadata"]["source_code_sha256"] == source_hashes
    historical = []
    for model, dirname in (("mantis", "pads_mantis_v2"), ("multirocket_hydra", "pads_mr_hydra")):
        old = PROJECT / "artifacts" / dirname
        assert json.loads((old / "splits.json").read_text()) == plan
        old_cohort = read_csv(old / "cohort.csv")
        assert [(r["participant_id"], int(r["target_et"])) for r in old_cohort] == [(i, target[i]) for i in ids]
        values = json.loads((old / "summary.json").read_text())["repeat_metrics"][0]
        historical.append({"model": model, "channels": 132, "metrics": values})
    return {"run_id": run.name, "manifest": manifest, "metrics": results, "folds": folds,
            "selections": choices, "split_counts": split_counts, "inner_train_et_range": [min(inner_et_counts), max(inner_et_counts)],
            "comparisons": comparisons, "encoder": encoder,
            "final_selected": json.loads((run / "final_selected.json").read_text()),
            "historical": historical,
            "audit": {"status": "passed", "actual_retraining_performed": False, "model_forward_passes_performed": 0,
                      "unique_participants": 304, "outer_predictions_checked": 2128,
                      "outer_folds_checked": 35, "selected_inner_predictions_checked": sum(len(s["train_ids"]) for s in plan) * 7,
                      "data_file_hashes_checked": len(manifest["dataset"]["verified_source_sha256"]),
                      "archived_source_hashes_checked": len(source_hashes),
                      "checks": ["Participant coverage, labels and nested split separation",
                                 "All pooled and outer-fold metrics recomputed independently",
                                 "Selected inner OOF mean BA and candidate/tie selection verified",
                                 "All six paired bootstrap differences and intervals recomputed",
                                 "Data/source/frozen-feature hashes verified",
                                 "Final bundle training IDs, finite tensors and source provenance checked"],
                      "limit": "Artifact/source consistency audit; not a rerun of training, external validation, or independent execution trace."}}, predictions


def rows_tex(rows):
    return "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)


def coordinates(x, y):
    return " ".join(f"({float(a):.8g},{float(b):.8g})" for a, b in zip(x, y))


def plot_bars(metrics):
    ba = coordinates([100 * metrics[m]["balanced_accuracy"] for m in MODELS], range(7, 0, -1))
    et = coordinates([100 * metrics[m]["et_recall"] for m in MODELS], range(7, 0, -1))
    return r"""\begin{tikzpicture}
\begin{axis}[width=.89\linewidth,height=7.4cm,xbar,bar width=5pt,
 xmin=0,xmax=104,ymin=.45,ymax=7.55,ytick={1,2,3,4,5,6,7},
 yticklabels={MR--HYDRA,MantisV2,Concat-MLP,Neither,No-interaction,No-temporal,Full},
 xlabel={Performance (\%)},xtick={0,20,40,60,80,100},xmajorgrids,
 tick label style={font=\small},legend style={at={(.5,-.20)},anchor=north,legend columns=2,font=\small},
 nodes near coords,nodes near coords align={horizontal},every node near coord/.append style={font=\scriptsize},
 point meta=x,/pgf/number format/fixed,/pgf/number format/precision=1]
\addplot[fill=navy,draw=navy] coordinates {""" + ba + r"""};
\addplot[fill=orangeplot,draw=orangeplot] coordinates {""" + et + r"""};
\draw[densely dashed,gray] (axis cs:50,.5)--(axis cs:50,7.5);
\legend{Balanced accuracy,ET sensitivity}
\end{axis}\end{tikzpicture}"""


def plot_curves(predictions):
    selected = ("full", "neither", "mantis", "multirocket_hydra")
    colors = ("redplot", "greenplot", "navy", "orangeplot")
    out = [r"\begin{tikzpicture}\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.3cm},width=.43\linewidth,height=6.2cm,xmin=0,xmax=1,ymin=0,ymax=1,grid=major,tick label style={font=\scriptsize},label style={font=\small},legend style={font=\scriptsize,draw=none,fill=none},legend cell align=left]",
           r"\nextgroupplot[xlabel={False positive rate},ylabel={ET sensitivity},title={Pooled ROC},legend pos=south east]"]
    for model, color in zip(selected, colors):
        rows = list(predictions[model].values())
        y = np.array([int(x["target_et"]) for x in rows]); score = np.array([float(x["raw_score_et"]) for x in rows])
        x, yy, _ = roc_curve(y, score)
        out += [rf"\addplot[thick,{color},no marks] coordinates {{{coordinates(x, yy)}}};", rf"\addlegendentry{{{LABEL[model]}}}"]
    out += [r"\addplot[dashed,gray,forget plot] coordinates {(0,0)(1,1)};",
            r"\nextgroupplot[xlabel={ET recall},ylabel={ET precision},title={Pooled precision--recall},legend pos=north east]"]
    for model, color in zip(selected, colors):
        rows = list(predictions[model].values())
        y = np.array([int(x["target_et"]) for x in rows]); score = np.array([float(x["raw_score_et"]) for x in rows])
        precision, recall, _ = precision_recall_curve(y, score)
        out += [rf"\addplot[thick,{color},no marks] coordinates {{{coordinates(recall, precision)}}};", rf"\addlegendentry{{{LABEL[model]}}}"]
    out += [r"\addplot[dashed,gray,forget plot] coordinates {(0,.092105263)(1,.092105263)};", r"\end{groupplot}\end{tikzpicture}"]
    return "\n".join(out)


def plot_forest(comparisons):
    out = [r"\begin{tikzpicture}\begin{axis}[width=.89\linewidth,height=6.2cm,xmin=-19,xmax=14,ymin=.5,ymax=6.5,ytick={1,2,3,4,5,6},yticklabels={MR--HYDRA,MantisV2,Concat-MLP,Neither,No-interaction,No-temporal},xlabel={Full minus comparator: balanced accuracy (percentage points)},xmajorgrids,tick label style={font=\small},label style={font=\small}]",
           r"\draw[densely dashed,gray] (axis cs:0,.5)--(axis cs:0,6.5);"]
    for row, model in zip(range(6, 0, -1), MODELS[1:]):
        value = comparisons[model]
        est = 100 * value["reference_minus_other_balanced_accuracy"]
        lo, hi = [100 * x for x in value["conditional_percentile_95_interval"]]
        out += [rf"\draw[navy,line width=.9pt] (axis cs:{lo},{row})--(axis cs:{hi},{row});",
                rf"\draw[navy] (axis cs:{lo},{row-.08})--(axis cs:{lo},{row+.08});",
                rf"\draw[navy] (axis cs:{hi},{row-.08})--(axis cs:{hi},{row+.08});",
                rf"\addplot[only marks,mark=*,redplot,mark size=2pt] coordinates {{({est},{row})}};"]
    out += [r"\end{axis}\end{tikzpicture}"]
    return "\n".join(out)


def plot_training(run):
    out = [r"\begin{tikzpicture}\begin{axis}[width=.94\linewidth,height=6.1cm,xlabel={Epoch},ylabel={Training weighted BCE},xmin=1,xmax=60,ymin=0,ymax=1.5,grid=major,legend columns=3,legend style={at={(.5,-.23)},anchor=north,font=\small},tick label style={font=\small}]"]
    for fold, color in zip(range(1, 6), ("navy", "orangeplot", "greenplot", "purpleplot", "redplot")):
        rows = read_csv(run / "full" / f"repeat_01_fold_{fold:02d}" / "training_history.csv")
        points = coordinates([int(x["epoch"]) for x in rows], [float(x["training_weighted_bce"]) for x in rows])
        out += [rf"\addplot[thick,{color},no marks] coordinates {{{points}}};", rf"\addlegendentry{{Fold {fold}}}"]
    out += [r"\end{axis}\end{tikzpicture}"]
    return "\n".join(out)


def author(run, data, predictions):
    results, manifest = data["metrics"], data["manifest"]
    selections = {(v["model"], v["fold"]): v for v in data["selections"]}
    def p(value): return f"{100 * value:.2f}"
    fields = {
        "RUN_ID": run.name,
        "PRIMARY_ROWS": rows_tex([[LABEL[m], p(v["balanced_accuracy"]), p(v["et_recall"]), p(v["pd_recall"]), f"{v['roc_auc']:.4f}", f"{v['mean_fold_roc_auc']:.4f}", f"{v['average_precision_et']:.4f}"] for m, v in results.items()]),
        "CONFUSION_ROWS": rows_tex([[LABEL[m], v["pd_correct"], v["pd_called_et"], v["et_called_pd"], v["et_correct"], v["pd_called_et"] + v["et_correct"]] for m, v in results.items()]),
        "SECONDARY_ROWS": rows_tex([[LABEL[m], p(v["accuracy"]), p(v["et_precision"]), f"{v['f1_et']:.4f}", f"{v['macro_f1']:.4f}", f"{v['matthews_correlation']:.4f}"] for m, v in results.items()]),
        "SPLIT_ROWS": rows_tex([[r["fold"], r["train_pd"], r["train_et"], r["test_pd"], r["test_et"]] for r in data["split_counts"]]),
        "PARAMETER_ROWS": rows_tex([[LABEL[m], f"{n:,}"] for m, n in manifest["parameter_counts"].items()]),
        "PAIRED_ROWS": rows_tex([[LABEL[m], f"{100*v['reference_minus_other_balanced_accuracy']:+.2f}", f"[{100*v['conditional_percentile_95_interval'][0]:.2f}, {100*v['conditional_percentile_95_interval'][1]:.2f}]"] for m, v in data["comparisons"].items()]),
        "BAR_PLOT": plot_bars(results), "CURVE_PLOT": plot_curves(predictions),
        "FOREST_PLOT": plot_forest(data["comparisons"]), "TRAINING_PLOT": plot_training(run),
        "VERSION_ROWS": rows_tex([[r"\code{" + name + "}", value] for name, value in manifest["package_versions"].items()]),
        "RUNTIME_ROWS": rows_tex([[LABEL[m], f"{v['nested_seconds']:.2f}", "CPU (고정 특징)" if m == "mantis" else "CPU" if m == "multirocket_hydra" else "MPS + CPU"] for m, v in results.items()]),
        "START_LOCAL": datetime.fromisoformat(manifest["started_at_utc"]).astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
        "END_LOCAL": datetime.fromisoformat(manifest["finished_at_utc"]).astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
        "TOTAL_SECONDS": f"{manifest['elapsed_seconds']:.2f}",
        "ENCODER_SECONDS": f"{data['encoder']['elapsed_seconds']:.2f}",
    }
    fold_rows = []
    for x in data["folds"]:
        selected = selections[x["model"], x["fold"]]
        setting = f"{selected['epochs']} / {selected['weight_decay']:g}" if "epochs" in selected else rf"$\alpha={selected['alpha']:g}$"
        fold_rows.append([LABEL[x["model"]], x["fold"], setting, p(selected["inner_mean_balanced_accuracy"]), p(x["balanced_accuracy"]), f"{x['et_correct']}/{x['n_et']}", f"{x['roc_auc']:.4f}"])
    fields["FOLD_ROWS"] = rows_tex(fold_rows)
    fields["HISTORICAL_ROWS"] = rows_tex(
        [[LABEL[h["model"]], 132, p(h["metrics"]["balanced_accuracy"]), f"{h['metrics']['et_correct']}/28", f"{h['metrics']['roc_auc']:.4f}"] for h in data["historical"]]
        + [[LABEL[m], 60, p(results[m]["balanced_accuracy"]), f"{results[m]['et_correct']}/28", f"{results[m]['roc_auc']:.4f}"] for m in ("mantis", "multirocket_hydra")])
    source = (HERE / "report_template.tex").read_text()
    for key, value in fields.items():
        source = source.replace("@@" + key + "@@", value)
    if "@@" in source:
        raise ValueError("Unresolved report placeholder.")
    (HERE / "report.tex").write_text(source)
    (HERE / "verification.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    flat = [{"model": m, **{k: v for k, v in values.items() if not isinstance(v, list)}} for m, values in results.items()]
    with (HERE / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    data, predictions = audit(run)
    author(run, data, predictions)
    print(json.dumps({"report": str(HERE / "report.tex"), "audit": data["audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
