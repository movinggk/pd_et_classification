"""Validate saved outer predictions and generate editable PGFPlots figures.

Run from the project root:
    .venv-sota/bin/python -m sota_pd_et.plot_evaluation --update-article

This command reads completed experiment outputs; it never fits a model. The
article embeds all figure coordinates, so it remains a single-file document.
Standalone figure sources can be compiled with XeLaTeX for vector PDF exports.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, roc_auc_score, roc_curve,
)


BEGIN = "% BEGIN GENERATED OUTER EVALUATION"
END = "% END GENERATED OUTER EVALUATION"
COLORS = r"""
\definecolor{evalteal}{HTML}{087F8C}
\definecolor{evalnavy}{HTML}{17324D}
\definecolor{evalgray}{HTML}{97A3B1}
\definecolor{evalorange}{HTML}{B95C16}
"""
METRIC_NAMES = (
    "balanced_accuracy", "accuracy", "roc_auc", "average_precision_et",
    "et_recall", "pd_recall", "et_precision", "f1_et", "macro_f1",
    "matthews_correlation",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(rows):
    y = np.array([int(row["target_et"]) for row in rows], dtype=int)
    s = np.array([float(row["decision_score_et"]) for row in rows])
    require(np.isfinite(s).all(), "Nonfinite decision score.")
    require(set(y.tolist()) == {0, 1}, "Every metric group needs both classes.")
    predicted = (s > 0).astype(int)
    require(all(int(r["predicted_et"]) == int(p)
                for r, p in zip(rows, predicted, strict=True)),
            "Stored prediction disagrees with the fixed strict s>0 rule.")
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision_et": float(average_precision_score(y, s)),
        "et_recall": tp / (tp + fn),
        "pd_recall": tn / (tn + fp),
        "et_precision": float(precision_score(y, predicted, zero_division=0)),
        "f1_et": float(f1_score(y, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(y, predicted)),
        "n_pd": tn + fp, "n_et": tp + fn,
        "pd_correct": tn, "pd_called_et": fp,
        "et_called_pd": fn, "et_correct": tp,
        "confusion_matrix_true_rows_predicted_columns_pd_et": cm.tolist(),
    }
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    precision, recall, _ = precision_recall_curve(y, s)
    # The AP rectangle on (R[l-1], R[l]] has height P[l]. Expand its
    # vertices explicitly rather than leaving the plotting step convention
    # implicit. Its area equals sklearn's non-interpolated AP.
    rx, py = recall[::-1], precision[::-1]
    pr_x, pr_y = [float(rx[0])], [float(py[0])]
    for index in range(1, len(rx)):
        pr_x.extend((float(rx[index - 1]), float(rx[index])))
        pr_y.extend((float(py[index]), float(py[index])))
    ap_area = float(np.sum(np.diff(rx) * py[1:]))
    require(abs(ap_area - result["average_precision_et"]) < 1e-12,
            "PR step area does not reproduce AP.")
    # Independent pairwise definition, including half credit for tied scores.
    comparisons = s[y == 1, None] - s[y == 0][None, :]
    pair_auc = float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))
    require(abs(pair_auc - result["roc_auc"]) < 1e-12,
            "Pairwise AUROC does not reproduce the saved-score AUROC.")
    return result, {
        "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(),
        "pr_recall": pr_x, "pr_precision": pr_y,
    }


def compare_metrics(actual, saved, context):
    for key, value in actual.items():
        if isinstance(value, list):
            require(value == saved[key], f"{context}: {key} disagrees.")
        else:
            require(abs(float(saved[key]) - value) < 1e-12,
                    f"{context}: {key} disagrees.")


def load_verified(run, repeat, project):
    manifest = read_json(run / "manifest.json")
    require(manifest["status"] == "complete", "The experiment is not complete.")
    summary = read_json(run / "summary.json")
    config = manifest["config"]
    require(1 <= repeat <= config["repeats"], "Requested repeat is unavailable.")
    cohort = read_csv(run / "cohort.csv")
    targets = {r["participant_id"]: int(r["target_et"]) for r in cohort}
    require(len(targets) == len(cohort), "Duplicate cohort IDs.")
    require(set(targets.values()) == {0, 1}, "Invalid binary cohort labels.")
    counts = Counter(targets.values())
    require(counts[0] == summary["cohort_counts"]["Parkinson's"]
            and counts[1] == summary["cohort_counts"]["Essential Tremor"],
            "Cohort class counts disagree.")
    all_rows = read_csv(run / "oof_predictions.csv")
    require(len(all_rows) == len(cohort) * config["repeats"],
            "Unexpected total number of outer predictions.")
    all_ids = set(targets)
    for r in range(1, config["repeats"] + 1):
        group = [p for p in all_rows if int(p["repeat"]) == r]
        require(Counter(p["participant_id"] for p in group)
                == Counter({pid: 1 for pid in targets}),
                f"Repeat {r} must contain every participant exactly once.")
        require(all(int(p["target_et"]) == targets[p["participant_id"]]
                    for p in group), "Outer target disagrees with cohort.")
    rows = [p for p in all_rows if int(p["repeat"]) == repeat]
    pooled, pooled_curves = calculate(rows)
    saved_pooled = next(r for r in summary["repeat_metrics"]
                        if r["repeat"] == repeat)
    compare_metrics(pooled, saved_pooled, "Pooled outer metrics")
    plans = [p for p in read_json(run / "splits.json")
             if p["repeat"] == repeat]
    require(sorted(p["fold"] for p in plans)
            == list(range(1, config["outer_folds"] + 1)),
            "Outer fold plan is incomplete.")
    table = read_csv(run / "fold_metrics.csv")
    folds, curves = [], []
    checked_files = ["manifest.json", "summary.json", "cohort.csv",
                     "oof_predictions.csv", "fold_metrics.csv", "splits.json"]
    for plan in sorted(plans, key=lambda p: p["fold"]):
        fold = plan["fold"]
        train, test = set(plan["train_ids"]), set(plan["test_ids"])
        require(not (train & test) and train | test == all_ids,
                f"Fold {fold}: outer membership is invalid.")
        group = [p for p in rows if int(p["fold"]) == fold]
        require(set(p["participant_id"] for p in group) == test,
                f"Fold {fold}: prediction membership disagrees with splits.")
        valid_counts = Counter()
        require(len(plan["inner_splits"]) == config["inner_folds"],
                "Inner fold count disagrees with manifest.")
        for inner in plan["inner_splits"]:
            a, b = set(inner["train_ids"]), set(inner["validation_ids"])
            require(not (a & b) and a | b == train,
                    f"Fold {fold}: inner membership is invalid.")
            valid_counts.update(b)
        require(valid_counts == Counter({pid: 1 for pid in train}),
                f"Fold {fold}: incomplete inner validation coverage.")
        computed, curve = calculate(group)
        saved = next(r for r in summary["outer_fold_metrics"]
                     if r["repeat"] == repeat and r["fold"] == fold)
        compare_metrics(computed, saved, f"Fold {fold} summary")
        flat = next(r for r in table
                    if int(r["repeat"]) == repeat and int(r["fold"]) == fold)
        for key in METRIC_NAMES:
            require(abs(float(flat[key]) - computed[key]) < 1e-12,
                    f"Fold {fold}: metric CSV disagrees.")
        folder = f"repeat_{repeat:02d}_fold_{fold:02d}"
        per_fold = read_csv(run / folder / "test_predictions.csv")
        signature = lambda p: (p["participant_id"], int(p["target_et"]),
                               float(p["decision_score_et"]), int(p["predicted_et"]))
        require(sorted(map(signature, group)) == sorted(map(signature, per_fold)),
                f"Fold {fold}: saved test rows disagree.")
        compare_metrics(computed, read_json(run / folder / "metrics.json"),
                        f"Fold {fold} metrics file")
        search = read_csv(run / folder / "alpha_search.csv")
        best = max(float(a["inner_mean_balanced_accuracy"]) for a in search)
        selected = max(float(a["alpha"]) for a in search
                       if abs(float(a["inner_mean_balanced_accuracy"]) - best) <= 1e-12)
        require(selected == saved["selected_alpha"] == float(flat["selected_alpha"]),
                f"Fold {fold}: alpha does not follow the saved inner selection.")
        folds.append({"fold": fold, "selected_alpha": selected, **computed})
        curves.append(curve)
        checked_files.extend(f"{folder}/{name}" for name in
                             ("test_predictions.csv", "metrics.json", "alpha_search.csv"))
    for name, expected in manifest["source_code_sha256"].items():
        path = project / name
        require(path.is_file() and digest(path) == expected,
                f"Owning evaluation source has changed since this run: {name}")
    baseline_rows = [{**p, "decision_score_et": "-1", "predicted_et": "0"}
                     for p in rows]
    baseline, _ = calculate(baseline_rows)
    compare_metrics(baseline, summary["always_predict_pd_baseline"], "PD baseline")
    return {
        "repeat": repeat, "manifest": manifest, "pooled": pooled, "folds": folds,
        "baseline": baseline, "pooled_curves": pooled_curves, "fold_curves": curves,
        "input_sha256": {name: digest(run / name) for name in checked_files},
        "checks": {
            "one_outer_prediction_per_participant_per_repeat": True,
            "stored_labels_and_fixed_zero_threshold": True,
            "outer_and_inner_membership_matches_split_plan": True,
            "recomputed_metrics_match_saved_summary_and_fold_files": True,
            "inner_alpha_selection_matches_saved_rule": True,
            "pairwise_auroc_and_ap_rectangle_area": True,
            "recorded_evaluation_source_hashes_match_current_files": True,
        },
    }


def coordinates(xs, ys):
    return " ".join(f"({float(x):.10g},{float(y):.10g})"
                    for x, y in zip(xs, ys, strict=True))


def plot(options, xs, ys):
    return rf"\addplot[{options}] coordinates {{" + coordinates(xs, ys) + "};\n"


def confusion_figure(data):
    m = data["pooled"]
    cm = np.asarray(m["confusion_matrix_true_rows_predicted_columns_pd_et"])
    rates = cm / cm.sum(axis=1, keepdims=True)
    parts = [r"\begin{tikzpicture}[font=\small]"]
    for panel, values in enumerate((cm, rates)):
        parts.append(rf"\begin{{scope}}[xshift={panel * 7.3}cm]")
        title = "(a) Number of participants" if panel == 0 else "(b) Within each true class"
        parts.append(rf"\node[font=\small\bfseries] at (2.75,0.8) {{{title}}};")
        for col, name in enumerate(("PD", "ET")):
            parts.append(rf"\node at ({1.85 + 1.8 * col},0) {{{name}}};")
        for row, name in enumerate(("PD", "ET")):
            y = -1.05 - 1.8 * row
            n = int(cm[row].sum())
            parts.append(rf"\node[anchor=east] at (0.8,{y}) {{{name} ($n={n}$)}};")
            for col in range(2):
                x = 1.85 + col * 1.8
                relative = float(values[row, col]) / float(values.max())
                shade = int(8 + 87 * relative)
                label = str(int(values[row, col])) if panel == 0 else (
                    rf"{100 * values[row, col]:.2f}\%")
                textcolor = "white" if shade > 75 else "evalnavy"
                parts.append(rf"\filldraw[fill=evalteal!{shade}!white,draw=white,line width=1pt]"
                             rf" ({x - 0.9},{y - 0.9}) rectangle ({x + 0.9},{y + 0.9});")
                parts.append(rf"\node[text={textcolor},font=\large\bfseries]"
                             rf" at ({x},{y}) {{{label}}};")
        parts.append(r"\node at (2.75,-4.1) {Predicted class};")
        parts.append(r"\node[rotate=90] at (-1.8,-1.95) {True class};")
        parts.append(r"\end{scope}")
    return "\n".join(parts) + "\n\\end{tikzpicture}\n"


def curves_figure(data):
    m, base = data["pooled"], data["baseline"]
    parts = [r"""\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=2 by 1,horizontal sep=1.35cm},
 width=6.85cm,height=6.05cm,
 xmin=0,xmax=1,ymin=0,ymax=1.03,
 xtick={0,0.2,0.4,0.6,0.8,1},ytick={0,0.2,0.4,0.6,0.8,1},
 axis lines=left,axis line style={evalgray},
 grid=major,grid style={black!8},
 tick label style={font=\footnotesize},
 label style={font=\small},title style={font=\small\bfseries},
 legend style={font=\scriptsize,draw=none,fill=white,fill opacity=0.94,
  text opacity=1,cells={anchor=west}},
 clip=true]
\nextgroupplot[title={(a) ROC},xlabel={False positive rate (PD)},
 ylabel={True positive rate (ET)},
 legend style={at={(0.5,-0.30)},anchor=north}]
"""]
    for index, curve in enumerate(data["fold_curves"]):
        parts.append(plot("evalgray,line width=0.6pt",
                          curve["roc_fpr"], curve["roc_tpr"]))
        if index == 0:
            parts.append(r"\addlegendentry{Outer-fold curves}" + "\n")
        else:
            # Remove additional fold curves from the legend list.
            parts[-1] = parts[-1].replace("[evalgray,", "[forget plot,evalgray,")
    curve = data["pooled_curves"]
    parts.append(plot("evalteal,line width=1.35pt", curve["roc_fpr"], curve["roc_tpr"]))
    parts.append(rf"\addlegendentry{{Pooled AUROC = {m['roc_auc']:.3f}}}" + "\n")
    parts.append(plot("evalorange!70,dashed,line width=0.7pt", [0, 1], [0, 1]))
    parts.append(r"\addlegendentry{Chance diagonal}" + "\n")
    fpr, tpr = 1 - m["pd_recall"], m["et_recall"]
    parts.append(plot("forget plot,only marks,mark=*,mark size=1.8pt,evalorange", [fpr], [tpr]))
    parts.append(rf"\node[anchor=west,font=\scriptsize,text=evalorange]"
                 rf" at (axis cs:{fpr + 0.055:.8g},{tpr - 0.075:.8g}) {{$s>0$}};" + "\n")
    parts.append(rf"\draw[evalorange,thin,->] (axis cs:{fpr + 0.065:.8g},{tpr - 0.03:.8g})"
                 rf" -- (axis cs:{fpr:.8g},{tpr:.8g});" + "\n")
    parts.append(r"""\nextgroupplot[title={(b) ET precision--recall},
 xlabel={Recall (ET)},ylabel={Precision (ET)},
 legend style={at={(0.5,-0.30)},anchor=north}]
""")
    for index, fold_curve in enumerate(data["fold_curves"]):
        opts = "evalgray,line width=0.6pt" if index == 0 else (
            "forget plot,evalgray,line width=0.6pt")
        parts.append(plot(opts, fold_curve["pr_recall"], fold_curve["pr_precision"]))
        if index == 0:
            parts.append(r"\addlegendentry{Outer-fold curves}" + "\n")
    parts.append(plot("evalteal,line width=1.35pt", curve["pr_recall"], curve["pr_precision"]))
    parts.append(rf"\addlegendentry{{Pooled AP = {m['average_precision_et']:.3f}}}" + "\n")
    prevalence = base["average_precision_et"]
    parts.append(plot("evalorange!70,dashed,line width=0.7pt", [0, 1], [prevalence] * 2))
    parts.append(rf"\addlegendentry{{ET prevalence = {prevalence:.3f}}}" + "\n")
    parts.append(plot("forget plot,only marks,mark=*,mark size=1.8pt,evalorange",
                      [m["et_recall"]], [m["et_precision"]]))
    parts.append(rf"\node[anchor=west,font=\scriptsize,text=evalorange]"
                 rf" at (axis cs:{m['et_recall'] + 0.075:.8g},{m['et_precision'] - 0.13:.8g})"
                 r" {$s>0$};" + "\n")
    parts.append(rf"\draw[evalorange,thin,->]"
                 rf" (axis cs:{m['et_recall'] + 0.085:.8g},{m['et_precision'] - 0.08:.8g})"
                 rf" -- (axis cs:{m['et_recall']:.8g},{m['et_precision']:.8g});" + "\n")
    return "".join(parts) + "\\end{groupplot}\n\\end{tikzpicture}\n"


def folds_figure(data):
    panels = [
        ("balanced_accuracy", "(a) Balanced accuracy"),
        ("et_recall", "(b) ET recall"),
        ("pd_recall", "(c) PD recall"),
        ("accuracy", "(d) Accuracy"),
        ("roc_auc", "(e) AUROC"),
        ("average_precision_et", "(f) ET average precision"),
    ]
    parts = [r"""\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=3 by 2,horizontal sep=0.9cm,vertical sep=1.2cm},
 width=4.55cm,height=4.3cm,xmin=0.6,xmax=5.4,ymin=-0.03,ymax=1.04,
 xtick={1,2,3,4,5},ytick={0,0.25,0.5,0.75,1},
 xlabel={Outer fold},axis lines=left,axis line style={evalgray},
 grid=major,grid style={black!8},
 tick label style={font=\scriptsize},label style={font=\footnotesize},
 title style={font=\footnotesize\bfseries},clip=true]
"""]
    xs = [f["fold"] for f in data["folds"]]
    for index, (key, title) in enumerate(panels):
        xlabel = ",xlabel={}" if index < 3 else ""
        parts.append(rf"\nextgroupplot[title={{{title}}}{xlabel}]" + "\n")
        baseline = data["baseline"][key]
        parts.append(plot("evalgray,densely dotted,line width=0.7pt",
                          [0.6, 5.4], [baseline, baseline]))
        pooled = data["pooled"][key]
        parts.append(plot("evalorange,dashed,line width=0.9pt",
                          [0.6, 5.4], [pooled, pooled]))
        parts.append(plot("only marks,mark=*,mark size=2pt,evalnavy",
                          xs, [f[key] for f in data["folds"]]))
    return "".join(parts) + "\\end{groupplot}\n\\end{tikzpicture}\n"


def percent(value):
    return rf"{100 * value:.2f}\%"


def results_section(data, figures, run):
    m, base, config = data["pooled"], data["baseline"], data["manifest"]["config"]
    metrics = [
        ("균형 정확도", "balanced_accuracy", True),
        ("ET 재현율", "et_recall", True),
        ("PD 재현율", "pd_recall", True),
        ("일반 정확도", "accuracy", True),
        ("ET 정밀도", "et_precision", True),
        ("ET F1", "f1_et", False),
        ("AUROC", "roc_auc", False),
        ("ET 평균 정밀도(AP)", "average_precision_et", False),
    ]
    table = []
    for label, key, is_percent in metrics:
        fmt = percent if is_percent else lambda v: f"{v:.4f}"
        table.append(f"{label} & {fmt(m[key])} & {fmt(base[key])}" + r"\\" + "\n")
    fold_table = []
    for fold in data["folds"]:
        fold_table.append(
            f"{fold['fold']} & {fold['n_pd']} & {fold['n_et']} & "
            f"{fold['et_correct']}/{fold['n_et']} & {fold['selected_alpha']:g}" + r"\\" + "\n")
    means = {key: float(np.mean([f[key] for f in data["folds"]])) for key in METRIC_NAMES}
    stds = {key: float(np.std([f[key] for f in data["folds"]], ddof=1)) for key in METRIC_NAMES}
    template = r"""
% BEGIN GENERATED OUTER EVALUATION
\clearpage
\section{완료된 실행의 외부 평가 결과}\label{sec:observed-results}
\subsection{실행 이력과 평가 단위}
이 절은 \code{__RUN__}에 저장된 완료 상태의 실험을 요약한다.
실행 종료 시각은 \code{__FINISHED__} (UTC)이며,
외부 __OUTER__폴드, 내부 __INNER__폴드, 반복 __REPEATS__회,
시드 \(__SEED__\)를 사용하였다.
아래 그림과 표는 반복 \(__REPEAT__\)에서 참가자별로 정확히 한 번 생성한
외부 보류 예측 \(__N__\)건에 근거한다.
실제 라벨과 \(\widehat y=\ind\{s>0\}\), 외부ㆍ내부 분할의 참가자 배정,
폴드별 예측 파일 및 저장 지표의 일치를 확인하였다.
최종 전체 자료 적합 모델의 학습 예측과 내부 선택 지표는 여기에 포함하지 않는다.

\begin{table}[H]
\centering\small
\caption{반복 __REPEAT__의 통합 외부 예측 지표와 항상 PD 기준선.
F1ㆍAUROCㆍAP는 \(0\)--\(1\) 척도이며, 나머지는 백분율이다.
ET 정밀도와 F1의 분모가 \(0\)인 기준선은 구현 규약에 따라 \(0\)으로 표시한다.}
\label{tab:observed-metrics}
\begin{tabular}{@{}lrr@{}}
\toprule 지표 & MultiROCKET--HYDRA & 항상 PD\\ \midrule
__METRIC_ROWS__
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
__CONFUSION__
\caption{통합 외부 혼동행렬. 행은 실제 진단, 열은 예측 진단이다.
(a)는 참가자 수, (b)는 각 실제 클래스 안의 백분율이다.
고정 임계값 \(s>0\)에서 PD __TN__/__NPD__명과 ET __TP__/__NET__명을 맞혔다.
색의 진하기는 각 패널 내부 최댓값을 기준으로 하므로 두 패널 사이의 색을
절대 인원 비교에 사용하지 않는다.}\label{fig:evaluation-confusion}
\end{figure}

일반 정확도는 __ACCURACY__이지만 ET __FN__/__NET__명은 PD로 예측되었다.
따라서 이 실행의 고정 결정 규칙에서는 PD 재현율(__PDRECALL__)과
ET 재현율(__ETRECALL__)이 크게 다르다.
균형 정확도 __BA__는 항상 PD 기준선보다 __BA_GAIN__ 퍼센트포인트 높다.
이는 이 자료ㆍ분할에서 관측한 차이이며 유의성 검정이나 독립 코호트의
개선 보장을 뜻하지 않는다.

\clearpage
\subsection{순위 지표와 고정 임계값의 구분}
\begin{figure}[H]
\centering
__CURVES__
\caption{외부 점수의 ROC와 ET 정밀도--재현율 곡선.
회색은 각 외부 폴드, 청록색은 참가자별 외부 점수의 통합 곡선이다.
주황색 점은 사전에 고정한 결정 규칙 \(s>0\)의 통합 작동점이다.
ROC 점선은 무정보 대각선, PR 점선은 전체 ET 비율
\(__PREVALENCE__\)이다. PR 계단의 면적은 비보간 AP 정의와 일치한다.
다른 폴드 모델의 점수 척도가 다를 수 있어 통합 순위 지표는 폴드별
지표와 함께 해석한다.}\label{fig:evaluation-curves}
\end{figure}

통합 AUROC는 \(__AUROC__\), 통합 ET AP는 \(__AP__\)이다.
외부 폴드별 AUROC의 단순 평균은 \(__AUCMEAN__\), 표본 표준편차는
\(__AUCSTD__\)이며, AP의 평균과 표본 표준편차는 각각
\(__APMEAN__\), \(__APSTD__\)이다.
통합 지표는 서로 다른 모델 사이의 점수까지 비교하지만 폴드별 지표는
각 모델 내부의 순위만 비교하므로 두 집계 방식의 값이 같을 필요는 없다.
이 표준편차는 겹치는 학습 집합에서 얻은 기술적 산포이며 신뢰구간이 아니다.

순위 성능과 고정 임계값의 ET 검출률은 별개의 성질이다.
예를 들어 외부 폴드 4의 AUROC와 AP는 모두 \(1\)이지만,
임계값 \(0\)에서는 ET 6명 중 2명만 ET로 예측하였다.
PD보다 높은 점수를 받은 ET라도 점수 자체가 \(0\) 이하일 수 있기 때문이다.
따라서 높은 AUROC를 높은 ET 재현율 또는 확률 보정의 근거로 바꾸어 읽으면
안 된다. 이 결과를 본 뒤 외부 자료에서 임계값을 골라 같은 자료의 성능을
확증적 결과로 보고하는 것도 피해야 한다.

\clearpage
\subsection{외부 폴드의 인원과 정규화 선택}
\begin{table}[H]
\centering\small
\caption{외부 테스트 폴드의 클래스별 인원, ET 정답 수와
내부 검증에서 선택한 릿지 강도.}
\label{tab:observed-folds}
\begin{tabular}{@{}rrrrr@{}}
\toprule 외부 폴드 & PD 인원 & ET 인원 & ET 정답/전체 & 선택 \(\alpha\)\\ \midrule
__FOLD_ROWS__
\bottomrule
\end{tabular}
\end{table}
다섯 외부 폴드 모두 탐색 격자의 상한인 \(\alpha=1000\)을 선택하였다.
이는 저장된 후보 안에서의 선택을 뜻하며 \(1000\)보다 큰 값이 최적인지는
이 실험으로 알 수 없다. 더 넓은 격자를 비교하려면 별도의 평가 계획과
중첩 선택 절차를 적용해야 한다.

\subsection{폴드별 지표의 산포}
\begin{figure}[H]
\centering
__FOLDS__
\caption{다섯 외부 테스트 폴드의 지표. 점은 개별 폴드, 주황색 점선은
통합 외부 지표, 회색 점선은 전체 코호트의 항상 PD 기준선이다.
일반 정확도와 AP의 기준선은 폴드마다 조금 다른 클래스 비율을 보정한 값이
아니라 전체 코호트 기준값이다. 모든 패널의 세로축은 \(0\)--\(1\) 척도이다.
폴드를 잇는 추세선이나 신뢰구간은 표시하지 않는다.}
\label{fig:evaluation-folds}
\end{figure}

폴드별 균형 정확도는 \(__BAMIN__\)--\(__BAMAX__\)이고,
단순 평균은 \(__BAMEAN__\), 표본 표준편차는 \(__BASTD__\)이다.
폴드별 ET 재현율은 \(__ETMIN__\)--\(__ETMAX__\)이다.
ET가 각 폴드에 5명 또는 6명뿐이므로 한 사람의 예측 변화가 재현율을
20 또는 약 16.67 퍼센트포인트 바꾼다.
한 번의 5폴드 평가를 독립 실험 다섯 번으로 해석하지 않는다.

\paragraph{재현성.}
그림의 모든 수치 좌표를 이 원고에 직접 포함하였다.
지표 재계산, 입력 해시와 그림 재생성 절차는 부록~\ref{app:evaluation-rebuild}에 제시한다.
\clearpage
% END GENERATED OUTER EVALUATION

"""
    values = {
        "RUN": run.as_posix(), "FINISHED": data["manifest"]["finished_at_utc"][:19],
        "OUTER": config["outer_folds"], "INNER": config["inner_folds"],
        "REPEATS": config["repeats"], "SEED": config["seed"], "REPEAT": data["repeat"],
        "N": m["n_pd"] + m["n_et"], "METRIC_ROWS": "".join(table),
        "CONFUSION": figures["evaluation_confusion"], "CURVES": figures["evaluation_curves"],
        "FOLDS": figures["evaluation_folds"], "FOLD_ROWS": "".join(fold_table),
        "TN": m["pd_correct"], "TP": m["et_correct"], "FN": m["et_called_pd"],
        "NPD": m["n_pd"], "NET": m["n_et"], "ACCURACY": percent(m["accuracy"]),
        "ETRECALL": percent(m["et_recall"]), "PDRECALL": percent(m["pd_recall"]),
        "BA": percent(m["balanced_accuracy"]),
        "BA_GAIN": f"{100 * (m['balanced_accuracy'] - base['balanced_accuracy']):.2f}",
        "PREVALENCE": f"{base['average_precision_et']:.4f}",
        "AUROC": f"{m['roc_auc']:.4f}", "AP": f"{m['average_precision_et']:.4f}",
        "AUCMEAN": f"{means['roc_auc']:.4f}", "AUCSTD": f"{stds['roc_auc']:.4f}",
        "APMEAN": f"{means['average_precision_et']:.4f}",
        "APSTD": f"{stds['average_precision_et']:.4f}",
        "BAMIN": f"{min(f['balanced_accuracy'] for f in data['folds']):.4f}",
        "BAMAX": f"{max(f['balanced_accuracy'] for f in data['folds']):.4f}",
        "BAMEAN": f"{means['balanced_accuracy']:.4f}",
        "BASTD": f"{stds['balanced_accuracy']:.4f}",
        "ETMIN": f"{min(f['et_recall'] for f in data['folds']):.4f}",
        "ETMAX": f"{max(f['et_recall'] for f in data['folds']):.4f}",
    }
    for key, value in values.items():
        template = template.replace(f"__{key}__", str(value))
    require("__" not in template, "Unresolved result template placeholder.")
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/pads_mr_hydra"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output/figures/pads_mr_hydra"))
    parser.add_argument("--article", type=Path,
                        default=Path("sota_pd_et/docs/model_explanation_ko.tex"))
    parser.add_argument("--update-article", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    data = load_verified(args.run, args.repeat, project)
    # The article discussion is explicitly about this five-fold, 304-person
    # experiment. Stop instead of silently reusing cohort-specific prose.
    require(data["manifest"]["config"]["outer_folds"] == 5
            and data["pooled"]["n_pd"] == 276 and data["pooled"]["n_et"] == 28,
            "This article template is specific to the PADS PD/ET five-fold run.")
    require(all(f["selected_alpha"] == 1000 for f in data["folds"]),
            "Revise the article's alpha-boundary interpretation for this run.")
    fourth = data["folds"][3]
    require(fourth["roc_auc"] == fourth["average_precision_et"] == 1
            and fourth["n_et"] == 6 and fourth["et_correct"] == 2,
            "Revise the article's fold-4 operating-point example for this run.")
    figures = {
        "evaluation_confusion": confusion_figure(data),
        "evaluation_curves": curves_figure(data),
        "evaluation_folds": folds_figure(data),
    }
    section = COLORS + results_section(data, figures, args.run)
    # Color declarations belong inside the replacement markers as well.
    section = section.replace(COLORS, "", 1).replace(BEGIN, BEGIN + "\n" + COLORS, 1)
    wrapper = r"""\documentclass[border=3mm]{standalone}
\usepackage{amsmath,amssymb,pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
""" + COLORS + "\n\\begin{document}\n"
    args.output.mkdir(parents=True, exist_ok=True)
    for name, body in figures.items():
        (args.output / f"{name}.tex").write_text(wrapper + body + "\\end{document}\n")
    (args.output / "article_evaluation_section.tex").write_text(section)
    report = {k: v for k, v in data.items() if k != "manifest"}
    report["run"] = args.run.as_posix()
    report["finished_at_utc"] = data["manifest"]["finished_at_utc"]
    report["config"] = data["manifest"]["config"]
    report["source_code_sha256"] = data["manifest"]["source_code_sha256"]
    (args.output / "evaluation_data.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.update_article:
        text = args.article.read_text()
        if BEGIN in text or END in text:
            require(text.count(BEGIN) == text.count(END) == 1,
                    "The article's generated section markers are ambiguous.")
            start, stop = text.index(BEGIN), text.index(END) + len(END)
            text = text[:start] + section.strip() + text[stop:]
        else:
            anchor = r"\section{실행 및 결과 파일}"
            require(text.count(anchor) == 1, "Article insertion point is ambiguous.")
            text = text.replace(anchor, section + anchor, 1)
        args.article.write_text(text)
    print(json.dumps({"output": str(args.output), "repeat": args.repeat,
                      "pooled": data["pooled"], "checks": data["checks"]}, indent=2))


if __name__ == "__main__":
    main()
