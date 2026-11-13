# 한국어 LaTeX 모델 설명서와 Beamer 발표자료

## MantisV2 Transformer 특징 추출 설명서

[한국어 원고](mantis_v2_explanation_ko.tex),
[PDF](../../output/pdf/mantis_v2_explanation_ko.pdf),
[원고·PDF·컴파일 안내 묶음](../../output/pdf/mantis_v2_explanation_ko_bundle.zip)을 제공합니다.
기존 사전학습 Transformer의 토큰 생성·attention·특징 결합부터,
가중 릿지와 참가자 단위 중첩 평가까지 수식으로 설명합니다.
완료된 MPS 실험과 MultiROCKET–HYDRA의 실제 외부 예측을 비교한
혼동행렬, ROC·PR 곡선, 폴드별 결과를 포함합니다.
별도 그림이나 BibTeX 파일 없이 컴파일할 수 있습니다.
[문서 구성·독립 컴파일 안내](mantis_v2_readme_ko.md)를 참고하세요.

## 상세 설명서

[model_explanation_ko.tex](model_explanation_ko.tex)는 현재 구현의 입력 표현,
MultiROCKET·HYDRA 특징, 두 정규화, 가중 릿지 목적함수, 참가자 단위 중첩
교차검증과 평가 지표를 설명하는 단일 LaTeX 원고입니다. 완료된 실험
`artifacts/pads_mr_hydra`의 외부 예측 304건을 검증하고 평가 결과 절을 추가했습니다.
수식, TikZ 흐름도, PGFPlots 평가 그림의 수치 좌표와 참고문헌을 모두 포함하므로
별도의 그림이나 BibTeX 파일이 필요하지 않습니다.

XeLaTeX, ko.TeX, PGFPlots를 포함하는 TeX 배포판에서 프로젝트 루트를 기준으로
컴파일합니다.

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf sota_pd_et/docs/model_explanation_ko.tex
```

`latexmk`가 없으면 같은 출력 디렉터리를 만든 뒤 `xelatex`를 여러 번 실행하여
목차·인용·교차참조를 해결할 수 있습니다. Codex의 LaTeX 편집기에서도 원고를 열어
소스와 PDF 미리보기를 함께 볼 수 있습니다.

추가된 평가 그림은 다음과 같습니다.

- 통합 외부 혼동행렬: 참가자 수와 실제 클래스별 백분율.
- ROC와 ET 정밀도–재현율 곡선: 각 외부 폴드, 통합 곡선, 고정 임계값의 작동점.
- 폴드별 지표: 균형 정확도, ET/PD 재현율, 일반 정확도, AUROC, ET AP.

[갱신된 문서 PDF](../../output/pdf/model_explanation_ko.pdf)와
[독립 컴파일 묶음](../../output/pdf/model_explanation_ko_bundle.zip)을 제공합니다.
단독 그림의 벡터 PDF·LaTeX 원고와 검증된 수치·입력 SHA-256은
`output/figures/pads_mr_hydra/`에 저장합니다. 폴드 간 산포는 신뢰구간으로
표시하지 않으며, 전체 적합 모델의 학습 성능은 평가 그림에 포함하지 않습니다.

기존 실험 출력에서 지표를 재계산하고 그림과 원고의 평가 절을 재생성하려면:

```sh
.venv-sota/bin/python -m sota_pd_et.plot_evaluation \
  --run artifacts/pads_mr_hydra --repeat 1 --update-article
```

이 명령은 모델을 학습하지 않습니다. 실험이 완료되었는지, 참가자별 외부 예측이
정확히 한 번인지, 분할·라벨·고정 임계값·저장 지표·실행 당시 코드 해시가
일치하는지 확인합니다. 현재 원고의 해석은 해당 304명·5폴드 실험에 맞춰져 있어,
선택한 정규화 강도나 폴드 4의 예시가 달라지면 자동으로 중단하고 문구 수정을
요구합니다. 생성한 단독 `.tex`는 XeLaTeX로 컴파일하여 벡터 PDF로 내보냅니다.

## 요약 발표자료

[model_summary_beamer_ko.tex](model_summary_beamer_ko.tex)는 상세 설명서의 핵심을
요약한 한국어 16:9 Beamer 원고입니다. 총 33장으로, 본문 23장과 계산 규약·수학
유도·참고문헌을 담은 부록 10장으로 구성되어 있습니다.

- 본문: 데이터와 라벨, 입력 구조, MultiROCKET·HYDRA, 분기별 정규화, 가중 릿지,
  참가자 단위 중첩 교차검증, 평가 지표, 실행과 결과 확인.
- 부록: 기호, 정확한 풀링과 경계 규약, HYDRA 커널과 희소 특징 척도,
  가중 정규방정식과 해의 유일성, AUROC·AP 정의, 원문과 구현 출처.

수식·표·TikZ 도식·참고문헌·발표자 메모가 하나의 `.tex` 파일에 들어 있습니다.
기본 PDF에서는 발표자 메모를 숨깁니다. 별도 그림, 데이터, `.bib` 파일 없이
컴파일할 수 있으며 모델 학습을 실행하지 않습니다.

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf sota_pd_et/docs/model_summary_beamer_ko.tex
```

[발표용 PDF](../../output/pdf/model_summary_beamer_ko.pdf),
[원고·PDF 묶음](../../output/pdf/model_summary_beamer_ko_bundle.zip),
[독립 컴파일 안내](beamer_readme_ko.md)를 함께 제공합니다.
