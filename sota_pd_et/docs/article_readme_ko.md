# 평가 그림을 포함한 한국어 모델 설명서

문서는 총 18쪽입니다. 모델과 수학적 구조를 설명한 뒤, 12–14쪽에서 완료된
PADS PD/ET 실험의 참가자별 외부 예측 304건을 그림과 표로 요약합니다.

## 묶음 구성

- `model_explanation_ko.tex`: 편집용 한국어 article 원고.
- `model_explanation_ko.pdf`: 평가 결과를 포함한 문서 PDF.
- `figures/evaluation_confusion.pdf`, `.tex`: 혼동행렬의 인원과 실제 클래스별 비율.
- `figures/evaluation_curves.pdf`, `.tex`: 폴드별·통합 ROC 및 ET 정밀도–재현율 곡선.
- `figures/evaluation_folds.pdf`, `.tex`: 외부 5폴드의 여섯 지표와 통합 값·기준선.
- `figures/evaluation_data.json`: 재계산한 지표, 곡선 좌표, 입력 SHA-256, 검증 항목.
- `plot_evaluation.py`: 원래 프로젝트에서 사용하는 그림 재생성 코드의 사본.
- `README_ko.md`: 이 안내문.

## 문서 컴파일

XeLaTeX, ko.TeX, TikZ, PGFPlots를 포함하는 TeX 배포판을 사용합니다.
수식·도식·모든 평가 그림 좌표와 참고문헌이 원고 안에 있어, 컴파일에는 외부 그림,
데이터, 체크포인트 또는 Python 환경이 필요하지 않습니다.

묶음을 풀고 해당 폴더에서:

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error model_explanation_ko.tex
```

원래 프로젝트 루트에서는:

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf sota_pd_et/docs/model_explanation_ko.tex
```

개별 그림의 `.tex`도 독립형 원고이므로 같은 방법으로 컴파일할 수 있습니다.
코드와 수치 좌표를 편집하면 벡터 그림의 글꼴·색·축·범례·패널 구성을 수정할 수
있습니다. 문서 원고의 그림은 자체 좌표를 사용하므로, 개별 그림 파일의 편집이
문서에 자동으로 반영되지는 않습니다.

## 원래 실험 출력에서 재생성

이 묶음은 원래 데이터나 모델 체크포인트를 포함하지 않습니다. 지표와 그림을
재생성할 때는 원래 프로젝트와 완료된 `artifacts/pads_mr_hydra` 결과가 필요합니다.
원래 프로젝트 루트에서 Python 환경을 사용하여 실행합니다.

```sh
.venv-sota/bin/python -m sota_pd_et.plot_evaluation \
  --run artifacts/pads_mr_hydra --repeat 1 --update-article
```

이 명령은 모델을 다시 학습하지 않습니다. 완료 상태, 참가자별 외부 예측의 유일성,
실제 라벨, 고정 임계값, 분할 배정, 폴드별 파일, 저장 지표 및 실행 당시 평가 코드
해시를 확인합니다. 지표와 독립형 그림 원고를
`output/figures/pads_mr_hydra/`에 저장하고, 문서 안의 생성 표시 사이에 평가 절을
갱신합니다. 벡터 PDF는 생성한 `.tex`를 별도로 컴파일하여 내보냅니다.

설명 문구는 현재 304명·5폴드 실험에 맞춰져 있습니다. 다른 코호트, 선택한 릿지
강도, 폴드 4의 예시로 인해 문구가 달라져야 하면 자동으로 중단합니다.

실수 점수는 확률이 아닙니다. 통합 AUROC·AP에는 서로 다른 폴드 모델의 점수 척도
차이가 들어가므로 폴드별 지표도 함께 읽습니다. 폴드 표준편차는 기술적 산포이며
신뢰구간으로 표시하지 않습니다.
