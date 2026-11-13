# 동결 MantisV2 특징 추출 한국어 설명서

이 안내문은 `mantis_v2_explanation_ko.tex`의 열람과 컴파일 방법을 설명합니다.
문서 작성 기준일은 **2026년 10월 25일**이고, 설명하는 완료 실험의 실행일은
**2026년 10월 24일**입니다. 문서의 수치와 그림은 저장된 실험 결과를 설명하며,
문서 컴파일은 특징 추출이나 모델 재학습을 실행하지 않습니다.

## 파일과 배포 묶음

프로젝트 안의 경로는 다음과 같습니다.

- `sota_pd_et/docs/mantis_v2_explanation_ko.tex`: 편집용 독립형 LaTeX 원고.
- `sota_pd_et/docs/mantis_v2_readme_ko.md`: 이 안내문.
- `output/pdf/mantis_v2_explanation_ko.pdf`: 컴파일한 한국어 설명서 PDF.
- `output/pdf/mantis_v2_explanation_ko_bundle.zip`: 원고, PDF와 안내문을 담은 배포 묶음.

원고에는 수식, TikZ 도식, PGFPlots 평가 그림의 수치 좌표와 참고문헌이 들어
있습니다. 다른 `.tex` 원고를 불러오거나 외부 이미지 및 `.bib` 파일을 읽을 필요가
없습니다. 원고만 별도 폴더로 옮겨도 필요한 TeX 환경이 있으면 컴파일할 수 있습니다.
배포 묶음은 설명 문서용이며, 원본 PADS 데이터, 특징 캐시, 모델 가중치와 최종
분류기를 포함하지 않습니다.

## 문서에서 설명하는 내용

문서는 참가자와 표적 라벨을 정의한 뒤, 입력에서 평가까지의 계산을 순서대로
설명합니다.

1. PADS 코호트의 PD 276명, ET 28명과 참가자별 `(132, 976)` 입력 행렬.
2. 채널별 길이 976 → 512 선형 보간과 고정된 MantisV2 인코더.
3. 신호·차분·통계 경로, 패치 토큰, Transformer attention, 세 번째 층의 표현.
4. 256차원 class token과 256차원 평균 patch token을 결합한 채널당 512차원
   표현 및 132개 채널을 연결한 참가자당 67,584차원 특징.
5. 학습 집합에서만 적합하는 특징 표준화, 클래스 가중 릿지의 목적함수와
   양수 마진일 때 ET로 분류하는 고정 결정 규칙.
6. 참가자 단위 외부 5폴드·내부 3폴드 중첩 교차검증, 정규화 강도 선택과
   전체 코호트 최종 적합의 구분.
7. 혼동행렬, 균형정확도, 재현율, 정밀도, AUROC, AP의 정의와 기존
   MultiROCKET–HYDRA 결과와의 비교.
8. 실제 Apple MPS 특징 추출, CPU 분류기 적합, 수치 검증, 버전 고정과 해석의 한계.

두 방법은 동일한 304명과 동일한 외부·내부 참가자 분할을 사용했습니다. 저장된
외부 검증 예측을 합치면 균형정확도는 67.13%에서 70.89%로, ET 정답 수는
10/28에서 12/28로 증가했고, AUROC는 0.9000에서 0.8876으로 감소했습니다.
이는 일부 지표의 개선을 보인 탐색적 내부 코호트 비교입니다. 보편적 최우수
모델이나 외부 임상 성능을 확립한 결과로 읽지 않습니다. 출력 마진은 보정된
확률이 아니며, 최종 전체 코호트 적합 모델의 학습 성능을 외부 검증 성능으로
대체하지 않습니다.

## 컴파일 환경과 명령

**XeLaTeX, ko.TeX, TikZ, PGFPlots**를 포함하는 TeX 배포판과 `latexmk`가
필요합니다. 한글 글꼴도 설치되어 있어야 합니다. 원고는 `UnBatang.ttf`가 있으면
이를 사용하고, 없으면 ko.TeX의 한글 글꼴 설정을 따릅니다. 그림은 TeX 안에서
생성하므로 Python 환경, GPU, 체크포인트 또는 네트워크 연결은 컴파일에 필요하지
않습니다.

프로젝트 루트에서 다음 명령으로 PDF를 생성합니다.

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf sota_pd_et/docs/mantis_v2_explanation_ko.tex
```

출력 PDF는 `output/pdf/mantis_v2_explanation_ko.pdf`입니다. `latexmk`는 목차와
상호 참조가 안정될 때까지 필요한 횟수만큼 XeLaTeX를 실행합니다. 배포 묶음을
풀어 원고와 같은 폴더에서 컴파일할 때는 다음 명령을 사용합니다.

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  mantis_v2_explanation_ko.tex
```

이 경우 PDF는 해당 폴더에 생성됩니다. TeX 설치에 `latexmk`가 없다면 XeLaTeX를
여러 번 실행하여 목차·참조가 안정되었는지 로그를 확인해야 합니다. 한글이나
패키지 관련 오류가 발생하면 먼저 ko.TeX, 한글 글꼴과 PGFPlots의 설치 상태를
확인합니다.

## 수치·구현의 원래 근거

다음 경로는 원래 프로젝트 루트를 기준으로 합니다. PDF를 컴파일하기 위해
이 파일들이 필요한 것은 아니며, 설명과 수치의 출처를 추적할 때 사용합니다.

| 대상 | 저장 경로 |
| --- | --- |
| MantisV2 실행 보고서와 전체 지표 | `artifacts/pads_mantis_v2/REPORT.md`, `summary.json` |
| MantisV2 참가자별 외부 검증 예측 | `artifacts/pads_mantis_v2/oof_predictions.csv` |
| MantisV2 분할·코호트·폴드 지표 | `artifacts/pads_mantis_v2/splits.json`, `cohort.csv`, `fold_metrics.csv` |
| 기존 방법의 참가자별 외부 검증 예측 | `artifacts/pads_mr_hydra/oof_predictions.csv` |
| 기존 방법의 지표와 분할 | `artifacts/pads_mr_hydra/summary.json`, `splits.json` |
| 기존 방법과의 짝지은 비교 | `artifacts/pads_mantis_v2/baseline_comparison.json`, `paired_predictions.csv` |
| 인코더·입력·실행 버전의 출처 | `artifacts/pads_mantis_v2/encoder_provenance.json`, `manifest.json`, `environment-lock.txt` |
| CPU–MPS 수치 비교와 완료 검증 | `artifacts/pads_mantis_v2/mps_validation.json`, `verification.json` |
| 실험 실행·재현 방법 | `sota_pd_et/TRANSFORMER_README.md` |
| 입력 로더 | `sota_pd_et/data.py` |
| Transformer 특징 추출 | `sota_pd_et/transformer.py` |
| Transformer 평가 실행 | `sota_pd_et/evaluate_transformer.py` |
| 공통 평가·분류 구현 | `sota_pd_et/evaluate.py`, `model.py` |

혼동행렬과 곡선은 두 방법의 외부 검증 예측에 근거합니다. 각 참가자는 실제
라벨 하나와 외부 폴드 예측 하나를 가지며, ET를 양성 클래스 `1`, PD를 음성
클래스 `0`으로 둡니다. 혼동행렬의 행은 실제 클래스, 열은 예측 클래스이고
두 축 모두 `[PD, ET]` 순서입니다. PR 기준선은 ET 비율 `28/304`입니다.
통합 AUROC·AP는 서로 다른 외부 폴드에서 적합한 모델의 점수를 합친 기술적
결과이므로 폴드별 결과도 함께 읽습니다.

원고 안의 지표와 곡선 좌표는 내장된 고정 값입니다. 원래 실험 파일이 바뀌어도
컴파일만으로 자동 갱신되지 않습니다. 새 실험을 설명하려면 저장 예측에서
지표·좌표를 다시 검증하고, 원고의 표·그림·설명과 실행 날짜를 함께 갱신해야
합니다. 모델 실험을 실제로 다시 실행하려면 위의 `TRANSFORMER_README.md`를
따르며, 기존 결과를 보존하도록 새 출력 디렉터리를 사용합니다.
