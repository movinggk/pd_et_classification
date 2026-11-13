# 시간 표현·과제 간 상호작용 PD/ET 실험 보고서

완료된 실행 `20261108T071720522119Z`를 분석한 **17쪽 한국어 LaTeX 보고서**다.
신경망 5개와 기준선 2개를 동일한 60채널 입력·참가자 단위 5×3 중첩 검증으로
비교한 사용자 실행의 결과를 다룬다. 보고서를 만드는 동안 새로운 학습이나
모델 예측은 수행하지 않았다.

- 편집·컴파일 원본: `report.tex`.
- 완성 PDF: 프로젝트의 `output/pdf/temporal_cross_task_report_ko.pdf`.
- 감사 기록과 분석 자료: `verification.json`, `metrics.csv`.
- 생성기와 편집용 템플릿: `build_report.py`, `report_template.tex`.
- 문서 검증 기록: `document_verification.json`.

`report.tex`에 모든 표와 PGFPlots/TikZ 그림 좌표가 내장되어 있다. 이 파일만으로
본문 PDF를 컴파일할 수 있으며, 학습 자료·모델 파일·외부 그림은 필요하지 않다.

## LaTeX 컴파일

XeLaTeX, ko.TeX, PGFPlots를 포함한 TeX Live 환경에서 이 폴더를 연 뒤 실행한다.

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
```

이 명령은 현재 폴더에 `report.pdf`와 보조 파일을 만든다. 기본 한국어 글꼴은
TeX Live에 포함된 UnBatang이다. 참고문헌은 원본에 내장되어 있어 BibTeX 단계가
필요하지 않다. 제공된 PDF는 모든 페이지를 이미지로 렌더링해 검토했다.

## 원본 실행 결과에서 표·좌표 다시 생성

프로젝트 폴더에서 다음 명령을 실행한다. 이 과정은 결과 파일의 감사와 문서
생성이며, 모델을 학습하거나 모델의 forward 함수를 호출하지 않는다.

```sh
.venv-transformer/bin/python Temporal_CrossTask_Report/build_report.py
```

생성기는 프로젝트 안의 기존 PADS 데이터, 완료된 실행, 과거 기준선 폴더와
NumPy·scikit-learn·PyTorch 환경을 읽는다. 기본값은 위 완료 실행이며,
`--run-dir`로 경로를 바꿀 수 있다. 템플릿은 이번 기본 실험 명세를 설명하므로
다른 입력·구조·검증 명세에는 본문과 생성기를 함께 수정해야 한다.
단독 LaTeX 컴파일에는 이 생성 단계를 실행할 필요가 없다.

## 보고서의 핵심 결과

Full의 외부 BA는 56.16%, ET 민감도는 25.00%(7/28), PD 특이도는 87.32%다.
Neither는 BA 61.57%, MantisV2는 BA 61.05%이며 MR–HYDRA의 pooled AUROC는
0.8612다. 이번 실행에서 제안 구조의 개선은 관측되지 않았다.
여섯 대조군에 대한 Full의 BA 차이의 조건부 대응 bootstrap 구간은 모두 0을
포함한다. 이 구간은 적합된 교차검증 모델을 고정한 조건부 구간이며,
새 데이터로 재학습할 때의 변동이나 외부 임상 유효성을 확립하지 않는다.

보고서는 2,128개 외부 예측, 35개 외부 fold 지표, 선택된 내부 검증 예측과
후보 선택, 대응 bootstrap 계산을 검사한다. 원 데이터·정의 파일 611개의
SHA-256, 실행 당시 보관 소스 23개의 SHA-256과 Mantis 특징 캐시도 확인했다.
원래 실행 폴더와 학습 코드는 수정하지 않았다.
