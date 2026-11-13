# MultiROCKET–HYDRA 한국어 Beamer 발표자료

`model_summary_beamer_ko.tex`는 PADS v1.0.0의 PD/ET 분류 모델과 평가 설계를
설명하는 독립형 한국어 Beamer 원고입니다. 총 33장(본문 23장, 부록 10장)이며,
슬라이드 비율은 16:9입니다.

본문은 데이터·입력 표현, 두 특징 추출기, 정규화와 가중 릿지, 중첩 교차검증,
평가 지표와 실행 절차를 요약합니다. 부록은 정확한 수치 규약, 정규방정식과
유일성, 순위 지표, 문헌 및 고정 버전 구현의 출처를 제공합니다.

## 포함 파일

- `model_summary_beamer_ko.tex`: 수식, 표, TikZ 도식, 참고문헌, 발표자 메모를
  모두 포함하는 편집용 원고.
- `model_summary_beamer_ko.pdf`: 발표용 PDF.
- `README_ko.md`: 묶음 파일 안의 이 안내문.

외부 그림, BibTeX 데이터베이스, 학습 데이터 또는 Python 환경은 컴파일에
필요하지 않습니다. 컴파일 과정은 모델을 학습하거나 평가하지 않습니다.

## 컴파일

XeLaTeX, Beamer, ko.TeX, TikZ 등을 포함하는 TeX 배포판을 사용합니다.
원고는 UnDotum 글꼴 파일을 찾으면 사용하고, 찾지 못하면 ko.TeX의 기본
한글 글꼴 설정을 사용합니다.

묶음을 풀어 같은 폴더에 있는 원고를 컴파일할 때:

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error model_summary_beamer_ko.tex
```

프로젝트 루트에서 컴파일할 때:

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf sota_pd_et/docs/model_summary_beamer_ko.tex
```

`latexmk`가 없으면 `xelatex`를 여러 번 실행하여 페이지 수와 내부 링크를
갱신할 수 있습니다. Codex LaTeX 편집기에서도 원고와 PDF 미리보기를 볼 수 있습니다.

## 편집과 발표자 메모

모든 슬라이드는 `frame` 환경에 들어 있습니다. 제목과 날짜는 원고 앞부분의
`title`, `subtitle`, `author`, `date` 명령에서 변경합니다. 모델 도식도 원고 안의
TikZ 코드로 편집할 수 있습니다.

기본 설정은 `\setbeameroption{hide notes}`입니다. 메모를 출력하려면 이 줄을
`\setbeameroption{show notes}`로 바꿉니다. 메모 출력 시에는 PDF 페이지 수가
발표 슬라이드 수보다 많아질 수 있습니다.

자료는 현재 구현과 평가 절차의 설명입니다. 다수 클래스 기준선과 특징 차원은
데이터 구성 및 설정에서 계산한 값이며, 학습한 모델의 실제 평가 성능을 뜻하지
않습니다. 실제 성능은 완료된 실험의 외부 예측 및 평가 파일에서 확인합니다.
