# PD/ET: 과제 내 시간 변화와 과제 간 상호작용 모델

이 구현은 **PADS의 안정 시, 인지 부하 안정 시, 자세 조건**에서 양측 손목의
가속도계·자이로스코프를 사용한다. 새로운 경량 신경망과 구성요소 제거 모델,
동일 입력의 Frozen MantisV2 및 MultiROCKET–HYDRA를 비교하는 실행 코드다.

**작성 단계에서는 실제 PADS 학습·평가를 실행하지 않았다.** 합성 참가자를 이용한
자동 테스트만 수행한다. 실제 성능 수치는 사용자가 아래 명령을 실행한 뒤 생성된다.
기존 실험 폴더의 결과는 이 모델의 결과가 아니다.

## 1. 시작하기

이 작업의 소스 코드 묶음은 `output/code/PD_ET_Temporal_CrossTask_Source.zip`이다.
다른 컴퓨터에서 풀어도 필요한 Python 모듈, 의존성 명세와 새 합성 테스트가 함께
들어 있다. 데이터와 사전학습 가중치는 코드 묶음에 포함되지 않는다.

프로젝트 최상위 폴더에서 Python 3.12를 사용하는 것을 권한다. 기존
`.venv-transformer`에는 이 구현을 테스트한 주요 의존성이 이미 설치되어 있다.

```sh
.venv-transformer/bin/python -m unittest \
  sota_pd_et.tests.test_task_interaction \
  sota_pd_et.tests.test_task_interaction_protocol -v
```

새 환경을 만들려면 다음과 같이 설치한다.

```sh
python3.12 -m venv .venv-task-interaction
.venv-task-interaction/bin/python -m pip install --upgrade pip
.venv-task-interaction/bin/python -m pip install -r sota_pd_et/requirements-task-interaction.txt
.venv-task-interaction/bin/python -m pip check
```

이후 예시의 `.venv-transformer/bin/python`을 새 환경의 Python 경로로 바꾸면 된다.
NVIDIA GPU를 사용할 때는 해당 환경에 적합한 CUDA PyTorch를 설치해야 한다.
`--device auto`는 CUDA, Apple MPS, CPU 순서로 사용 가능한 장치를 고른다.
명시한 CUDA/MPS가 없으면 조용히 다른 장치로 바꾸지 않고 오류를 낸다.

새 컴퓨터에 데이터가 없으면 포함된 다운로드 도구로 분류에 필요한 공식 파일만
받을 수 있다. 설치 환경의 Python으로 실행하며, 공개 네트워크 접속이 필요하다.

```sh
.venv-task-interaction/bin/python -m sota_pd_et.colab_download data/pads/1.0.0 --workers 4
```

전체 비교와 별도의 최종 모델 생성을 실행하려면 다음 명령을 사용한다.

```sh
.venv-transformer/bin/python -m sota_pd_et.evaluate_task_interaction \
  --data-root data/pads/1.0.0 \
  --output artifacts/pads_temporal_cross_task/run_01 \
  --device auto \
  --outer-folds 5 --inner-folds 3 --repeats 1 \
  --seed 20261017 \
  --fit-final
```

`--output`에는 **아직 존재하지 않는 폴더**를 지정한다. 이미 있는 실험 결과,
원본 데이터, 소스 코드, 기존 기준선 폴더는 덮어쓰지 않는다. 중단된 실행은
자동으로 이어서 실행하지 않는다. 완료한 fold의 파일은 남고, `manifest.json`에
실패 상태가 기록된다. 다시 실행할 때는 새 출력 폴더를 사용한다.

기본 설정은 신경망 5개와 기준선 2개를 모두 평가한다. 신경망 하나만 먼저
직접 실행하려면 다음과 같이 범위를 줄일 수 있다. 이 결과만으로 기준선 대비
향상을 주장할 수는 없다.

```sh
.venv-transformer/bin/python -m sota_pd_et.evaluate_task_interaction \
  --output artifacts/pads_temporal_cross_task/full_only_01 \
  --models full --device auto --fit-final
```

전체 비교는 많은 학습을 포함하며, 실제 소요 시간과 GPU 속도 향상은 아직 측정하지
않았다. 기본 설정에서 각 신경망 변형은 외부 fold마다 내부 학습 궤적 6개와
외부 재학습 1개를 만든다. 5개 변형·외부 5-fold는 총 175개 학습 궤적이며,
`--fit-final`은 전체 참가자에서 7개를 추가한다. HYDRA/MultiROCKET 변환은
별도로 내부/외부 학습마다 다시 적합한다.

## 2. 정확한 입력과 라벨

참가자 선택은 공식 `condition` 문자열을 사용한다.

- PD: `Parkinson's`, 이진 목표값 `0`, 276명.
- ET: `Essential Tremor`, 이진 목표값 `1`, 28명.
- 원래 숫자 라벨 `2`는 다른 감별진단도 포함하므로 ET 라벨로 사용하지 않는다.
- 공식 PD 진단이면서 설명에 ET가 언급된 사례도 공식 라벨을 유지한다. 임의로 재분류하지 않는다.
- 인구학적 변수, 설문, 진단 설명, 참가자 ID는 모델 입력으로 사용하지 않는다.

기존 `load_cohort`를 사용하여 입력 데이터와 정의 스크립트의 SHA-256, 라벨,
모든 선택된 신호의 유한성 등을 확인한다. 공식 전처리 입력은 참가자당
`(132, 976)`이며, 여기에서 다음 **5개 구간 × 12개 신체 센서 채널**을 선택한다.

| 모델 조건 | 공식 구간 | 구간당 물리 채널 | 사용 샘플 |
|---|---|---|---|
| 안정 시 | `Relaxed1`, `Relaxed2` | 양측 손목 × 가속도/각속도 × XYZ = 12 | 각 구간 976 |
| 인지 부하 안정 시 | `RelaxedTask1`, `RelaxedTask2` | 12 | 각 구간 976 |
| 자세 조건 | `StretchHold` | 12 | 976 |

인지 조건은 **안정 자세에서 연속 7 빼기(serial sevens)를 수행한 조건**이다.
‘인지 과제’와 ‘안정 자세’가 서로 배타적인 운동 상태라고 가정하지 않는다.
자세 조건에는 이번 명세에서 `StretchHold`만 사용하며, `LiftHold`나 무게 들기를
자동으로 추가하지 않는다.

공식 132채널 순서에서 선택하는 인덱스는 `0..59`이며, 정확한 정의는
`task_interaction.CHANNEL_INDICES`에 있다. 각 구간의 채널 순서는 왼쪽 손목
가속도 XYZ, 왼쪽 손목 각속도 XYZ, 오른쪽 손목 가속도 XYZ, 오른쪽 손목
각속도 XYZ다. 신경망 입력은 `(N, 5, 12, 976)`, 두 기준선 입력은 동일한 값을
재배열한 `(N, 60, 976)`이다. **세 방법 모두 같은 신호와 참가자를 사용한다.**

기존 실험의 `(N, 132, 976)`은 더 많은 과제를 포함한다. 기존 보고서의 BA
70.89%와 67.13%를 이번 60채널 비교의 기준선 수치로 그대로 옮기지 않는다.
이 평가기는 두 기준선을 같은 60채널 입력으로 다시 학습·평가한다.

## 3. 시간축 처리와 전처리의 범위

공식 전처리는 긴 기록을 두 구간으로 나눈 뒤 각 구간 앞 48샘플을 제거한다.
따라서 `Relaxed1`과 `Relaxed2`를 아무 표시 없이 하나의 연속 파형으로 연결하면
중간의 잘린 부분을 숨기게 된다.

이번 구현은 각 976샘플 구간을 **4개 × 244샘플**로 나눈다. 명목상 100 Hz에서
윈도 하나는 2.44초다. CNN은 각 윈도를 별도로 처리하므로 과제·구간 경계를
가로질러 합성곱하지 않는다. Transformer에는 구간의 원래 명목 위치를 사용한
위치 정보를 넣는다. 두 번째 반쪽은 원래 1,024샘플 오프셋을 가지며,
잘려 나간 48샘플을 시간 위치에서 고려한다.

이 처리는 **실제 원자료 타임스탬프 복원이나 결측 구간 보간을 수행하지 않는다.**
기존 공식 바이너리는 샘플 인덱스로 표현되며, 원자료에서 발견되는 비균일
샘플링·간격 문제는 그대로 남는다. 물리적 초/Hz에 관한 정밀 해석과 긴 결측
구간의 영향은 원자료 기반 별도 분석으로 검증해야 한다.

신경망의 표준화는 각 학습 fold의 참가자만 사용하여 물리 채널별 평균과
표준편차 12개를 구한다. 같은 변환을 모든 과제와 검증/평가 참가자에 적용한다.
**각 사람·각 과제 신호를 자기 표준편차로 나누지 않으므로 과제 간 진폭 차이가
제거되지 않는다.** CNN 특징과 별도로 윈도별 log-RMS도 보존한다. 이 RMS는
공유된 학습 표준화 이후의 신호에서 계산하며, 원래 물리 단위의 진폭 추정치는 아니다.

## 4. 신경망의 명시적 정의

참가자를 `i`, 과제 조건을 `q ∈ {rest, cognitive_rest, posture}`라 하자.
각 과제의 윈도 집합은 같은 참가자에게서만 온다.

1. **공유 윈도 CNN:** 모든 과제에 동일한 1D CNN을 적용한다. 입력 채널은 12개,
   세 합성곱의 커널 크기는 7·5·3, stride는 각각 2다. 기본 은닉 차원은 32다.
   각 윈도에서 합성곱 출력의 시간 평균·최댓값과 12개 log-RMS를 결합한다.
2. **과제 내 시간 인코더:** 과제별 윈도 특징에 명목 시간 위치를 더하고
   가중치를 공유하는 1-layer, 4-head Transformer로 처리한다. 안정 시와
   인지 조건은 각각 8개 윈도, 자세 조건은 4개 윈도다.
3. **과제 특징:** Transformer 출력의 윈도 간 평균·표준편차와 log-RMS의
   윈도 간 평균·표준편차를 결합한다. 기본 과제 특징 차원은
   `d = 2 × 32 + 2 × 12 = 88`이다.
4. **과제 간 결합:** 세 과제 특징의 선형 주효과에 낮은 랭크의 쌍선형 곱항을
   추가한다. 이를 ET 방향의 로짓으로 사용한다.

과제 특징을 $z_i^q\in\mathbb R^d$, 과제쌍 집합을
$\mathcal P=\{(rest,cognitive\_rest),(rest,posture),(cognitive\_rest,posture)\}$라 하자.
전체 모델의 분류 점수는 다음과 같다.

$$
s_i=b+\sum_q a_q^\top z_i^q+
\sum_{(q,r)\in\mathcal P}
\frac{(U_{qr}^\top z_i^q)^\top(V_{qr}^\top z_i^r)}{\sqrt{k}}.
$$

여기서 $b\in\mathbb R$는 편향, $a_q\in\mathbb R^d$는 과제별 가중치,
$U_{qr},V_{qr}\in\mathbb R^{d\times k}$는 서로 다른 과제를 결합하는 학습 행렬이다.
기본 랭크는 $k=4$다. `s_i > threshold`이면 ET, 그렇지 않으면 PD다.

곱항은 서로 다른 과제 특징이 함께 나타날 때의 효과를 표현한다.
반면 차이 $z^q-z^r$를 단순히 선형 분류기에 추가하는 것은 원래 두 특징의
선형결합으로 흡수할 수 있으므로 그 자체로 새 상호작용 표현력을 주지 않는다.

이 인코더가 주파수 변화·간헐성 등 임상적으로 의도한 현상을 실제로 사용한다는
사실은 구조만으로 보장되지 않는다. 분류 라벨로 학습하는 잠재 특징이며,
시간순서 대조군과 추후 해석 분석으로 확인해야 한다. 서로 다른 과제의 동일
샘플 번호가 동일한 생리 사건을 뜻한다고 가정하지 않고, 과제별 요약 특징을 결합한다.

## 5. 비교하는 7개 모델

| 이름 | 윈도 간 순서 인코더 | 과제 결합 | 검증 목적 |
|---|---|---|---|
| `full` | 사용 | 명시적 저랭크 쌍선형 곱항 + 과제별 주효과 | 제안 모델 |
| `no_temporal` | 제거 | 동일한 쌍선형 결합 | 과제 간 결합이 있을 때 순서 인코더 기여 |
| `no_interaction` | 사용 | 과제별 로짓의 합 | 시간 인코더가 있을 때 명시적 결합 기여 |
| `neither` | 제거 | 과제별 로짓의 합 | 두 모듈을 모두 제거한 대조군 |
| `concat_mlp` | 사용 | 특징 연결 + MLP | 일반적인 비선형 결합과 비교 |
| `mantis` | 고정 사전학습 MantisV2 | 채널별 특징 연결 + 가중 ridge | 동일 신호의 사전학습 기준선 |
| `multirocket_hydra` | 원래 MultiROCKET/HYDRA 변환 | 변환 특징 + 가중 ridge | 동일 신호의 강한 기존 기준선 |

`no_temporal`과 `neither`는 윈도 간 순서 없는 평균·표준편차를 사용한다.
윈도 내부의 CNN 시간 패턴은 남는다. 그러므로 이 제거 실험은 **윈도 간 순서
모델링의 기여**를 검증하며, 모든 시간 정보를 없애는 실험은 아니다.

`no_interaction`의 과제별 점수는 다른 과제를 보지 않은 특징의 합이다.
`concat_mlp`는 암묵적 비선형 상호작용을 학습할 수 있으므로 ‘상호작용 없음’으로
부르지 않는다. 변형별 파라미터 수는 `manifest.json`에 저장한다. 파라미터 수가
완전히 일치하는 실험은 아니므로 성능 차이를 모듈만의 인과 효과로 해석하지 않는다.

MantisV2는 기존 구현과 같은 고정 revision, layer index 2, combined token,
채널당 512차원 특징을 사용한다. 각 길이 976 채널을 512로 선형 보간한다.
사전학습 인코더 자체는 PADS에서 적합하지 않아 고정 특징을 fold 사이에
공유할 수 있지만, 특징 StandardScaler와 ridge는 fold마다 새로 적합한다.
보간은 원자료 타임스탬프 문제를 해결하지 않는다.

## 6. 중첩 검증과 선택 규칙

외부 5-fold·내부 3-fold는 모두 참가자 단위의 계층화 분할이다.
한 참가자의 양손, 과제, 모든 윈도는 반드시 같은 fold에 머문다.
기본 seed `20261017`은 기존 평가기와 같은 분할 생성 함수를 사용한다.
기존 기본 분할 파일이 로컬에 있으면 정확한 일치를 자동 확인한다.
`--reference-splits path/to/splits.json`을 지정하면 해당 파일과 다를 때 중단한다.

각 외부 학습 집합에서 다음을 수행한다.

1. 내부 학습 참가자만으로 표준화 통계와 학습 객체를 만든다.
2. 신경망은 weight decay `0.001`, `0.01`을 비교한다. 각 내부 학습 궤적은
   최대 60 epoch까지 진행하며, epoch `15`, `30`, `60`의 검증 예측을 저장한다.
   학습률 `0.001`, 배치 크기 16, dropout 0.2, gradient clipping 1.0이 기본값이다.
3. 각 `(weight_decay, epoch, threshold)` 조합의 **내부 fold BA 산술평균**으로
   선택한다. 동점은 더 강한 weight decay, 더 적은 epoch 순으로 선택한다.
4. 두 기준선은 동일한 내부 분할에서 ridge alpha
   `{0.001, 0.01, 0.1, 1, 10, 100, 1000}`을 비교한다. HYDRA/MultiROCKET과
   두 변환의 scaler도 내부 학습 참가자만으로 새로 적합한다. 동점은 더 큰 alpha다.
5. 선택한 설정으로 외부 학습 참가자 전체에 **새 모델**을 적합한다.
   신경망의 epoch 수는 내부에서 이미 고정되어 있으며 외부 평가 손실을 보지 않는다.
6. 외부 평가 참가자에게 한 번 예측하고 지표를 저장한다.

모든 학습의 클래스 가중치는 해당 학습 참가자 수로 다시 계산한다.
클래스 $c\in\{0,1\}$의 학습 인원을 $n_c$, 총 학습 인원을 $n$이라 하면
$w_c=n/(2n_c)$이다. 신경망은 이 가중치를 적용한 이진 교차엔트로피와 AdamW를
사용한다. 예측 시 sigmoid를 적용해 얻을 수 있는 수를 보정된 임상 확률로 취급하지 않는다.

기본 임계값은 모든 방법에서 **0으로 고정**한다. 외부 결과를 보고 임계값을
바꾸지 않는다. 임계값 선택도 비교하고 싶으면 실행 전에 모든 방법에 공통인
후보를 명시한다.

```sh
.venv-transformer/bin/python -m sota_pd_et.evaluate_task_interaction \
  --output artifacts/pads_temporal_cross_task/inner_threshold_01 \
  --thresholds -1 -0.5 0 0.5 1 --device auto --fit-final
```

이 경우 임계값도 내부 fold 평균 BA로 선택한다. 동점은 0에 더 가까운 값,
그다음 더 큰 값을 선택한다. 각 방법의 원점수 척도는 다를 수 있다.
기본 실험과 임계값 탐색 실험은 별개의 사전 명세로 해석해야 한다.

`--fit-final`은 평가와 별도로 전체 304명에서 다시 내부 검증 후 `full` 모델을
적합한다. 이 최종 모델의 학습 자료 성능을 독립 평가로 보고하지 않는다.

## 7. 저장 결과와 읽는 방법

| 파일/폴더 | 의미 |
|---|---|
| `manifest.json` | 상태, 설정, 버전, 파라미터 수, 데이터·소스 해시, 한계 |
| `splits.json`, `cohort.csv` | 모든 내부·외부 참가자 분할과 코호트 |
| `evaluated_sources.zip` | 실제 실행 시점의 Python 소스 |
| `summary.json`, `REPORT.md` | 일곱 방법의 외부 평가 요약 |
| `<model>/oof_predictions.csv` | 참가자별 원점수, 임계값, 예측, fold/repeat |
| `<model>/fold_metrics.csv` | 외부 fold별 BA·AUROC·AP·ET/PD 지표 |
| `<model>/repeat_XX_fold_XX/inner_search.csv` | 내부 선택 결과; 독립 평가 성능이 아님 |
| `<model>/repeat_XX_fold_XX/selected.json` | 해당 외부 fold에서 선택한 설정 |
| `<model>/repeat_XX_fold_XX/test_predictions.csv` | 해당 외부 평가 참가자의 예측 |
| `paired_comparisons.json` | `full`과 각 대조군의 참가자 대응 BA 차이와 조건부 구간 |
| `final_model.pt` | `--fit-final`로 생성한 전체 코호트 학습 모델 |
| `final_inner_splits.json`, `final_selected.json` | 최종 학습에 사용한 별도 내부 분할·설정 |

ET를 양성으로 정한다. 혼동행렬은 **행=실제, 열=예측, 순서=[PD, ET]**다.
ET 민감도는 ET 정분류율, PD 특이도는 PD 정분류율이다.

$$
\mathrm{Sensitivity}_{ET}=\frac{TP}{TP+FN},\qquad
\mathrm{Specificity}_{PD}=\frac{TN}{TN+FP},\qquad
\mathrm{BA}=\frac{\mathrm{Sensitivity}_{ET}+\mathrm{Specificity}_{PD}}2.
$$

BA가 주 지표다. AUROC, ET average precision(AP), ET precision/F1,
macro-F1, MCC, 일반 정확도, 혼동행렬과 ET 정분류 인원도 저장한다.
AUROC/AP는 임계값을 뺀 점수가 아니라 원점수로 계산한다. 여러 fold의 원점수
척도가 다를 수 있으므로 pooled AUROC/AP와 fold별 값의 평균을 함께 확인한다.
AP는 average precision이며, 다른 방식의 PR 곡선 사다리꼴 면적과 구별한다.

쌍대 비교는 진단별 참가자를 층화하여 같은 참가자를 두 방법에서 함께
재표집한다. 반복 CV를 수행해도 한 참가자의 반복 예측을 하나의 묶음으로
취급하며 독립 표본 수를 늘리지 않는다. 저장된 적합 모델을 고정한 percentile
95% bootstrap 구간이므로 **새 데이터로 재학습할 때의 불확실성을 포함하지 않는다.**
이 구간이나 fold 표준편차만으로 통계적 우월성·임상 유효성을 주장하지 않는다.

ET가 28명이므로 한 명의 정오분류 변화는 ET 민감도 약 3.57%p, PD 판정이
같다면 BA 약 1.79%p에 해당한다. 외부 fold 하나에는 ET가 5~6명뿐이다.

## 8. 학습된 모델로 추론하기

새 참가자 신호에 **동일한 공식 PADS 전처리와 채널 순서**를 적용한 `.npy`가
필요하다. 원자료 CSV를 그대로 넣는 명령은 아니다. 입력 형태는
`(N,132,976)`, 선택 채널 순서의 `(N,60,976)`, 또는 `(N,5,12,976)`을 허용한다.

```sh
.venv-transformer/bin/python -m sota_pd_et.predict_task_interaction \
  --model artifacts/pads_temporal_cross_task/run_01/final_model.pt \
  --inputs new_participants.npy \
  --output new_predictions.csv --device auto
```

저장 형식은 tensor와 기본 자료형으로 구성하며 `torch.load(weights_only=True)`로
읽는다. 모델 설정, 공유 표준화 통계, 임계값을 함께 복원한다.
버전 정보는 모델에 저장되어 있으므로 재현할 때 실행 당시 환경을 복원한다.

## 9. 설계 근거와 독창성의 범위

주파수의 시간적 변동을 PD/ET 감별에 이용하는 발상 자체는 새롭지 않다.
[di Biase 등의 Tremor Stability Index 연구(2017)](https://discovery.ucl.ac.uk/id/eprint/1556738/)가
그 예다. 해당 연구의 기록 길이와 이번 짧은 PADS 구간은 다르므로 보고된
성능이나 cut-off를 그대로 적용하지 않는다.

[MFCA-Net 연구(2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564626/)는
여러 자세의 가속도·sEMG 신호와 multiscale CNN·cross-attention을 사용했다.
또한 [PADS 양측 손목 attention 사전공개 논문의 v2(2026)](https://arxiv.org/html/2604.18372v2)는
양측 결합과 새로운 운동 과제로의 일반화를 분석한다. 해당 논문은 PD/건강인 및
PD/여러 감별진단의 문제를 다루므로 현재 PD/ET만의 실험과 성능을 직접 비교하지 않는다.

이 구현의 검증 대상은 **같은 참가자의 명시적으로 구분된 세 과제 특징을,
과제 내 순서 표현과 과제 간 저랭크 곱항으로 결합하는 특정 구조가 동일 입력의
대조군보다 나은가**이다. 저랭크 쌍선형 결합 자체의 최초 발명이나 기존 모델이
시간적 특징을 배우지 못한다는 주장을 하지 않는다. 동일 코호트가 과거에 분석된
상태에서 설계한 후속 실험이므로, 이번 nested CV도 탐색적 근거다.

이 데이터는 수초~수십초의 과제 기록이다. 여기서 말하는 시간적 변화는
기록 내 변화이며, 수개월의 질병 진행이나 치료 효과의 인과 추정을 의미하지 않는다.
결과가 좋아도 외부 코호트·전향적 평가 전에는 임상 성능 향상을 확정하지 않는다.

데이터 출처와 사용 조건:
[PADS v1.0.0 / PhysioNet](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/),
[DOI 10.13026/m0w9-zx22](https://doi.org/10.13026/m0w9-zx22), CC BY-NC-SA 4.0.
배포·발표 시 공식 페이지의 데이터 및 원 연구 인용 지침을 따른다.

사전학습 기준선의 원 연구:
[MantisV2(2026)](https://arxiv.org/abs/2602.17868),
[공식 저장소](https://github.com/vfeofanov/mantis).
