# Best Model Methodology

이 문서는 현재 최종 best 모델의 학습 설정과 open-set 방법론을 한 번에 정리한 요약본이다.

## 1. Goal

- `train`에서는 known 3개 클래스만 사용:
  - `DIE_BROKEN`
  - `NORMAL`
  - `NO_DIE`
- `val/test`에서는 unknown 2개 클래스 포함:
  - `DIE_CRACK`
  - `DIE_INK`
- 목표:
  - known 3-class 분류 성능을 유지하면서
  - unknown 샘플을 `UNKNOWN`으로 reject하는 open-set 성능을 최대화

## 2. Final Best Setting

기준 checkpoint:

- [resnet18_best_val_loss_epoch29.pth](best_run/resnet18_best_val_loss_epoch29.pth)

관련 결과 파일:

- [open_set_summary.json](best_run/open_set_summary.json)
- [layer2 patch results json](best_run/resnet18_best_val_loss_epoch29_layer2_patch_topk12_global_norm_test_results.json)
- [ablation study](Ablation%20study/ablation_study.md)

최종 설정:

- backbone: `ResNet18`
- batch size: `8`
- epochs: `30`
- label smoothing: `0.05`
- model selection: `val_loss`
- selected checkpoint epoch: `29`

augmentation:

- rotation: `5`
- contrast strength: `0.08`
- sharpness factor: `1.10`
- sharpness probability: `0.20`
- autocontrast probability: `0.17`

## 3. Training Pipeline

기본 분류기는 일반적인 3-class closed-set classifier로 학습했다.

1. `train` split에서 known 3개 클래스만 사용한다.
2. `ResNet18` 마지막 `fc`를 3-class output으로 바꿔 fine-tuning한다.
3. loss는 `CrossEntropyLoss + label_smoothing=0.05`를 사용한다.
4. validation에서는 `val_loss`, `val_f1`를 추적하고 checkpoint를 저장한다.
5. 최종 분류기 자체는 `softmax argmax`로 known class를 예측한다.

중요한 점:

- 학습 자체는 `UNKNOWN` 클래스를 따로 학습하지 않는다.
- unknown reject는 학습 후 별도의 open-set scoring 단계에서 수행한다.

## 4. Open-Set Method

최종적으로 가장 잘 된 방법은:

- `layer2 patch-based feature distance [global threshold]`

구체적으로는 아래 순서다.

1. 이미지가 들어오면 먼저 분류기 `softmax argmax`로 known class를 예측한다.
2. `ResNet18 layer2` feature map을 추출한다.
3. `train known` 샘플들로 각 known class의 `layer2 prototype map`을 만든다.
4. 현재 이미지의 `layer2` feature map과, 예측된 class의 prototype map을 같은 위치끼리 비교한다.
5. 각 spatial location에서 cosine distance를 계산해 patch-level distance map을 만든다.
6. distance map에서 `topk=12`개의 큰 값 평균을 anomaly score로 사용한다.
7. class마다 score scale이 다르기 때문에, `train known`만으로 예측 class별 `mean/std`를 계산해 z-score normalization을 한다.
8. `val`에서 threshold를 선택한다.
9. `test`에서는 이 threshold를 고정 적용한다.
10. normalized score가 threshold보다 크면 `UNKNOWN`, 아니면 argmax class를 유지한다.

한 줄로 요약하면:

- `softmax argmax`로 known class를 예측하고,
- `layer2 patch mismatch score`가 크면 `UNKNOWN`으로 reject하는 구조다.

## 5. Why Layer2

실험 결과 `layer2`가 `layer3`, `layer4`보다 확실히 좋았다.

이유는 다음과 같다.

- `layer2`는 spatial detail이 더 살아 있어서 국소 patch mismatch를 잘 잡는다.
- `layer3`, `layer4`로 갈수록 feature가 추상적이 되어 unknown과 known의 local difference가 덜 드러난다.
- 실제 ablation에서도 `layer2`가 `open_macro_f1`, `overall_acc`, `unknown_acc` 모두 가장 높았다.

## 6. Threshold / Normalization Rule

최종 실험에서 확인한 규칙은 다음과 같다.

- z-score mean/std는 `train known`만으로 계산
- threshold는 `val`에서만 선택
- `test`는 threshold를 다시 찾지 않고 고정 평가만 수행

즉 데이터 분리는:

- `train`: classifier 학습 + z-score 통계
- `val`: threshold tuning
- `test`: final evaluation

이다.

## 7. Final Performance

최종 best 모델의 test 성능:

- `open_macro_f1 = 0.8447`
- `overall_acc = 0.8667`
- `known_acc = 0.8000`
- `unknown_acc = 0.9250`

추가 수치:

- selected threshold: `0.09808522504334695`
- raw known classifier accuracy before reject: `0.9714`

recall 관점에서 보면:

- unknown recall: `0.9250`
  - unknown 40장 중 37장을 `UNKNOWN`으로 correctly reject
- known class recall:
  - `DIE_BROKEN`: `0.6250` (`5 / 8`)
  - `NORMAL`: `0.8500` (`17 / 20`)
  - `NO_DIE`: `0.8571` (`6 / 7`)
- unknown subclass recall:
  - `DIE_CRACK`: `0.9000` (`18 / 20`)
  - `DIE_INK`: `0.9500` (`19 / 20`)

해석:

- raw classifier만 보면 known 분류는 매우 좋다.
- 하지만 open-set에서는 unknown을 known으로 오분류할 수 있다.
- 최종 방법론은 known 일부를 희생하는 대신 unknown reject를 크게 개선했다.

## 8. Interpretation of Visualizations

관련 시각화는 [Ablation study](Ablation%20study/ablation_study.md)에 정리했다.

시각화는 두 종류로 해석하면 된다.

- `Grad-CAM`
  - 분류기가 왜 그 known class로 예측했는지
  - 즉 classifier decision evidence
- `patch-distance heatmap`
  - 예측된 class prototype과 어디가 많이 어긋나는지
  - 즉 open-set reject evidence

따라서:

- `Grad-CAM`은 `why known class?`
- `patch-distance`는 `why reject as unknown?`

를 보여준다.

## 9. Main Conclusion

현재 best 방법론은 다음 조합으로 정리할 수 있다.

- `ResNet18 3-class classifier`
- `mild augmentation + weak autocontrast`
- `label smoothing 0.05`
- `layer2 patch-based prototype distance`
- `topk=12`
- `predicted-class z-score normalization from train known`
- `global threshold selected on val`

실험 전체를 통틀어 보면:

- `baseline`보다 open-set 성능이 크게 향상되었고
- `softmax threshold`, `global feature distance`, `class-wise threshold`보다도 더 좋았으며
- 최종적으로 가장 안정적인 조합은 `layer2 patch-based global threshold`였다.
