# Open-set Comparison: `label_smoothing=0.1`

대상 checkpoint:
- `resnet18_best_val_loss_epoch30.pth`

비교한 결과 파일:
- [plain](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_test_results.json)
- [class-wise threshold](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_classwise_threshold_test_results.json)
- [class-wise threshold independent](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_classwise_threshold_independent_test_results.json)
- [feature-distance threshold](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_feature_distance_test_results.json)
- [feature-distance threshold class-wise independent](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_feature_distance_classwise_test_results.json)
- [layer2 patch summary](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_layer2_patch_summary.json)
- [layer3 patch summary](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_layer3_patch_summary.json)
- [layer4 patch summary](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_layer4_patch_summary.json)
- [layer2 patch class-wise summary](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_layer2_patch_classwise_summary.json)
- [layer2 patch predicted-class normalization summary](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_layer2_patch_global_norm_summary.json)
- [layer2 patch z-score train-known summary](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_layer2_patch_global_norm_train_known_summary.json)
- [layer2 patch z-score class-wise summary](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_layer2_patch_classwise_norm_train_known_summary.json)
- [global + layer2 ensemble summary](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_ensemble_global_patch_topk12_summary.json)

기준:
- test 전체: 75장
- known: 35장 (`DIE_BROKEN`, `NORMAL`, `NO_DIE`)
- unknown: 40장 (`DIE_CRACK`, `DIE_INK`)

## Summary Table

| Method | Threshold | Overall Acc | Known Acc | Unknown Acc | Known Rejected | Unknown Rejected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Plain 3-class | 없음 | 0.4533 | 0.9714 | 0.0000 | 0 / 35 | 0 / 40 |
| Class-wise threshold | `DIE_BROKEN=0.9482`, `NORMAL=0.8645`, `NO_DIE=0.3000` | 0.5467 | 0.6000 | 0.5000 | 13 / 35 | 20 / 40 |
| Class-wise threshold independent | `DIE_BROKEN=0.9342`, `NORMAL=0.3000`, `NO_DIE=0.3000` | 0.5600 | 0.9429 | 0.2250 | 2 / 35 | 9 / 40 |
| Feature-distance threshold | global distance threshold `0.0364` | 0.8000 | 0.7714 | 0.8250 | 7 / 35 | 33 / 40 |
| Feature-distance threshold class-wise independent | `DIE_BROKEN=0.0364`, `NORMAL=0.0401`, `NO_DIE=0.0194` | 0.7867 | 0.7429 | 0.8250 | 8 / 35 | 33 / 40 |

## Per-class Prediction Patterns

### 1. Plain 3-class

- `DIE_BROKEN`: `7 -> DIE_BROKEN`, `1 -> NORMAL`
- `DIE_CRACK`: `20 -> NORMAL`
- `DIE_INK`: `16 -> DIE_BROKEN`, `4 -> NORMAL`
- `NORMAL`: `20 -> NORMAL`
- `NO_DIE`: `7 -> NO_DIE`

해석:
- known 분류는 매우 좋다.
- unknown을 전혀 `UNKNOWN`으로 거르지 못한다.
- open-set 기준 전체 accuracy는 낮다.

### 2. Class-wise threshold

threshold:
- `DIE_BROKEN`: `0.9481818181818182`
- `NORMAL`: `0.8645454545454545`
- `NO_DIE`: `0.3`

test 예측:
- `DIE_BROKEN`: `2 -> DIE_BROKEN`, `1 -> NORMAL`, `5 -> UNKNOWN`
- `DIE_CRACK`: `17 -> NORMAL`, `3 -> UNKNOWN`
- `DIE_INK`: `2 -> DIE_BROKEN`, `1 -> NORMAL`, `17 -> UNKNOWN`
- `NORMAL`: `12 -> NORMAL`, `8 -> UNKNOWN`
- `NO_DIE`: `7 -> NO_DIE`

해석:
- unknown 일부를 거르기는 한다.
- 하지만 known도 너무 많이 `UNKNOWN`으로 보낸다.
- 특히 `DIE_BROKEN`와 `NORMAL` 손실이 크다.
- `NO_DIE` threshold가 `0.3`으로 낮아 사실상 거의 reject하지 않는다.

### 3. Feature-distance threshold

threshold:
- predicted-class prototype cosine distance threshold: `0.036403640364036406`

test 예측:
- `DIE_BROKEN`: `2 -> DIE_BROKEN`, `1 -> NORMAL`, `5 -> UNKNOWN`
- `DIE_CRACK`: `4 -> NORMAL`, `16 -> UNKNOWN`
- `DIE_INK`: `1 -> DIE_BROKEN`, `2 -> NORMAL`, `17 -> UNKNOWN`
- `NORMAL`: `18 -> NORMAL`, `2 -> UNKNOWN`
- `NO_DIE`: `7 -> NO_DIE`

추가 거리 통계:
- 전체 mean predicted-class distance: `0.0407`
- `DIE_BROKEN`: `0.0418`
- `DIE_CRACK`: `0.0489`
- `DIE_INK`: `0.0568`
- `NORMAL`: `0.0260`
- `NO_DIE`: `0.0120`

해석:
- unknown을 가장 잘 거른다.
- known 손실도 있지만 class-wise threshold보다 덜하다.
- 현재 세 방법 중 open-set 목적에 가장 적합하다.

### 4. Class-wise threshold independent

threshold:
- `DIE_BROKEN`: `0.9342424242424243`
- `NORMAL`: `0.3`
- `NO_DIE`: `0.3`

test 예측:
- `DIE_BROKEN`: `6 -> DIE_BROKEN`, `1 -> NORMAL`, `1 -> UNKNOWN`
- `DIE_CRACK`: `20 -> NORMAL`
- `DIE_INK`: `7 -> DIE_BROKEN`, `4 -> NORMAL`, `9 -> UNKNOWN`
- `NORMAL`: `20 -> NORMAL`
- `NO_DIE`: `7 -> NO_DIE`

해석:
- known 보존은 매우 좋다.
- 특히 joint class-wise threshold보다 known accuracy가 크게 좋아진다.
- 하지만 unknown reject는 약하다.
- `NORMAL`, `NO_DIE` threshold가 `0.3`까지 내려가서 사실상 reject를 거의 하지 않는다.
- 결과적으로 `DIE_CRACK`는 하나도 `UNKNOWN`으로 못 보낸다.

### 5. Feature-distance threshold class-wise independent

threshold:
- `DIE_BROKEN`: `0.036403640364036406`
- `NORMAL`: `0.0401040104010401`
- `NO_DIE`: `0.0194019401940194`

test 예측:
- `DIE_BROKEN`: `2 -> DIE_BROKEN`, `1 -> NORMAL`, `5 -> UNKNOWN`
- `DIE_CRACK`: `4 -> NORMAL`, `16 -> UNKNOWN`
- `DIE_INK`: `1 -> DIE_BROKEN`, `2 -> NORMAL`, `17 -> UNKNOWN`
- `NORMAL`: `18 -> NORMAL`, `2 -> UNKNOWN`
- `NO_DIE`: `6 -> NO_DIE`, `1 -> UNKNOWN`

해석:
- unknown reject 수는 global feature-distance와 동일하다.
- 하지만 `NO_DIE` threshold가 더 작게 잡히면서 known 1장을 추가로 잃었다.
- overall, known accuracy 모두 global feature-distance보다 소폭 낮다.
- 현재 데이터에서는 class-wise distance threshold의 추가 이득이 없었다.

## Confidence Comparison

세 JSON 모두 softmax 예측 자체는 같은 checkpoint에서 나온 것이므로, `confidence` 분포는 동일하다.
차이는 threshold / distance 규칙이 `UNKNOWN`으로 reject하느냐 여부다.

mean confidence:

| True Class | Mean Confidence |
| --- | ---: |
| DIE_BROKEN | 0.9359 |
| DIE_CRACK | 0.8850 |
| DIE_INK | 0.9019 |
| NORMAL | 0.8570 |
| NO_DIE | 0.9532 |

이 값만 보면 unknown confidence도 높아서, softmax confidence만으로는 known/unknown 분리가 어렵다.

## Patch-Level Distance

설정:
- softmax argmax class는 유지
- predicted class의 prototype feature map과 same-location cosine distance map 계산
- top-k patch distance 평균을 anomaly score로 사용
- val에서 threshold 선택 후 test에 고정 적용

### Patch Summary Table

| Feature Layer | topk | Threshold | Open Macro F1 | Overall Acc | Known Acc | Unknown Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer2` | 3 | `0.428445` | `0.7706` | `0.7333` | `0.8571` | `0.6250` |
| `layer2` | 5 | `0.376607` | `0.7874` | `0.7467` | `0.8571` | `0.6500` |
| `layer2` | 7 | `0.354317` | `0.7874` | `0.7467` | `0.8571` | `0.6500` |
| `layer2` | 12 | `0.289030` | `0.7876` | `0.7733` | `0.8000` | `0.7500` |
| `layer3` | 3 | `0.176670` | `0.6676` | `0.6800` | `0.7143` | `0.6500` |
| `layer3` | 5 | `0.172692` | `0.6687` | `0.6800` | `0.7429` | `0.6250` |
| `layer3` | 7 | `0.170708` | `0.6687` | `0.6800` | `0.7429` | `0.6250` |
| `layer3` | 12 | `0.212396` | `0.6844` | `0.6267` | `0.9143` | `0.3750` |
| `layer4` | 3 | `0.159403` | `0.6351` | `0.6000` | `0.7143` | `0.5000` |
| `layer4` | 5 | `0.132994` | `0.6987` | `0.6933` | `0.6286` | `0.7500` |
| `layer4` | 7 | `0.127241` | `0.7175` | `0.7200` | `0.6571` | `0.7750` |
| `layer4` | 12 | `0.119612` | `0.7200` | `0.7200` | `0.6857` | `0.7500` |

### Patch-Level Interpretation

- `layer2`가 patch-level 실험 중 가장 좋았다.
- 최고 성능은 `layer2, topk=12`였고, `open macro F1=0.7876`, `overall acc=0.7733`였다.
- `layer3`는 세 레이어 중 가장 약했다.
- `layer4`는 `layer3`보다는 낫지만 `layer2`보다는 일관되게 낮았다.
- `topk`를 너무 작게 잡으면 일부 극단 patch만 보게 되고, `topk=12`처럼 조금 넓게 평균할 때 더 안정적이었다.

### Patch-Level vs Feature-Distance Global

- 기존 global feature-distance 결과:
  - `open macro F1=0.7472`
  - `overall acc=0.8000`
  - `known acc=0.7714`
  - `unknown acc=0.8250`
- patch-level best (`layer2, topk=12`) 결과:
  - `open macro F1=0.7876`
  - `overall acc=0.7733`
  - `known acc=0.8000`
  - `unknown acc=0.7500`

해석:
- patch-level은 `open macro F1`와 `known acc`가 더 좋다.
- global feature-distance는 `overall acc`와 `unknown acc`가 더 좋다.
- 즉 patch-level은 known/unknown 균형이 더 좋고, global feature-distance는 unknown reject를 더 공격적으로 하는 쪽이다.

### Layer2 Patch With Class-Wise Threshold

설정:
- feature layer: `layer2`
- patch anomaly score: top-k patch distance 평균
- threshold: predicted class별로 독립 탐색

결과:

| Method | topk | Thresholds | Open Macro F1 | Overall Acc | Known Acc | Unknown Acc |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `layer2 patch + class-wise` | 7 | `BROKEN=0.4042`, `NORMAL=0.2647`, `NO_DIE=0.1889` | `0.7768` | `0.7867` | `0.8000` | `0.7750` |
| `layer2 patch + class-wise` | 12 | `BROKEN=0.3705`, `NORMAL=0.2425`, `NO_DIE=0.1737` | `0.7605` | `0.7733` | `0.8000` | `0.7500` |
| `layer2 patch + class-wise` | 15 | `BROKEN=0.3598`, `NORMAL=0.2325`, `NO_DIE=0.1676` | `0.7605` | `0.7733` | `0.8000` | `0.7500` |

비교:
- `layer2 patch + global topk=12`: `open macro F1=0.7876`, `overall=0.7733`, `known=0.8000`, `unknown=0.7500`
- `layer2 patch + class-wise topk=7`: `open macro F1=0.7768`, `overall=0.7867`, `known=0.8000`, `unknown=0.7750`

해석:
- class-wise로 바꾸면 `topk=7`에서 `overall`과 `unknown`은 조금 좋아졌다.
- 하지만 최고 `open macro F1`는 여전히 global threshold(`topk=12`)가 더 높다.
- `topk=12, 15`의 class-wise 결과는 global과 거의 같거나 약간 낮다.
- 즉 `layer2 patch`에서는 class-wise threshold의 이득이 크지 않다.

### Layer2 Patch With Predicted-Class Score Normalization

설정:
- feature layer: `layer2`
- patch anomaly score: top `1%`, `2%`, `3%`, `5%` patch distance 평균
- predicted class별 `val known score`의 mean/std로 z-score normalization
- normalized score에 global threshold 적용

결과:

| Method | Patch Pooling | Threshold | Open Macro F1 | Overall Acc | Known Acc |
| --- | --- | ---: | ---: | ---: | ---: |
| `layer2 patch + pred-class norm` | `top 1%` | `1.276902` | `0.7284` | `0.6933` | `0.8571` |
| `layer2 patch + pred-class norm` | `top 2%` | `1.505152` | `0.6980` | `0.6533` | `0.8857` |
| `layer2 patch + pred-class norm` | `top 3%` | `1.600474` | `0.6463` | `0.5867` | `0.9429` |
| `layer2 patch + pred-class norm` | `top 5%` | `1.412734` | `0.5849` | `0.5200` | `0.8857` |

비교:
- 이 설정의 최고 결과는 `top 1%`였다.
- 하지만 `layer2 patch + global topk12`의 `open macro F1=0.7876`보다 낮다.
- `layer2 patch + class-wise topk7`의 `open macro F1=0.7768`보다도 낮다.

해석:
- predicted-class normalization은 현재 데이터에선 개선을 주지 못했다.
- percent pooling으로 갈수록 known 보존은 일부 좋아질 수 있지만, unknown 분리력이 빠르게 떨어졌다.
- 현재 결과만 보면 `top-k raw score`가 `top-percent normalized score`보다 더 낫다.

### Layer2 Patch Raw vs Z-Score

설정:
- feature layer: `layer2`
- fixed `topk`
- z-score normalization 통계는 `train known`만 사용
- val에서 threshold 선택 후 test에 고정 적용

결과:

| Method | Open Macro F1 | Overall Acc | Known Acc | Unknown Acc |
| --- | ---: | ---: | ---: | ---: |
| `layer2 topk=5 raw` | `0.7874` | `0.7467` | `0.8571` | `0.6500` |
| `layer2 topk=5 z-score (train known)` | `0.8114` | `0.8133` | `0.8000` | `0.8250` |
| `layer2 topk=7 raw` | `0.7874` | `0.7467` | `0.8571` | `0.6500` |
| `layer2 topk=7 z-score (train known)` | `0.8114` | `0.8133` | `0.8000` | `0.8250` |
| `layer2 topk=12 raw` | `0.7876` | `0.7733` | `0.8000` | `0.7500` |
| `layer2 topk=12 z-score (train known)` | `0.8301` | `0.8400` | `0.8000` | `0.8750` |

해석:
- `train known` 기준 z-score normalization은 raw보다 확실히 좋아졌다.
- 특히 `topk=12`에서 가장 좋았고,
  - `open macro F1: 0.7876 -> 0.8301`
  - `overall acc: 0.7733 -> 0.8400`
  - `unknown acc: 0.7500 -> 0.8750`
- `known acc`는 `topk=12`에서 동일했고, `topk=5,7`에서는 다소 내려갔지만 전체적으로 open-set 지표는 크게 개선됐다.
- 즉 `z-score normalization 자체의 효과`는 있었고, 이전 실패는 `top-percent pooling` 조합의 문제에 더 가깝다.
- `topk` 근방 추가 탐색(`8~16`)에서도 최고 `open macro F1`는 `0.8301`이었고, `topk=11~14`가 동률이었다.
- 따라서 대표 설정은 해석이 가장 간단한 `topk=12 + z-score(train known)`로 유지했다.

### Threshold Selection Rule Comparison

기준:
- best method: `layer2 patch topk=12 + z-score(train known)`
- threshold는 항상 `val`에서만 선택
- `test`는 선택된 threshold 고정 평가

| Selection Rule | Val Threshold | Val Open Macro F1 | Val Overall Acc | Val Known Acc | Test Open Macro F1 | Test Overall Acc | Test Known Acc | Test Unknown Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `open_macro_f1` 최대 | `0.250573` | `0.8677` | `0.8667` | `0.8000` | `0.8301` | `0.8400` | `0.8000` | `0.8750` |
| `known_acc >= 0.80` 조건 후 `open_macro_f1` 최대 | `0.250573` | `0.8677` | `0.8667` | `0.8000` | `0.8301` | `0.8400` | `0.8000` | `0.8750` |
| `known_acc >= 0.85` 조건 후 `open_macro_f1` 최대 | `0.723930` | `0.8570` | `0.8267` | `0.9143` | `0.7571` | `0.7467` | `0.8000` | `0.7000` |
| `overall_acc` 최대 | `0.250573` | `0.8677` | `0.8667` | `0.8000` | `0.8301` | `0.8400` | `0.8000` | `0.8750` |

해석:
- 현재 실험에서는 `open_macro_f1 최대`, `known_acc >= 0.80`, `overall_acc 최대`가 모두 같은 threshold를 선택했다.
- `known_acc >= 0.85` 제약을 강하게 걸면 val에선 known을 더 보존하지만, test에선 unknown reject가 줄어들어 전체 성능이 내려갔다.
- 따라서 현재 데이터에선 threshold rule을 바꾸기보다, `open_macro_f1 최대` 규칙을 유지하는 게 가장 낫다.

### Layer2 Patch Z-Score With Class-Wise Threshold

설정:
- feature layer: `layer2`
- `topk=12`
- score: predicted-class z-score (`train known` 기준 mean/std)
- threshold: predicted class별 독립 탐색

결과:
- thresholds:
  - `DIE_BROKEN = 1.146664`
  - `NORMAL = 0.263585`
  - `NO_DIE = -0.166378`
- `open macro F1 = 0.7605`
- `overall acc = 0.7733`
- `known acc = 0.8000`
- `unknown acc = 0.7500`

비교:
- `layer2 topk=12 z-score global`: `open macro F1 = 0.8301`, `overall = 0.8400`, `known = 0.8000`, `unknown = 0.8750`
- `layer2 topk=12 z-score class-wise`: `open macro F1 = 0.7605`, `overall = 0.7733`, `known = 0.8000`, `unknown = 0.7500`

해석:
- z-score까지 적용한 뒤에는 class-wise threshold가 오히려 성능을 크게 깎았다.
- 현재 설정에서는 `global threshold + z-score(train known)`가 class-wise threshold보다 확실히 낫다.

### Global Feature-Distance + Layer2 Z-Score Ensemble

설정:
- global feature-distance와 `layer2 topk=12 z-score(train known)`를 결합
- global score도 predicted-class 기준으로 `train known` z-score로 맞춤
- ensemble score:
  - `alpha * patch_zscore + (1 - alpha) * global_zscore`
- `alpha`는 val에서 탐색

최적 결과:
- best `alpha_patch = 0.95`
- best `alpha_global = 0.05`
- threshold `0.687915`
- `open macro F1 = 0.7571`
- `overall acc = 0.7467`
- `known acc = 0.8000`
- `unknown acc = 0.7000`

비교:
- `layer2 topk=12 z-score global`: `open macro F1 = 0.8301`, `overall = 0.8400`, `unknown = 0.8750`
- `ensemble`: `open macro F1 = 0.7571`, `overall = 0.7467`, `unknown = 0.7000`

해석:
- 단순 weighted-sum ensemble은 현재 데이터에서 도움이 되지 않았다.
- val 기준 최적 alpha도 `patch` 쪽으로 거의 몰려서, global score가 실질적으로 추가 정보를 주지 못했다.

## Pros / Cons

### Plain 3-class

장점:
- known accuracy가 가장 높다.
- 구조가 단순하고 재현이 쉽다.

단점:
- unknown rejection이 전혀 없다.
- open-set 실험으로는 의미가 약하다.

### Class-wise threshold

장점:
- 구현이 단순하다.
- 클래스별로 임계값을 다르게 둘 수 있다.

단점:
- val이 작아서 class별 threshold가 과적합되기 쉽다.
- known reject가 너무 커서 실용성이 낮다.
- 현재 결과에선 feature-distance보다 확실히 열세다.

### Class-wise threshold independent

장점:
- joint search보다 훨씬 빠르다.
- known accuracy를 거의 유지한다.
- 해석이 직관적이다.

단점:
- 클래스별 threshold를 따로 찾다 보니 전체 open-set 목적 최적화가 약하다.
- 현재 결과에서는 unknown reject가 충분히 안 된다.
- 특히 `NORMAL`, `NO_DIE`는 threshold가 낮게 고정돼 reject 효과가 거의 없다.

### Feature-distance threshold

장점:
- 세 방법 중 overall / unknown 성능이 가장 좋다.
- prototype과의 거리로 unknown을 거르는 방식이 softmax confidence보다 더 정보성이 있다.

단점:
- known accuracy가 plain baseline보다 낮아진다.
- `DIE_BROKEN` known 샘플 reject가 아직 많다.
- threshold 선택 기준이 더 보수적으로 조정될 여지가 있다.

### Feature-distance threshold class-wise independent

장점:
- global distance threshold와 비슷한 unknown reject를 유지한다.
- 클래스별 거리 분포 차이를 반영할 수 있다.

단점:
- 현재 결과에서는 global threshold보다 성능이 좋아지지 않았다.
- `NO_DIE` threshold가 지나치게 작아져 known 손실이 늘었다.
- val이 작아서 클래스별 거리 threshold 튜닝이 불안정할 수 있다.

## Main Gaps

현재 가장 부족한 점:

1. `DIE_BROKEN` known 샘플이 open-set 방법에서 많이 `UNKNOWN`으로 떨어진다.
2. val split이 작아서 threshold tuning이 불안정할 수 있다.
3. softmax confidence 기반 방법은 unknown confidence가 높아서 분리가 잘 안 된다.
4. feature-distance가 가장 낫지만, known 보존(`0.7714`)은 아직 충분히 높지 않다.
5. independent class-wise threshold는 known 보존은 좋지만 unknown 분리가 부족하다.
6. class-wise feature-distance는 global feature-distance 대비 추가 이득이 없었다.
7. patch-level distance는 `layer2`에서 강했지만, `layer3/4`는 일관된 개선을 주지 못했다.
8. `layer2 patch`에서도 class-wise threshold는 global threshold를 확실히 이기지 못했다.
9. predicted-class score normalization + top-percent pooling도 현재 기준에선 성능 향상이 없었다.
10. 하지만 `train known` 기준 z-score normalization + fixed top-k는 `layer2`에서 유의미한 개선을 보였다.
11. `layer2 topk=12 z-score` 이후에는 class-wise threshold와 simple ensemble 모두 추가 개선을 주지 못했다.

## Recommendation

현재 결과 기준 추천:
- 3-class closed-set 성능만 필요하면: plain baseline
- known 보존을 가장 우선하면: class-wise threshold independent
- open-set 실험 결과를 보여줘야 하면: feature-distance threshold
- known/unknown 균형을 더 강조하면: `layer2 patch topk=12`
- 현재 patch-level 최우선 후보: `layer2 patch topk=12 + z-score(train known)`

우선순위:
1. `feature-distance threshold`와 `layer2 patch topk=12`를 함께 비교 결과로 제시
2. threshold selection rule을 더 보수적으로 조정
3. known accuracy 하한을 두는 방식(`known_acc >= x`) 검토
4. class-wise distance threshold는 현재 우선순위가 낮다
