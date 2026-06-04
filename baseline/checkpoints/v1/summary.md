# V1 Checkpoint Test Summary

기준:
- 모든 결과는 `test_all_classes.py`로 평가한 `test` split 75장 기준
- known-class metric은 `DIE_BROKEN`, `NORMAL`, `NO_DIE` 35장 기준
- 아래 표는 `*_test_results.json`에서 집계한 결과

## Overall

모든 `val_loss` checkpoint는 동일한 test 예측 결과를 냈다.

- known-class test accuracy: `0.9714`
- confusion pattern:
  - `DIE_BROKEN`: `7 -> DIE_BROKEN`, `1 -> NORMAL`
  - `DIE_CRACK`: `20 -> NORMAL`
  - `DIE_INK`: `16 -> DIE_BROKEN`, `4 -> NORMAL`
  - `NORMAL`: `20 -> NORMAL`
  - `NO_DIE`: `7 -> NO_DIE`

차이는 confidence만 있다. label smoothing이 커질수록 confidence가 내려간다.

## No Smoothing

| Checkpoint | Best Epoch | Val Loss | Known Acc | DIE_BROKEN | DIE_CRACK | DIE_INK | NORMAL | NO_DIE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_smoothing/resnet18_best_val_loss_epoch18.pth` | 18 | 0.000710 | 0.9714 | 0.9996 | 0.9997 | 0.9983 | 0.9968 | 0.9999 |
| `no_smoothing/resnet18_best_val_loss_epoch92.pth` | 92 | 0.000427 | 0.9714 | 0.9999 | 1.0000 | 0.9970 | 0.9996 | 0.9998 |

JSON:
- [epoch18](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/no_smoothing/resnet18_best_val_loss_epoch18_test_results.json)
- [epoch92](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/no_smoothing/resnet18_best_val_loss_epoch92_test_results.json)

## Label Smoothing

| Label Smoothing | Checkpoint | Best Epoch | Val Loss | Known Acc | DIE_BROKEN | DIE_CRACK | DIE_INK | NORMAL | NO_DIE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | `label_smoothing/0.05/resnet18_best_val_loss_epoch17.pth` | 17 | 0.215489 | 0.9714 | 0.9803 | 0.9591 | 0.9620 | 0.9250 | 0.9734 |
| 0.10 | `label_smoothing/0.1/resnet18_best_val_loss_epoch30.pth` | 30 | 0.356897 | 0.9714 | 0.9359 | 0.8850 | 0.9019 | 0.8570 | 0.9532 |
| 0.20 | `label_smoothing/0.2/resnet18_best_val_loss_epoch29.pth` | 29 | 0.575026 | 0.9714 | 0.8665 | 0.7796 | 0.8289 | 0.7416 | 0.9119 |

JSON:
- [0.05](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.05/resnet18_best_val_loss_epoch17_test_results.json)
- [0.10](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30_test_results.json)
- [0.20](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v1/label_smoothing/0.2/resnet18_best_val_loss_epoch29_test_results.json)

## Notes

- `no_smoothing`의 `epoch18`과 `epoch92`는 예측 결과가 같고 confidence만 미세하게 다르다.
- label smoothing은 분류 결과를 바꾸지 않았고, confidence만 낮췄다.
- 현재 test 기준으로는 `label_smoothing=0.1`이 과신 완화와 confidence 유지 사이에서 가장 무난하다.
