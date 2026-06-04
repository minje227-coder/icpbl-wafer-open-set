# V2 Augmentation Comparison

평가 기준은 모두 동일하다.

- 학습 ckpt: `baseline/resnet18.ipynb`에서 저장한 `best val_loss`
- open-set 방법: `layer2 patch distance + topk=12 + predicted-class z-score`
- z-score mean/std: `train` known 3-class만 사용
- threshold 선택: `val`에서 `open_macro_f1` 최대 기준
- `test`는 선택된 threshold를 고정 적용

## Training Summary

| Experiment | Augmentation | Best epoch | Best val loss |
| --- | --- | ---: | ---: |
| `aug_cs_mild` | rot=5, contrast=0.08, sharpness=1.15@0.30 | 29 | 0.2173 |
| `aug_cs_medium` | rot=8, contrast=0.15, sharpness=1.35@0.50 | 25 | 0.2156 |
| `aug_cs_autocontrast` | rot=6, contrast=0.08, sharpness=1.15@0.30, autocontrast=0.50 | 18 | 0.2185 |

## Test Comparison

| Experiment | Raw known acc | Selected threshold | Open macro F1 | Overall acc | Known acc | Unknown acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `aug_cs_mild` | 0.9714 | 0.231561 | 0.8347 | 0.8533 | 0.7714 | 0.9250 |
| `aug_cs_medium` | 0.9714 | 0.144109 | 0.8133 | 0.8133 | 0.8000 | 0.8250 |
| `aug_cs_autocontrast` | 0.9714 | 0.274850 | 0.8294 | 0.8533 | 0.8000 | 0.9000 |

`Raw known acc`는 threshold 적용 전 softmax argmax 기준의 known 35장 정확도다.

## Reference To V1 Best

기존 최고 기준은 `v1/label_smoothing/0.1/resnet18_best_val_loss_epoch30.pth`에 대한
`layer2 patch + topk=12 + z-score(train known) + global threshold`였다.

| Reference | Raw known acc | Open macro F1 | Overall acc | Known acc | Unknown acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v1 best` | 0.9714 | 0.8301 | 0.8400 | 0.8000 | 0.8750 |

## Notes

- `aug_cs_mild`가 이번 v2 중 최고 `open_macro_f1`를 기록했다.
  - `0.8347`, v1 best `0.8301`보다 소폭 상승
- `aug_cs_autocontrast`는 `overall_acc`가 `aug_cs_mild`와 같은 `0.8533`이지만,
  `unknown_acc`가 `0.9000`으로 mild의 `0.9250`보다 낮아서 `open_macro_f1`는 약간 뒤진다.
- `aug_cs_medium`은 known 보존은 나쁘지 않지만 unknown rejection이 약해져 전체적으로 가장 불리했다.
- 세 실험 모두 raw known acc는 동일하게 `0.9714`였다.
  차이는 threshold 적용 이후 unknown rejection과 known rejection의 균형에서 갈렸다.

## Output Files

- `aug_cs_mild`
  - `aug_cs_mild/resnet18_best_val_loss_epoch29_layer2_patch_global_norm_train_known_summary.json`
  - `aug_cs_mild/resnet18_best_val_loss_epoch29_layer2_patch_topk12_global_norm_test_results.json`
- `aug_cs_medium`
  - `aug_cs_medium/resnet18_best_val_loss_epoch25_layer2_patch_global_norm_train_known_summary.json`
  - `aug_cs_medium/resnet18_best_val_loss_epoch25_layer2_patch_topk12_global_norm_test_results.json`
- `aug_cs_autocontrast`
  - `aug_cs_autocontrast/resnet18_best_val_loss_epoch18_layer2_patch_global_norm_train_known_summary.json`
  - `aug_cs_autocontrast/resnet18_best_val_loss_epoch18_layer2_patch_topk12_global_norm_test_results.json`
