# V2 Fine Search Final Result

## Search Setup

- model: `ResNet18`
- known train classes: `DIE_BROKEN`, `NORMAL`, `NO_DIE`
- train setting:
  - `batch_size=8`
  - `epochs=30`
  - `learning_rate=1e-4`
  - `weight_decay=1e-4`
  - `label_smoothing=0.05`
- open-set evaluation:
  - `layer2 patch distance`
  - `topk=12`
  - predicted-class z-score
  - z-score stats from `train known` only
  - threshold selected on `val` by `open_macro_f1`
  - fixed threshold on `test`

## Baseline Reference

| Setting | Raw known acc | Open macro F1 | Overall acc | Known acc | Unknown acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v1 best` | 0.9714 | 0.8301 | 0.8400 | 0.8000 | 0.8750 |
| `v2 aug_cs_mild` | 0.9714 | 0.8347 | 0.8533 | 0.7714 | 0.9250 |
| `v2 aug_cs_autocontrast` | 0.9714 | 0.8294 | 0.8533 | 0.8000 | 0.9000 |

## Round Winners

| Round | Best config | Open macro F1 | Overall acc | Known acc | Unknown acc | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `round1` | `fs_r1_ac20` | 0.8362 | 0.8533 | 0.8000 | 0.9000 | 0.464571 |
| `round2` | `fs_r2_ac18_sharp110_p20` | 0.8375 | 0.8533 | 0.7714 | 0.9250 | 0.101082 |
| `round3` | `fs_r3_ac17_sharp110_p20` | 0.8447 | 0.8667 | 0.8000 | 0.9250 | 0.288528 |
| `round4` | `fs_r4_ac172_sharp110_p20` | 0.8123 | 0.8267 | 0.8000 | 0.8500 | 0.376250 |

`round4`가 `round3` 최고를 넘지 못했으므로, 국소 탐색은 `round3`에서 포화된 것으로 보고 종료했다.

## Final Best

- experiment: `fs_r3_ac17_sharp110_p20`
- checkpoint: [resnet18_best_val_loss_epoch25.pth](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v2/fine_search_round3/fs_r3_ac17_sharp110_p20/resnet18_best_val_loss_epoch25.pth)
- config: [config.json](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v2/fine_search_round3/fs_r3_ac17_sharp110_p20/config.json)
- combined summary: [combined_summary.json](/home/minje/icpbl-wafer-open-set/baseline/checkpoints/v2/fine_search_round3/fs_r3_ac17_sharp110_p20/combined_summary.json)

최종 augmentation:

- `rotation_degrees = 5`
- `contrast_strength = 0.08`
- `sharpness_factor = 1.10`
- `sharpness_p = 0.20`
- `autocontrast_p = 0.17`

최종 성능:

| Setting | Best epoch | Best val loss | Raw known acc | Open macro F1 | Overall acc | Known acc | Unknown acc | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fs_r3_ac17_sharp110_p20` | 25 | 0.2173 | 0.9714 | 0.8447 | 0.8667 | 0.8000 | 0.9250 | 0.288528 |

## Improvement Summary

- vs `v1 best`
  - `open_macro_f1`: `0.8301 -> 0.8447` (`+0.0146`)
  - `overall_acc`: `0.8400 -> 0.8667` (`+0.0267`)
  - `known_acc`: `0.8000 -> 0.8000` (same)
  - `unknown_acc`: `0.8750 -> 0.9250` (`+0.0500`)

- vs initial `v2 aug_cs_mild`
  - `open_macro_f1`: `0.8347 -> 0.8447` (`+0.0100`)
  - `overall_acc`: `0.8533 -> 0.8667` (`+0.0134`)
  - `known_acc`: `0.7714 -> 0.8000` (`+0.0286`)
  - `unknown_acc`: `0.9250 -> 0.9250` (same)

## Interpretation

- 강한 augmentation 방향은 계속 불리했고, `mild` 축을 유지한 채 `autocontrast`를 약하게 섞는 방향이 가장 잘 맞았다.
- `autocontrast`는 `0.17~0.20` 부근에서만 유효했고, 더 높이거나 더 낮추면 성능이 바로 떨어졌다.
- `sharpness`는 초깃값 `1.15 / 0.30`보다 약간 낮춘 `1.10 / 0.20`이 더 안정적이었다.
- 마지막 국소 탐색(`round4`)에서 성능이 다시 내려갔기 때문에, 현재 기준 최종 설정은 `fs_r3_ac17_sharp110_p20`으로 두는 것이 합리적이다.
