# icpbl-wafer-open-set

## Project Goal

Build an open-set wafer defect classifier for the G5 wafer dataset.

- Known classes:
  - `DIE_BROKEN` -> `0`
  - `NORMAL` -> `1`
  - `NO_DIE` -> `2`
- Unknown classes:
  - `DIE_CRACK`
  - `DIE_INK`

Training should use only the known classes, while validation and test should evaluate both known and unknown samples.

## Deliverables

- `report.pdf`
  - NeurIPS format
  - Maximum 6 pages excluding references
- `code.zip`
  - `notebook.ipynb`
  - `model.pth`

## Required Work

- Build a 3-class baseline classifier for known classes.
- Validate the model on the validation split.
- Evaluate on the test split including unknown classes.
- Add an open-set recognition method so unseen defects can be rejected as `Unknown`.
- Log training and evaluation metrics with WandB.
- Include dataset visualization and result plots in the report.

## Dataset Overview

This repository currently contains the `prepared_dataset_G5_622` wafer dataset.
The dataset is organized by split, wafer group, class label, and image files.

```text
prepared_dataset_G5_622/
├── train/
│   └── G5/
│       ├── DIE_BROKEN/images/
│       ├── DIE_CRACK/images/
│       ├── DIE_INK/images/
│       ├── NORMAL/images/
│       └── NO_DIE/images/
├── val/
│   └── G5/
│       ├── DIE_BROKEN/images/
│       ├── DIE_CRACK/images/
│       ├── DIE_INK/images/
│       ├── NORMAL/images/
│       └── NO_DIE/images/
└── test/
    └── G5/
        ├── DIE_BROKEN/images/
        ├── DIE_CRACK/images/
        ├── DIE_INK/images/
        ├── NORMAL/images/
        └── NO_DIE/images/
```

## Image Counts

Total images: 379

| Split | Images |
| --- | ---: |
| train | 229 |
| val | 75 |
| test | 75 |

## Class Distribution

| Class | Type | Train | Val | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| DIE_BROKEN | Known | 26 | 8 | 8 | 42 |
| DIE_CRACK | Unknown | 60 | 20 | 20 | 100 |
| DIE_INK | Unknown | 60 | 20 | 20 | 100 |
| NORMAL | Known | 60 | 20 | 20 | 100 |
| NO_DIE | Known | 23 | 7 | 7 | 37 |

## Recommended Workflow

1. Build a dataset loader with known/unknown label handling.
2. Train a baseline classifier using only `DIE_BROKEN`, `NORMAL`, and `NO_DIE`.
3. Evaluate the baseline on `val` and `test`.
4. Add open-set rejection, for example confidence-threshold-based `Unknown` prediction.
5. Compare baseline and improved model with an ablation study.
6. Export the notebook, trained weights, and final report.

## TODO

- [ ] Create `notebook.ipynb` for training and evaluation.
- [ ] Implement dataset loading for `prepared_dataset_G5_622`.
- [ ] Map labels for the three known classes.
- [ ] Exclude unknown classes from baseline training.
- [ ] Train a baseline 3-class classifier.
- [ ] Log `loss`, `accuracy`, `precision`, `recall`, and `f1-score` to WandB.
- [ ] Visualize sample wafer images for each class.
- [ ] Evaluate on validation and test splits with known + unknown samples.
- [ ] Implement an open-set method to reject unknown defects.
- [ ] Run an ablation study for the open-set improvement.
- [ ] Save trained weights as `model.pth`.
- [ ] Prepare the final report PDF.
