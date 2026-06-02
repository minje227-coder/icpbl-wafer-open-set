# icpbl-wafer-open-set

## 프로젝트 목표

G5 웨이퍼 데이터셋을 대상으로 open-set 불량 분류 모델을 구현하는 프로젝트입니다.

- Known 클래스:
  - `DIE_BROKEN` -> `0`
  - `NORMAL` -> `1`
  - `NO_DIE` -> `2`
- Unknown 클래스:
  - `DIE_CRACK`
  - `DIE_INK`

학습은 known 클래스만 사용하고, validation 및 test에서는 known과 unknown 샘플을 함께 평가합니다.

## 제출물

- `report.pdf`
  - NeurIPS 형식
  - 참고문헌 제외 최대 6페이지
- `code.zip`
  - `notebook.ipynb`
  - `model.pth`

## 해야 할 일

- known 클래스 기준 3-class baseline 분류기 구현
- validation split에서 모델 검증
- unknown 클래스를 포함한 test split 평가
- 보지 못한 결함을 `Unknown`으로 거절할 수 있도록 open-set 기법 추가
- WandB로 학습 및 평가 지표 기록
- 데이터셋 시각화와 결과 그래프를 보고서에 포함

## 데이터셋 개요

현재 저장소에는 `prepared_dataset_G5_622` 웨이퍼 데이터셋이 포함되어 있습니다.
데이터셋은 split, wafer 그룹, 클래스 라벨, 이미지 파일 기준으로 구성되어 있습니다.

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

## 이미지 개수

총 이미지 수: 379

| Split | Images |
| --- | ---: |
| train | 229 |
| val | 75 |
| test | 75 |

## 클래스 분포

| Class | Type | Train | Val | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| DIE_BROKEN | Known | 26 | 8 | 8 | 42 |
| DIE_CRACK | Unknown | 60 | 20 | 20 | 100 |
| DIE_INK | Unknown | 60 | 20 | 20 | 100 |
| NORMAL | Known | 60 | 20 | 20 | 100 |
| NO_DIE | Known | 23 | 7 | 7 | 37 |

## 권장 진행 순서

1. known/unknown 라벨 처리가 가능한 데이터로더를 만든다.
2. `DIE_BROKEN`, `NORMAL`, `NO_DIE`만 사용해 baseline 분류기를 학습한다.
3. baseline을 `val`, `test`에서 평가한다.
4. confidence threshold 기반 `Unknown` 예측 등 open-set rejection을 추가한다.
5. baseline과 개선 모델을 ablation study로 비교한다.
6. notebook, 학습된 가중치, 최종 보고서를 정리한다.

## TODO

- [ ] 학습 및 평가용 `notebook.ipynb` 작성
- [ ] `prepared_dataset_G5_622`용 데이터 로딩 구현
- [ ] known 3개 클래스 라벨 매핑 구현
- [ ] baseline 학습 시 unknown 클래스 제외
- [ ] baseline 3-class 분류기 학습
- [ ] `loss`, `accuracy`, `precision`, `recall`, `f1-score`를 WandB에 기록
- [ ] 클래스별 wafer 이미지 샘플 시각화
- [ ] known + unknown 샘플을 포함한 validation/test 평가
- [ ] unknown 결함 거절을 위한 open-set 기법 구현
- [ ] open-set 개선안에 대한 ablation study 수행
- [ ] 학습된 가중치를 `model.pth`로 저장
- [ ] 최종 보고서 PDF 작성
