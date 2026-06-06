# ICPBL Guide Remaining Checklist

검토 기준: `icpbl-wafer-open-set/ICPBL+ Project Introduction and Guidelines.pdf`

검토 대상:

- Report: 사용자가 대화에 붙여준 LaTeX 초안 기준
- Notebook: `icpbl-wafer-open-set/best_model/code/notebook.ipynb`
- Report용 통합 curve: `icpbl-wafer-open-set/best_model/combined_training_curves.png`

우선순위:

- `P0`: 제출 전 반드시 처리 권장

## P0. 반드시 보완할 항목

- [ ] Report에 class별 wafer dataset visualization 추가
  - Guide 근거: `Visualize wafer dataset for each class`
  - 현재 상태: notebook cell 5에는 dataset visualization이 있지만, report LaTeX 초안에는 없음.
  - 보완 위치: `Dataset and task setting` section, class definition table 직후.
  - 권장 내용: `DIE_BROKEN`, `NORMAL`, `NO_DIE`, `DIE_CRACK`, `DIE_INK` 각 class 예시 이미지를 한 figure로 삽입.
  - 권장 파일 경로: `icpbl-wafer-open-set/best_model/dataset_examples_by_class.png`

- [ ] Report에 epoch별 training/validation curve 추가
  - Guide 근거: `Plot loss, acc, precision, recall, f1-score per each epoch`
  - 현재 상태: notebook cell 9에는 curve가 있지만, report LaTeX 초안에는 없음.
  - 사용할 파일: `icpbl-wafer-open-set/best_model/combined_training_curves.png`
  - 보완 위치: `Experiments` section 안에 `Training curves and WandB visualization` subsection 추가.
  - TeX가 repo root에 있으면 사용할 상대 경로: `best_model/combined_training_curves.png`
  - TeX가 `best_model` 안에 있으면 사용할 상대 경로: `combined_training_curves.png`

```latex
\subsection{Training curves and WandB visualization}
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{best_model/combined_training_curves.png}
  \caption{Training and validation curves of the best ResNet18 classifier. The figure shows loss, accuracy, precision, recall, and F1-score for each epoch using the best-run history logged to WandB.}
  \label{fig:training_curves}
\end{figure}
```

- [ ] Report의 qualitative figure 경로 수정
  - 현재 LaTeX 경로:

```latex
\includegraphics[width=\linewidth]{die_crack_gradcam_layers234.png}
\includegraphics[width=\linewidth]{die_crack_patch_distance_layers234.png}
```

  - 문제: repo root에 위 파일명 그대로는 없음. 컴파일 시 이미지 누락 가능성이 큼.
  - 실제 존재하는 예시 경로:

```text
icpbl-wafer-open-set/best_model/Ablation study/class_by_image_seed61/DIE_CRACK/test__R1C84X0Y0L0W0/resnet18_best_val_loss_epoch29_test__R1C84X0Y0L0W0_gradcam_layers234.png
icpbl-wafer-open-set/best_model/Ablation study/class_by_image_seed61/DIE_CRACK/test__R1C84X0Y0L0W0/resnet18_best_val_loss_epoch29_test__R1C84X0Y0L0W0_patch_distance_layers234.png
```

  - 권장 처리: report용으로 짧은 파일명으로 복사한 뒤 include path를 수정.
  - 권장 복사 후 경로:

```text
icpbl-wafer-open-set/best_model/die_crack_gradcam_layers234.png
icpbl-wafer-open-set/best_model/die_crack_patch_distance_layers234.png
```

- [ ] Report Appendix training config에 class weights 추가
  - 현재 상태: notebook은 class weights를 사용하지만 report Appendix의 `Final training configuration` table에는 없음.
  - Guide 관련성: training 설정 재현성에 직접 관련됨.
  - 사용하는 이유: known class train sample 수가 `DIE_BROKEN=26`, `NORMAL=60`, `NO_DIE=23`으로 불균형하기 때문에, inverse-frequency class weights로 minority class loss 기여도를 보정함.
  - 보완 위치: Appendix `Implementation details`의 `Table~\ref{tab:training_config}`.
  - 권장 추가 row:

```latex
    Class weights & Inverse-frequency weights from train known class counts \\
```

## 가장 중요한 결론

현재 notebook은 대부분 준비되어 있고, 남은 핵심 보완은 report 쪽이다.

1. Report에 `icpbl-wafer-open-set/best_model/combined_training_curves.png` 삽입.
2. Report에 class별 wafer dataset example figure 삽입.
3. Report qualitative figure의 실제 이미지 경로 수정.
4. Report Appendix training config table에 class weights row 추가.

최종 확인사항: PDF를 컴파일한 뒤 references 제외 6 page 안에 들어가는지만 확인하면 된다.
