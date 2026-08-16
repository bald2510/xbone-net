@echo off
setlocal EnableExtensions
chcp 65001 >nul

rem Rebuild the report-ready Chapter 4 tables and figures from current results.
rem Usage:
rem   scripts\generate_report_assets.bat
rem   scripts\generate_report_assets.bat --refresh-statistics

cd /d "%~dp0\.."

set "PYTHON=%XBONE_PYTHON%"
if not defined PYTHON set "PYTHON=python"

set "VIS=tools\visualize\results.py"
set "SUMMARY=results\summary"
set "CLASS=%SUMMARY%\classification"
set "OOD=%SUMMARY%\ood"
set "ATTN=%SUMMARY%\attention"
set "ABL=%SUMMARY%\ablation"
set "RUN_ALL=%SUMMARY%\run_all_table.csv"
set "REPORT=docs\report\generated\chapter4_results"
set "APPENDIX_SUMMARY=%SUMMARY%\appendix_seed_results"
set "APPENDIX_REPORT=docs\report\generated\appendix_seed_results"

if not exist "%RUN_ALL%" (
    echo [ERROR] Khong tim thay "%RUN_ALL%".
    exit /b 1
)

echo.
echo [1/6] Tao cac bang phan loai zero-shot, full-shot va mau hoc han che...

"%PYTHON%" "%VIS%" latex ^
  --input "%RUN_ALL%" ^
  --where category="Baselines / Zero-shot" ^
  --where-in "config=biomedclip_zeroshot|clip_zeroshot|medclip_zeroshot|pubmedclip_zeroshot" ^
  --columns dataset config accuracy_mean balanced_accuracy_mean f1_macro_mean auroc_macro_mean auprc_macro_mean ^
  --value-columns accuracy_mean balanced_accuracy_mean f1_macro_mean auroc_macro_mean auprc_macro_mean ^
  --column-label "dataset=Dữ liệu" ^
  --column-label "config=Mô hình" ^
  --column-label "accuracy_mean=Acc $\uparrow$" ^
  --column-label "balanced_accuracy_mean=BAcc $\uparrow$" ^
  --column-label "f1_macro_mean=Macro-F1 $\uparrow$" ^
  --column-label "auroc_macro_mean=Macro-AUROC $\uparrow$" ^
  --column-label "auprc_macro_mean=Macro-AUPRC $\uparrow$" ^
  --bold-within dataset ^
  --multirow dataset ^
  --resize-to-textwidth ^
  --precision 4 ^
  --position H ^
  --caption "Kết quả phân loại zero-shot với một câu nhắc cho mỗi lớp trên BTXRD và CTCH; mỗi mô hình được đánh giá một lần tại hạt giống 42." ^
  --label tab:ch4draft_zeroshot ^
  --output "%CLASS%\table_zero_shot.tex"

if not exist docs\report\generated\chapter4_draft mkdir docs\report\generated\chapter4_draft
copy /Y "%CLASS%\table_zero_shot.tex" docs\report\generated\chapter4_draft\table_zero_shot.tex >nul

"%PYTHON%" "%VIS%" latex ^
  --input "%RUN_ALL%" ^
  --where-in "category=Baselines / Full fine-tuning|Proposed" ^
  --where-in "config=fft_resnet50|fft_densenet|fft_medclip|fft_clip|fft_pubmedclip|fft_biomedclip|ours_xbone_net" ^
  --columns dataset config accuracy_mean balanced_accuracy_mean f1_macro_mean auroc_macro_mean auprc_macro_mean ^
  --value-columns accuracy_mean balanced_accuracy_mean f1_macro_mean auroc_macro_mean auprc_macro_mean ^
  --std-map accuracy_mean=accuracy_std ^
  --std-map balanced_accuracy_mean=balanced_accuracy_std ^
  --std-map f1_macro_mean=f1_macro_std ^
  --std-map auroc_macro_mean=auroc_macro_std ^
  --std-map auprc_macro_mean=auprc_macro_std ^
  --column-label "dataset=Dữ liệu" ^
  --column-label "config=Mô hình" ^
  --column-label "accuracy_mean=Acc $\uparrow$" ^
  --column-label "balanced_accuracy_mean=BAcc $\uparrow$" ^
  --column-label "f1_macro_mean=Macro-F1 $\uparrow$" ^
  --column-label "auroc_macro_mean=Macro-AUROC $\uparrow$" ^
  --column-label "auprc_macro_mean=Macro-AUPRC $\uparrow$" ^
  --bold-within dataset ^
  --multirow dataset ^
  --precision 4 ^
  --position H ^
  --resize-to-textwidth ^
  --caption "Kết quả phân loại khi sử dụng toàn bộ tập huấn luyện" ^
  --label tab:classification_full_data ^
  --output "%CLASS%\table_full_shot_classification.tex"


"%PYTHON%" "%VIS%" latex ^
  --input "%RUN_ALL%" ^
  --where-in "category=Baselines / Full fine-tuning|Baselines / PEFT|Proposed" ^
  --where-in "config=fft_biomedclip|fft_clip|fft_densenet|fft_medclip|fft_pubmedclip|fft_resnet50|lora_biomedclip|lora_pubmedclip|ours_xbone_net" ^
  --columns dataset config auroc_macro_mean auprc_macro_mean ^
  --value-columns auroc_macro_mean auprc_macro_mean ^
  --std-map auroc_macro_mean=auroc_macro_std ^
  --std-map auprc_macro_mean=auprc_macro_std ^
  --column-label "dataset=Dữ liệu" ^
  --column-label "config=Mô hình" ^
  --column-label "auroc_macro_mean=Macro-AUROC $\uparrow$" ^
  --column-label "auprc_macro_mean=Macro-AUPRC $\uparrow$" ^
  --bold-within dataset ^
  --multirow dataset ^
  --precision 4 ^
  --position H ^
  --caption "Các độ đo xếp hạng xác suất khi sử dụng toàn bộ tập huấn luyện" ^
  --label tab:classification_full_data_ranking ^
  --output "%CLASS%\table_full_shot_ranking.tex"

"%PYTHON%" "%VIS%" latex ^
  --input "%RUN_ALL%" ^
  --where-in "category=Few-shot / 1 shot|Few-shot / 10 shot|Few-shot / 20 shot" ^
  --where-in "config=lora_biomedclip|lora_pubmedclip|ours_xbone_net" ^
  --columns dataset category config accuracy_mean balanced_accuracy_mean f1_macro_mean ^
  --value-columns accuracy_mean balanced_accuracy_mean f1_macro_mean ^
  --std-map accuracy_mean=accuracy_std ^
  --std-map balanced_accuracy_mean=balanced_accuracy_std ^
  --std-map f1_macro_mean=f1_macro_std ^
  --column-label "dataset=Dữ liệu" ^
  --column-label "category=Thiết lập" ^
  --column-label "config=Mô hình" ^
  --column-label "accuracy_mean=Accuracy $\uparrow$" ^
  --column-label "balanced_accuracy_mean=Balanced Accuracy $\uparrow$" ^
  --column-label "f1_macro_mean=Macro-F1 $\uparrow$" ^
  --bold-within dataset category ^
  --multirow dataset category ^
  --resize-to-textwidth ^
  --precision 4 ^
  --position H ^
  --caption "Kết quả phân loại trong các thiết lập mẫu học hạn chế" ^
  --label tab:classification_few_shot ^
  --output "%CLASS%\table_few_shot_classification.tex"

"%PYTHON%" "%VIS%" latex ^
  --input "%RUN_ALL%" ^
  --where-in "category=Few-shot / 1 shot|Few-shot / 10 shot|Few-shot / 20 shot" ^
  --where-in "config=lora_biomedclip|lora_pubmedclip|ours_xbone_net" ^
  --columns dataset category config auroc_macro_mean auprc_macro_mean ^
  --value-columns auroc_macro_mean auprc_macro_mean ^
  --std-map auroc_macro_mean=auroc_macro_std ^
  --std-map auprc_macro_mean=auprc_macro_std ^
  --column-label "dataset=Dữ liệu" ^
  --column-label "category=Thiết lập" ^
  --column-label "config=Mô hình" ^
  --column-label "auroc_macro_mean=Macro-AUROC $\uparrow$" ^
  --column-label "auprc_macro_mean=Macro-AUPRC $\uparrow$" ^
  --bold-within dataset category ^
  --multirow dataset category ^
  --precision 4 ^
  --position H ^
  --caption "Các độ đo xếp hạng xác suất trong các thiết lập mẫu học hạn chế" ^
  --label tab:classification_few_shot_ranking ^
  --output "%CLASS%\table_few_shot_ranking.tex"

"%PYTHON%" "%VIS%" latex ^
  --input "%RUN_ALL%" ^
  --where category=Proposed ^
  --where config=ours_xbone_net ^
  --columns dataset config ece_15_mean ^
  --value-columns ece_15_mean ^
  --std-map ece_15_mean=ece_15_std ^
  --column-label "dataset=Dữ liệu" ^
  --column-label "config=Mô hình" ^
  --column-label "ece_15_mean=ECE $\downarrow$" ^
  --minimize-columns ece_15_mean ^
  --no-bold-best ^
  --precision 4 ^
  --position H ^
  --caption "Sai số hiệu chỉnh kỳ vọng của XBone-Net với 15 khoảng xác suất" ^
  --label tab:classification_ece ^
  --output "%CLASS%\table_calibration_results.tex"

echo.
echo [2/6] Tao bang tham so, hieu qua suy luan va cac bieu do phan loai...

"%PYTHON%" "%VIS%" efficiency ^
  --full-shot-output "%CLASS%\table_efficiency_full_shot.tex" ^
  --few-shot-output "%CLASS%\table_efficiency_few_shot.tex"
if errorlevel 1 goto :error
if not exist "%CLASS%\table_efficiency_full_shot.tex" goto :error
if not exist "%CLASS%\table_efficiency_few_shot.tex" goto :error

"%PYTHON%" "%VIS%" full-shot-comparison ^
  --input "%RUN_ALL%" ^
  --datasets BTXRD CTCH ^
  --title-template "Phân loại với toàn bộ dữ liệu trên {dataset}" ^
  --dpi 300 ^
  --output "%CLASS%\full_shot_comparison.png"

"%PYTHON%" "%VIS%" aggregate-curves ^
  --inputs ^
    results\ctch\proposed\ours_xbone_net\seed_42\analysis\features\ctch_test.npz ^
    results\ctch\proposed\ours_xbone_net\seed_123\analysis\features\ctch_test.npz ^
    results\ctch\proposed\ours_xbone_net\seed_456\analysis\features\ctch_test.npz ^
  --output-dir "%CLASS%\curves_3seed" ^
  --prefix ctch_xbone_net ^
  --title-prefix "" ^
  --n-bins 15 ^
  --dpi 300

"%PYTHON%" "%VIS%" aggregate-confusion ^
  --inputs ^
    results\ctch\proposed\ours_xbone_net\seed_42\analysis\features\ctch_test.npz ^
    results\ctch\proposed\ours_xbone_net\seed_123\analysis\features\ctch_test.npz ^
    results\ctch\proposed\ours_xbone_net\seed_456\analysis\features\ctch_test.npz ^
  --class-names-file data\CTCH\labels.txt ^
  --title "" ^
  --dpi 300 ^
  --output "%CLASS%\curves_3seed\ctch_xbone_net_confusion_matrix_3seed.png"

"%PYTHON%" "%VIS%" bar ^
  --input "%RUN_ALL%" ^
  --where dataset=BTXRD ^
  --where-in "category=Few-shot / 1 shot|Few-shot / 10 shot|Few-shot / 20 shot" ^
  --where-in "config=lora_biomedclip|lora_pubmedclip|ours_xbone_net" ^
  --x category ^
  --y f1_macro_mean ^
  --hue config ^
  --error f1_macro_std ^
  --highlight ours_xbone_net ^
  --title "Phân loại mẫu học hạn chế trên BTXRD" ^
  --x-label "Số mẫu huấn luyện cho mỗi lớp" ^
  --y-label "Macro-F1" ^
  --annotate ^
  --precision 3 ^
  --dpi 300 ^
  --output "%CLASS%\few_shot_btxrd_f1_bar.png"

"%PYTHON%" "%VIS%" bar ^
  --input "%RUN_ALL%" ^
  --where dataset=CTCH ^
  --where-in "category=Few-shot / 1 shot|Few-shot / 10 shot|Few-shot / 20 shot" ^
  --where-in "config=lora_biomedclip|lora_pubmedclip|ours_xbone_net" ^
  --x category ^
  --y f1_macro_mean ^
  --hue config ^
  --error f1_macro_std ^
  --highlight ours_xbone_net ^
  --title "Phân loại mẫu học hạn chế trên CTCH" ^
  --x-label "Số mẫu huấn luyện cho mỗi lớp" ^
  --y-label "Macro-F1" ^
  --annotate ^
  --precision 3 ^
  --dpi 300 ^
  --output "%CLASS%\few_shot_ctch_f1_bar.png"

echo.
echo [3/6] Tao bang va bieu do OOD cho Semantic OOD va BTXRD...

"%PYTHON%" "%VIS%" ood ^
  --input results\ctch\proposed\ours_xbone_net\analysis\aggregated_results.json ^
  --scenarios semantic_ood domain_ood_btxrd ^
  --methods cosine_centroids mahalanobis_centroid knn entropy ^
  --csv-output "%OOD%\ood_semantic_btxrd.csv" ^
  --output "%OOD%\table_ood_semantic_btxrd.tex"

"%PYTHON%" "%VIS%" ood-metrics ^
  --input results\ctch\proposed\ours_xbone_net\analysis\aggregated_results.json ^
  --scenarios semantic_ood domain_ood_btxrd ^
  --method mahalanobis_centroid ^
  --title "Mahalanobis theo tâm lớp trên OOD ngữ nghĩa và BTXRD" ^
  --dpi 300 ^
  --output "%OOD%\mahalanobis_semantic_btxrd_metrics.png"

echo.
echo [4/6] Tao hinh Integrated Gradients...

"%PYTHON%" "%VIS%" explainability-report ^
  --aggregate-input results\ctch\proposed\ours_xbone_net\analysis\aggregated_results.json ^
  --experiment-root results\ctch\proposed\ours_xbone_net ^
  --seeds 42 123 456 ^
  --output-dir "%ATTN%" ^
  --dpi 300

"%PYTHON%" tools\visualize\render_multimodal_ig.py ^
  --image-id 2842_img-33484-00001.jpg ^
  --experiment ctch/proposed/ours_xbone_net ^
  --seed 42 ^
  --ig-steps 24 ^
  --output "%ATTN%\attention_case_2842_img-33484-00001.png"

"%PYTHON%" tools\visualize\render_multimodal_ig.py ^
  --image-id 3021_img-84263-00001.jpg ^
  --experiment ctch/proposed/ours_xbone_net ^
  --seed 42 ^
  --ig-steps 24 ^
  --output "%ATTN%\attention_case_3021_img-84263-00001.png"

if not exist docs\report\generated\explainability mkdir docs\report\generated\explainability
copy /Y "%ATTN%\attention_case_2842_img-33484-00001.png" docs\report\generated\explainability\attention_case_2842_img-33484-00001.png >nul
copy /Y "%ATTN%\attention_case_3021_img-84263-00001.png" docs\report\generated\explainability\attention_case_3021_img-84263-00001.png >nul

echo.
echo [5/6] Tao bang leave-one-out va forest plot khoang tin cay...

"%PYTHON%" "%VIS%" ablation-leave-one-out ^
  --input "%RUN_ALL%" ^
  --precision 4 ^
  --output "%ABL%\table_ablation_leave_one_out_classification.tex"

if /I "%~1"=="--refresh-statistics" (
  echo Dang tinh lai bootstrap/permutation; buoc nay co the mat vai phut...
  "%PYTHON%" "%VIS%" ablation-statistics ^
    --results-root results ^
    --seeds 42 123 456 ^
    --metrics f1_macro balanced_accuracy accuracy ^
    --n-bootstrap 10000 ^
    --n-permutations 10000 ^
    --test-method permutation ^
    --output-dir "%ABL%\statistics" ^
    --dpi 300
  
  "%PYTHON%" "%VIS%" ablation-statistics ^
    --results-root results ^
    --seeds 42 123 456 ^
    --metrics auroc_macro auprc_macro ^
    --n-bootstrap 10000 ^
    --test-method bootstrap ^
    --output-dir "%ABL%\statistics_auc" ^
    --dpi 300
  )

if not exist "%ABL%\statistics\paired_bootstrap_results.csv" goto :missing_statistics
if not exist "%ABL%\statistics_auc\paired_bootstrap_results.csv" goto :missing_statistics

"%PYTHON%" "%VIS%" ablation-forest ^
  --input "%ABL%\statistics\paired_bootstrap_results.csv" ^
  --metric accuracy ^
  --dpi 300 ^
  --output "%ABL%\statistics\forest_accuracy.png"

"%PYTHON%" "%VIS%" ablation-forest ^
  --input "%ABL%\statistics\paired_bootstrap_results.csv" ^
  --metric balanced_accuracy ^
  --dpi 300 ^
  --output "%ABL%\statistics\forest_balanced_accuracy.png"

"%PYTHON%" "%VIS%" ablation-forest ^
  --input "%ABL%\statistics\paired_bootstrap_results.csv" ^
  --metric f1_macro ^
  --dpi 300 ^
  --output "%ABL%\statistics\forest_f1_macro.png"

"%PYTHON%" "%VIS%" ablation-forest ^
  --input "%ABL%\statistics_auc\paired_bootstrap_results.csv" ^
  --metric auroc_macro ^
  --dpi 300 ^
  --output "%ABL%\statistics_auc\forest_auroc_macro.png"

"%PYTHON%" "%VIS%" ablation-forest ^
  --input "%ABL%\statistics_auc\paired_bootstrap_results.csv" ^
  --metric auprc_macro ^
  --dpi 300 ^
  --output "%ABL%\statistics_auc\forest_auprc_macro.png"

if not exist "%REPORT%" mkdir "%REPORT%"

copy /Y "%CLASS%\table_zero_shot.tex" "%REPORT%\table_zero_shot.tex" >nul
copy /Y "%CLASS%\table_full_shot_classification.tex" "%REPORT%\table_full_shot_classification.tex" >nul
copy /Y "%CLASS%\table_full_shot_ranking.tex" "%REPORT%\table_full_shot_ranking.tex" >nul
copy /Y "%CLASS%\table_few_shot_classification.tex" "%REPORT%\table_few_shot_classification.tex" >nul
copy /Y "%CLASS%\table_few_shot_ranking.tex" "%REPORT%\table_few_shot_ranking.tex" >nul
copy /Y "%CLASS%\table_calibration_results.tex" "%REPORT%\table_calibration_results.tex" >nul
copy /Y "%CLASS%\table_efficiency_full_shot.tex" "%REPORT%\table_efficiency_full_shot.tex" >nul
copy /Y "%CLASS%\table_efficiency_few_shot.tex" "%REPORT%\table_efficiency_few_shot.tex" >nul
copy /Y "%ABL%\table_ablation_leave_one_out_classification.tex" "%REPORT%\table_ablation_leave_one_out_classification.tex" >nul
copy /Y "%OOD%\table_ood_semantic_btxrd.tex" "%REPORT%\table_ood_semantic_btxrd.tex" >nul

copy /Y "%CLASS%\curves_3seed\ctch_xbone_net_confusion_matrix_3seed.png" "%REPORT%\ctch_xbone_net_confusion_matrix_3seed.png" >nul
copy /Y "%CLASS%\curves_3seed\ctch_xbone_net_pr_3seed.png" "%REPORT%\ctch_xbone_net_pr_3seed.png" >nul
copy /Y "%CLASS%\curves_3seed\ctch_xbone_net_roc_3seed.png" "%REPORT%\ctch_xbone_net_roc_3seed.png" >nul
copy /Y "%CLASS%\full_shot_comparison.png" "%REPORT%\full_shot_comparison.png" >nul
copy /Y "%CLASS%\few_shot_btxrd_f1_bar.png" "%REPORT%\few_shot_btxrd_f1_bar.png" >nul
copy /Y "%CLASS%\few_shot_ctch_f1_bar.png" "%REPORT%\few_shot_ctch_f1_bar.png" >nul
copy /Y "%CLASS%\curves_3seed\ctch_xbone_net_calibration_3seed.png" "%REPORT%\ctch_xbone_net_calibration_3seed.png" >nul
copy /Y "%OOD%\mahalanobis_semantic_btxrd_metrics.png" "%REPORT%\mahalanobis_semantic_btxrd_metrics.png" >nul
copy /Y "%ABL%\statistics\forest_accuracy.png" "%REPORT%\forest_accuracy.png" >nul
copy /Y "%ABL%\statistics\forest_balanced_accuracy.png" "%REPORT%\forest_balanced_accuracy.png" >nul
copy /Y "%ABL%\statistics\forest_f1_macro.png" "%REPORT%\forest_f1_macro.png" >nul
copy /Y "%ABL%\statistics_auc\forest_auroc_macro.png" "%REPORT%\forest_auroc_macro.png" >nul
copy /Y "%ABL%\statistics_auc\forest_auprc_macro.png" "%REPORT%\forest_auprc_macro.png" >nul
copy /Y "%ATTN%\attention_case_2842_img-33484-00001.png" "%REPORT%\attention_case_2842_img-33484-00001.png" >nul
copy /Y "%ATTN%\attention_case_3021_img-84263-00001.png" "%REPORT%\attention_case_3021_img-84263-00001.png" >nul

echo.
echo [6/6] Tao bang va bieu do ket qua chi tiet theo tung hat giong...

"%PYTHON%" tools\visualize\appendix_seed_results.py ^
  --results-root results ^
  --output-dir "%APPENDIX_SUMMARY%" ^
  --dpi 300
if errorlevel 1 goto :error

if not exist "%APPENDIX_REPORT%" mkdir "%APPENDIX_REPORT%"
copy /Y "%APPENDIX_SUMMARY%\table_full_shot_per_seed.tex" "%APPENDIX_REPORT%\table_full_shot_per_seed.tex" >nul
copy /Y "%APPENDIX_SUMMARY%\table_few_shot_per_seed.tex" "%APPENDIX_REPORT%\table_few_shot_per_seed.tex" >nul
copy /Y "%APPENDIX_SUMMARY%\table_ablation_per_seed.tex" "%APPENDIX_REPORT%\table_ablation_per_seed.tex" >nul
copy /Y "%APPENDIX_SUMMARY%\table_ood_per_seed.tex" "%APPENDIX_REPORT%\table_ood_per_seed.tex" >nul
copy /Y "%APPENDIX_SUMMARY%\full_shot_macro_f1_by_seed.png" "%APPENDIX_REPORT%\full_shot_macro_f1_by_seed.png" >nul
copy /Y "%APPENDIX_SUMMARY%\ood_mahalanobis_by_seed.png" "%APPENDIX_REPORT%\ood_mahalanobis_by_seed.png" >nul
copy /Y "%APPENDIX_SUMMARY%\manifest.json" "%APPENDIX_REPORT%\manifest.json" >nul

set "MISSING=0"
for %%F in (
  "%CLASS%\table_full_shot_classification.tex"
  "%CLASS%\table_zero_shot.tex"
  "%CLASS%\table_full_shot_ranking.tex"
  "%CLASS%\table_efficiency_full_shot.tex"
  "%CLASS%\table_few_shot_classification.tex"
  "%CLASS%\table_few_shot_ranking.tex"
  "%CLASS%\table_efficiency_few_shot.tex"
  "%CLASS%\table_calibration_results.tex"
  "%CLASS%\curves_3seed\ctch_xbone_net_calibration_3seed.png"
  "%CLASS%\curves_3seed\ctch_xbone_net_roc_3seed.png"
  "%CLASS%\curves_3seed\ctch_xbone_net_pr_3seed.png"
  "%CLASS%\curves_3seed\ctch_xbone_net_confusion_matrix_3seed.png"
  "%CLASS%\full_shot_comparison.png"
  "%CLASS%\few_shot_btxrd_f1_bar.png"
  "%CLASS%\few_shot_ctch_f1_bar.png"
  "%OOD%\table_ood_semantic_btxrd.tex"
  "%OOD%\mahalanobis_semantic_btxrd_metrics.png"
  "%ATTN%\attention_case_2842_img-33484-00001.png"
  "%ATTN%\attention_case_3021_img-84263-00001.png"
  "%ABL%\table_ablation_leave_one_out_classification.tex"
  "%ABL%\statistics\forest_accuracy.png"
  "%ABL%\statistics\forest_balanced_accuracy.png"
  "%ABL%\statistics\forest_f1_macro.png"
  "%ABL%\statistics_auc\forest_auroc_macro.png"
  "%ABL%\statistics_auc\forest_auprc_macro.png"
  "%APPENDIX_SUMMARY%\table_full_shot_per_seed.tex"
  "%APPENDIX_SUMMARY%\table_few_shot_per_seed.tex"
  "%APPENDIX_SUMMARY%\table_ablation_per_seed.tex"
  "%APPENDIX_SUMMARY%\table_ood_per_seed.tex"
  "%APPENDIX_SUMMARY%\full_shot_macro_f1_by_seed.png"
  "%APPENDIX_SUMMARY%\ood_mahalanobis_by_seed.png"
) do (
  if not exist "%%~F" (
    echo [MISSING] %%~F
    set "MISSING=1"
  )
)

if "%MISSING%"=="1" goto :error

echo.
echo [DONE] Da tao va kiem tra day du cac bang, bieu do trong "%SUMMARY%".
exit /b 0

:missing_statistics
echo [ERROR] Chua co ket qua kiem dinh thong ke.
echo Chay lai: scripts\generate_report_assets.bat --refresh-statistics
exit /b 1

:error
echo.
echo [ERROR] Qua trinh tao bang va bieu do da dung do co loi.
exit /b 1
