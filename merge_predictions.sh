### Example: Merge rubbish vs rest, followed by healthy vs unhealthy+bothcells
python merge_binary_predictions.py \
--csv_paths \
    submission/predictions_binary_best_model_efficientnet_b0_binary_rubbish_vs_healthy+unhealthy+bothcells.csv \
    submission/predictions_binary_best_model_efficientnet_b0_binary_healthy_vs_unhealthy+bothcells.csv \
--class_0 \
    rubbish \
    healthy \
--class_1 \
    "healthy unhealthy bothcells" \
    "unhealthy bothcells" \
--output_path submission/merged_predictions_efficientnet_b0.csv


### Example: Different merging order or groups
# python merge_binary_predictions.py \
# --csv_paths \
#     submission/predictions_binary_healthy_vs_rest.csv \
#     submission/predictions_binary_unhealthy_vs_bothcells.csv \
# --class_0 \
#     healthy \
#     unhealthy \
# --class_1 \
#     "unhealthy bothcells rubbish" \
#     "bothcells" \
# --output_path submission/merged_predictions_alternate.csv 