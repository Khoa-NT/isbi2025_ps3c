### ------------------------------------------- EfficientNet-B0 ------------------------------------------- ###
### Example: Merge rubbish vs rest, followed by healthy vs unhealthy+bothcells
# python merge_binary_predictions.py \
# --csv_paths \
#     submission/predictions_binary_best_model_efficientnet_b0_binary_rubbish_vs_healthy+unhealthy+bothcells.csv \
#     submission/predictions_binary_best_model_efficientnet_b0_binary_healthy_vs_unhealthy+bothcells.csv \
# --class_0 \
#     rubbish \
#     healthy \
# --class_1 \
#     "healthy unhealthy bothcells" \
#     "unhealthy bothcells" \
# --output_path submission/merged_predictions_efficientnet_b0.csv




### ------------------------------------------- EVA-02 ------------------------------------------- ###
# ### Merge Stage 1 and Stage 2
# python merge_binary_predictions.py \
# --csv_paths \
#     submission/predictions_binary_best_model_eva02_base_patch14_448_binary_rubbish_vs_healthy+unhealthy+bothcells.csv \
#     submission/predictions_binary_best_model_eva02_base_patch14_448_binary_healthy_vs_unhealthy+bothcells.csv \
# --class_0 \
#     rubbish \
#     healthy \
# --class_1 \
#     "healthy unhealthy bothcells" \
#     "unhealthy bothcells" \
# --output_path submission/merged_predictions_eva02_base_patch14_448.csv


### Merge Stage 3 and Stage 2
python merge_binary_predictions.py \
--csv_paths \
    submission/predictions_binary_best_model_eva02_base_patch14_448_binary_rubbish_vs_healthy.csv \
    submission/predictions_binary_best_model_eva02_base_patch14_448_binary_healthy_vs_unhealthy+bothcells.csv \
--class_0 \
    rubbish \
    healthy \
--class_1 \
    "healthy" \
    "unhealthy bothcells" \
--output_path submission/merged_predictions_stage3_stage2_eva02_base_patch14_448.csv