### Classifier token features
classifier_token_train_path="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/train_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
classifier_token_test_path="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/test_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"

### Pooled features
pooled_features_train_path="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/train_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
pooled_features_test_path="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/test_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"

### Pooled_all features
pooled_all_features_train_path="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/train_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
pooled_all_features_test_path="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/test_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"

hidden_dim_multiplier=4.0


# ### Train SAE for classifier token features
# python train_sae.py \
# --csv_path_train $classifier_token_train_path \
# --csv_path_test $classifier_token_test_path \
# --hidden_dim_multiplier $hidden_dim_multiplier \
# --batch_size 2048 \
# --num_workers 16 \
# --num_epochs 10000 \
# --learning_rate 1e-4 \
# --l1_coefficient 1e-3 \
# --notes "Training SAE classifier_token_features ${hidden_dim_multiplier}x features with 10000 epochs"

# ### Train SAE for pooled features
python train_sae.py \
--csv_path_train $pooled_features_train_path \
--csv_path_test $pooled_features_test_path \
--hidden_dim_multiplier $hidden_dim_multiplier \
--batch_size 2048 \
--num_workers 32 \
--num_epochs 1000 \
--learning_rate 1e-2 \
--l1_coefficient 1 \
--decorr_coefficient 0 \
--entropy_coefficient 1e-2 \
--activation "gelu" \
--notes "Training SAE pooled_features ${hidden_dim_multiplier}x features with 1000 epochs OneCycleLR"

### Train SAE for pooled_all features
# python train_sae.py \
# --csv_path_train $pooled_all_features_train_path \
# --csv_path_test $pooled_all_features_test_path \
# --hidden_dim_multiplier $hidden_dim_multiplier \
# --batch_size 2048 \
# --num_workers 16 \
# --num_epochs 10000 \
# --learning_rate 1e-4 \
# --l1_coefficient 1e-3 \
# --notes "Training SAE pooled_all_features ${hidden_dim_multiplier}x features with 10000 epochs CosineAnnealingLR"














# ### For loop to train SAE for all features
# declare -a train_paths=("$classifier_token_train_path" "$pooled_features_train_path" "$pooled_all_features_train_path")
# declare -a test_paths=("$classifier_token_test_path" "$pooled_features_test_path" "$pooled_all_features_test_path")
# declare -a feature_types=("classifier_token" "pooled" "pooled_all")

# for i in "${!train_paths[@]}"; do
#     echo "Training SAE for ${feature_types[$i]} features..."
#     python train_sae.py \
#         --csv_path_train "${train_paths[$i]}" \
#         --csv_path_test "${test_paths[$i]}" \
#         --hidden_dim_multiplier 4.0 \
#         --batch_size 2048 \
#         --num_workers 16 \
#         --num_epochs 10000 \
#         --learning_rate 1e-4 \
#         --l1_coefficient 1e-3 \
#         --notes "Training SAE ${feature_types[$i]} 4x features with 10000 epochs"
# done
