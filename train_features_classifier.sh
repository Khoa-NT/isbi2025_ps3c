### Dictionary of feature paths
declare -A classifier_token_paths
classifier_token_paths[train]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/train_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
classifier_token_paths[test]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/test_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"

declare -A pooled_paths
pooled_paths[train]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/train_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
pooled_paths[test]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/test_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"

declare -A pooled_all_paths
pooled_all_paths[train]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/train_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
pooled_all_paths[test]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/test_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"


### --------------------------------------------------------------------- Manual --------------------------------------------------------------------- ###

# ### Select the feature type ['classifier_token_paths', 'pooled_paths', 'pooled_all_paths']
# data_path="pooled_paths"

# ### Select the masking method ['Gradient_Boosting', 'Random_Forest', 'Logistic_Regression']
# masking_method="Gradient_Boosting" 

# ### Train classifier on masked features (3-class, with class weights)
# python train_features_classifier.py \
# --csv_path_train "dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-train-dataset.csv" \
# --features_train_csv $(eval echo \${$data_path[train]}) \
# --masking_path "/home/khoa/workspace/Project/isbi2025_ps3c/gaeun/masking.csv" \
# --masking_method $masking_method \
# --merge_bothcells \
# --hidden_dims 1024 512 256 \
# --dropout 0.5 \
# --batch_size 512 \
# --num_workers 32 \
# --num_epochs 100 \
# --learning_rate 1e-3 \
# --notes "Training classifier on masked $data_path features with class weights" \
# --use_class_weights 

### --------------------------------------------------------------------- For-loop --------------------------------------------------------------------- ###

### Loop through all feature types and masking methods
for data_path in "classifier_token_paths" "pooled_paths" "pooled_all_paths"; do
    # for masking_method in "Gradient_Boosting" "Random_Forest" "Logistic_Regression"; do
    for masking_method in "None"; do

        echo "Training $masking_method on $data_path features without class weights"

        ### Train without class weights
        python train_features_classifier.py \
        --csv_path_train "dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-train-dataset.csv" \
        --features_train_csv $(eval echo \${$data_path[train]}) \
        --masking_path "/home/khoa/workspace/Project/isbi2025_ps3c/gaeun/masking.csv" \
        --masking_method $masking_method \
        --merge_bothcells \
        --hidden_dims 1024 512 256 \
        --dropout 0.5 \
        --batch_size 512 \
        --num_workers 32 \
        --num_epochs 100 \
        --learning_rate 1e-3 \
        --notes "Training classifier on masked $data_path features without class weights"

        echo "Training $masking_method on $data_path features with class weights"

        ### Train with class weights
        python train_features_classifier.py \
        --csv_path_train "dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-train-dataset.csv" \
        --features_train_csv $(eval echo \${$data_path[train]}) \
        --masking_path "/home/khoa/workspace/Project/isbi2025_ps3c/gaeun/masking.csv" \
        --masking_method $masking_method \
        --merge_bothcells \
        --hidden_dims 1024 512 256 \
        --dropout 0.5 \
        --batch_size 512 \
        --num_workers 32 \
        --num_epochs 100 \
        --learning_rate 1e-3 \
        --notes "Training classifier on masked $data_path features with class weights" \
        --use_class_weights
    done
done
