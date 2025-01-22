### Dictionary of feature paths
declare -A classifier_token_paths
classifier_token_paths[train]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/train_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
classifier_token_paths[test]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/test_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
classifier_token_paths[Gradient_Boosting_unweighted]="ckpt/feature_classifier/train_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Gradient_Boosting/best_model_feature_classifier_Gradient_Boosting_3class.pth"
classifier_token_paths[Gradient_Boosting_weighted]="ckpt/feature_classifier/train_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Gradient_Boosting/use_class_weights/best_model_feature_classifier_Gradient_Boosting_3class_weighted.pth"
classifier_token_paths[Random_Forest_unweighted]="ckpt/feature_classifier/train_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Random_Forest/best_model_feature_classifier_Random_Forest_3class.pth"
classifier_token_paths[Random_Forest_weighted]="ckpt/feature_classifier/train_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Random_Forest/use_class_weights/best_model_feature_classifier_Random_Forest_3class_weighted.pth"
classifier_token_paths[Logistic_Regression_unweighted]="ckpt/feature_classifier/train_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Logistic_Regression/best_model_feature_classifier_Logistic_Regression_3class.pth"
classifier_token_paths[Logistic_Regression_weighted]="ckpt/feature_classifier/train_classifier_token_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Logistic_Regression/use_class_weights/best_model_feature_classifier_Logistic_Regression_3class_weighted.pth"


declare -A pooled_paths
pooled_paths[train]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/train_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
pooled_paths[test]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/test_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
pooled_paths[Gradient_Boosting_unweighted]="ckpt/feature_classifier/train_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Gradient_Boosting/best_model_feature_classifier_Gradient_Boosting_3class.pth"
pooled_paths[Gradient_Boosting_weighted]="ckpt/feature_classifier/train_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Gradient_Boosting/use_class_weights/best_model_feature_classifier_Gradient_Boosting_3class_weighted.pth"
pooled_paths[Random_Forest_unweighted]="ckpt/feature_classifier/train_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Random_Forest/best_model_feature_classifier_Random_Forest_3class.pth"
pooled_paths[Random_Forest_weighted]="ckpt/feature_classifier/train_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Random_Forest/use_class_weights/best_model_feature_classifier_Random_Forest_3class_weighted.pth"
pooled_paths[Logistic_Regression_unweighted]="ckpt/feature_classifier/train_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Logistic_Regression/best_model_feature_classifier_Logistic_Regression_3class.pth"
pooled_paths[Logistic_Regression_weighted]="ckpt/feature_classifier/train_pooled_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Logistic_Regression/use_class_weights/best_model_feature_classifier_Logistic_Regression_3class_weighted.pth"


declare -A pooled_all_paths
pooled_all_paths[train]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/train_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
pooled_all_paths[test]="extracted_features/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class/test_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class.csv"
pooled_all_paths[Gradient_Boosting_unweighted]="ckpt/feature_classifier/train_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Gradient_Boosting/best_model_feature_classifier_Gradient_Boosting_3class.pth"
pooled_all_paths[Gradient_Boosting_weighted]="ckpt/feature_classifier/train_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Gradient_Boosting/use_class_weights/best_model_feature_classifier_Gradient_Boosting_3class_weighted.pth"
pooled_all_paths[Random_Forest_unweighted]="ckpt/feature_classifier/train_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Random_Forest/best_model_feature_classifier_Random_Forest_3class.pth"
pooled_all_paths[Random_Forest_weighted]="ckpt/feature_classifier/train_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Random_Forest/use_class_weights/best_model_feature_classifier_Random_Forest_3class_weighted.pth"
pooled_all_paths[Logistic_Regression_unweighted]="ckpt/feature_classifier/train_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Logistic_Regression/best_model_feature_classifier_Logistic_Regression_3class.pth"
pooled_all_paths[Logistic_Regression_weighted]="ckpt/feature_classifier/train_pooled_all_features_ckpt_backup_20Epoch_best_model_eva02_base_patch14_448_3class/Logistic_Regression/use_class_weights/best_model_feature_classifier_Logistic_Regression_3class_weighted.pth"


### --------------------------------------------------------------------- Manual --------------------------------------------------------------------- ###

# ### Select the feature type
# data_path="pooled_paths"

# ### Select the masking method ['Gradient_Boosting', 'Random_Forest', 'Logistic_Regression']
# masking_method="Gradient_Boosting"

# ### [weighted, unweighted]
# loss_type="weighted"


# ### Create the key for the dictionary
# selected_ckpt=${masking_method}_${loss_type}


# ### Run inference
# python infer_features_classifier.py \
# --csv_path "dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-test-dataset.csv" \
# --features_csv $(eval echo \${$data_path[test]}) \
# --masking_path "/home/khoa/workspace/Project/isbi2025_ps3c/gaeun/masking.csv" \
# --masking_method $masking_method \
# --load_ckpt $(eval echo \${${data_path}[${selected_ckpt}]}) \
# --merge_bothcells \
# --hidden_dims 1024 512 256 \
# --dropout 0.5 \
# --batch_size 512 \
# --num_workers 32 




### --------------------------------------------------------------------- For-loop --------------------------------------------------------------------- ###


### Loop through all feature types and masking methods
### There is a problem in the submission directory that the prediction is ovelapped because we didn't distinguish between `data_path`
### We have to copy the result and move to the specific folder. For example, we run a case of `data_path` = `classifier_token_paths`, then we move all the prediction to `submission/feature_classifier/classifier_token_paths`
for data_path in "classifier_token_paths" "pooled_paths" "pooled_all_paths"; do
    for masking_method in "Gradient_Boosting" "Random_Forest" "Logistic_Regression"; do
        for loss_type in "weighted" "unweighted"; do
            
            ### Create the key for the dictionary
            selected_ckpt=${masking_method}_${loss_type}
            
            echo "Running inference with $masking_method on $data_path features with $loss_type loss"
            
            ### Run inference
            python infer_features_classifier.py \
            --csv_path "dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-test-dataset.csv" \
            --features_csv $(eval echo \${$data_path[test]}) \
            --masking_path "/home/khoa/workspace/Project/isbi2025_ps3c/gaeun/masking.csv" \
            --masking_method $masking_method \
            --load_ckpt $(eval echo \${${data_path}[${selected_ckpt}]}) \
            --merge_bothcells \
            --hidden_dims 1024 512 256 \
            --dropout 0.5 \
            --batch_size 512 \
            --num_workers 32
            
        done
    done

    ### Delete and create the prediction folder
    rm -rf submission/feature_classifier/$data_path 
    mkdir -p submission/feature_classifier/$data_path

    ### Move the prediction to the specific folder
    mv submission/feature_classifier/predictions_*.csv submission/feature_classifier/$data_path/
done









