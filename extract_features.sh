model_name="eva02_base_patch14_448"

### Load the ckpt
### Uncomment to use the pretrained model. Remember to uncomment the `--load_ckpt $load_ckpt \` below too.
# load_ckpt="ckpt/eva02_base_patch14_448/backup_20Epoch_best_model_eva02_base_patch14_448_3class.pth"

### Dataset
datasets=("train" "test")

### Extract mode
extract_mode="pooled" # pooled, pooled_all, classifier_token
extract_modes=("pooled" "pooled_all" "classifier_token")


### ------------------------------------------- Process ------------------------------------------- ###
### For running individual mode
# python infer.py \
# --infer_mode extract_features \
# --extract_mode $extract_mode \
# --dataset test \
# --merge_bothcells \
# --model_name $model_name \
# --load_ckpt $load_ckpt \
# --batch_size 512 --num_workers 32 ### 35GB GPU + 118GB RAM


### Extract features with specific mode for all datasets 
# for dataset in ${datasets[@]}; do
#     echo -e "\033[33m\nExtracting features from $dataset dataset...\033[0m"
#     python infer.py \
#     --infer_mode extract_features \
#     --extract_mode $extract_mode \
#     --dataset $dataset \
#     --merge_bothcells \
#     --model_name $model_name \
#     --load_ckpt $load_ckpt \
#     --batch_size 512 --num_workers 32 ### 35GB GPU + 118GB RAM
# done


### Run all modes for all datasets
for extract_mode in ${extract_modes[@]}; do
    for dataset in ${datasets[@]}; do
        echo -e "\033[33m\nExtracting features from $dataset dataset with $extract_mode mode...\033[0m"
        python infer.py \
        --infer_mode extract_features \
        --extract_mode $extract_mode \
        --dataset $dataset \
        --merge_bothcells \
        --model_name $model_name \
        # --load_ckpt $load_ckpt \
        --batch_size 512 --num_workers 32 ### 35GB GPU + 118GB RAM
    done
done