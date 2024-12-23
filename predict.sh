### ------------------------------------------- EVA-02 ------------------------------------------- ###
# python predict.py \
# --merge_bothcells \
# --model_name eva02_base_patch14_448 \
# --load_ckpt ckpt/backup_20Epoch_best_model_eva02_base_patch14_448_3class.pth


### ------------------------------------------- EfficientNet-B0 ------------------------------------------- ###
python predict.py \
--merge_bothcells \
--model_name efficientnet_b0 \
--load_ckpt ckpt/efficientnet_b0/best_model_efficientnet_b0_3class.pth

