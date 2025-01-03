### ------------------------------------------- EVA-02 ------------------------------------------- ###
python infer.py \
--dataset test \
--merge_bothcells \
--model_name eva02_base_patch14_448 \
--load_ckpt ckpt/eva02_base_patch14_448/best_model_eva02_base_patch14_448_3class.pth \
--batch_size 512 --num_workers 32 ### 35GB GPU + 118GB RAM


### ------------------------------------------- EfficientNet-B0 ------------------------------------------- ###
# python infer.py \
# --dataset test \
# --merge_bothcells \
# --model_name efficientnet_b0 \
# --load_ckpt ckpt/efficientnet_b0/best_model_efficientnet_b0_3class.pth

