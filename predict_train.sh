### ------------------------------------------- EVA-02 ------------------------------------------- ###
### Predicting on pretrained model
python infer.py \
--infer_mode prediction \
--dataset train \
--merge_bothcells \
--model_name eva02_base_patch14_448 \
--batch_size 256 --num_workers 32 ### 35GB GPU + 118GB RAM


### ------------------------------------------- EfficientNet-B0 ------------------------------------------- ###
# python infer.py \
# --infer_mode prediction \
# --dataset train \
# --merge_bothcells \
# --model_name efficientnet_b0 \
# --batch_size 512 --num_workers 32 ### 35GB GPU + 118GB RAM
