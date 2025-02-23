# Train with bothcells merged with unhealthy (recommended)
# python train.py --merge_bothcells --num_epochs 100

# Or train keeping bothcells as separate class
# python train.py --num_epochs 100


### Optional parameters
# --export_each_epoch: Export each epoch model for analysis. For example:
#                      python train.py --merge_bothcells --num_epochs 100 --export_each_epoch
# --use_class_weights: Use class weights for training. For example:
#                      python train.py --merge_bothcells --num_epochs 100 --use_class_weights
# --notes: Add notes for the experiment. For example:
#          python train.py --merge_bothcells --num_epochs 100 --notes "This is a test experiment"

### ------------------------------------------- EVA-02 ------------------------------------------- ###
### https://github.com/baaivision/EVA/tree/master/EVA-02
### https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
### https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eva.py
### If you would like to use / fine-tune EVA-02 in your project, 
### please start with a shorter schedule & smaller learning rate (compared with the baseline setting) first.
### This model is very large and requires a lot of memory if the batch size is large.
### With Batch size = 52, it requires 41GB of memory.

### Large EVA-02
# python train.py \
# --merge_bothcells \
# --batch_size 18 \
# --num_epochs 10 \
# --lr 1e-4 \
# --model_name eva02_large_patch14_448

### Base EVA-02
python train.py \
--merge_bothcells \
--batch_size 32 \
--num_epochs 10 \
--lr 1e-5 \
--model_name eva02_base_patch14_448 \
--csv_path "/home/khoa/workspace/Project/isbi2025_ps3c/gaeun/train_removed_fp_fn.csv" \
--notes "Training with removed FPs and FNs"


### ------------------------------------------- EfficientNet-B0 ------------------------------------------- ###
# python train.py \
# --merge_bothcells \
# --batch_size 32 \
# --num_epochs 10 \
# --lr 1e-4 \
# --model_name efficientnet_b0