# Train with bothcells merged with unhealthy (recommended)
# python train.py --merge_bothcells --num_epochs 100

# Or train keeping bothcells as separate class
# python train.py --num_epochs 100


### ------------------------------------------- EVA-02 ------------------------------------------- ###
### https://github.com/baaivision/EVA/tree/master/EVA-02
### https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
### If you would like to use / fine-tune EVA-02 in your project, 
### please start with a shorter schedule & smaller learning rate (compared with the baseline setting) first.
# python train.py \
# --merge_bothcells \
# --batch_size 18 \
# --num_epochs 10 \
# --lr 1e-4 \
# --model_name eva02_large_patch14_448

# python train.py \
# --merge_bothcells \
# --batch_size 50 \
# --num_epochs 20 \
# --lr 1e-5 \
# --model_name eva02_base_patch14_448


### ------------------------------------------- EfficientNet-B0 ------------------------------------------- ###
python train.py \
--merge_bothcells \
--batch_size 32 \
--num_epochs 10 \
--lr 1e-4 \
--model_name efficientnet_b0