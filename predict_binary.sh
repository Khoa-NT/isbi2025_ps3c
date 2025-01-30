### ------------------------------------------- EfficientNet-B0 ------------------------------------------- ###
# ### Stage 1: Predict rubbish vs all other classes
# python infer_binary.py \
# --dataset test \
# --class_0 rubbish \
# --class_1 healthy unhealthy bothcells \
# --model_name efficientnet_b0 \
# --load_ckpt ckpt/efficientnet_b0/binary_rubbish_vs_healthy+unhealthy+bothcells/best_model_efficientnet_b0_binary_rubbish_vs_healthy+unhealthy+bothcells.pth \
# --batch_size 512 --num_workers 32

# ### Stage 2: Predict healthy vs unhealthy+bothcells
# python infer_binary.py \
# --dataset test \
# --class_0 healthy \
# --class_1 unhealthy bothcells \
# --model_name efficientnet_b0 \
# --load_ckpt ckpt/efficientnet_b0/binary_healthy_vs_unhealthy+bothcells/best_model_efficientnet_b0_binary_healthy_vs_unhealthy+bothcells.pth \
# --batch_size 512 --num_workers 32


### ------------------------------------------- EVA-02 ------------------------------------------- ###
# ### Stage 1: Predict rubbish vs all other classes
# python infer_binary.py \
# --dataset test \
# --class_0 rubbish \
# --class_1 healthy unhealthy bothcells \
# --model_name eva02_base_patch14_448 \
# --load_ckpt ckpt/eva02_base_patch14_448/binary_rubbish_vs_healthy+unhealthy+bothcells/best_model_eva02_base_patch14_448_binary_rubbish_vs_healthy+unhealthy+bothcells.pth \
# --batch_size 512 --num_workers 32

# ### Stage 2: Predict healthy vs unhealthy+bothcells
# python infer_binary.py \
# --dataset test \
# --class_0 healthy \
# --class_1 unhealthy bothcells \
# --model_name eva02_base_patch14_448 \
# --load_ckpt ckpt/eva02_base_patch14_448/binary_healthy_vs_unhealthy+bothcells/best_model_eva02_base_patch14_448_binary_healthy_vs_unhealthy+bothcells.pth \
# --batch_size 512 --num_workers 32

# ### Stage 3: Classify rubbish vs healthy
# python infer_binary.py \
# --dataset test \
# --class_0 rubbish \
# --class_1 healthy \
# --model_name eva02_base_patch14_448 \
# --load_ckpt ckpt/eva02_base_patch14_448/binary_rubbish_vs_healthy/best_model_eva02_base_patch14_448_binary_rubbish_vs_healthy.pth \
# --batch_size 512 --num_workers 32


### Stage 4: Classify unhealthy vs bothcells
python infer_binary.py \
--dataset test \
--class_0 unhealthy \
--class_1 bothcells \
--model_name eva02_base_patch14_448 \
--load_ckpt ckpt/eva02_base_patch14_448/binary_unhealthy_vs_bothcells/best_model_eva02_base_patch14_448_binary_unhealthy_vs_bothcells.pth \
--batch_size 512 --num_workers 32