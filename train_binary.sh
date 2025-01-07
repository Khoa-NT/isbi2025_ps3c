### ------------------------------------------- EfficientNet-B0 ------------------------------------------- ###
### Stage 1: Classify rubbish vs all other classes
# python train_binary.py \
# --class_0 rubbish \
# --class_1 healthy unhealthy bothcells \
# --batch_size 128 \
# --num_epochs 10 \
# --lr 1e-4 \
# --model_name efficientnet_b0 \
# --notes "Binary classification: rubbish vs healthy+unhealthy+bothcells"

### Stage 2: Classify healthy vs unhealthy bothcells
# python train_binary.py \
# --class_0 healthy \
# --class_1 unhealthy bothcells \
# --batch_size 256 \
# --num_epochs 10 \
# --lr 1e-4 \
# --model_name efficientnet_b0 \
# --notes "Binary classification: healthy vs unhealthy+bothcells"

### ------------------------------------------- EVA-02 ------------------------------------------- ###
### Stage 1: Classify rubbish vs all other classes
python train_binary.py \
--class_0 rubbish \
--class_1 healthy unhealthy bothcells \
--batch_size 32 \
--num_epochs 10 \
--lr 1e-5 \
--model_name eva02_base_patch14_448 \
--notes "Binary classification: rubbish vs healthy+unhealthy+bothcells"

### Stage 2: Classify healthy vs unhealthy bothcells
python train_binary.py \
--class_0 healthy \
--class_1 unhealthy bothcells \
--batch_size 32 \
--num_epochs 10 \
--lr 1e-5 \
--model_name eva02_base_patch14_448 \
--notes "Binary classification: healthy vs unhealthy+bothcells"