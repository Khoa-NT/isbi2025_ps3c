### Train student model with fixed teacher
echo "Training student model with fixed teacher"

python train_student.py \
--student_model efficientnet_b0 \
--teacher_model eva02_base_patch14_448 \
# --teacher_ckpt "ckpt/eva02_base_patch14_448/binary_unhealthy_vs_bothcells/best_model_eva02_base_patch14_448_binary_unhealthy_vs_bothcells.pth" \
# --batch_size 128 \
# --num_workers 32 \
# --num_epochs 10 \
# --student_lr 1e-4


## Train student model together with teacher
echo "Training student model together with teacher"

python train_student.py \
--student_model efficientnet_b0 \
--teacher_model eva02_base_patch14_448 \
--teacher_ckpt "ckpt/eva02_base_patch14_448/binary_unhealthy_vs_bothcells/best_model_eva02_base_patch14_448_binary_unhealthy_vs_bothcells.pth" \
--batch_size 32 \
--num_workers 32 \
--num_epochs 10 \
--student_lr 1e-4 \
--teacher_lr 1e-5 \
--train_teacher