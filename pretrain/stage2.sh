CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision bf16 src/blip_fine_tune_2.py \
   --dataset 'discriminator' \
   --blip-model-name 'blip2_cir_full' \
   --num-epochs 2 \
   --num-workers 4 \
   --learning-rate '2e-4' \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 \
#>backup_dot.log 2>&1 &