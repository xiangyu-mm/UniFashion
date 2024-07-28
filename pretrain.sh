CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision bf16 src/blip_fine_tune_2.py \
   --dataset 'FashionIQ' \
   --blip-model-name 'blip2_cir_cls' \
   --num-epochs 10 \
   --num-workers 4 \
   --learning-rate '2e-5' \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 \
#>backup_dot.log 2>&1 &