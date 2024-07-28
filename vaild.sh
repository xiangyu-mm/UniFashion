#CUDA_VISIBLE_DEVICES=0 python src/blip_validate.py \
#   --dataset 'fashioniq' \
#   --blip-model-name 'blip2_cir_cls' \
#   --model-path '/home/data2/xiangyu/Code/SPRC/models/only_code2code_blip2_cir_cls_rqvae8_2024-05-03_04:57:26/saved_models/tuned_clip_best.pt'

CUDA_VISIBLE_DEVICES=3 accelerate launch --mixed_precision bf16 src/generate.py \
   --dataset 'discriminator' \
   --blip-model-name 'blip2_cir_full' \
   --num-epochs 2 \
   --num-workers 4 \
   --learning-rate '2e-5' \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1