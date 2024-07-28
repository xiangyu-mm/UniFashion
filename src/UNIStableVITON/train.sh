# VITONHD base
#CUDA_VISIBLE_DEVICES=2,3 python train.py \
# --config_name VITONHD \
# --transform_size shiftscale3 hflip \
# --transform_color hsv bright_contrast \
# --save_name Base_test


# VITONHD ATVloss
CUDA_VISIBLE_DEVICES=2,3 python train.py \
 --config_name VITONHD \
 --transform_size shiftscale3 hflip \
 --transform_color hsv bright_contrast \
 --use_atv_loss \
 --resume_path /home/data2/xiangyu/Code/StableVITON/logs/20240623_Base_test/models/resume.ckpt \
 --save_name ATVloss_test
