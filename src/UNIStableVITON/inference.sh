#### paired
CUDA_VISIBLE_DEVICES=3 python inference.py \
 --config_path /home/data2/xiangyu/Code/StableVITON/configs/VITONHD.yaml \
 --batch_size 1 \
 --model_load_path /home/data2/xiangyu/Code/StableVITON/logs/20240610_Base_test/models/VITONHD.ckpt \
 --save_dir /home/data2/xiangyu/Code/StableVITON/test_dir

##### unpaired
#CUDA_VISIBLE_DEVICES=4 python inference.py \
# --config_path ./configs/VITONHD.yaml \
# --batch_size 4 \
# --model_load_path <model weight path> \
# --unpair \
# --save_dir <save directory>
#
##### paired repaint
#CUDA_VISIBLE_DEVICES=4 python inference.py \
# --config_path ./configs/VITONHD.yaml \
# --batch_size 4 \
# --model_load_path <model weight path>t \
# --repaint \
# --save_dir <save directory>
#
##### unpaired repaint
#CUDA_VISIBLE_DEVICES=4 python inference.py \
# --config_path ./configs/VITONHD.yaml \
# --batch_size 4 \
# --model_load_path <model weight path> \
# --unpair \
# --repaint \
# --save_dir <save directory>