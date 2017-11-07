set -e
log=`date +%Y%m%d`'_train_obj.log'

python -u ../code/train_obj.py \
  --list=../list/train.list \
  --training_images=100 \
  --training_patches=512 \
  --batch_size=64 \
  --max_steps=400000 \
  --checkpoint_dir=./checkpoints/obj | tee $log
