set -e
log=`date +%Y%m%d`'_train_score.log'

python -u ../code/train_score.py \
  --list=../list/train_score.list \
  --training_images=100 \
  --training_hyps=16 \
  --batch_size=64 \
  --max_steps=4000 \
  --obj_model=./checkpoints/obj \
  --checkpoint_dir=./checkpoints/score | tee $log
