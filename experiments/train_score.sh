set -e
log=`date +%Y%m%d`'_train_score.log'

python -u ../code/train_score.py \
  --gpu_no=1 \
  --list=../list/train_score.list \
  --training_images=100 \
  --training_hyps=16 \
  --batch_size=1600 \
  --learning_rate_decay=0.5 \
  --max_steps=40000 \
  --obj_model=./checkpoints/obj/model-400000 \
  --checkpoint_dir=./checkpoints/score | tee $log
