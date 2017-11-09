python -m pdb ../code/train_score.py \
  --gpu_no=1 \
  --list=../list/train_score.list \
  --training_images=100 \
  --training_hyps=16 \
  --batch_size=64 \
  --max_steps=4000 \
  --obj_model=./checkpoints/obj/model-400000 \
  --checkpoint_dir=./checkpoints/score
