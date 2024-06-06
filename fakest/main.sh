export semi_setting='COVERAGE_AUG/1-8'

CUDA_VISIBLE_DEVICES=0 python -W ignore main.py \
  --dataset COVERAGE_AUG --data-root ./ \
  --batch-size 16 --backbone resnet50 --model deeplabv3plus \
  --labeled-id-path dataset/splits/$semi_setting/labeled.txt \
  --unlabeled-id-path dataset/splits/$semi_setting/unlabeled.txt \
  --pseudo-mask-path outdir/pseudo_masks/$semi_setting \
  --save-path outdir/models/$semi_setting \
  --reliable-id-path dataset/splits/$semi_setting