
python main_train.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--visual_extractor resnet101 \
--d_vf 2048 \
--monitor_metric CIDEr \
--n_gpu 1 \
--original \
--frozen
mv results resultsfrozenbaselinenopretrained
mv records recordsfrozenbaselinenopretrained

python main_train.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--visual_extractor resnet101 \
--weights IMAGENET1K_V1 \
--d_vf 2048 \
--monitor_metric CIDEr \
--n_gpu 1 \
--original
mv results resultsbaseline
mv records recordsbaseline

python main_train.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--visual_extractor vit_b_16 \
--weights IMAGENET1K_V1 \
--d_vf 768 \
--monitor_metric CIDEr \
--n_gpu 1 \
--original
mv results resultstrans
mv records recordstrans

nvtop