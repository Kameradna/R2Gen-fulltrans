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
--frozen \
--record_dir recordsfrozentrans3 \
--save_dir resultsfrozentrans3


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
--record_dir recordsbaselinenopretrained2 \
--save_dir resultsbaselinenopretrained2

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
--original \
--record_dir recordsbaseline3 \
--save_dir resultsbaseline3

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
--original \
--frozen \
--record_dir recordsfrozenbaseline3 \
--save_dir resultsfrozenbaseline3


nvtop