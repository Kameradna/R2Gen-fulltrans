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
--frozen \
--original \
--record_dir recordsfrozenbaselinenopretrained2 \
--save_dir resultsfrozenbaselinenopretrained2

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
--d_vf 768 \
--monitor_metric CIDEr \
--n_gpu 1 \
--frozen \
--record_dir recordsfrozentransnopretrained2 \
--save_dir resultsfrozentransnopretrained2

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
--record_dir recordsbaseline2 \
--save_dir resultsbaseline2

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
--record_dir recordsfrozenbaseline2 \
--save_dir resultsfrozenbaseline2

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
--record_dir recordsfrozentrans2 \
--save_dir resultsfrozentrans2

nvtop