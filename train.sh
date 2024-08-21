# This script uses two NVIDIA 3090 GPUs to run two experiments simultaneously. If you only have one 3090, please comment out one of the experiments.

# train on the VisA dataset
(
nohup python -u train.py --dataset visa --train_data_path ./dataset/mvisa/data \
--val_data_path ./dataset/mvisa/data \
--save_path ./my_exps/train_visa --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt \
--prompt_len 2 --deep_prompt_len 1 --device_id 0 --learning_rate 0.00004 --features_list 6 12 18 24 --pretrained openai --image_size 518 \
--batch_size 32 --epoch 10 --group_id_list 2 --seed 333 --config_path ./models/model_configs/ViT-L-14-336.json --model ViT-L-14-336
) > ./log_train_visa.out 2>&1 &

# train on the MVTec-AD dataset
(
nohup python -u train.py --dataset mvtec --train_data_path ./dataset/mvisa/data \
--val_data_path ./dataset/mvisa/data \
--save_path ./my_exps/train_mvtec --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt \
--prompt_len 2 --deep_prompt_len 1 --device_id 1 --learning_rate 0.00002 --features_list 6 12 18 24 --pretrained openai --image_size 518 \
--batch_size 32 --epoch 10 --group_id_list 2 --seed 333 --config_path ./models/model_configs/ViT-L-14-336.json --model ViT-L-14-336
) > ./log_train_mvtec.out 2>&1 &