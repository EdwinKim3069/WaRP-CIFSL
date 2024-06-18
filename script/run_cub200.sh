echo "Start Pretraining"
seed_num=-1
gpu_num=0

python train.py -project base \
        -dataset cub200 \
        -base_mode ft_dot \
        -new_mode avg_cos \
        -gamma 0.1 \
        -lr_base 0.01 \
        -decay 0.0005 \
        -epochs_base 120 \
        -schedule Milestone \
        -milestones 30 60 90 \
        -gpu $gpu_num \
        -temperature 16 \
        -start_session 0 \
        -batch_size_base 128 \
        -seed $seed_num


echo "Run WaRP for Incremental Sessions"
model_directory="checkpoint/cub200/base/ft_dot-avg_cos-data_init-start_0/Epo_120-Lr_0.0100-MS_30_60_90-Gam_0.10-Bs_128-Mom_0.90-Wd_0.00050-seed_$seed_num-T_16.00/session0_last_epoch.pth"

python train.py -project base \
        -dataset cub200 \
        -new_mode ft_cos \
        -gamma 0.1 \
        -lr_base 0.1 \
        -decay 0.0005 \
        -epochs_base 0 \
        -temperature 16 \
        -start_session 0 \
        -model_dir $model_directory \
        -gpu $gpu_num \
        -lr_new 0.01 \
        -epochs_new 30 \
        -fraction_to_keep 0.1 \
        -seed $seed_num
