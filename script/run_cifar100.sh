echo "Start Pretraining"
seed_num=-1
gpu_num=0

python train.py -project base \
        -dataset cifar100 \
        -base_mode ft_dot \
        -new_mode avg_cos \
        -gamma 0.1 \
        -lr_base 0.1 \
        -decay 0.0005 \
        -epochs_base 210 \
        -schedule Milestone \
        -milestones 120 150 180 \
        -gpu $gpu_num \
        -temperature 16 \
        -start_session 0 \
        -batch_size_base 128 \
        -seed $seed_num


echo "Run WaRP for Incremental Sessions"
model_directory="checkpoint/cifar100/base/ft_dot-avg_cos-data_init-start_0/Epo_210-Lr_0.1000-MS_120_150_180-Gam_0.10-Bs_128-Mom_0.90-Wd_0.00050-seed_$seed_num-T_16.00/session0_last_epoch.pth"

python train.py -project base \
        -dataset cifar100 \
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
