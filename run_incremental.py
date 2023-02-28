import os



seed = -1
gpu_num = 0

project = 'base'
# dataset = 'mini_imagenet'
dataset = 'cifar100'
# dataset = 'cub200'

lr_new = 0.01
epochs_new = 30
frac_to_keep_list = [0.1]
# new_mode = 'avg_cos'
new_mode = 'ft_cos'
model_dir = 'checkpoint/{}/{}/ft_dot-avg_cos-data_init-start_0/Epo_210-Lr_0.1000-MS_120_150_180-Gam_0.10-Bs_128-Mom_0.90-Wd_0.00050-seed_{}-T_16.00/session0_last_epoch.pth'.format(dataset, project, seed)
# model_dir = 'checkpoint/{}/{}/ft_dot-avg_cos-data_init-start_0/Epo_120-Lr_0.0100-MS_30_60_90-Gam_0.10-Bs_128-Mom_0.90-Wd_0.00050-seed_{}-T_16.00/session0_last_epoch.pth'.format(dataset, project, seed)
# model_dir = 'checkpoint/{}/{}/ft_dot-avg_cos-data_init-start_0/Epo_300-Lr_0.1000-MS_210_240_270-Gam_0.10-Bs_128-Mom_0.90-Wd_0.00050-seed_{}-T_16.00/session0_last_epoch.pth'.format(dataset, project, seed)
# for epochs_new in epochs_new_list:
for frac_to_keep in frac_to_keep_list:
    os.system(''
              'python train.py '
              '-project {} '
              '-dataset {} '
              '-base_mode ft_dot '
              '-new_mode {} '
              '-gamma 0.1 '
              '-lr_base 0.1 '
              '-decay 0.0005 '
              '-epochs_base 0 '
              '-schedule Milestone '
              '-milestones 40 70 '
              '-temperature 16 '
              '-start_session 0 '
              '-model_dir {} '
              '-gpu {} '
              '-lr_new {} '
              '-epochs_new {} '
              '-fraction_to_keep {} '.format(project, dataset, new_mode, model_dir, gpu_num, lr_new, epochs_new, frac_to_keep)
              )