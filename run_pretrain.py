import os


seed = -1
gpu_num = 0

project = 'base'

# dataset = 'mini_imagenet'
dataset = 'cifar100'
# dataset = 'cub200'
if 'cub' in dataset:
    lr_base = 0.01
else:
    lr_base = 0.1

epochs_bases = [210]
milestones_list = ['120 150 180']
# epochs_bases = [120]
# milestones_list = ['30 60 90']
# epochs_bases = [300]
# milestones_list = ['210 240 270']
for i, epochs_base in enumerate(epochs_bases):
    os.system(''
              'python train.py '
              '-project {} '
              '-dataset {} '
              '-base_mode ft_dot '
              '-new_mode avg_cos '
              '-gamma 0.1 '
              '-lr_base {} '
              '-decay 0.0005 '
              '-epochs_base {} '
              '-schedule Milestone '
              '-milestones {} '
              '-gpu {} '
              '-temperature 16 '
              '-start_session 0 '
              '-batch_size_base 128 '
              '-seed {}'.format(project, dataset, lr_base, epochs_base, milestones_list[i], gpu_num, seed)
              )
