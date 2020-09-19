from collections import OrderedDict

cfg = OrderedDict()
cfg['img_base'] = '/data/lucasxu/Dataset/ImageDataset'
cfg['epoch'] = 300
cfg['init_lr'] = 0.01
cfg['lr_decay_step'] = 50
cfg['weight_decay'] = 1e-4
cfg['out_num'] = 425
cfg['batch_size'] = 64
cfg['data_aug_samples'] = 1000