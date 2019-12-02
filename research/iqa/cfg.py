from collections import OrderedDict

cfg = OrderedDict()
cfg['iqa_img_base'] = '/data/lucasxu/Dataset/DeblurDataset'
cfg['epoch'] = 300
cfg['init_lr'] = 1e-2
cfg['lr_decay_step'] = 50
cfg['weight_decay'] = 1e-4
cfg['out_num'] = 2
cfg['batch_size'] = 64
