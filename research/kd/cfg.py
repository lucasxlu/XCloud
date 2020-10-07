from collections import OrderedDict

cfg = OrderedDict()
cfg['img_base'] = '/path/to/dir'
cfg['use_lsr'] = True
cfg['epoch'] = 250
cfg['init_lr'] = 0.01
cfg['lr_decay_step'] = 70
cfg['weight_decay'] = 1e-5
cfg['out_num'] = 75
cfg['batch_size'] = 64
