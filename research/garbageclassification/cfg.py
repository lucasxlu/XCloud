from collections import OrderedDict

cfg = OrderedDict()

cfg['garbage_classification_root'] = '/home/xulu/DataSet/garbage_classify'
cfg['batch_size'] = 64
cfg['epoch'] = 200
cfg['init_lr'] = 0.01
cfg['weight_decay'] = 1e-4
cfg['lr_decay_step'] = 80
cfg['out_num'] = 102
