from collections import OrderedDict

cfg = OrderedDict()

cfg['root'] = '/home/xulu/Project/nsfw_data_scrapper/raw_data'
cfg['batch_size'] = 64
cfg['epoch'] = 200
cfg['init_lr'] = 0.001
cfg['weight_decay'] = 1e-4
cfg['lr_decay_step'] = 50
cfg['out_num'] = 5
