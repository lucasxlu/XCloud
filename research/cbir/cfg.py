from collections import OrderedDict

cfg = OrderedDict()
cfg['tissue_physiology_img_base'] = '/data/lucasxu/Dataset/TissuePhysiologySku'
cfg['light_clothing_img_base'] = '/data/lucasxu/Dataset/LightClothingSku'
cfg['epoch'] = 300
cfg['init_lr'] = 0.01
cfg['lr_decay_step'] = 50
cfg['weight_decay'] = 1e-4
cfg['out_num'] = 51
# cfg['out_num'] = 162
cfg['batch_size'] = 128
cfg['data_aug_samples'] = 1000
