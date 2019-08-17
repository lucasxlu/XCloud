import os
import sys
import pickle

from PIL import Image
import numpy as np
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import gray2rgb, rgba2rgb
from torchvision import models
from torchvision.transforms import transforms

sys.path.append('../../')
from research.cbir import dataloaders

USE_GPU = True


def load_model_with_weights(model, model_path):
    print(model)
    model = model.float()
    model_name = model.__class__.__name__
    device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if model_path is not None and model_name != "":
        model.load_state_dict(torch.load(model_path))
    model.eval()

    if torch.cuda.device_count() > 1:
        model = model.module

    return model


def ext_deep_feat(model_with_weights, img_filepath):
    """
    extract deep feature from an image filepath
    :param model_with_weights:
    :param img_filepath:
    :return:
    """
    model_name = model_with_weights.__class__.__name__
    device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
    model_with_weights = model_with_weights.to(device)
    model_with_weights.eval()
    if model_name.startswith('DenseNet'):
        # print('extracting deep features of {}...'.format(model_name))

        with torch.no_grad():
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = io.imread(img_filepath)
            if len(list(image.shape)) < 3:
                image = gray2rgb(image)
            elif len(list(image.shape)) > 3:
                image = rgba2rgb(image)

            img = preprocess(Image.fromarray(image.astype(np.uint8)))
            img.unsqueeze_(0)
            img = img.to(device)

            inputs = img.to(device)

            feat = model_with_weights.features(inputs)
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)

            # print('feat size = {}'.format(feat.shape))

    return feat.to('cpu').detach().numpy()


def batch_ext_deep_feats(model, dataloader, model_path):
    """
    [UNFINISHED] batch extract deep features
    Note: it may brings you some trouble!!!
    :param model:
    :param dataloader:
    :param model_path:
    :param: batch_size:
    :return:
    """
    print(model)
    model = model.float()
    model_name = model.__class__.__name__
    device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if model_path is not None and model_name != "":
        model.load_state_dict(torch.load(model_path))
    model.eval()

    idx_filename = {}

    if model_name.startswith('DenseNet'):
        print('[INFO] extracting deep features of {}...'.format(model_name))
        feats = torch.FloatTensor().to(device)

        with torch.no_grad():
            for bat_id, data in enumerate(dataloader):
                images, filename, idx = data['image'], data['filename'], data['idx']
                inputs = images.to(device)

                for i in range(len(idx)):
                    idx_filename[int(idx[i])] = filename[i]

                if torch.cuda.device_count() > 1:
                    model_ext = model.module
                else:
                    model_ext = model

                batch_feat = model_ext.features(inputs)
                feat = F.relu(batch_feat, inplace=True)
                feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(batch_feat.size(0), -1)

                # for idx, module in model_ext.features.named_children():
                #     batch_feat = torch.FloatTensor().to(device)
                #     inputs = module.forward(inputs)
                #     if idx in layers:
                #         print('Concatenating {} features!'.format(idx))
                #         # reshape from N*C*W*H to N*CWH
                #         print('batch_feat size = {}'.format(inputs.shape))
                #         batch_feat = torch.cat((batch_feat, inputs.view(batch_size, -1)), dim=1)

                print(
                    '[INFO] {0}/{1} extracting deep features, feat size = {2}'.format(bat_id, dataloader.__len__(),
                                                                                      feat.shape))
                feats = torch.cat((feat, feats), dim=0)
            print('feats size = {}'.format(feats.shape))

        print('Finish extracting deep features.\nSerializing to .pkl file...')
        with open('./feats.pkl', mode='wb') as f:
            pickle.dump(feats.to('cpu').detach().numpy(), f)

        with open('./filenames.pkl', mode='wb') as f:
            pickle.dump(idx_filename, f)

        return feats


def ext_feats_in_dir(model_with_weights, dir_name):
    """
    extract deep features in a directory
    :param model_with_weights:
    :param dir_name:
    :return:
    """
    print('[INFO] start extracting features')
    idx_filename = {}
    feats = []
    for i, f in enumerate(sorted(os.listdir(dir_name))):
        feat = ext_deep_feat(model_with_weights, os.path.join(dir_name, f))
        print(
            '[INFO] {0}/{1} extracting deep features, feat size = {2}'.format(i, len(os.listdir(dir_name)), feat.shape))
        idx_filename[i] = os.path.join(dir_name, f)
        feats.append(feat.ravel().tolist())
    print('[INFO] finish extracting features')

    with open('./feats.pkl', mode='wb') as f:
        pickle.dump(np.array(feats).astype('float32'), f)

    with open('./filenames.pkl', mode='wb') as f:
        pickle.dump(idx_filename, f)


if __name__ == '__main__':
    densenet121 = models.densenet121(pretrained=True)
    # batch_ext_deep_feats(model=densenet121, dataloader=dataloaders.load_patch_dataset(), model_path=None)
    ext_feats_in_dir(densenet121, '/home/xulu/DataSet/LightClothingCrops/LightClothing')

    # feat = ext_deep_feat(densenet121,
    #                      img_filepath='/data/lucasxu/Dataset/TissuePhysiologyCrops/TissuePhysiology/b624d45a22b98e34f8a44071.orig_TissuePhysiology_3.jpg')
    # print(feat)
