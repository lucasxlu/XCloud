import sys
import copy
from collections import defaultdict
import random

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

sys.path.append('../')
from research.cbir.cfg import cfg
from research.cbir.datasets import ImageDataset


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, skucode).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size=cfg['batch_size'], num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def load_imagedataset_data():
    """
    load ImageDataset Crop dataset
    :return:
    """
    batch_size = cfg['batch_size']
    print('loading ImageDataset...')
    train_dataset = ImageDataset(type='train',
                                 transform=transforms.Compose([
                                     transforms.Resize((224, 224)),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.1,
                                                            hue=0.1),
                                     transforms.RandomRotation(90),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     transforms.RandomErasing(p=0.5, scale=(0.1, 0.3), value='random')
                                 ]))
    # trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=50, drop_last=True,
    #                          pin_memory=True)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=RandomIdentitySampler(
        data_source=[[f, t] for f, t in zip(train_dataset.filelist, train_dataset.typelist)]), num_workers=50,
                             drop_last=True, pin_memory=True)

    val_dataset = ImageDataset(type='val',
                               transform=transforms.Compose([
                                   transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ]))
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=True,
                           pin_memory=True)

    test_dataset = ImageDataset(type='test',
                                transform=transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=50, drop_last=True,
                            pin_memory=True)

    return trainloader, valloader, testloader
