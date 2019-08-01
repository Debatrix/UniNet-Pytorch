# coding=utf-8
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import savemat
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from dataset import BaseDataset


class LoadConfig(object):
    def __init__(self):
        self.dataset_path = '/home/dl/wangleyuan/dataset'
        self.dataset = 'CASIA-Iris-Thousand'
        self.model = 'CASIA'

        self.mode = 'test'
        self.save = 'mat'

        self.batch_size = 16
        self.device = "cuda:0"
        self.num_workers = 2

        self._change_cfg()

    def _change_cfg(self):
        parser = ArgumentParser()
        for name, value in vars(self).items():
            parser.add_argument('--' + name, type=type(value), default=value)
        args = parser.parse_args()

        for name, value in vars(args).items():
            if self.__dict__[name] != value:
                self.__dict__[name] = value

        self.featnet_path = 'models/UniNet_' + self.model + '_FeatNet.pth'
        self.masknet_path = 'models/UniNet_' + self.model + '_MaskNet.pth'

    def __str__(self):
        config = ""
        for name, value in vars(self).items():
            config += ('%s=%s\n' % (name, value))
        return config


def extraction(cfg):
    # cpu or gpu?
    if torch.cuda.is_available() and cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

    dataset = BaseDataset(path=cfg.dataset_path, dataset=cfg.dataset, mode=cfg.mode)
    data_loader = DataLoader(
        dataset,
        cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers)

    featnet = model.FeatNet()
    featnet.load_state_dict(torch.load(cfg.featnet_path, map_location=device))
    featnet.to(device)
    masknet = model.MaskNet()
    masknet.load_state_dict(torch.load(cfg.masknet_path, map_location=device))
    masknet.to(device)

    with torch.no_grad():
        featnet.eval()
        masknet.eval()
        if not os.path.exists('feature/{}'.format(cfg.dataset)):
            os.makedirs('feature/{}'.format(cfg.dataset))
        for img_batch, _, label_vec_batch, img_name_batch in tqdm(
                data_loader, ncols=80, ascii=True):
            img_batch = img_batch.to(device)
            feature_batch = featnet(img_batch)
            mask_batch = masknet(img_batch)
            for idx, img_name in enumerate(img_name_batch):
                feature = feature_batch[idx].cpu().numpy()
                label_vec = label_vec_batch.cpu().numpy()
                mask = F.softmax(mask_batch[idx], dim=0).cpu().numpy()
                mask = (mask[0] < mask[1]).astype(np.bool)
                if cfg.save == 'pic':
                    feature_img = (feature - feature.min()) / (feature.max() - feature.min())
                    Image.fromarray(feature_img * 255).convert('L').save(
                        'feature/{}/{}_feature.png'.format(cfg.dataset, img_name))
                    Image.fromarray(mask * 255).convert('L').save(
                        'feature/{}/{}_mask.png'.format(cfg.dataset, img_name))
                else:
                    ft_load = {'feature': feature, 'mask': mask, 'label_vec': label_vec}
                    savemat('feature/{}/{}.mat'.format(cfg.dataset, img_name), ft_load)


if __name__ == '__main__':
    config = LoadConfig()
    extraction(config)
