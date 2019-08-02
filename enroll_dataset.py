# coding=utf-8
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from util.img_enhance import enh_contrast


class LoadConfig(object):
    def __init__(self):
        self.dataset_path = '/home/dl/wangleyuan/dataset/CASIA-Iris-Thousand'

        self.mode = 'test'

        self.model = 'CASIA'
        self.save = 'mat'

        self.batch_size = 1
        self.device = "cuda:0"
        self.num_workers = 1

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

        self.dataset_path = os.path.normcase(self.dataset_path)
        self.dataset_name = os.path.split(self.dataset_path)[1]

    def __str__(self):
        config = ""
        for name, value in vars(self).items():
            config += ('%s=%s\n' % (name, value))
        return config


class Dataset(data.Dataset):
    def __init__(self, path, mode='test', seed=7548):
        np.random.seed(seed)

        self.path = path
        self.mode = mode

        # protocol containing the training/validation/test file
        with open(os.path.join(path, mode + '.txt'), 'r') as f:
            self.img_list = [
                tuple(line.strip().split(' ')) for line in f.readlines()
            ]
            label_list = sorted(list(set([item[1] for item in self.img_list])))

        np.random.shuffle(self.img_list)
        self.label_list = sorted(label_list)

    def __len__(self):
        return len(self.img_list)

    @property
    def class_num(self):
        return len(self.label_list)

    def __getitem__(self, item):
        img_name, label = self.img_list[item]
        img = cv2.imread(os.path.join(self.path, 'NormIm', img_name), 0)
        img = cv2.resize(img, (512, 64))
        img = enh_contrast(img).astype(np.float32) / 255
        img = torch.unsqueeze(torch.from_numpy(img), 0).to(torch.float32)
        return img, label, img_name


def extraction(cfg):
    # cpu or gpu?
    if torch.cuda.is_available() and cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

    dataset = Dataset(path=cfg.dataset_path, mode=cfg.mode)
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
        if not os.path.exists('feature/{}'.format(cfg.dataset_name)):
            os.makedirs('feature/{}'.format(cfg.dataset_name))
        for img_batch, label_batch, img_name_batch in tqdm(
                data_loader, ncols=80, ascii=True):
            img_batch = img_batch.to(device)
            feature_batch = featnet(img_batch)
            mask_batch = masknet(img_batch)
            for idx, img_name in enumerate(img_name_batch):
                img = torch.squeeze(img_batch[idx]).cpu().numpy()
                feature = torch.squeeze(feature_batch[idx]).cpu().numpy()
                template = np.ones_like(feature).astype(np.bool)
                template[feature < feature.mean()] = False
                tmask = (np.abs(feature - feature.mean()) > 0.6)
                mask = F.softmax(mask_batch[idx], dim=0).cpu().numpy()
                mask = np.logical_and((mask[0] < mask[1]).astype(np.bool), tmask)
                if cfg.save == 'pic':
                    cv2.imwrite('feature/{}/{}_img.png'.format(cfg.dataset_name, img_name), img * 255)
                    cv2.imwrite('feature/{}/{}_template.png'.format(cfg.dataset_name, img_name), template * 255)
                    cv2.imwrite('feature/{}/{}_mask.png'.format(cfg.dataset_name, img_name), mask * 255)
                else:
                    ft_load = {'template': template, 'mask': mask, 'label': label_batch[idx]}
                    savemat('feature/{}/{}.mat'.format(cfg.dataset_name, img_name), ft_load)


if __name__ == '__main__':
    config = LoadConfig()
    extraction(config)
