# coding=utf-8
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from dataset import BaseDataset


class LoadConfig(object):
    def __init__(self):
        self.dataset_path = 'E:/dataset'
        self.dataset = 'UniNet-test'
        self.model = 'CASIA'

        self.mode = 'test'
        self.save = 'pic'

        self.batch_size = 1
        self.device = "cpu"
        self.num_workers = 4

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
        labels = []
        img_names = []
        labels_vec = np.zeros((len(dataset), dataset.class_num))
        features = np.zeros((len(dataset), 64, 512))
        masks = np.zeros((len(dataset), 3, 64, 512))
        didx = -1
        for didx, (img_batch, label_batch, label_vec_batch, img_name_batch) in tqdm(
                enumerate(data_loader), ncols=80, ascii=True):
            img_batch = img_batch.to(device)
            feature_batch = featnet(img_batch)
            mask_batch = masknet(img_batch)
            for idx in range(feature_batch.shape[0]):
                labels_vec[idx + didx, :] = label_vec_batch[idx, :].numpy()
                labels.append(label_batch[idx])
                img_names.append(img_name_batch[idx])
                features[idx + didx, :, :] = feature_batch[idx].cpu().numpy()
                masks[idx + didx, :2, :, :] = mask_batch[idx].cpu().numpy()
                mask = F.softmax(mask_batch[idx], dim=0).cpu().numpy()
                masks[idx + didx, 2, :, :] = mask[0] < mask[1]
        if cfg.save == 'pth':
            ft_path = 'feature/UniNet_{}__{}.pth'.format(cfg.model, cfg.dataset)
            ft_load = {'features': features, 'masks': masks, 'labels_vec': labels_vec, 'labels': labels}
            torch.save(ft_load, ft_path)
        elif cfg.save == 'pic':
            if not os.path.exists('feature/{}'.format(cfg.dataset)):
                os.makedirs('feature/{}'.format(cfg.dataset))
            for idx in range(len(dataset)):
                feature_img = features[idx, :, :]
                feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
                Image.fromarray(feature_img * 255).convert('L').save(
                    'feature/{}/{}_feature.png'.format(cfg.dataset, img_names[idx]))
                Image.fromarray(masks[idx, 2, :, :] * 255).convert('L').save(
                    'feature/{}/{}_mask.png'.format(cfg.dataset, img_names[idx]))

    return features, masks, labels, labels_vec


if __name__ == '__main__':
    config = LoadConfig()
    extraction(config)
