# coding=utf-8
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat

import model
from util.img_enhance import enh_contrast


class LoadConfig(object):
    def __init__(self):
        self.dataset_path = '/home/dl/wangleyuan/dataset/CASIA-Iris-Thousand'

        self.mode = 'test'

        self.model = 'CASIA'
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

        self.dataset_path = os.path.normcase(self.dataset_path)
        if not self.dataset_name:
            self.dataset_name = os.path.split(self.dataset_path)[1]

    def __str__(self):
        config = ""
        for name, value in vars(self).items():
            config += ('%s=%s\n' % (name, value))
        return config


def get_img(img_path, label='unknown'):
    img_name = os.path.split(img_path)[1]
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (512, 64))
    img = enh_contrast(img).astype(np.float32) / 255
    img = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(img), 0), 0).to(torch.float32)
    return img, label, img_name


def extraction(img, model_name):
    # cpu or gpu?
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")

    featnet_path = 'models/UniNet_' + model_name + '_FeatNet.pth'
    masknet_path = 'models/UniNet_' + model_name + '_MaskNet.pth'

    featnet = model.FeatNet()
    featnet.load_state_dict(torch.load(featnet_path, map_location=device))
    featnet.to(device)
    masknet = model.MaskNet()
    masknet.load_state_dict(torch.load(masknet_path, map_location=device))
    masknet.to(device)

    with torch.no_grad():
        featnet.eval()
        masknet.eval()
        img = img.to(device)
        feature_batch = featnet(img)
        mask_batch = masknet(img)
        feature = torch.squeeze(feature_batch[0]).cpu().numpy()
        mask = F.softmax(mask_batch[0], dim=0).cpu().numpy()
        mask = (mask[0] < mask[1]).astype(np.bool)
        template = np.ones_like(feature).astype(np.bool)
        template[feature < feature.mean()] = False
        return template, mask


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img', '-i', type=str)
    parser.add_argument('--path', '-p', type=str, default='.')
    parser.add_argument('--model', '-m', type=str, default='ND')
    parser.add_argument('--label', '-l', type=str, default='unknown')
    parser.add_argument('--show', '-s', action="store_true")
    args = parser.parse_args()

    img, label, img_name = get_img(args.img, args.label)
    template, mask = extraction(img, args.model)
    if args.show:
        # cv2.imwrite(os.path.join(args.path, '{}_img.png'.format(img_name)), img * 255)
        cv2.imwrite(os.path.join(args.path, '{}_template.png'.format(img_name)), template * 255)
        cv2.imwrite(os.path.join(args.path, '{}_mask.png'.format(img_name)), mask * 255)
    else:
        ft_load = {'template': template, 'mask': mask, 'label': label}
        savemat(os.path.join(args.path, '{}.mat'.format(img_name)), ft_load)
