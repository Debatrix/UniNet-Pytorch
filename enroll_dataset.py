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
from evaluation import MatchBinary


class LoadConfig(object):
    def __init__(self):
        self.dataset_path = '/home/leyuan.wang/dataset/CASIA-Complex-CX3'

        self.mode = 'test'
        self.is_eval = False
        self.shift_bits = 10

        self.model = 'CASIA'
        self.save = 'pth'

        self.batch_size = 32
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
    print('\nload model and dataset...')
    # cpu or gpu?
    if torch.cuda.is_available() and cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

    dataset = Dataset(path=cfg.dataset_path, mode=cfg.mode)
    data_loader = DataLoader(dataset,
                             cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.num_workers)

    featnet = model.FeatNet()
    featnet.load_state_dict(torch.load(cfg.featnet_path, map_location=device))
    featnet.to(device)
    masknet = model.MaskNet()
    masknet.load_state_dict(torch.load(cfg.masknet_path, map_location=device))
    masknet.to(device)

    print('\nfeature extraction...')
    feature_dict = {}
    with torch.no_grad():
        featnet.eval()
        masknet.eval()
        if not os.path.exists('feature/{}'.format(cfg.dataset_name)):
            os.makedirs('feature/{}'.format(cfg.dataset_name))
        for img_batch, label_batch, img_name_batch in tqdm(data_loader,
                                                           ncols=80,
                                                           ascii=True):
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
                mask = np.logical_and((mask[0] < mask[1]).astype(np.bool),
                                      tmask)
                feature_save = {
                    'template': template,
                    'mask': mask,
                    'label': label_batch[idx]
                }
                feature_dict[img_name] = feature_save
                if cfg.save == 'pic':
                    cv2.imwrite(
                        'feature/{}/{}_img.png'.format(cfg.dataset_name,
                                                       img_name), img * 255)
                    cv2.imwrite(
                        'feature/{}/{}_template.png'.format(
                            cfg.dataset_name, img_name), template * 255)
                    cv2.imwrite(
                        'feature/{}/{}_mask.png'.format(
                            cfg.dataset_name, img_name), mask * 255)
                elif cfg.save == 'pth':
                    pass
                elif cfg.save == 'mat':
                    savemat(
                        'feature/{}/feature_UniNet_{}_{}.mat'.format(
                            cfg.dataset_name, cfg.model, img_name),
                        feature_save)
                else:
                    raise NotImplementedError
    if cfg.save == 'pth':
        torch.save(
            feature_dict,
            'feature/feature_UniNet_{}_{}.pth'.format(cfg.model,
                                                      cfg.dataset_name))

    if cfg.is_eval:
        print('\nevaluate...')
        data_feature = {'features': [], 'masks': [], 'labels': []}
        for v in feature_dict.values():
            data_feature['features'].append(torch.tensor(v['template']))
            data_feature['labels'].append(v['label'])
            data_feature['masks'].append(torch.tensor(v['mask']))
        data_feature['features'] = torch.stack(data_feature['features'], 0)
        data_feature['masks'] = torch.stack(data_feature['masks'], 0)

        FAR, FRR, T, EER, T_eer, FNMR_FMR, acc_rank1, acc_rank5, acc_rank10 = MatchBinary(
            data_feature, cfg.shift_bits, cfg.batch_size)
        DET_data = dict(FAR=FAR,
                        FRR=FRR,
                        T=T,
                        EER=EER,
                        T_eer=T_eer,
                        FNMR_FMR=FNMR_FMR,
                        acc_rank1=acc_rank1,
                        acc_rank5=acc_rank5,
                        acc_rank10=acc_rank10)

        savemat(
            'feature/evaluation_UniNet_{}_{}.mat'.format(
                cfg.model, cfg.dataset_name), DET_data)
        print('-' * 50)
        print('\nEER:{:.4f}%\nAcc: rank1 {:.4f}% rank5 {:.4f}% rank10 {:.4f}%'.
              format(EER * 100, acc_rank1 * 100, acc_rank5 * 100,
                     acc_rank10 * 100))
        print('-' * 50)
        for fmr, fnmr in FNMR_FMR.items():
            print('FNMR:{:.2f}%% @FMR:{:.2f}%%'.format(100.0 * fnmr,
                                                       100.0 * fmr))
        print('-' * 50)


if __name__ == '__main__':
    config = LoadConfig()
    extraction(config)
