import os

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from torch.utils import data
from torchvision.transforms import transforms


class BaseDataset(data.Dataset):
    def __init__(self, path, dataset, mode='train', seed=7548):
        np.random.seed(seed)

        self.path = os.path.join(path, dataset, 'NormIm')
        self.mode = mode

        self.enc = None

        with open(os.path.join(path, dataset, mode + '.txt'), 'r') as f:
            self.img_list = [
                tuple(line.strip().split(' ')) for line in f.readlines()
            ]
            label_list = sorted(list(set([item[1] for item in self.img_list])))

        np.random.shuffle(self.img_list)
        self.label_list = sorted(label_list)

        self.enc = LabelBinarizer()
        self.enc.fit(self.label_list)

        self.transform = transforms.Compose([
            transforms.Resize((64, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_list)

    @property
    def class_num(self):
        return len(self.label_list)

    def __getitem__(self, item):
        img_name, label = self.img_list[item]
        img = Image.open(os.path.join(
            self.path, img_name)).convert("L")
        img = self.transform(img)
        label_vec = self.enc.transform([label])
        return img, label, label_vec, img_name
