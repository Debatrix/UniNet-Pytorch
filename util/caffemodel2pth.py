from glob import glob

import caffe
import torch

from model import FeatNet, MaskNet

if __name__ == '__main__':
    test_tensor = torch.rand((1, 1, 64, 512))
    for name in glob('ICCV17_release/models/*.caffemodel'):
        caffe_net = caffe.Net(
            'ICCV17_release/models/UniNet_deploy.prototxt', 1, weights=name)

        print(name)
        feat_net = FeatNet()
        feat_dict = feat_net.state_dict()
        for layer in feat_dict.keys():
            layer_name, t = layer.split('.')[-2:]
            t = 0 if t == 'weight' else 1
            print(layer, feat_dict[layer].shape,
                  torch.from_numpy(caffe_net.params[layer_name][t].data).shape)
            feat_dict[layer] = torch.from_numpy(
                caffe_net.params[layer_name][t].data)
        feat_net.load_state_dict(feat_dict)
        print(feat_net(test_tensor).shape)
        torch.save(feat_dict, name.split('.')[0] + '_FeatNet.pth')

        mask_net = MaskNet()
        mask_dict = mask_net.state_dict()
        for layer in mask_dict.keys():
            layer_name, t = layer.split('.')[-2:]
            t = 0 if t == 'weight' else 1
            print(layer, mask_dict[layer].shape,
                  torch.from_numpy(caffe_net.params[layer_name][t].data).shape)
            mask_dict[layer] = torch.from_numpy(
                caffe_net.params[layer_name][t].data)
        mask_net.load_state_dict(mask_dict)
        print(mask_net(test_tensor).shape)
        torch.save(mask_dict, name.split('.')[0] + '_MaskNet.pth')
