# try to compose a stand-alone script for CenterNet pth->onnx->tensorrt test.

import torch
import cv2
import numpy as np

from lib.models.networks.resnet import get_pose_net as get_pose_net_resnet


def get_model(num_layers, head, head_conv):
    model = get_pose_net_resnet(num_layers, head, head_conv)
    return model


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == '__main__':
    layers = 18  # Resnet18.
    head = {
      'hm': 80,
      'wh': 2,
      'reg': 2,
    }  # For detection task.
    head_conv = 64  # For detection task.
    pth_path = '/workspace/tangyang.sy/pytorch_CV/pytorch_CenterNet/models/ctdet_coco_resnet18.pth'
    img_path = '/workspace/tangyang.sy/pytorch_CV/test_imgs/combined_v2_20190301_cont_mod/Images/' \
               '20190117_all_0153.png'
    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

    model = get_model(
        num_layers=layers,
        head=head,
        head_conv=head_conv
    )

    print(model)

    model = load_model(model, pth_path)
    model.eval()

    img = cv2.imread(img_path)
    img = cv2.resize(img, (512, 512))
    inp_img = ((img / 255. - mean) / std).astype(np.float32)
    image = inp_img.transpose(2, 0, 1).reshape(1, 3, 512, 512)
    image = torch.from_numpy(image)

    res = model(image)[-1]
