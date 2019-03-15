import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/Library')
import cv2
import yaml
import time
import datetime

from Library import Model
from Library import Dataset
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

from Viewpoint import compute_location


dims_avg = {
            'Cyclist': np.array([1.73532436, 0.58028152, 1.77413709]),
            'Van': np.array([2.18928571, 1.90979592, 5.07087755]),
            'Tram': np.array([3.56092896, 2.39601093, 18.34125683]),
            'Car': np.array([1.52159147, 1.64443089, 3.85813679]),
            'Pedestrian': np.array([1.75554637, 0.66860882, 0.87623049]),
            'Truck': np.array([3.07392252, 2.63079903, 11.2190799])
        }

WRITE_RESULT = False

if __name__ == '__main__':
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        print('No folder named \"models/\"')
        exit()

    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    path = config['kitti_path']
    epochs = config['epochs']
    batches = config['batches']
    bins = config['bins']
    alpha = config['alpha']
    w = config['w']
    result_path = config['result_path']

    data = Dataset.ImageDataset(path + '/training')
    data = Dataset.BatchDataset(data, batches, bins, mode='eval')
    
    if len(model_lst) == 0:
        print('No previous model found, please check it')
        exit()
    else:
        print('Find previous model %s' % model_lst[-1])
        vgg = vgg.vgg19_bn(pretrained=False)
        model = Model.Model(features=vgg.features, bins=bins, mode='test').cuda()
        params = torch.load(store_path + '/%s' % model_lst[-1])
        model.load_state_dict(params)
        model.eval()

    angle_error = []
    dimension_error = []
    for i in range(data.num_of_patch):
        batch, centerAngle, info = data.EvalBatch()
        dimGT = info['Dimension']
        angle = info['LocalAngle'] / np.pi * 180
        Ry = info['Ry']
        batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()

        [orient, conf, dim, vp] = model(batch)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]
        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        cos = orient[0]
        sin = orient[1]
        vp = vp.cpu().data.numpy()[0, :]
        vp = np.argmax(vp)

        # 计算角度
        theta = np.arctan2(sin, cos) / np.pi * 180
        theta = theta + centerAngle[argmax] / np.pi * 180
        theta = 360 - info['ThetaRay'] - theta
        if theta > 180: theta -= 360
        alpha = theta - (90 - info['ThetaRay'])

        dim_error = np.mean(abs(np.array(dimGT) - dim))
        dimension_error.append(dim_error)

        # 计算尺寸
        if info['Class'] in dims_avg:
            dim = dims_avg[info['Class']] + dim
        else:
            dim = dims_avg['Car'] + dim

        # 计算目标空间位置
        loc = compute_location(info['Box_2D'], dim, theta * np.pi / 180, info['Intrinsic'], vp)

        # write result to file
        if WRITE_RESULT:
            with open(result_path + '/' + info['ID'] + '.txt', 'a') as box_3d:
                line = [info['Class'],
                        info['Truncated'],
                        info['Occluded'],
                        alpha * np.pi / 180,  # Alpha
                        info['Box_2D_float'][0], info['Box_2D_float'][1], info['Box_2D_float'][2], info['Box_2D_float'][3],
                        dim[0], dim[1], dim[2],
                        loc[0, 0], loc[1, 0], loc[2, 0],
                        # info['Location'][0], info['Location'][1], info['Location'][2],
                        theta * np.pi / 180,  # Ry
                        conf[argmax],  # score
                        ]
                line = map(str, line)
                line = ' '.join(line) + '\n'
                box_3d.write(line)
        
        theta_error = abs(Ry - theta)
        if theta_error > 180: theta_error = 360 - theta_error
        angle_error.append(theta_error)
        
        #if i % 60 == 0:
        #    print (theta, Ry)
        #    print (dim.tolist(), dimGT)
        if i % 1000 == 0:
            now = datetime.datetime.now()
            now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
            print('------- %s %.5d -------'%(now_s, i))
            print('Angle error: %lf'%(np.mean(angle_error)))
            print('Dimension error: %lf'%(np.mean(dimension_error)))
            print('-----------------------------')

    print('Angle error: %lf'%(np.mean(angle_error)))
    print('Dimension error: %lf'%(np.mean(dimension_error)))
