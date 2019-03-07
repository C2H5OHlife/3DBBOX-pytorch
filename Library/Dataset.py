import os
import sys
import cv2
import glob
import numpy as np
from Viewpoint import viewpoint
import random


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            if line == '\n': break
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data


class ImageDataset:
    def __init__(self, path):
        self.img_path = path + '/image_2'
        self.label_path = path + '/label_2'
        self.calib_path = path + '/calib'

        self.IDLst = [x.split('.')[0] for x in sorted(os.listdir(self.img_path))]  # 图像list

    def __getitem__(self, index):  # 返回一张图的解析（多个sample）
        tmp = {}
        # img = cv2.imread(self.img_path + '/%s.png'%self.IDLst[index], cv2.IMREAD_COLOR)
        with open(self.label_path + '/%s.txt'%self.IDLst[index], 'r') as f:
            calib_data = read_calib_file(self.calib_path + '/%s.txt'%self.IDLst[index])
            buf = []
            for line in f:
                line = line[:-1].split(' ')
                for i in range(1, len(line)):
                    line[i] = float(line[i])
                Class = line[0]
                Truncated = int(line[1])
                Occluded = int(line[2])
                Alpha = line[3] / np.pi * 180  # 这个角度根本没用到 (与LocalAngle有关)
                Ry = line[14] / np.pi * 180  # 目标的全局朝向theta
                top_left = (int(round(line[4])), int(round(line[5])))  # (x, y)
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]
                Dimension = [line[8], line[9], line[10]]  # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                # 目标中心反射光线射入镜头的角度theta_ray （Alpha + ThetaRay = 90°） 在x-z平面内计算
                ThetaRay = (np.arctan2(Location[2], Location[0])) / np.pi * 180
                #if Ry > 0:
                #    LocalAngle = (180 - Ry) + (180 - ThetaRay)
                #else:
                #    LocalAngle = 360 - (ThetaRay + Ry)
                LocalAngle = 360 - (ThetaRay + Ry)  # 参考穿过图像块中心光线的局部朝向theta_l
                if LocalAngle > 360:
                    LocalAngle -= 360
                #LocalAngle = Ry - ThetaRay
                LocalAngle = LocalAngle / 180 * np.pi
                v = viewpoint(Dimension, Location, line[14], calib_data['P2'].reshape(3, 4))  # Ry需要输入弧度
                if LocalAngle < 0:
                    LocalAngle += 2 * np.pi
                buf.append({
                        'Class': Class,
                        'Truncated': Truncated,
                        'Occluded': Occluded,
                        'Box_2D': Box_2D,
                        'Dimension': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry,
                        'ThetaRay': ThetaRay,
                        'LocalAngle': LocalAngle,
                        'Viewpoint': v,
                        'Intrinsic': calib_data['P2'].reshape(3, 4)
                    })
        tmp['ID'] = self.IDLst[index]
        tmp['Label'] = buf
        return tmp

    def GetImage(self, idx):
        name = '%s/%s.png'%(self.img_path, self.IDLst[idx])
        # 归一化
        img = cv2.imread(name, cv2.IMREAD_COLOR).astype(np.float)
        if random.random() < 0.5:
            img = np.clip((random.uniform(0.5, 1.5) * img + random.uniform(0, 20)), 0, 255)
        img = img / 255
        img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229
        return img

    def __len__(self):
        return len(self.IDLst)


class BatchDataset:
    def __init__(self, imgDataset, batchSize=1, bins=3, overlap=25/180.0*np.pi, mode='train'):
        self.imgDataset = imgDataset
        self.batchSize = batchSize
        self.bins = bins
        self.overlap = overlap
        self.mode = mode
        self.imgID = None
        centerAngle = np.zeros(bins)  # 每个bin的中心角度
        interval = 2 * np.pi / bins  # 每个bin的跨度
        for i in range(1, bins):
            centerAngle[i] = i * interval
        self.centerAngle = centerAngle
        # print centerAngle / np.pi * 180
        self.intervalAngle = interval
        self.dims_avg = {
            'Cyclist': [1.73532436, 0.58028152, 1.77413709],
            'Van': [2.18928571, 1.90979592, 5.07087755],
            'Tram': [3.56092896, 2.39601093, 18.34125683],
            'Car': [1.52159147, 1.64443089, 3.85813679],
            'Pedestrian': [1.75554637, 0.66860882, 0.87623049],
            'Truck': [3.07392252, 2.63079903, 11.2190799]
        }

        self.info = self.getBatchInfo()
        self.Total = len(self.info)
        print('%d samples used!' % self.Total)

        # 样本id
        if mode == 'train':
            self.idx = 0
            self.num_of_patch = 35570
        else:
            self.idx = 0
            self.num_of_patch = self.Total
            # self.idx = 35570
            # self.num_of_patch = self.Total - 35570
        #print len(self.info)
        #print self.info
    def getBatchInfo(self):
        #
        # get info of all crop image
        #   
        data = []
        total = len(self.imgDataset)
        centerAngle = self.centerAngle
        intervalAngle = self.intervalAngle
        for idx, one in enumerate(self.imgDataset):
            ID = one['ID']
            #img = one['Image']
            allLabel = one['Label']
            for label in allLabel:
                if label['Class'] != 'DontCare' and (not (label['Viewpoint'] == -1 and self.mode == 'train')):
                    #crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
                    LocalAngle = label['LocalAngle']
                    confidence = np.zeros(self.bins)  # 给每个bin算confidence（无overlap）
                    confidence_multi = np.zeros(self.bins)  # 给每个bin算confidence（有overlap）
                    for i in range(self.bins):
                        diff = abs(centerAngle[i] - LocalAngle)
                        if diff > np.pi:
                            diff = 2 * np.pi - diff
                        if diff <= intervalAngle / 2 + self.overlap:  # 落入bin内置信度为1，否则为0
                            confidence_multi[i] = 1
                        if diff < intervalAngle / 2:
                            confidence[i] = 1
                    n = np.sum(confidence)
                    if label['Class'] in self.dims_avg:
                        for i in range(len(label['Dimension'])):
                            label['Dimension'][i] -= self.dims_avg[label['Class']][i]
                    else:
                        for i in range(len(label['Dimension'])):
                            label['Dimension'][i] -= self.dims_avg['Car'][i]
                    data.append({
                                'ID': ID, # img ID
                                'Index': idx, # id in Imagedataset
                                'Box_2D': label['Box_2D'],
                                'Truncated': label['Truncated'],
                                'Occluded': label['Occluded'],
                                'Dimension': label['Dimension'],
                                'Location': label['Location'],
                                'Class': label['Class'],
                                'Alpha': label['Alpha'],
                                'LocalAngle': LocalAngle,
                                'Confidence': confidence,
                                'ConfidenceMulti': confidence_multi,
                                'Ntheta': n,
                                'Ry': label['Ry'],
                                'ThetaRay': label['ThetaRay'],
                                'Viewpoint': label['Viewpoint'],
                                'Intrinsic': label['Intrinsic'],
                            })
        return data

    def Next(self):
        batch = np.zeros([self.batchSize, 3, 224, 224], np.float) 
        confidence = np.zeros([self.batchSize, self.bins], np.float)
        confidence_multi = np.zeros([self.batchSize, self.bins], np.float)
        ntheta = np.zeros(self.batchSize, np.float)
        angleDiff = np.zeros([self.batchSize, self.bins], np.float)  # 每个bin都要计算一个bin中心角度到真实角度的差值
        dim = np.zeros([self.batchSize, 3], np.float)
        viewpoint = np.zeros([self.batchSize], np.int)
        # vp_one_hot = np.zeros([self.batchSize, 16], np.float64)
        record = None
        for one in range(self.batchSize):  # 采够batch_size个样本
            data = self.info[self.idx]
            imgID = data['Index']
            if imgID != record:
                # 用GetImage方法得到的是归一化的图像
                img = self.imgDataset.GetImage(imgID)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #cv2.namedWindow('GG')
                #cv2.imshow('GG', img)
                #cv2.waitKey(0)
            pt1 = data['Box_2D'][0]
            pt2 = data['Box_2D'][1]
            crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]  # crop出目标
            # cv2.imshow('crop', crop)
            # cv2.waitKey()

            # original
            crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            batch[one, 0, :, :] = crop[:, :, 2]
            batch[one, 1, :, :] = crop[:, :, 1]
            batch[one, 2, :, :] = crop[:, :, 0]

            # modified
            # padding or masking
            # offset_x = (self.max_width - crop.shape[1]) // 2
            # offset_y = (self.max_height - crop.shape[0]) // 2
            # mask_crop = np.zeros(shape=(self.max_height, self.max_width, 3), dtype=np.float64)
            # mask_crop[offset_y:offset_y + crop.shape[0], offset_x:offset_x + crop.shape[1]] = crop
            # mask_crop = cv2.resize(src=mask_crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)  # resize到224 x 224
            # batch[one, 0, :, :] = mask_crop[:, :, 2]  # 注意OpenCV的channel在最后一唯，PyTorch在第1维
            # batch[one, 1, :, :] = mask_crop[:, :, 1]
            # batch[one, 2, :, :] = mask_crop[:, :, 0]
            confidence[one, :] = data['Confidence'][:]
            confidence_multi[one, :] = data['ConfidenceMulti'][:]
            # confidence[one, :] /= np.sum(confidence[one, :])
            ntheta[one] = data['Ntheta']
            # 如果angleDiff > 180°，其实应该处理成angle = angle - 360，但是因为正余弦以360°为周期，所以没关系
            angleDiff[one, :] = data['LocalAngle'] - self.centerAngle
            dim[one, :] = data['Dimension']
            viewpoint[one] = data['Viewpoint']

            # 样本id + 1
            if self.mode == 'train':
                if self.idx + 1 < self.num_of_patch:
                    self.idx += 1
                else:
                    self.idx = 0  # 循环使用
            else:
                if self.idx + 1 < self.Total:
                    self.idx += 1
                else:
                    self.idx = 35570
        # vp_one_hot[np.arange(self.batchSize), viewpoint[:,0]] = 1
        return batch, confidence, confidence_multi, angleDiff, dim, viewpoint

    def EvalBatch(self):
        batch = np.zeros([1, 3, 224, 224], np.float)
        info = self.info[self.idx]
        imgID = info['Index']
        if imgID != self.imgID:
            self.img = self.imgDataset.GetImage(imgID)
            self.imgID = imgID
        pt1 = info['Box_2D'][0]
        pt2 = info['Box_2D'][1]
        crop = self.img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]

        # original
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        batch[0, 0, :, :] = crop[:, :, 2]
        batch[0, 1, :, :] = crop[:, :, 1]
        batch[0, 2, :, :] = crop[:, :, 0]
        # modified
        # padding or masking
        # offset_x = (self.max_width - crop.shape[1]) // 2
        # offset_y = (self.max_height - crop.shape[0]) // 2
        # mask_crop = np.zeros(shape=(self.max_height, self.max_width, 3), dtype=np.float64)
        # mask_crop[offset_y:offset_y + crop.shape[0], offset_x:offset_x + crop.shape[1]] = crop
        #
        # mask_crop = cv2.resize(src=mask_crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)  # resize到224 x 224
        # batch[0, 0, :, :] = mask_crop[:, :, 2]
        # batch[0, 1, :, :] = mask_crop[:, :, 1]
        # batch[0, 2, :, :] = mask_crop[:, :, 0]

        if self.mode == 'train':
            if self.idx + 1 < self.num_of_patch:
                self.idx += 1
            else:
                self.idx = 0
        else:
            if self.idx + 1 < self.Total:
                self.idx += 1
            else:
                self.idx = 35570
        return batch, self.centerAngle, info 


if __name__ == '__main__':
    import yaml
    with open('../config.yaml', 'r') as f:
        config = yaml.load(f)
    path = config['kitti_path']
    batches = 1  # batch size
    bins = config['bins']

    # 如此老旧的写法
    data = ImageDataset(path + '/training')
    data = BatchDataset(data, batches, bins)
    iter_each_time = round(float(data.num_of_patch) / batches)
    for i in range(iter_each_time):
        batch, confidence, confidence_multi, angleDiff, dimGT, vp = data.Next()
        print(vp)
        break
