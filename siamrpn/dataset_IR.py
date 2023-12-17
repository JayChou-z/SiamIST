# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import torch
import random
import numpy as np
import os.path as osp

from got10k.datasets.ir_train import IR_train
from got10k.datasets.seqs_22_train import seqs_22_train
from .utils import *
from PIL import Image
from .config import config
from torch.utils.data import Dataset
from got10k.datasets import ImageNetVID, GOT10k
from torchvision import datasets, transforms, utils
from .transforms import Normalize, ToTensor, RandomStretch,RandomCrop, CenterCrop, RandomBlur, ColorAug

class IR_Dataset(Dataset):
    def __init__(self, seq_dataset, z_transforms, x_transforms, name = 'NUDT-SIRST'):

        self.max_inter     = config.frame_range_got #100 ?
        self.z_transforms  = z_transforms
        self.x_transforms  = x_transforms
        self.sub_class_dir = seq_dataset   #22 sequence
        self.ret           = {}
        self.count         = 0
        self.index         = 3000
        self.name          = name
        self.anchors       = generate_anchors( config.total_stride,
                                                    config.anchor_base_size,
                                                    config.anchor_scales,
                                                    config.anchor_ratios,
                                                    config.score_size)

    def _pick_img_pairs(self, index_of_subclass):  #index 1-18 =》5

        assert index_of_subclass < len(self.sub_class_dir), 'index_of_subclass should less than total classes'

        video_name = self.sub_class_dir[index_of_subclass][0]   #图片一个序列的（index） 标注

        video_num  = len(video_name)  #第index_of_subclass+1个序列的图片量
        video_gt   = self.sub_class_dir[index_of_subclass][1]   #标签

        status = True
        while status:
            if self.max_inter >= video_num-1:    #最大 比其图片数大
                self.max_inter = video_num//2     #取其一半

            template_index = np.clip(random.choice(range(0, max(1, video_num - self.max_inter))), 0, video_num-1)   #0-301 选择任意如 100

            detection_index= np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0, video_num-1)    # 随机104

            template_img_path, detection_img_path  = video_name[template_index], video_name[detection_index]  #得到模板图片第100 和检测图片104  成对出现

            template_gt  = video_gt[template_index]   #相继获得标注

            detection_gt = video_gt[detection_index]

            if template_gt[2]*template_gt[3]*detection_gt[2]*detection_gt[3] != 0:   # w h 没问题
                status = False
            else:
                #print('Warning : Encounter object missing, reinitializing ...')
                print(  'index_of_subclass:', index_of_subclass, '\n',
                        'template_index:', template_index, '\n',
                        'template_gt:', template_gt, '\n',
                        'detection_index:', detection_index, '\n',
                        'detection_gt:', detection_gt, '\n')                #该序列的某个 有问题

        # load infomation of template and detection
        self.ret['template_img_path']      = template_img_path       #给ret  字典  图片 标注
        self.ret['detection_img_path']     = detection_img_path
        self.ret['template_target_x1y1wh'] = template_gt
        self.ret['detection_target_x1y1wh']= detection_gt
        t1, t2 = self.ret['template_target_x1y1wh'].copy(), self.ret['detection_target_x1y1wh'].copy()
        self.ret['template_target_xywh']   = np.array([t1[0]+t1[2]//2, t1[1]+t1[3]//2, t1[2], t1[3]], np.float32)   # 左上角 w,h  =》中心点
        self.ret['detection_target_xywh']  = np.array([t2[0]+t2[2]//2, t2[1]+t2[3]//2, t2[2], t2[3]], np.float32)
        self.ret['anchors'] = self.anchors    #17*17*5 4
        #self._average()

    def open(self):

        '''template'''
        #template_img = cv2.imread(self.ret['template_img_path']) if you use cv2.imread you can not open .JPEG format
        template_img = Image.open(self.ret['template_img_path']).convert('RGB')  #pil  读取到的是图片对象
        template_img = np.array(template_img)  #转为numpy的

        detection_img = Image.open(self.ret['detection_img_path']).convert('RGB')
        detection_img = np.array(detection_img)

        if np.random.rand(1) < config.gray_ratio:   #rand  0-1均匀分布   randn 服从均值0-方差1标准正态分布

            template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
            template_img = cv2.cvtColor(template_img, cv2.COLOR_GRAY2RGB)
            detection_img = cv2.cvtColor(detection_img, cv2.COLOR_RGB2GRAY)
            detection_img = cv2.cvtColor(detection_img, cv2.COLOR_GRAY2RGB)

        img_mean = np.mean(template_img, axis=(0, 1))    #256 256 3   轴为前两个 即对三通道 每一个通道求解均值
        #img_mean = tuple(map(int, template_img.mean(axis=(0, 1))))

        exemplar_img, scale_z, s_z, w_x, h_x = self.get_exemplar_image( template_img,
                                                                        self.ret['template_target_xywh'],
                                                                        config.exemplar_size,
                                                                        config.tem_amount, img_mean ) #模板图片 中心  127/255 0,。5 图片均值
        # resize后的图片  127/64  64  映射到原图后的宽高
        size_x = config.exemplar_size
        x1, y1 = int((size_x + 1) / 2 - w_x / 2), int((size_x + 1) / 2 - h_x / 2)
        x2, y2 = int((size_x + 1) / 2 + w_x / 2), int((size_x + 1) / 2 + h_x / 2)
        #frame = cv2.rectangle(exemplar_img, (x1,y1), (x2,y2), (0, 255, 0), 1)
        #cv2.imwrite('exemplar_img.png',frame)
        #cv2.waitKey(0)

        self.ret['exemplar_img'] = exemplar_img  #模板图片赋值

        '''detection'''    #检测图片
        #detection_img = cv2.imread(self.ret['detection_img_path'])
        d = self.ret['detection_target_xywh']
        cx, cy, w, h = d  # float type

        wc_z = w + 0.5 * (w + h)
        hc_z = h + 0.5 * (w + h)
        s_z = np.sqrt(wc_z * hc_z)

        s_x = s_z / (config.instance_size//2)
        img_mean_d = tuple(map(int, detection_img.mean(axis=(0, 1))))   #map 函数 《=序列   tuple 转换成元组

        a_x_ = np.random.choice(range(-12,12))
        a_x = a_x_ * s_x

        b_y_ = np.random.choice(range(-12,12))
        b_y = b_y_ * s_x

        instance_img, a_x, b_y, w_x, h_x, scale_x = self.get_instance_image(  detection_img, d,
                                                                    config.exemplar_size, # 127
                                                                    config.instance_size,# 255
                                                                    config.sch_amount,           # 0.5
                                                                    a_x, b_y,
                                                                    img_mean_d )

        size_x = config.instance_size

        x1, y1 = int((size_x + 1) / 2 - w_x / 2), int((size_x + 1) / 2 - h_x / 2)

        x2, y2 = int((size_x + 1) / 2 + w_x / 2), int((size_x + 1) / 2 + h_x / 2)

        #frame_d = cv2.rectangle(instance_img, (int(x1+(a_x*scale_x)),int(y1+(b_y*scale_x))), (int(x2+(a_x*scale_x)),int(y2+(b_y*scale_x))), (0, 255, 0), 1)
        #cv2.imwrite('detection_img_ori.png',frame_d)

        w  = x2 - x1
        h  = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2

        #print('[a_x_, b_y_, w, h]', [int(a_x_), int(b_y_), w, h])

        self.ret['instance_img'] = instance_img
        #self.ret['cx, cy, w, h'] = [int(a_x_*0.16), int(b_y_*0.16), w, h]
        self.ret['cx, cy, w, h'] = [int(a_x_), int(b_y_), w, h]

    def get_exemplar_image(self, img, bbox, size_z, context_amount, img_mean=None):
        cx, cy, w, h = bbox

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = size_z / s_z

        exemplar_img, scale_x = self.crop_and_pad_old(img, cx, cy, size_z, s_z, img_mean)   #中心点
        #127  127/18
        w_x = w * scale_x
        h_x = h * scale_x

        return exemplar_img, scale_z, s_z, w_x, h_x    #映射回去的宽高

    def get_instance_image(self, img, bbox, size_z, size_x, context_amount, a_x, b_y, img_mean=None):

        cx, cy, w, h = bbox  # float type

        #cx, cy = cx - a_x , cy - b_y
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z) # the width of the crop box

        scale_z = size_z / s_z

        s_x = s_z * size_x / size_z    #18*255/127
        instance_img, gt_w, gt_h, scale_x, scale_h, scale_w = self.crop_and_pad(img, cx, cy, w, h, a_x, b_y,  size_x, s_x, img_mean)
        w_x = gt_w #* scale_x #w * scale_x
        h_x = gt_h #* scale_x #h * scale_x

        #cx, cy = cx/ scale_w *scale_x, cy/ scale_h *scale_x
        #cx, cy = cx/ scale_w, cy/ scale_h
        a_x, b_y = a_x*scale_w, b_y*scale_h
        x1, y1 = int((size_x + 1) / 2 - w_x / 2), int((size_x + 1) / 2 - h_x / 2)
        x2, y2 = int((size_x + 1) / 2 + w_x / 2), int((size_x + 1) / 2 + h_x / 2)
        '''frame = cv2.rectangle(instance_img, (   int(x1+(a_x*scale_x)),
                                                int(y1+(b_y*scale_x))),
                                                (int(x2+(a_x*scale_x)),
                                                int(y2+(b_y*scale_x))),
                                                (0, 255, 0), 1)'''
        #cv2.imwrite('1.jpg', frame)
        return instance_img, a_x, b_y, w_x, h_x, scale_x

    def crop_and_pad(self, img, cx, cy, gt_w, gt_h, a_x, b_y, model_sz, original_sz, img_mean=None):

        #random = np.random.uniform(-0.15, 0.15)
        scale_h = 1.0 + np.random.uniform(-0.15, 0.15)   #取任意一个值
        scale_w = 1.0 + np.random.uniform(-0.15, 0.15)

        im_h, im_w, _ = img.shape

        xmin = (cx-a_x) - ((original_sz - 1) / 2)* scale_w    #左边*
        xmax = (cx-a_x) + ((original_sz - 1) / 2)* scale_w    #右边

        ymin = (cy-b_y) - ((original_sz - 1) / 2)* scale_h
        ymax = (cy-b_y) + ((original_sz - 1) / 2)* scale_h

        #print('xmin, xmax, ymin, ymax', xmin, xmax, ymin, ymax)

        left   = int(self.round_up(max(0., -xmin)))
        top    = int(self.round_up(max(0., -ymin)))
        right  = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))

        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))

        r, c, k = img.shape
        if any([top, bottom, left, right]):
            te_im_ = np.zeros((int((r + top + bottom)), int((c + left + right)), k), np.uint8)  # 0 is better than 1 initialization
            te_im = np.zeros((int((r + top + bottom)), int((c + left + right)), k), np.uint8)  # 0 is better than 1 initialization

            #cv2.imwrite('te_im1.jpg', te_im)
            te_im[:, :, :] = img_mean
            #cv2.imwrite('te_im2_1.jpg', te_im)
            te_im[top:top + r, left:left + c, :] = img
            #cv2.imwrite('te_im2.jpg', te_im)

            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean

            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]

            #cv2.imwrite('te_im3.jpg',   im_patch_original)

        else:
            im_patch_original = img[int(ymin):int((ymax) + 1), int(xmin):int((xmax) + 1), :]

            #cv2.imwrite('te_im4.jpg', im_patch_original)

        if not np.array_equal(model_sz, original_sz):

            h, w, _ = im_patch_original.shape


            if h < w:
                scale_h_ = 1
                scale_w_ = h/w
                scale = config.instance_size/h
            elif h > w:
                scale_h_ = w/h
                scale_w_ = 1
                scale = config.instance_size/w
            elif h == w:
                scale_h_ = 1
                scale_w_ = 1
                scale = config.instance_size/w

            gt_w = gt_w * scale_w_
            gt_h = gt_h * scale_h_

            gt_w = gt_w * scale
            gt_h = gt_h * scale

            #im_patch = cv2.resize(im_patch_original_, (shape))  # zzp: use cv to get a better speed
            #cv2.imwrite('te_im8.jpg', im_patch)

            s = SideWindowFilter(radius=1, iteration=3)
            im_patch_original = s.forward(im_patch_original)

            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
            #cv2.imwrite('te_im9.jpg', im_patch)


        else:
            s = SideWindowFilter(radius=1, iteration=3)
            im_patch_original = s.forward(im_patch_original)

            im_patch = im_patch_original
        #scale = model_sz / im_patch_original.shape[0]
        return im_patch, gt_w, gt_h, scale, scale_h_, scale_w_




    def crop_and_pad_old(self, img, cx, cy, model_sz, original_sz, img_mean=None):  #original_sz=18
        im_h, im_w, _ = img.shape

        xmin = cx - (original_sz - 1) / 2  #左
        xmax = xmin + original_sz - 1     #右
        ymin = cy - (original_sz - 1) / 2   #上
        ymax = ymin + original_sz - 1    #下

        left = int(self.round_up(max(0., -xmin)))     #若left=-xmin >0则   xmin<0 也就是 目标在边上 18*18的框会溢出
        top = int(self.round_up(max(0., -ymin)))
        right = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))

        xmin = int(self.round_up(xmin + left))  #溢出的地方 将左为0
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))
        r, c, k = img.shape
        if any([top, bottom, left, right]):
            te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
            te_im[top:top + r, left:left + c, :] = img
            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean
            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        else:
            im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]  #截取框住的部分  18*18
        if not np.array_equal(model_sz, original_sz):

            s = SideWindowFilter(radius=1, iteration=3)
            im_patch_original = s.forward(im_patch_original)

            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
        else:

            s = SideWindowFilter(radius=1, iteration=3)
            im_patch_original = s.forward(im_patch_original)

            im_patch = im_patch_original
        scale = model_sz / im_patch_original.shape[0]
        return im_patch, scale

    def round_up(self, value):
        return round(value + 1e-6 + 1000) - 1000

    def _target(self):

        regression_target, conf_target = self.compute_target(self.anchors,
                                                             np.array(list(map(round, self.ret['cx, cy, w, h']))))  #将坐标 宽高四舍五入

        return regression_target, conf_target             #返回 回归 和 分类

    def compute_target(self, anchors, box):
        #box = [-(box[0]), -(box[1]), box[2], box[3]]
        regression_target = self.box_transform(anchors, box)   #依据逆运算得到regression

        iou = self.compute_iou(anchors, box).flatten()      #计算iou
        #print(np.max(iou))
        pos_index = np.where(iou > config.pos_threshold)[0]  # iou>0.6  占0.6的为正样本
        neg_index = np.where(iou < config.neg_threshold)[0]   #<0.3
        label = np.ones_like(iou) * -1   #返回一个用1填充的跟输入 形状和类型 一致的数组。

        label[pos_index] = 1     #正样本为1
        label[neg_index] = 0     #负样本为0
        '''print(len(neg_index))
        for i, neg_ind in enumerate(neg_index):
            if i % 40 == 0:
                label[neg_ind] = 0'''

        # max_index = np.argsort(iou.flatten())[-20:]

        return regression_target, label

    def box_transform(self, anchors, gt_box):
        anchor_xctr = anchors[:, :1]    #依据公式逆运算
        anchor_yctr = anchors[:, 1:2]
        anchor_w = anchors[:, 2:3]
        anchor_h = anchors[:, 3:]
        gt_cx, gt_cy, gt_w, gt_h = gt_box

        target_x = (gt_cx - anchor_xctr) / anchor_w     #计算回归 也就是偏移的坐标占宽高的比  Xpro=Xan+Xreg*Wan  现在求reg  训练时 gt就是测试的pro
        target_y = (gt_cy - anchor_yctr) / anchor_h
        target_w = np.log(gt_w / anchor_w)
        target_h = np.log(gt_h / anchor_h)
        regression_target = np.hstack((target_x, target_y, target_w, target_h))
        return regression_target

    def compute_iou(self, anchors, box):   #1445 4    4
        if np.array(anchors).ndim == 1:
            anchors = np.array(anchors)[None, :]
        else:
            anchors = np.array(anchors)
        if np.array(box).ndim == 1:
            box = np.array(box)[None, :]
        else:
            box = np.array(box)
        gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))   #将box沿y方向平铺 与anchor的1445 一致

        anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5   #中心值减去宽的一半 得左
        anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5    #右
        anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5    #上
        anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5    #下

        gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
        gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
        gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
        gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

        xx1 = np.max([anchor_x1, gt_x1], axis=0)
        xx2 = np.min([anchor_x2, gt_x2], axis=0)
        yy1 = np.max([anchor_y1, gt_y1], axis=0)
        yy2 = np.min([anchor_y2, gt_y2], axis=0)

        inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                               axis=0)
        area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
        area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
        return iou

    def _tranform(self):

        self.ret['train_x_transforms'] = self.x_transforms(self.ret['instance_img'])
        self.ret['train_z_transforms'] = self.z_transforms(self.ret['exemplar_img'])

    def __getitem__(self, index):
        index = random.choice(range(len(self.sub_class_dir)))      #返回随机项 故训练时的打乱   在18的大小里面 进行随机
        '''if len(self.sub_class_dir) > 180:
            index = self.index
            self.index += 1

            if self.index >= 8000:
                self.index = 3000

            index = random.choice(range(3000, 8000))

            if index in self.index:
                index = random.choice(range(3000, 8000))
                print("index in self.index")

            if not index in self.index:
                self.index.append(index)
            if len(self.index) >= 3000:
                self.index = []
        else:
            index = random.choice(range(len(self.sub_class_dir)))'''

        # if self.name == 'GOT-10k':
        #     if  index == 4418 or index == 8627 or index == 8629 or index == 9057 or index == 9058 or index==7787 or index==5911:
        #         index += 3
        self._pick_img_pairs(index)   #在__giitem里执行  依据 随机的index 获取 模板 检测对
        self.open()                   #将图片 变换成
        self._tranform()               #255 归一化到0-1的
        regression_target, conf_target = self._target()
        self.count += 1

        return self.ret['train_z_transforms'], self.ret['train_x_transforms'], regression_target, conf_target.astype(np.int64)

    def __len__(self):
        return config.train_epoch_size*32   # 1000*64

if __name__ == "__main__":  #直接运行时

    root_dir = '../data/IR_data'
    seq_dataset = IR_train(root_dir, subset='train')
    train_data  = IR_Dataset(seq_dataset)
    train_data.__getitem__(180)
