import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
import os
import cv2
import pickle
import lmdb
import torch.nn as nn
import time

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from collections import OrderedDict

import setproctitle
import argparse

import sys

from got10k.datasets.ir_train import IR_train
from got10k.datasets.seqs_22_train import seqs_22_train
from siamrpn.dataset_IR import IR_Dataset

sys.path.append(os.getcwd())

from IPython import embed

from siamrpn.config import config
from siamrpn.network3 import SiamRPNNet
# from .dataset import ImagnetVIDDataset
from got10k.datasets import GOT10k
from siamrpn.dataset import seq_22Dataset
from siamrpn.transforms import Normalize, ToTensor, RandomStretch, RandomCrop, CenterCrop, RandomBlur, ColorAug
from siamrpn.loss import rpn_smoothL1, rpn_cross_entropy_balance
from siamrpn.visual import visual
from siamrpn.utils import get_topk_box, add_box_img, compute_iou, box_transform_inv, adjust_learning_rate, freeze_layers

from IPython import embed

torch.manual_seed(config.seed)


def train(data_dir, resume_path=None, vis_port=None, init=None):  # resume_path init vis_port
    # -----------------------
    # name = 'GOT-10k'
    # name = '22_sequences'
    name = 'IR_DATA'
    # seq_dataset_train = GOT10k(data_dir, subset='train')
    seq_dataset_train = IR_train(data_dir, subset='train')  # 图片和标签
    # seq_dataset_val = GOT10k(data_dir, subset='val')
    print('seq_dataset_train', len(seq_dataset_train))  # train 个文件   22

    # define transforms 
    train_z_transforms = transforms.Compose([
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])
    # valid_z_transforms = transforms.Compose([
    #     ToTensor()
    # ])
    # valid_x_transforms = transforms.Compose([
    #     ToTensor()
    # ])

    # create dataset 
    # -----------------------------------------------------------------------------------------------------
    # train_dataset = ImagnetVIDDataset(db, train_videos, data_dir, train_z_transforms, train_x_transforms)
    train_dataset = IR_Dataset(
        seq_dataset_train, train_z_transforms, train_x_transforms, name)  # 生成anchor 以及转换tensor   图片对  以及 回归和分类值

    # valid_dataset = GOT10kDataset(
    #     seq_dataset_val, valid_z_transforms, valid_x_transforms, name)

    anchors = train_dataset.anchors  # -17/2*8    w h 64

    # create dataloader

    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=config.train_batch_size,
                             shuffle=True, num_workers=config.train_num_workers,
                             pin_memory=True, drop_last=True)  # train_batch_size =16   总共有64000张  需要4000个batch

    # validloader = DataLoader(dataset=valid_dataset, batch_size=config.valid_batch_size,
    #                          shuffle=False, pin_memory=True,
    #                          num_workers=config.valid_num_workers, drop_last=True)

    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)
    if vis_port:
        vis = visual(port=vis_port)

    # start training
    # -----------------------------------------------------------------------------------------------------#
    model = SiamRPNNet()

    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # load model weight
    # -----------------------------------------------------------------------------------------------------#
    start_epoch = 1
    if resume_path and init:  # 不加载optimizer   resume_path就是上次的中断训练
        print("init training with checkpoint %s" % resume_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(resume_path)
        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'])
        else:
            model_dict = model.state_dict()  # 获取网络参数
            model_dict.update(checkpoint)  # 更新网络参数
            model.load_state_dict(model_dict)  # 加载网络参数
        del checkpoint
        torch.cuda.empty_cache()  # 清空缓存
        print("inited checkpoint")
    elif resume_path and not init:  # 获取某一个checkpoint恢复训练
        print("loading checkpoint %s" % resume_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(resume_path)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1  # 恢复迭代
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            model.load_state_dict(checkpoint)

        del checkpoint
        torch.cuda.empty_cache()  # 缓存清零
        print("loaded checkpoint")
    elif not resume_path and config.pretrained_model:
        print("loading pretrained model %s" % config.pretrained_model + '\n')  # 将alexnet加载进来
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(config.pretrained_model)  # 加载预训练模型参数
        # change name and load parameters
        checkpoint = {k.replace('features.features', 'featureExtract'): v for k, v in checkpoint.items()}  #针对于alexnet 如果用siamrpn预训练 不需要
        model_dict = model.state_dict()  # 获取模型参数
        model_dict.update(checkpoint)  # 将模型参数更新
        # model_dict.update(checkpoint['model'])   #针对全局
        model.load_state_dict(model_dict)  # 将参数部署到模型中  针对局部
        # optimizer.load_state_dict(checkpoint['optimizer'])

    # print(model.featureExtract[:10])
    # 如果有多块GPU，则开启多GPU模式
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # 并行计算

    for epoch in range(start_epoch, config.epoch + 1):  # 50次
        train_loss = []
        model.train()  # 进入训练模式将训练过程中每个 batch 的 μ \muμ 和 σ \sigmaσ 都保存下来，然后加权平均当做整个训练数据集的 μ \muμ 和 σ \sigmaσ ，同时用于测试。 BN DROPOUT

        # True，固定模型的前10层参数不变
        if config.fix_former_3_layers:  # 前三层卷积固定
            if torch.cuda.device_count() > 1:  # 多GPU
                freeze_layers(model.module)
            else:  # 单GPU
                freeze_layers(model)

        loss_temp_cls = 0
        loss_temp_reg = 0
        for i, data in enumerate(tqdm(trainloader)):  # data 四个值
            exemplar_imgs, instance_imgs, regression_target, conf_target = data  # 16 3 127127   16 3 255255
            # conf_target (8,1125) (8,225x5)    16张  batch_size=16
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()  # 从cpu中转到gpu
            # pre_score=16,10,19,19 ； pre_regression=[16,20,19,19]
            pred_score, pred_regression = model(exemplar_imgs.cuda(),
                                                instance_imgs.cuda())  # test时仅用track init 没有使用forward
            # [16, 5x17x17, 2]=[16,1445,2]
            pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                     2,
                                                                                                                     1)
            # [16,5x17x17,4] =[16,1445,4]
            pred_offset = pred_regression.reshape(-1, 4,
                                                  config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                     2,
                                                                                                                     1)
            cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors,
                                                 ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
            # （16 1805 2）预测的分类置信度  （16 1805）目标置信度 16 16 （1805 4）anchor false false
            reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)
            loss = cls_loss + config.lamb * reg_loss  # 分类权重和回归权重
            optimizer.zero_grad()  # 梯度清零
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.clip)  # config.clip=10 ，clip_grad_norm_梯度裁剪，防止梯度爆炸
            optimizer.step()  # 模型更新

            step = (epoch - 1) * len(trainloader) + i  # 64000
            summary_writer.add_scalar('train/cls_loss', cls_loss.data, step)
            summary_writer.add_scalar('train/reg_loss', reg_loss.data, step)
            train_loss.append(loss.detach().cpu())  # 当前计算图中分离下来的，但是仍指向原变量的存放位置,requires_grad=false
            loss_temp_cls += cls_loss.detach().cpu().numpy()  # 如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。 numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
            loss_temp_reg += reg_loss.detach().cpu().numpy()
            # if vis_port:
            #     vis.plot_error({'rpn_cls_loss': cls_loss.detach().cpu().numpy().ravel()[0],
            #                     'rpn_regress_loss': reg_loss.detach().cpu().numpy().ravel()[0]}, win=0)
            if (i + 1) % config.show_interval == 0:  # 展示间隔
                # if (i + 1) % 5 == 0:
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f lr: %.2e"
                           % (epoch, i, loss_temp_cls / config.show_interval, loss_temp_reg / config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0  # 结束清零

        train_loss = np.mean(train_loss)

        # valid_loss = []
        #
        # valid_loss = 0

        # print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        print("EPOCH %d train_loss: %.4f" % (epoch, train_loss))
        # summary_writer.add_scalar('valid/loss', valid_loss, (epoch + 1) * len(trainloader))

        adjust_learning_rate(optimizer, config.gamma)  # adjust before save, and it will be epoch+1's lr when next load

        if epoch % config.save_interval == 0 or epoch == config.epoch:  # 保存每个epoch
            if not os.path.exists('../models/'):  # ./当前文件所在目录  ../ 当前文件所在的上一层
                os.makedirs("../models/")
            save_name = "../models/half/swf_7_0001_5/siamrpn_{}.pth".format(epoch)
            # new_state_dict = model.state_dict()
            if torch.cuda.device_count() > 1:  # 多GPU训练
                new_state_dict = model.module.state_dict()
            else:  # 单GPU训练
                new_state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('save model: {}'.format(save_name))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 多卡情况下默认多卡训练,如果想单卡训练,设置为"0"

if __name__ == '__main__':
    # 参 数
    parser = argparse.ArgumentParser(description=" SiamRPN Train")

    parser.add_argument('--resume_path', default='', type=str,
                        help=" input gpu id ")  # resume_path 为空, 默认加载预训练模型alexnet,在config中有配置

    # parser.add_argument('--data', default='../data/GOT-10k', type=str, help=" the path of data")
    parser.add_argument('--data', default='../data/IR_data', type=str, help=" the path of data")
    args = parser.parse_args()

    # 训 练 
    train(args.data, args.resume_path)
