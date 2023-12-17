from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

sys.path.append('../')
sys.path.append(os.getcwd())
import argparse
import cv2
import torch
from glob import glob
from siamrpn import SiamRPNTracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamRPN demo')
parser.add_argument('--snapshot', type=str, default='../models/siamrpn_50.pth', help='model name')
parser.add_argument('--video_name', default='../data/22data/test/data05', type=str, help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:  # 如果没有的话
        cap = cv2.VideoCapture(0)  # VideoCapture()中参数是0，表示打开笔记本的内置摄像头

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):  # 判断其以什么结尾   视频的话
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()  # cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。
            # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, '*.bmp')))  # key=lambda x: int(os.path.basename(x).split('.')[0])
        if len(images) > 300:
            image = images[:300]
        else:
            image = images
        for img in image:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config

    # create model

    # build tracker
    tracker = SiamRPNTracker(args.snapshot)

    # hp = {'lr': 0.3, 'penalty_k': 0.04, 'window_lr': 0.4}

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False,
                                          False)  # cv2.selectROI可以让用户框出感兴趣的区域，以便对这个区域进行截取和后续处理。
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            bbox = tracker.update(frame)  # hp
            # bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (0, 255, 0), 1)  # 左上 右下 字体颜色 字体粗细
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)


if __name__ == '__main__':
    main()
