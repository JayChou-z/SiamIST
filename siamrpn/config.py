import numpy as np


class Config:
    # dataset related
    exemplar_size = 127  # exemplar size 样例大小
    instance_size = 255  # 默认271             # instance size  修改成255试试?   检测帧大小  与论文不同论文是255
    context_amount = 0.5  # context amount 图像模板扩充大小  18 1/2 ==> 36 3/2
    tem_amount=0.5
    sch_amount=0.5
    sample_type = 'uniform'

    # --------
    train_epoch_size = 1000  # 默认1000, 尝试设置成500训练的少一些
    val_epoch_size = 100  #
    out_feature = 17  # 19 对应论文，互相关后的尺寸大小；；与论文不同，这里是19，因为detection_frame_size的不同
    # max_inter   = 80# 使用   frame_range_got = 100 替换
    eps = 0.01  #
    # --------

    # training related
    exem_stretch = False

    ohem_pos = False  # 原始都是False 训练的时候没有对anchors非极大值抑制，具体参考loss.py函数rpn_cross_entropy_balance（）
    ohem_neg = False  # 原始都是False 训练的时候没有对anchors非极大值抑制，具体参考loss.py函数rpn_cross_entropy_balance（）
    ohem_reg = False  # 原始都是False 训练的时候没有对anchors非极大值抑制，具体参考loss.py函数rpn_smoothL1（）

    fix_former_3_layers = True
    scale_range = (0.001, 0.7)
    ratio_range = (0.1, 10)
    pairs_per_video_per_epoch = 2  # pairs per video
    train_ratio = 0.99  # training ratio of VID dataset
    frame_range_vid = 100  # frame range of choosing the instance
    frame_range_ytb = 1  # training batch size
    frame_range_got = 100  # max_inter   = 80#
    train_batch_size = 16  # 16 双           # training batch size
    valid_batch_size = 16  # 16 双            # validation batch size
    train_num_workers = 8  # number of workers of train dataloader
    valid_num_workers = 8  # number of workers of validation dataloader
    clip = 10  # grad clip

    start_lr = 1e-4 # siamrpn++ 里是  0.005 0.01
    end_lr = 1e-7 # siamrpn++ 里是  0.0005  0.000005
    # --------------
    warm_lr = 1e-3  #
    warm_scale = warm_lr / start_lr  #
    # ---------------

    epoch = 40  # 1

    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[1] / \
            np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]  # 构造等比数列 start end 基底默认为10
    # decay rate of LR_Schedular
    step_size = 1  # step size of  LR_Schedular
    momentum = 0.9  # momentum of SGD
    weight_decay = 0.0005  # weight decay of optimizator

    seed = 6666  # seed to sample training videos
    log_dir = '../log'  # log dirs
    max_translate = 12  # max translation of random shift  随机移动
    scale_resize = 0.15  # scale step of instance image
    total_stride = 8  # total stride of backbone
    valid_scope = int((instance_size - exemplar_size) / total_stride + 1)  # (271-127)/8+1  ==18+1==19  等于最后互相关后的尺寸大小

    anchor_scales = np.array([8, ])  # Backbone会进行8倍下采样
    anchor_ratios = np.array([1, 1, 1, 1, 1])  # 5个anchor的 尺度大小
    anchor_num = len(anchor_scales) * len(anchor_ratios)  # ==5 论文中对应K的数字
    anchor_base_size = 8  # Backbone会进行8倍下采样
    pos_threshold = 0.6
    neg_threshold = 0.3
    num_pos = 16
    num_neg = 48
    lamb = 5  # 原始 cls:res = 1:5   5 default  4ke
    save_interval = 1
    show_interval = 500  # 100
    show_topK = 3
    pretrained_model = '../models/alexnet.pth'

    # tracking related
    gray_ratio = 0.25
    blur_ratio = 0.15
    score_size = int((instance_size - exemplar_size) / total_stride + 1)  # (271-127)/8+1  ==18+1==19 等于最后互相关后的尺寸大小
    penalty_k = 0.22  # 0.22 #0.22          #0.16
    window_influence = 0.40  # 0.40 #0.40  #0.40
    lr_box = 0.30  # 0.30                  #0.30
    min_scale = 0.1
    max_scale = 10

    def update(self, cfg):
        for k, v in cfg.items():  # 键值对
            setattr(self, k, v)  # self.k=v
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1  #
        # self.valid_scope = int((self.instance_size - self.exemplar_size) / self.total_stride / 2)#anchor的范围
        self.valid_scope = self.score_size


config = Config()
