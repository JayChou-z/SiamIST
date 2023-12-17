import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
from tqdm import tqdm

from bin.sidewinFiltering import SideWindowFilter
from .network3 import SiamRPNNet
from .config import config
from .transforms import ToTensor
from .utils import generate_anchors, get_exemplar_image, get_instance_image, box_transform_inv, add_box_img, \
    add_box_img_left_top, show_image
from IPython import embed

torch.set_num_threads(1)  # otherwise pytorch will take all cpus


class SiamRPNTracker:
    def __init__(self, model_path, cfg=None, is_deterministic=False):

        self.name = 'SiamRPN'

        # 更新一下config 的参数,这个非常重要
        if cfg:
            config.update(cfg)  # 更新参数数据

        self.model = SiamRPNNet()  #
        self.is_deterministic = is_deterministic

        checkpoint = torch.load(model_path)  # 加载save的模型参数

        if 'model' in checkpoint.keys():  # checkpoint.keys()  'epoch' 'model' 'optimizer'
            self.model.load_state_dict(torch.load(model_path)['model'])  # 加载训练好的模型参数进入该model
        else:
            self.model.load_state_dict(torch.load(model_path))

        self.model = self.model.cuda()  # 默认是0   模型送入cuda

        self.model.eval()
        self.transforms = transforms.Compose([ToTensor()])
#作用是将导入的图片转换为Tensor的格式，导入的图片为PIL image 或者 numpy.nadrry格式的图片，其shape为（HxWxC）数值范围在[0,255],转换之后shape为（CxHxw）,数值范围在[0,1].
        # valid_scope = 2 * config.valid_scope + 1 # 2x9+1=19 or 2x8+1=17 ； 2x7+1=15

        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        config.valid_scope)  # 8,8,8,[0.33, 0.5, 1, 2, 3],19
        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()  # tile 平铺的意思 np.tile(a,2)沿x方向复制扩大倍数，（a，（2,1））沿y复制扩大2倍，x扩大一倍  outer两向量外积 即4个*4个=16个
        # hanning（m) 生成余弦窗函数或者高斯函数，用于过滤或者突出某个物体；通常与outer合用 生成高斯矩阵  [None, :] 将其弄成无行，即一行 两个维度  none增加一个维度 该位置为1
        # score_size=19，anchor_num=5,经过outer后形成19*19的高斯矩阵（19,19） 经过[None, :]  （1,19,19）[1个[19个[ 19个]]] =>(5,19,19)  经过flatten，5*19*19的数据

    def _cosine_window(self, size):
        """
            get the cosine window  生成余弦窗函数
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])    #方向上增加新的维度
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ initialize siamrpn tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        # self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]]) #cx,cy,w,h

        # self.bbox = np.array(
        #     [bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[1] - 1 + (bbox[3] - 1) / 2, bbox[2], bbox[3]])  # cx,cy,w,h    目标中心  第一帧  说明先前是左上角的坐标

        self.bbox = np.array(
                     [bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[1] - 1 + (bbox[3] - 1) / 2, bbox[2], bbox[3]])
        self.pos = np.array(
               [bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[1] - 1 + (bbox[3] - 1) / 2])

        #由于数据集提供的坐标就是目标中心 故：

        # self.pos = np.array(
        #     [bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[1] - 1 + (bbox[3] - 1) / 2])  # center x, center y, zero based   目标中心


        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height  目标大小

        self.origin_target_sz = np.array([bbox[2], bbox[3]])  # w,h   原目标大小

        # get exemplar img
        self.img_mean = np.mean(frame, axis=(0, 1))#对每个通道进行均值计算
        # 获取模板图像
        exemplar_img, scale_z, _ = get_exemplar_image(frame, self.bbox, config.exemplar_size, config.context_amount,
                                                      self.img_mean)  #第一帧  [213.5,253,34,81]  127 0.5   exemplar_img(127,127,3)



        exemplar_img = self.transforms(exemplar_img)[None, :, :]  #[1,:,:,:]  1 3 127 127 在测试阶段，转换成tensor类型就可以了 	convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
        self.model.track_init(exemplar_img.cuda())

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        instance_img_np, _, _, scale_x = get_instance_image(frame, self.bbox, config.exemplar_size,
                                                            config.instance_size,
                                                    config.context_amount, self.img_mean)




        instance_img = self.transforms(instance_img_np)[None, :, :, :]  #1 3 271 271  扩围
        # pred_score=1,2x5,17,17 ; pre_regression=1,4x5,17,17 
        pred_score, pred_regression = self.model.track(instance_img.cuda())  # （1 10 19 19）（1 20 19 19）
        # [1,5x17x17,2] 5x17x17=1445   能不能输出看看是什么
        pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0, 2,
                                                                                                                 1)
        # [1,5x17x17,4]  premute 置换 交换各个维度的大小      1 2 19*19*5      1 19*19*5 2
        pred_offset = pred_regression.reshape(-1, 4, config.anchor_num * config.score_size * config.score_size).permute(
            0, 2, 1) # 1 4 19*19*5  1 19*19*5 4   pred_offset[0]=第一维的 [1805,4]

        delta = pred_offset[0].cpu().detach().numpy()
        #返回一个new Tensor，只不过不再有梯度。如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。 numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
        # 使用detach()函数来切断一些分支的反向传播;返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
        # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad #这样我们就会继续使用这个新的Variable进行计算，后面当我们进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播

        box_pred = box_transform_inv(self.anchors, delta)  # 通过 初始化的anchors 和 offset 来预测box
        #1805 4
        # 传入的anchor(1805,4) delta(1805,4),delta是回归参数，对anchor进行调整，返回调整后的anchor，即pre_box(1805,4)
        # pred_conf=[1,1805,2];
        # hah=F.softmax(pred_conf, dim=2)
        score_pred = F.softmax(pred_conf, dim=2)[0, :, 1].cpu().detach().numpy()  # 计算预测分类得分  对第三维做归一化
        #1805
        # ?
        def change(r):
            return np.maximum(r, 1. / r)  # x 和 y 逐位进行比较选择最大值  惩罚函数的内容

        # ?
        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        # ?
        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)
        #1805 4
        s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) / (sz_wh(self.target_sz * scale_x)))  # scale penalty     微调后的w h==》s    s'
        r_c = change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty  目标 大小 w h  r r'
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)  # 尺度惩罚和比例惩罚
        pscore = penalty * score_pred  # 对每一个anchors的分类预测×惩罚因子   尺度惩罚和比例惩罚 penalty_k=0.22,penalty最大为1，即不惩罚
        # 惩罚的前提是假设目标在相邻帧的大小(尺度以及高宽比例)变化，所以增加了尺度和比例两个惩罚项，又假设目标在相邻帧的位置变化也不会太大，所以使用余弦窗来抑制大位移，正如论文所言，使用
        # penalty 来抑制尺度和比例的大变化，余弦窗口抑制大位移。
        #
        pscore = pscore * (1 - config.window_influence) + self.window * config.window_influence  # 再乘以余弦窗
        best_pscore_id = np.argmax(pscore)  # 得到最大的得分

        target = box_pred[best_pscore_id, :] / scale_x  # target（x,y,w,h）是以上一帧的pos为（0,0）  最好的

        lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box  # 预测框的学习率
        # 关于clip的用法， 对象 起点 终点 小于起点皆为起点
        res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[1])  #0 576 w=frame.shape[1]
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])  # 0  432 h=frame.shape[0]

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])  # 更新之后的坐标

        self.target_sz = np.array([res_w, res_h])

        bbox = np.array([res_x, res_y, res_w, res_h])

        self.bbox = (  # cx, cy, w, h
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))

        # 这个用来画图使用
        bbox = np.array([  # tr-x,tr-y w,h
            self.pos[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.pos[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.target_sz[0], self.target_sz[1]])

        # return self.bbox, score_pred[best_pscore_id]
        return bbox

    #   数据集测试
    def track(self, img_files, box, visualize=False):   #box 第一帧的目标坐标 宽高
        frame_num = len(img_files)  #这个序列的所有图片
        boxes = np.zeros((frame_num, 4))  #如（ 725，4） 725*[0，0，0，0]
        boxes[0] = box     #第一帧的数据
        times = np.zeros(frame_num)

        # for f, img_file in tqdm(enumerate(img_files),total=len(img_files)):
        for f, img_file in enumerate(img_files):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            begin = time.time()
            if f == 0:
                self.init(img, box)    #第一张图片
                # bbox = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]) # 1-idx
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                show_image(img, boxes[f, :])

        return boxes, times
