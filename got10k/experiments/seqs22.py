from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from tqdm import tqdm
from PIL import Image

from ..datasets.seqs22 import seqs22
from ..utils.metrics import rect_iou, center_error
from ..utils.viz import show_frame


class Experiment22seqs(object):
    """Experiment pipeline and evaluation toolkit for OTB dataset.

    Args:
        root_dir (string): Root directory of OTB dataset.
        version (integer or string): Specify the benchmark version, specify as one of
            ``2013``, ``2015``, ``tb50`` and ``tb100``. Default is ``2015``.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, version=22,
                 result_dir='results', report_dir='reports'):
        super(Experiment22seqs, self).__init__()
        self.dataset = seqs22(root_dir, version, download=False)  # 默认不下载  获取数据集
        self.result_dir = os.path.join(result_dir, '22seqs' + str(version))
        self.report_dir = os.path.join(report_dir, '22seqs' + str(version))
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21   #重叠像素
        self.nbins_ce = 11  #中心距离像素

    def run(self, tracker, visualize=False):

        print('Running tracker %s on %s...' % (
        tracker.name, type(self.dataset).__name__))  # type(xxx).__name__ 类型的名字    即找到数据集所属类型 OTB

        # loop over the complete dataset
        for s, (img_files, anno) in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            seq_name = self.dataset.seq_names[s]

            # print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))
            # skip if results exist
            record_file = os.path.join(self.result_dir, tracker.name, '%s.txt' % seq_name)  # results OTB100 siam
            # if os.path.exists(record_file):
            #     print('Found results, skipping', seq_name)
            #     continue
            # tracking loop

            boxes, times = tracker.track(img_files, anno[0, :],
                                         visualize=visualize)  # 用一个序列的所有图片 用第一帧的坐标长宽 =》tracker   anno[0, :]=box

            assert len(boxes) == len(anno)
            # record results
            self._record(record_file, boxes, times)   #将时间 和预测后的框 box

    def report(self, tracker_names):

        assert isinstance(tracker_names, (list, tuple))  # ‘SiamFC’

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])

        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)   #创建report 22seqs22 siamrpn 目录

        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)  # 视频的总数量OTB--100    seqs22 22
            succ_curve = np.zeros((seq_num, self.nbins_iou))  # IOU阈值21 成功率  （22，21）0   重叠部分占双方
            prec_curve = np.zeros((seq_num, self.nbins_ce))  # 21，51      中心误差
            speeds = np.zeros(seq_num)  #22个序列
            #
            performance.update({name: {'overall': {}, 'seq_wise': {}}})      #样式

            for s, (_, anno) in enumerate(self.dataset):    #仅取标注部分

                seq_name = self.dataset.seq_names[s]    #序列data01 0da02......

                record_file = os.path.join(self.result_dir, name, '%s.txt' % seq_name)   #result里的  存储经过跟踪器后的左上 宽高

                boxes = np.loadtxt(record_file, delimiter=',')

                boxes[0] = anno[0]



                assert len(boxes) == len(anno)

                ious, center_errors = self._calc_metrics(boxes, anno)     #将跟踪器后的结果和人工标注 计算重叠所占比例 中心距离  但是第一个都一样

                succ_curve[s], prec_curve[s] = self._calc_curves(ious, center_errors)     #s 即为第几个序列 对应每张图的iou rate  距离    整个序列 iou占比大于0.1有多少比列均值 中心像素距离小于等于1 2 3 4 像素有多少百分比 均值

                # calculate average tracking speed
                time_file = os.path.join(self.result_dir, name, 'times/%s_time.txt' % seq_name)

                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)
                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': succ_curve[s].tolist(),
                    'precision_curve': prec_curve[s].tolist(),
                    'success_score': np.mean(succ_curve[s]),   #分数求平均
                    'precision_score': prec_curve[s][10],     #有待商榷 取距离最远的占比
                    'success_rate': succ_curve[s][self.nbins_iou // 3],  #iou即重叠占0.5的有多少
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})     #记录每个序列

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)
            succ_score = np.mean(succ_curve)    #平均
            prec_score = prec_curve[10]           #有待商榷  所有序列距离为10
            succ_rate = succ_curve[self.nbins_iou // 3]  #  iou  重叠占比取中间  取0-1的多少合适
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'success_rate': succ_rate,
                'speed_fps': avg_speed})
            # print('prec_score:%s --succ_score:%s --succ_rate:%s' % (prec_score,succ_score,succ_rate)) # type(xxx).__name__ 类型的名字
        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        self.plot_curves(tracker_names)

        return prec_score, succ_score, succ_rate

    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0

        for s, seq_name in enumerate(seq_names):
            # print('[%d/%d] Showing results on %s...' % (
            #     s + 1, len(seq_names), seq_name))

            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]



            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')  #保存左上 宽高

        # print('  Results recorded at', record_file)

        # record running times
        time_dir = os.path.join(record_dir, 'times')
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(time_dir, os.path.basename(
            record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')  #保存时间

    def _calc_metrics(self, boxes, anno):
        # can be modified by children classes
        ious = rect_iou(boxes, anno)   #得到每张图片的对应 iou占比
        center_errors = center_error(boxes, anno)       #得到中心误差的距离
        return ious, center_errors

    def _calc_curves(self, ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]  #np.newaxis 增加维度 300,1   np.asarray  也是转化为数组

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]   #0-1 分为iou个
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]  #中心距离[0,1,2,3......  (1,nbins_ce)

        bin_iou = np.greater(ious, thr_iou)  #判断参数一是否大于参数二。 300 11
        bin_ce = np.less_equal(center_errors, thr_ce) #判断参数一是否小于等于参数二 300 11

        succ_curve = np.mean(bin_iou, axis=0)  #求均值300行的均值 11
        prec_curve = np.mean(bin_ce, axis=0)  #11   得到整个序列的均值 距离为1 2 3 小于等于百分比

        return succ_curve, prec_curve

    def plot_curves(self, tracker_names):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        assert os.path.exists(report_dir), \
            'No reports found. Run "report" first' \
            'before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), \
            'No reports found. Run "report" first' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots.png')
        prec_file = os.path.join(report_dir, 'precision_plots.png')
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots of OPE')
        ax.grid(True)
        fig.tight_layout()

        # print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots of OPE')
        ax.grid(True)
        fig.tight_layout()

        # print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, dpi=300)
