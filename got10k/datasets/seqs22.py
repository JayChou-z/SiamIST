from __future__ import absolute_import, print_function, unicode_literals

import os
import glob
import numpy as np
import io
import six
from itertools import chain

from ..utils.ioutils import download, extract


class seqs22(object):
    r"""`OTB <http://cvlab.hanyang.ac.kr/tracker_benchmark/>`_ Datasets.

    Publication:
        ``Object Tracking Benchmark``, Y. Wu, J. Lim and M.-H. Yang, IEEE TPAMI 2015.

    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        version (integer or string): Specify the benchmark version, specify as one of
            ``2013``, ``2015``, ``tb50`` and ``tb100``.
        download (boolean, optional): If True, downloads the dataset from the internet
            and puts it in root directory. If dataset is downloaded, it is not
            downloaded again.
    """

    __22_seqs = ['data05', 'data06', 'data07', 'data08', 'data09', 'data10', 'data11', 'data12',
                 'data13', 'data14', 'data15', 'data16', 'data17', 'data18', 'data19', 'data20', 'data21', 'data22']

    __version_dict = {
        22: __22_seqs,
        '22seqs': __22_seqs}

    def __init__(self, root_dir, version=22, download=True):
        super(seqs22, self).__init__()
        assert version in self.__version_dict  # 版本存在其中

        self.root_dir = root_dir
        self.version = version
        if download:
            self._download(root_dir, version)  # 获得数据
        self._check_integrity(root_dir, version)
        # chain.from_iterable 接受一个可以迭代的对象作为参数，返回一个迭代器
        valid_seqs = self.__version_dict[version]  ##glob.glob获取指定路径下的所有.txt文件，查找符合特征规则的文件路径名
        self.anno_files = sorted(list(chain.from_iterable(
            glob.glob(  # chain（） 接收多个可迭代对象作为参数，将它们『连接』起来，作为一个新的迭代器返回  chain.from_iterable（）接收一个可迭代对象作为参数，返回一个迭代器：
                os.path.join(root_dir, s, 'groundtruth.txt')) for s in valid_seqs)))  # 获得其标注 即ground truth坐标长宽
        # remove empty annotation files
        # (e.g., groundtruth_rect.1.txt of Human4)
        self.anno_files = self._filter_files(self.anno_files)  # 过滤空文件
        self.seq_dirs = [os.path.dirname(f) for f in
                         self.anno_files]  # 去掉文件名，单独返回目录路径    ...../basketball/groundtruth.txt  去掉groundtruth.txt
        self.seq_names = [os.path.basename(d) for d in
                          self.seq_dirs]  # basename用法是去掉目录路径，单独返回文件名  ..../basketball  ==>得到basketball  序列名称
        # rename repeated sequence names
        # (e.g., Jogging and Skating2)
        self.seq_names = self._rename_seqs(self.seq_names)

    def __getitem__(self, index):  # 依据index找key  依据key找index  找到序列下的所有图片   __方法会调用  enumerate会访问从index=0开始
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):  # 判断是否为字符串 python2:basestring python3:str
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)  # 依据index找key  依据key找index  找到序列

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], '*.bmp')))  # 得到每个序列下的img下的*.jpg文件

        # special sequences
        # (visit http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html for detail)

        # to deal with different delimeters
        with open(self.anno_files[index], 'r') as f:  # 某个序列
            anno = np.loadtxt(io.StringIO(f.read().replace(',',
                                                           ' ')))  # loadtxt用于从文本加载数据。 StringIO是以字符串的方式从  内存 中的文件读取数据。 replace用' '取代','如 101,91,32,81=>101 91 32 81
        assert len(img_files) == len(anno)  # 对比图片长度 和 注解长度是否相等
        assert anno.shape[1] == 4

        if len(img_files) > 300:
            img_file = img_files[:300]
            anno1 = anno[:300]
        else:
            img_file = img_files
            anno1 = anno

        return img_file, anno1

    def __len__(self):
        return len(self.seq_names)

    # 过滤掉不符合条件的groundtruth文件
    def _filter_files(self, filenames):
        filtered_files = []
        for filename in filenames:
            with open(filename, 'r') as f:
                if f.read().strip() == '':  # strip剔除文件中的首尾空格 但并不改变原文件  所以groundtruth.txt多出一行   返回文件内容
                    print('Warning: %s is empty.' % filename)
                else:
                    filtered_files.append(filename)  # 不为空就添加此文件

        return filtered_files

    def _rename_seqs(self, seq_names):
        # in case some sequences may have multiple targets
        renamed_seqs = []
        for i, seq_name in enumerate(seq_names):  # 返回0,1,2,3，和内容
            if seq_names.count(seq_name) == 1:
                renamed_seqs.append(seq_name)
            else:
                ind = seq_names[:i + 1].count(seq_name)
                renamed_seqs.append('%s.%d' % (seq_name, ind))  # 出现不止一次 则改名 第几次 car car1 car2

        return renamed_seqs

    # 下载数据集
    def _download(self, root_dir, version):
        assert version in self.__version_dict
        seq_names = self.__version_dict[version]

        if not os.path.isdir(root_dir):  # 判断是否为目录 不为
            os.makedirs(root_dir)
        elif all([os.path.isdir(os.path.join(root_dir, s)) for s in
                  seq_names]):  # 将目录 / 序列名   all() :如果bool(x)对于可迭代对象中的所有值 x为 True，则返回 True；如果可迭代对象为空，则返回 True
            print('Files already downloaded.')  # 不为空的话 就已经下载过了
            return

        url_fmt = 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/%s.zip'
        for seq_name in seq_names:
            seq_dir = os.path.join(root_dir, seq_name)  # 目录/序列   用于路径拼接文件路径，可以传入多个路径
            if os.path.isdir(seq_dir):  # 如果有了 就进行下一个序列的循环
                continue
            url = url_fmt % seq_name  # http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/%seq_name.zip  %用于嵌入其中
            zip_file = os.path.join(root_dir, seq_name + '.zip')  # 此时只是有一个路径的字样 还未创建
            print('Downloading to %s...' % zip_file)
            download(url, zip_file)  # 依据url下载到zip_file  .zip
            print('\nExtracting to %s...' % root_dir)
            extract(zip_file, root_dir)  # 抽取文件 zip 存储抽取文件

        return root_dir

    # 检查数据集的完整性
    def _check_integrity(self, root_dir, version):
        assert version in self.__version_dict
        seq_names = self.__version_dict[version]

        if os.path.isdir(root_dir) and len(os.listdir(root_dir)) > 0:  # os.listdir 获取指定文件夹下的所有文件 已经有文件
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)  # 用于路径拼接文件路径，可以传入多个路径
                if not os.path.isdir(seq_dir):  # 目录真实存在    判断是否下载过
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')
