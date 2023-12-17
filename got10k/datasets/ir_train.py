from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six


class IR_train(object):
    r"""`GOT-10K <http://got-10k.aitestunion.com//>`_ Dataset.

    Publication:
        ``GOT-10k: A Large High-Diversity Benchmark for Generic Object
        Tracking in the Wild``, L. Huang, X. Zhao and K. Huang, ArXiv 2018.

    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
    """

    def __init__(self, root_dir, subset='test', return_meta=False):  # 数据集路径

        super(IR_train, self).__init__()
        assert subset in ['train', 'val', 'test'], 'Unknown subset.'  # train val

        self.root_dir = root_dir
        self.subset = subset
        self.return_meta = False if subset == 'test' else return_meta
        self._check_integrity(root_dir, subset)

        list_file = os.path.join(root_dir, subset, 'list.txt')  # 路径的拼接   得到路径
        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')  # 按行读取所有的视频文件夹名  000001 000002
        self.seq_dirs = [os.path.join(root_dir, subset, s)  # 获取所有视频文件的绝对路径
                         for s in self.seq_names]
        self.anno_files = [os.path.join(d, 'groundtruth.txt')  # 在绝对路径下得到000001下的txt 每行对应一张图片
                           for d in self.seq_dirs]

    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``meta`` is a dict contains meta information about the sequence.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*[.png,.bmp]')))  # init中已经有self.seq_dirs 得到一个序列所有图片
        anno = np.loadtxt(self.anno_files[index], delimiter=',')  # init中已经有self.anno_files

        if self.subset == 'test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta:
            meta = self._fetch_meta(self.seq_dirs[index])
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, subset):
        assert subset in ['train', 'val', 'test']
        list_file = os.path.join(root_dir, subset, 'list.txt')  # 得到字符串 还未创建

        if os.path.isfile(list_file):  # 已创建
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')  # 条 删除给定字符仅针对开头和结尾   没有字符 则删除空格 开头 结尾  split拆分字符串 遇到换行  得到序列名称

            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, subset, seq_name)  # 合并路径  得到
                if not os.path.isdir(seq_dir):  # 是否创建
                    print('Warning: sequence %s not exists.' % seq_name)
        else:  # 未创建
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

    def _fetch_meta(self, seq_dir):
        # meta information
        meta_file = os.path.join(seq_dir, 'meta_info.ini')
        with open(meta_file) as f:
            meta = f.read().strip().split('\n')[1:]
        meta = [line.split(': ') for line in meta]
        meta = {line[0]: line[1] for line in meta}

        # attributes
        attributes = ['cover', 'absence', 'cut_by_image']
        for att in attributes:
            meta[att] = np.loadtxt(os.path.join(seq_dir, att + '.label'))

        return meta
