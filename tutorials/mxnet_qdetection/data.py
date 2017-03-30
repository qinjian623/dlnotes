import mxnet as mx
import random
import numpy as np
import matplotlib.pyplot as plt
import unittest
import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class QDetectionIter(mx.io.DataIter):
    def warm_up_data(self):
        for pn, rects in self._pnl:
            data, original_shape = load_img2np(self._root+"/"+pn,
                                               self._pic_size)
            oh = original_shape[1]
            ow = original_shape[0]
            label = generate_labels(rects, (oh, ow), self._pic_size)

    def pic_size(pic):
        pass

    def __init__(self,
                 labels_file,
                 pic_root,
                 mean,
                 resize_scale=(1024, 576),
                 label_size=(256, 144),
                 batch_size=4,
                 shuffle=True):
        self._pnl = load_labels_file(labels_file)
        pic_names = self._pnl.keys()
        if shuffle:
            random.shuffle(pic_names)
        self._batch_size = batch_size
        self._root = pic_root
        self._pic_size = resize_scale
        self._lb_sizie = label_size

    @property
    def provide_data(self):
        return [DataDesc('data',
                         tuple([self._batch_size, 3]+list(self._pic_size),
                               np.dtype(np.uint8)))]

    @property
    def provide_label(self):
        pass


class TestMethods(unittest.TestCase):
    pass


def test():
    unittest.main()
    # data, size = load_img2np('/home/qin/图片/layers.jpg', (576, 1024))
    # print(data.shape)
    # print(size)
    

if __name__ == '__main__':
    test()
