import mxnet as mx
import random
import numpy as np
import unittest
import logging
from functools import partial
from qdio import load_img2np, load_labels_file
from qd_label_gen_prcoess import generate_labels_worker
from qd_label_factory import LabelFactory
import cv2

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class QDetectionIter(mx.io.DataIter):
    def __init__(self,
                 labels_file,
                 pic_root,
                 mean_file,
                 resize_scale=(1024, 576),
                 label_size=(256, 144),
                 batch_size=4,
                 shuffle=True):
        self._pnl = load_labels_file(labels_file)
        pic_names = [key for key in self._pnl.keys()]
        if shuffle:
            random.shuffle(pic_names)
        lb_for_factory = []
        for name in pic_names:
            pp = pic_root + "/" + name
            rects = self._pnl[name]
            lb_for_factory.append((pp, rects))
            # img = cv2.imread(pp)
            # for x, y, w, h in rects:
            #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            # cv2.imshow("raw img", img)
            # cv2.waitKey()
        mean = np.load(mean_file)
        generator = partial(generate_labels_worker,
                            resize_scale,
                            label_size,
                            mean)
        self._factory = LabelFactory(generator)
        self._factory.feed(lb_for_factory)

        self._batch_size = batch_size
        self._root = pic_root
        self._pic_size = resize_scale
        self._lb_size = label_size
        self._labels_name = ['softmax_label',
                             'x0',
                             'y0',
                             'x1',
                             'y1',
                             'norm_x',
                             'norm_y']

    def reset(self):
        self._factory.reset()

    @property
    def provide_data(self):
        return [mx.io.DataDesc('data',
                               ([self._batch_size, 3]
                                + list(self._pic_size),
                                np.dtype(np.uint8)))]

    @property
    def provide_label(self):
        ret = []
        for idx, name in enumerate(self._labels_name):
            ret.append(mx.io.DataDesc(name, (
                [self._batch_size] + list(self._lb_size)), np.float))
        return ret

    def __next__(self):
        return self.next()

    def next(self):
        # 1 image
        nd_data = []
        # 7 items
        labels_by_batch = []
        for i in range(self._batch_size):
            data, lb = self._factory.get()
            if data is None and lb is None:
                break
            labels = self.seperate_labels(lb)
            data = data.reshape(([1]+list(data.shape)))
            nd_data.append(mx.nd.array(data))
            labels_by_batch.append(labels)
        if len(nd_data) == 0 and len(labels_by_batch) == 0:
            raise StopIteration
        # Merge nd_* list to one
        # print(nd_data, nd_lb)
        batch_data = mx.nd.concatenate(nd_data, axis=0)
        labels_by_name = []
        for i in range(len(self._labels_name)):
            label_by_name = []
            for j in range(self._batch_size):
                label_by_name.append(labels_by_batch[j][i])
            labels_by_name.append(mx.nd.array(label_by_name))

        # For DEBUG......
        # softmax_label = labels_by_name[0].asnumpy()
        # img = nd_data[0].asnumpy().reshape((3, 576, 1024))
        # img = np.transpose(img, (1, 2, 0))
        # print(softmax_label.shape)
        # print(nd_data[0].asnumpy().shape)
        # import cv2
        # cv2.imshow("img", img)
        # cv2.imshow("hello", softmax_label[0].reshape((144, 256)))
        # cv2.imshow("asdf", [0, :, :, :])
        # cv2.waitKey()
        return mx.io.DataBatch(data=[batch_data],
                               label=labels_by_name)

    def seperate_labels(self, labels):
        return [labels[i, :, :].reshape(([1] +
                                         list(labels[i, :, :].shape)))
                for i in range(labels.shape[0])]

    def getpad(self):
        remains = self._factory.num_remains()
        if remains < self._batch_size:
            return self._batch_size - remains
        else:
            return 0


class TestMethods(unittest.TestCase):
    def test_iter(self):
        labels_file = '/home/qin/uniq_labels_top100.txt'
        pic_root = '/home/qin/all'
        mean_file = './mean_file.npy'
        qditer = QDetectionIter(
            labels_file,
            pic_root,
            mean_file,
            resize_scale=(1024, 576),
            label_size=(256, 144),
            batch_size=4,
            shuffle=True)
        print("Data shape:", qditer.provide_data)
        print("Label shape:", qditer.provide_label)
        for bn, batch in enumerate(qditer):
            print("Batch: ", bn)


def test():
    unittest.main()
    # data, size = load_img2np('/home/qin/图片/layers.jpg', (576, 1024))
    # print(data.shape)
    # print(size)


if __name__ == '__main__':
    test()
