from multiprocessing import Queue, Process
from functools import partial
from label_gen_prcoess import generate_labels_worker
import unittest
import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class LabelsClearException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class LabelFactory():
    def __init__(self,
                 generator,
                 maxsize=10,
                 num_process=4):
        self._labels = Queue()
        self._output = Queue(maxsize=maxsize)
        self._labels_num = 0
        self._processes = []
        for pid in range(num_process):
            p = Process(target=generator,
                        args=(pid, self._labels, self._output))
            self._processes.append(p)
            p.daemon = True
            p.start()

    def feed(self, labels):
        self._labels_num += len(labels)
        for pair in labels:
            logger.info(pair)
            self._labels.put(pair)

    def get(self):
        if (self._labels_num == 0):
            raise LabelsClearException("")
        self._labels_num -= 1
        return self._output.get()


class LabelFactoryTest(unittest.TestCase):
    def test_factory(self):
        labels_file = '/home/qin/uniq_labels_top100.txt'
        root = '/home/qin/all'
        import qdio
        import numpy as np
        labels = qdio.load_labels_file(labels_file)
        mean = np.load('./mean_file.npy')
        lb_for_factory = []
        for pn, rects in labels.items():
            pp = root + "/" + pn
            lb_for_factory.append((pp, rects))
        logger.info("asdf")
        generator = partial(generate_labels_worker,
                            # (1280, 720),
                            (1024, 576),
                            (256, 144),
                            mean)
        factory = LabelFactory(generator)
        factory.feed(lb_for_factory)
        # import timeit
        # print(timeit.timeit(factory.get, number=20))
        for i in range(20):
            data, lb = factory.get()
            if np.count_nonzero(lb[0, :, :]) > 0:
                # print("Zero pos")
                continue
            # import cv2
            # cv2.imshow('lb', lb[3, :, :])
            # cv2.waitKey()
            assert(np.min(lb[1, :, :]) < 0.0)
            assert(np.max(lb[1, :, :]) >= 0.0)
            assert(np.min(lb[2, :, :]) < 0.0)
            assert(np.max(lb[2, :, :]) >= 0.0)
            assert(np.min(lb[3, :, :]) <= 0.0)
            assert(np.max(lb[3, :, :]) > 0.0)
            assert(np.min(lb[4, :, :]) <= 0.0)
            assert(np.max(lb[4, :, :]) > 0.0)
            assert(np.max(lb[5, :, :]) > 0.0)
            assert(np.max(lb[6, :, :]) > 0.0)


if __name__ == '__main__':
    unittest.main()
