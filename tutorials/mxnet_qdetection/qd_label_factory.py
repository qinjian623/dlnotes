from multiprocessing import Queue, Process
from functools import partial
from qd_label_gen_prcoess import generate_labels_worker
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
                 max_queue_size=10,
                 num_process=4):
        self._num_process = num_process
        self._max_queue_size = max_queue_size
        self._generator = generator
        self._processes = []
        self._reset()

    def _reset(self):
        for process in self._processes:
            process.terminate()
        self._labels_queue = Queue()
        self._output = Queue(maxsize=self._max_queue_size)
        self._labels_num = 0
        self._processes = []
        for pid in range(self._num_process):
            p = Process(target=self._generator,
                        args=(pid, self._labels_queue, self._output))
            self._processes.append(p)
            p.daemon = True
            p.start()

    def num_remains(self):
        return self._labels_num

    def feed(self, labels):
        self._labels = labels
        self._append_labels_to_queue(labels)

    def _append_labels_to_queue(self, labels):
        self._labels_num += len(labels)
        for pair in labels:
            self._labels_queue.put(pair)

    def get(self):
        if (self._labels_num == 0):
            return None, None
            # raise LabelsClearException("")
        self._labels_num -= 1
        return self._output.get()

    def reset(self):
        self._reset()
        self._append_labels_to_queue(self._labels)


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
        for i in range(3):
            data, lb = factory.get()
            if np.count_nonzero(lb[0, :, :]) == 0:
                print("Zero pos")
                continue
            # import cv2
            # cv2.imshow('lb', lb[0, :, :])
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
        factory.reset()
        for i in range(3):
            data, lb = factory.get()
            if np.count_nonzero(lb[0, :, :]) == 0:
                print("Zero pos")
                continue
            import cv2
            cv2.imshow('lb', lb[0, :, :])
            cv2.waitKey()
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
    # import profile
    # profile.run('unittest.main()')
    unittest.main()
