# import cv2
import numpy as np
import math
import logging
import unittest
from qdio import load_img2np, rescale_lb, preprocess


logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def generate_one_image_labels(rects,
                              p_osize,
                              p_nsize,
                              lbsize):
    ow, oh = p_osize
    nw, nh = p_nsize
    lbw, lbh = lbsize
    # print(p_osize, p_nsize, lbsize)

    pic_shrink_ratio = float(nh)/oh    # 576/720.0
    feat_shrink_ratio = float(lbh)/nh  # 0.25
    lb = np.zeros((7, lbh, lbw), dtype=np.float)
    for rect in rects:
        x, y, w, h = rescale_lb(rect, pic_shrink_ratio)
        x0, y0, x1, y1 = x+0.25*w, y+0.25*h, x+0.75*w, y+0.75*h
        # Don't care small obj.
        if w < 12 or h < 12:
            continue
        f_x0, f_y0, f_x1, f_y1 = [int(i*feat_shrink_ratio)
                                  for i in [x0, y0, x1, y1]]
        bb_x0, bb_y0, bb_x1, bb_y1 = [int(i*feat_shrink_ratio)
                                      for i in [x, y, x+w, y+h]]
        k = lb[0, bb_y0:bb_y1, bb_x0:bb_y1]
        k[:][:] = 2
        k = lb[0,
               math.ceil(f_y0): math.floor(f_y1),
               math.ceil(f_x0): math.floor(f_x1)]
        k[:][:] = 1
        for ix in range(int(math.ceil(f_x0)),
                        int(math.floor(f_x1))):
            for iy in range(int(math.ceil(f_y0)),
                            int(math.floor(f_y1))):
                lb[1, iy, ix] = x-(ix/feat_shrink_ratio)
                lb[2, iy, ix] = y-(iy/feat_shrink_ratio)
                lb[3, iy, ix] = (x + w -
                                 (ix/feat_shrink_ratio))
                lb[4, iy, ix] = (y + h -
                                 (iy/feat_shrink_ratio))
                lb[5, iy, ix] = 1.0/w
                lb[6, iy, ix] = 1.0/h
    num_of_points = np.count_nonzero(lb[0])
    if num_of_points != 0:
        lb[5] /= num_of_points
        lb[5] /= num_of_points
    else:
        print("Zero positive")
    # cv2.imshow('Label feature', lb[0, :, :])
    # cv2.waitKey()
    return lb


def generate_labels_worker(  # p_osize,
                           p_nsize,
                           lbsize,
                           mean,
                           pid, lbs, output):
    while True:
        pn, rects = lbs.get()
        np_img, p_osize = load_img2np(pn, p_nsize)
        oh, ow, _ = p_osize
        np_img = preprocess(np_img, mean)
        lb_feat = generate_one_image_labels(rects,
                                            (ow, oh),
                                            p_nsize,
                                            lbsize)
        output.put((np_img, lb_feat))


class LabelGenTestMethods(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
