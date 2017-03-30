import cv2
import numpy as np
import unittest
import matplotlib.pyplot as plt


def rescale_lb(rect, ratio):
    return [i*ratio for i in rect]


def load_img2np(pic_path, resize):
    img = cv2.imread(pic_path)
    ori_size = img.shape
    img = cv2.resize(img, resize)
    img = np.transpose(img, (2, 0, 1))
    return img, ori_size


def calc_mean(pic_names, pic_root, p_nsize):
    nh, nw = p_nsize
    mean = np.zeros((3, nw, nh), dtype=np.float)
    for pic_name in pic_names:
        pic_path = pic_root + "/" + pic_name
        img, _ = load_img2np(pic_path, p_nsize)
        mean += img
    mean = mean/len(pic_names)
    return mean


def write_mean(pic_names, pic_root, p_nsize, outfile_name):
    mean = calc_mean(pic_names, pic_root, p_nsize)
    np.save(outfile_name, mean)
    return mean


def load_labels_file(labels_file):
    ret = {}
    # Nevermind any exceptions.
    lf = open(labels_file)
    for line in lf:
        raw_segs = line.split(' ')
        pic_name = raw_segs[0]
        ret[pic_name] = [
            map(int, raw_rect.split(','))
            for raw_rect in raw_segs[1:]]
    lf.close()
    return ret


def preprocess(np_img, mean):
    # Normalize
    np_img = np_img.astype(np.float)
    np_img -= mean
    np_img /= 255
    return np_img


class IOTest(unittest.TestCase):
    def disable_test_write_mean(self):
        lbs = load_labels_file('/home/qin/uniq_labels.txt')
        mean = write_mean(lbs.keys(),
                          '/home/qin/all/',
                          (1024, 576),
                          './mean_file')
        mean_img = np.transpose(mean, (1, 2, 0))
        print(mean_img.shape)
        mean_img /= 255
        plt.imshow(mean_img)
        plt.show()

    def test_zload_mean(self):
        mean = np.load('./mean_file.npy')
        print("Mean file shape:", mean.shape)
        mean_img = np.transpose(mean, (1, 2, 0))
        plt.imshow(mean_img/255)
        plt.show()

    def test_loadimg2np(self):
        data, size = load_img2np('/home/qin/all/1.jpg', (1024, 576))
        img = np.transpose(data, (1, 2, 0))
        plt.imshow(img)
        plt.show()
        print("Transformed image shape: [C, W, H]", data.shape)
        print("Original image shape: [H,W,C]", size)

    def test_preprocess(self):
        mean = np.load('./mean_file.npy')
        data, size = load_img2np('/home/qin/all/1.jpg', (1024, 576))
        data = preprocess(data, mean)
        # data += data.min()
        data_img = np.transpose(data, (1, 2, 0))
        plt.imshow(data_img)
        plt.show()


if __name__ == '__main__':
    unittest.main()
