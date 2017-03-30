import mxnet as mx
from qd_symbol import build_network
from qd_data_iter import QDetectionIter


def train(net, train_iter, val_iter):
    pass


def main():
    net = build_network()
    labels_file = '/home/qin/uniq_labels_top100.txt'
    pic_root = '/home/qin/all'
    mean_file = './mean_file.npy'
    train_iter = QDetectionIter(
        labels_file,
        pic_root,
        mean_file,
        resize_scale=(1024, 576),
        label_size=(256, 144),
        batch_size=4,
        shuffle=True)
    test_iter = QDetectionIter(
        labels_file,
        pic_root,
        mean_file,
        resize_scale=(1024, 576),
        label_size=(256, 144),
        batch_size=2,
        shuffle=True)
    mx.viz.plot_network(net, shape={4, 3, 576, 1024})
    train(net, train_iter, test_iter)


if __name__ == '__main__':
    main()
