import os
import struct
from array import array


class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

        self.keys = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render

    @classmethod
    def load_labels(cls, dataset_lbl):
        with open(dataset_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            testing_labels = array("B", file.read())
        return testing_labels

    @classmethod
    def load_images(cls, dataset_img):
        with open(dataset_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            keys = ['Atr-%d' % i for i in range(1, rows * cols + 1)]
            keys.append('Class')
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())
        return image_data, size, rows, cols, keys

    def load_yield(self):
        training_img, training_lbl = os.path.join(self.path, self.train_img_fname), \
                                     os.path.join(self.path, self.train_lbl_fname)

        testing_img, testing_lbl = os.path.join(self.path, self.test_img_fname), os.path.join(self.path,
                                                                                              self.test_lbl_fname)

        testing_labels = MNIST.load_labels(testing_lbl)
        testing_image_data, size, rows, cols, keys = MNIST.load_images(testing_img)
        self.keys = keys

        for i in range(size):
            yield dict(zip(keys, list(testing_image_data[i * rows * cols:(i + 1) * rows * cols]) +
                           [testing_labels[i]]))

        training_labels = MNIST.load_labels(training_lbl)
        training_image_data, size, _, _, _ = MNIST.load_images(training_img)

        for i in range(size):
            yield dict(zip(keys, list(training_image_data[i * rows * cols:(i + 1) * rows * cols]) +
                           [training_labels[i]]))

    def get_keys(self):
        return self.keys
