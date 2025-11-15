import gzip
import os
from os.path import isfile, join
import numpy as np


# me dio pereza codear esto xd, creditos a Pepe Cantoral, Ph.D.
def list_files(mnist_path):
    return [join(mnist_path, f) for f in os.listdir(mnist_path) if isfile(join(mnist_path, f))]


def get_images(mnist_path, percentage=1):
    for f in list_files(mnist_path):
        if 'train-images' in f:
            with gzip.open(f, 'rb') as data:
                _ = int.from_bytes(data.read(4), 'big')
                num_images = int.from_bytes(data.read(4), 'big')
                rows = int.from_bytes(data.read(4), 'big')
                cols = int.from_bytes(data.read(4), 'big')
                train_images = data.read()
                x_train = np.frombuffer(train_images, dtype=np.uint8)
                x_train = x_train.reshape((num_images, rows, cols))
        elif 'train-labels' in f:
            with gzip.open(f, 'rb') as data:
                train_labels = data.read()[8:]
                y_train = np.frombuffer(train_labels, dtype=np.uint8)
        if 't10k-images' in f:
            with gzip.open(f, 'rb') as data:
                _ = int.from_bytes(data.read(4), 'big')
                num_images = int.from_bytes(data.read(4), 'big')
                rows = int.from_bytes(data.read(4), 'big')
                cols = int.from_bytes(data.read(4), 'big')
                test_images = data.read()
                x_test = np.frombuffer(test_images, dtype=np.uint8)
                x_test = x_test.reshape((num_images, rows, cols))
        elif 't10k-labels' in f:
            with gzip.open(f, 'rb') as data:
                test_labels = data.read()[8:]
                y_test = np.frombuffer(test_labels, dtype=np.uint8)

    SampleData = np.concatenate([x_train, x_test])
    Labels = np.concatenate([y_train, y_test])

    total_data = int(len(SampleData) * percentage)
    print("Cantidad de datos a usar: ", total_data)

    return SampleData[0:total_data], Labels[0:total_data]
