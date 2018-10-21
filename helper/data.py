import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys


num_classes=10


def get_data_set(name="train"):
    x = None
    y = None

    folder_name = "cifar_10"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            X = datadict["data"]
            Y = datadict['labels']

            X = np.array(X, dtype=float) / 255.0
            X = X.reshape([-1, 3, 32, 32])
            X = X.transpose([0, 2, 3, 1])
            X = X.reshape(-1, 32*32*3)

            if x is None:
                x = X
                y = Y
            else:
                x = np.concatenate((x, X), axis=0)
                y = np.concatenate((y, Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    num_labels = y.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + y.ravel()] = 1

    return x, labels_one_hot