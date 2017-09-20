# coding=utf-8
# import modules
import numpy as np
import cv2
import math
from scipy import ndimage
import os
from signal_preprocessing import *
import scipy.io as sio
import matplotlib.pyplot as plt

class data_class:
    def __init__(self, data, label, proportion=0.8):
        # implement data_class from npy_dir or raw npy file
        if type(data) is str:
            data = np.load(data)
            label = np.load(label)
        self.image = data
        self.label = label
        # split training and test data according to proportion
        perm = np.arange(self.image.shape[0])
        np.random.shuffle(perm)
        self.image = self.image[perm]
        self.label = self.label[perm]
        self.train_images = self.image[:int(self.image.shape[0]*proportion)]
        self.train_labels = self.label[:int(self.image.shape[0]*proportion)]
        self.test_images = self.image[int(self.image.shape[0]*proportion):]
        self.test_labels = self.label[int(self.image.shape[0]*proportion):]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        # Shuffle the data
        perm = np.arange(self.train_images.shape[0])
        np.random.shuffle(perm)
        self.train_images = self.train_images[perm]
        self.train_labels = self.train_labels[perm]
        return self.train_images[0:batch_size], self.train_labels[0:batch_size]


def read_data_from_text(text_dir):
    """
    read text data from [text_dir] and format it for training
    :param text_dir: []
    :return: None, data are saved with npy format
    """
    data = []
    label = []
    for item in text_dir:
        with open(item, 'r') as text:
            lines = text.readlines()
            lines_arr = [i.split() for i in lines]
            data_npy = np.array(lines_arr, np.float32)
            file_name = text_dir.split('/')[-1].split('.')[0]
            # print data_npy.shape
            # 剔除空数据
            # data_npy = np.delete(data_npy, data_npy[:, 0:2]==np.array([0, 0]), axis=0)
            # print data_npy.shape
            #  extract train data
            # left_hand = data_npy[data_npy[:, 2] == -1][:, :2].reshape(-1, 800*2)
            # right_hand = data_npy[data_npy[:, 2] == 1][:, :2].reshape(-1, 800*2)
            # rest = data_npy[data_npy[:, 2] == 0][:, :2][:171*800, :].reshape(-1, 800*2)
            # data_right.append(right_hand)
            # data_left.append(left_hand)
            data.append(data_npy[:, :2]).reshape(-1, 800*2)
            label.append(data_npy[:, 2].flatten())
            # 保存信息
            # np.save('data/npy/'+file_name+'_left_hand.npy', left_hand)
            # np.save('data/npy/'+file_name+'_right_hand.npy', right_hand)
            # np.save('data/npy/' + file_name + '_rest_hand.npy', rest)
            # 保存类别
            left_label_sparse = generate_sparse(left_hand.shape[0], 3, 0)
            right_label_sparse = generate_sparse(right_hand.shape[0], 2, 1)
            rest_label_sparse = generate_sparse(rest.shape[0], 3, 2)
            np.save('data/npy/' + file_name + '_left_label.npy', left_label_sparse)
            np.save('data/npy/' + file_name + '_right_label.npy', right_label_sparse)
            np.save('data/npy/' + file_name + '_rest_label.npy', rest_label_sparse)

    return


def generate_sparse(num, class_num, label):
    left_label_sparse = np.zeros((num, class_num), np.float32)
    left_label_sparse[:, label] = 1.0
    return left_label_sparse


def merge_npy(npy_folder):
    npy_list = os.listdir(npy_folder)
    left_list = filter(lambda x:'left_hand' in x, npy_list)
    right_list = filter(lambda x:'right_hand' in x, npy_list)
    rest_list = filter(lambda x:'rest_hand' in x, npy_list)

    left_data = np.vstack([np.load(npy_folder+i) for i in left_list])
    right_data = np.vstack([np.load(npy_folder+i) for i in right_list])
    rest_data = np.vstack([np.load(npy_folder+i) for i in rest_list])

    data = np.vstack([left_data, right_data, rest_data])
    # data = np.vstack([left_data, right_data])
    # np.save(npy_folder+'left_data.npy', left_data)
    # np.save(npy_folder+'rest_data.npy', rest_data)
    # np.save(npy_folder+'right_data.npy', right_data)
    np.save(npy_folder+'data.npy', data)
    # print left_data.shape
    # print right_data.shape
    # print rest_data.shape
    left_label = generate_sparse(left_data.shape[0], 3, 0)
    right_label = generate_sparse(right_data.shape[0], 3, 1)
    rest_label = generate_sparse(rest_data.shape[0], 3, 2)

    label = np.vstack([left_label, right_label, rest_label])
    np.save(npy_folder+'label.npy', label)
    return



def read_mat(mat_dir=None):
    """
    read a batch of dataset from bci and output interested channel data
    :param mat_dir:
    :return:
    """
    # c3:26 c4:30 cZ:28
    if mat_dir is None:
        mat_dir = 'BCI_competion_data/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1a_1000Hz.mat'
    data = sio.loadmat(mat_dir)
    data['cnt'] = 0.1 * np.float32(data['cnt'])
    data_MI_4s = []
    data_MI_4s_c34 = []
    data_MI_4s_c34z = []
    for i in xrange(data['mrk']['pos'][0][0][0].size):
        item = data['mrk']['pos'][0][0][0][i]
        data_MI_4s.append(data['cnt'][item+500:item+4000-500, :])
        # drop the first and the last 0.5s data
        c34 = data['cnt'][item + 500:item + 4000 - 500, :][:, (26, 30)]
        c34z = data['cnt'][item + 500:item + 4000 - 500, :][:, (26, 28, 30)]
        data_MI_4s_c34.append(c34)
        data_MI_4s_c34z.append(c34z)
    # take every 1.5 data as a single trail
    data_MI_4s = np.vstack(data_MI_4s).reshape(-1, 1500, 59)
    data_MI_4s_c34 = np.vstack(data_MI_4s_c34).reshape(-1, 1500, 2)
    data_MI_4s_c34z = np.vstack(data_MI_4s_c34z).reshape(-1, 1500, 3)
    labels_or = data['mrk']['y'][0][0][0]
    labels = np.zeros(labels_or.size*2)
    for i in xrange(labels_or.size):
        labels[i*2] = labels_or[i]
        labels[i*2+1] = labels_or[i]
    print 'output from read_mat function:'
    print 'all channel data: '
    print data_MI_4s.shape
    print 'c34 channel data:'
    print data_MI_4s_c34.shape
    print 'c34z channel data:'
    print data_MI_4s_c34z.shape
    print 'labels:'
    print labels.shape
    return data_MI_4s, data_MI_4s_c34, data_MI_4s_c34z, labels


def feature_average(feature, verbo=True):
    """
    计算某类的平均特征, feature of size(trail_count, h, w, channel)
    :param feature:
    :return:
    """
    c3 = feature[:, :, :, 0]
    c4 = feature[:, :, :, 1]
    c3_avg = np.average(c3, axis=0)
    c4_avg = np.average(c4, axis=0)
    if verbo:
        plt.figure('feature')
        plt.subplot(211)
        plt.imshow(c3_avg, 'gray')
        plt.subplot(212)
        plt.imshow(c4_avg, 'gray')
        plt.show()
    return c3_avg, c4_avg


def bci_data_prepare(file_dir=None):
    """
    extract feature from a batch of bci dataset and format it for training
    """
    _, c34, _, labels = read_mat(file_dir)
    print c34.shape
    feature_all = []
    for trail_count in xrange(c34.shape[0]):
        channel_all = c34[trail_count, :, :]
        channel_c3 = channel_all[:, 0]
        # plt.figure('c3')
        # plt.plot(channel_c3)
        channel_c4 = channel_all[:, 1]
        # plt.figure('c4')
        # plt.plot(channel_c4)
        # plt.show()
        feature_c3 = feature_extract(channel_c3)
        feature_c4 = feature_extract(channel_c4)
        feature = cv2.merge([feature_c3, feature_c4]).reshape(1, feature_c3.shape[0], feature_c3.shape[1], 2)
        feature_all.append(feature)
    feature_all = np.vstack(feature_all)
    labels_sparse = np.zeros((labels.size, 2), np.float32)
    for i in xrange(labels.size):
        if labels[i] == -1:
            labels_sparse[i, 0] = 1.0
        elif labels[i] == 1:
            labels_sparse[i, 1] = 1.0
    labels_sparse = np.float32(labels_sparse)
    feature_left = feature_all[labels == -1]
    feature_right = feature_all[labels == 1]
    # left = feature_average(feature_left)
    # right = feature_average(feature_right)
    # all = feature_average(feature_all)
    # output
    print 'output from bci_data_prepare function:'
    print 'feature_all:'
    print feature_all.shape
    print 'feature_left:'
    print feature_left.shape
    print 'feature_right:'
    print feature_right.shape
    print 'labels:'
    print labels_sparse.shape
    return feature_all, feature_left, feature_right, labels, labels_sparse

def bci_data_prepare_all():
    mat_dir = 'BCI_competion_data/BCICIV_1calib_1000Hz_mat/'
    file_list = os.listdir(mat_dir)
    feature_all = []
    feature_left = []
    feature_right = []
    labels_all = []
    labels_sparse = []
    for item in file_list:
        file_dir = mat_dir + item
        name = item.split('.')[0]
        feature, feature_left_item, feature_right_item, labels, labels_sparse_item = bci_data_prepare(file_dir)
        # feature_visualization(feature_left_item, 'left')
        # feature_visualization(feature_right_item, 'right')
        # feature_visualization(feature, 'all')
        # plt.show()
        feature_all.append(feature)
        feature_left.append(feature_left_item)
        feature_right.append(feature_right_item)
        labels_all.append(labels)
        labels_sparse.append(labels_sparse_item)
        np.save(name + '.npy', feature)
        np.save(name + '_labels.npy', labels)
    feature_all = np.vstack(feature_all)
    labels_all = np.hstack(labels_all)
    labels_sparse = np.vstack(labels_sparse)
    np.save('bci_data.npy', feature_all)
    np.save('bci_labels.npy', labels_all)
    np.save('bci_labels_sparse.npy', labels_sparse)
    print 'data overall'
    print feature_all.shape
    print labels_all.shape
    print labels_sparse.shape
    print 'done'
    return

def feature_visualization(feature_array, string_):
    """
    visualize feature array by averaging or other method
    :param feature_array:
    :return:
    """
    trails_average = np.average(feature_array, axis=0)
    c3 = trails_average[:, :, 0]
    c4 = trails_average[:, :, 1]
    mu_c3 = np.sum(c3[1:8, :], axis=0)
    beta_c3 = np.sum(c3[12:25, :], axis=0)
    mu_c4 = np.sum(c4[1:8, :], axis=0)
    beta_c4 = np.sum(c4[12:25, :], axis=0)
    plt.figure(string_)
    plt.subplot(211)
    plt.plot(c3.T)
    plt.xlabel('time')
    plt.ylabel('c3 channel f A')
    plt.subplot(212)
    plt.plot(c4.T)
    plt.ylabel('c4 channel f A')
    plt.xlabel('times')
    plt.figure(string_+'_avarage mu band power')
    plt.subplot(211)
    plt.plot(mu_c3)
    plt.ylabel('c3')
    plt.subplot(212)
    plt.plot(mu_c4)
    plt.ylabel('c4')
    plt.figure(string_ + '_avarage beta band power')
    plt.subplot(211)
    plt.plot(beta_c3)
    plt.ylabel('c3')
    plt.subplot(212)
    plt.plot(beta_c4)
    plt.ylabel('c4')
    return

if __name__ == '__main__':
    pass