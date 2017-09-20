# coding=utf-8
import scipy.signal as signal
import pylab as pl
from sklearn.decomposition import PCA
import numpy as np
from data_prepare import *

class Pca:
    def __init__(self, dimension=256):
        self.data = np.load('data/npy/07-12/data.npy')
        self.right_hand = np.load('data/npy/07-12/Eeg_MI_0_right_hand.npy')
        self.left_hand = np.load('data/npy/07-12/Eeg_MI_0_left_hand.npy')
        self.rest = np.load('data/npy/07-12/Eeg_MI_0_rest_hand.npy')
        self.train = np.load('data/npy/train/train_data.npy')
        self.test = np.load('data/npy/test/test_data.npy')
        self.pca_ = PCA(n_components=dimension)
        self.pca_.fit(self.data)

        # self.train_data = self.pca_.transform(self.train)
        # self.test_data = self.pca_.transform(self.test)

    def transform(self, data):
        result = self.pca_.transform(data)
        return result


def average_fft_of_singal_trail(x, fft_size):
    n = len(x) // fft_size * fft_size
    tmp = x[:n].reshape(-1, fft_size)
    # tmp *= signal.hann(fft_size, sym=0)
    xf = np.abs(np.fft.rfft(tmp)/fft_size)
    avgf = np.average(xf, axis=0)
    return avgf


def feature_extract(x, fft_size=500, step=50):
    signal_len = x.size
    times = (signal_len-500)//step
    feature = []
    for i in xrange(times):
        tmp = x[i*step:(i*step+fft_size)]
        # tmp *= signal.hann(200, sym=0)
        xf = np.abs(np.fft.rfft(tmp) / fft_size)[5:45].reshape(-1, 1)# 提取5-45频域
        feature.append(xf)
    feature = np.hstack(feature)
    # x_ = np.linspace(5, 44, 40)
    # plt.figure('f_all')
    # plt.plot(feature.T)
    # plt.xlabel('time')
    # plt.figure('f_average')
    # plt.plot(x_, f_average)
    # plt.xlabel('freqs')
    # plt.show()
    return feature

def data_analyse():
    fft_size = 200
    freqs = np.linspace(0, 250, fft_size / 2 + 1)
    pca = Pca()

    data = pca.right_hand
    avgf_c3_list = []
    avgf_c4_list = []
    feature_right = []
    for i in xrange(data.shape[0]):
        c3 = data.reshape(-1, 800, 2)[i][:, 0]
        c4 = data.reshape(-1, 800, 2)[i][:, 1]
        avgf_c3 = average_fft_of_singal_trail(c3, fft_size)
        avgf_c4 = average_fft_of_singal_trail(c4, fft_size)
        avgf = np.vstack([avgf_c3, avgf_c4])
        feature_right.append(avgf)
        avgf_c3_list.append(avgf_c3)
        avgf_c4_list.append(avgf_c4)
    pl.figure(1)
    pl.subplot(211)
    for item in avgf_c3_list:
        pl.plot(freqs, item)
    pl.subplot(212)
    for item in avgf_c4_list:
        pl.plot(freqs, item)

    data = pca.left_hand
    avgf_c3_list = []
    avgf_c4_list = []
    feature_left = []
    for i in xrange(data.shape[0]):
        c3 = data.reshape(-1, 800, 2)[i][:, 0]
        c4 = data.reshape(-1, 800, 2)[i][:, 1]
        avgf_c3 = average_fft_of_singal_trail(c3, fft_size)
        avgf_c4 = average_fft_of_singal_trail(c4, fft_size)
        avfg = np.vstack([avgf_c3, avgf_c4])
        feature_left.append(avgf)
        avgf_c3_list.append(avgf_c3)
        avgf_c4_list.append(avgf_c4)
    pl.figure(2)
    pl.subplot(211)
    for item in avgf_c3_list:
        pl.plot(freqs, item)
    pl.subplot(212)
    for item in avgf_c4_list:
        pl.plot(freqs, item)

    data = pca.rest
    avgf_c3_list = []
    avgf_c4_list = []
    feature_rest = []
    for i in xrange(data.shape[0]):
        c3 = data.reshape(-1, 800, 2)[i][:, 0]
        c4 = data.reshape(-1, 800, 2)[i][:, 1]
        avgf_c3 = average_fft_of_singal_trail(c3, fft_size)
        avgf_c4 = average_fft_of_singal_trail(c4, fft_size)
        avgf = np.vstack([avgf_c3, avgf_c4])
        feature_rest.append(avgf)
        avgf_c3_list.append(avgf_c3)
        avgf_c4_list.append(avgf_c4)
    pl.figure(3)
    pl.subplot(211)
    for item in avgf_c3_list:
        pl.plot(freqs, item)
    pl.subplot(212)
    for item in avgf_c4_list:
        pl.plot(freqs, item)


    # feature_left = np.vstack(feature_left).reshape(-1, 2, 26)
    # feature_rest = np.vstack(feature_rest).reshape(-1, 2, 26)
    # feature_right = np.vstack(feature_right).reshape(-1, 2, 26)
    # feature_all = np.vstack([feature_left, feature_right, feature_rest])
    # print feature_right.shape
    # print feature_rest.shape
    # print feature_left.shape
    # print feature_all.shape
    # np.save('feature_p.npy', feature_all)
    # pl.figure(4)
    return


def bci_competition_analyse():
    all, c34, c34z = read_mat()
    data = c34
    fft_size = 1000
    sampling_rate = 1000
    freqs = np.linspace(0, sampling_rate/2, fft_size / 2 + 1)
    freqs_filter = freqs#[freqs<35]
    avgf_c3_list = []
    avgf_c4_list = []
    pl.figure(2)
    for i in xrange(data.shape[0]):
        c3 = data[i][:, 0]
        c3_avg = np.average(c3)
        if -100 < c3_avg < 250:
            pl.plot(c3)
        c4 = data[i][:, 1]
        avgf_c3 = average_fft_of_singal_trail(c3, fft_size)#[freqs < 35]
        avgf_c4 = average_fft_of_singal_trail(c4, fft_size)#[freqs < 35]
        avgf = np.vstack([avgf_c3, avgf_c4])
        avgf_c3_list.append(avgf_c3)
        avgf_c4_list.append(avgf_c4)
    pl.figure(1)
    pl.subplot(211)
    for item in avgf_c3_list:
        pl.plot(freqs_filter, item)
    pl.subplot(212)
    for item in avgf_c4_list:
        pl.plot(freqs_filter, item)
    pl.show()
    pl.plot()
    return


if __name__ == '__main__':
    all, c34, c34z = read_mat()
    c3_test = c34[0][:, 0]
    feature_test = feature_extract(c3_test)
    feature_test += 125
    pl.imshow(feature_test, 'gray')
    pl.show()
    # data_analyse()
    # bci_competition_analyse()
    pass