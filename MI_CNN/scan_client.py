"""
a scan client, receiving data from scan server
"""
# -*- coding: utf-8 -*-
from socket import *
import struct
import time
import multiprocessing
import numpy as np
from LSTM_inference import lstm_server_ob


class client_process(multiprocessing.Process):
    HOST = '192.168.1.108'
    PORT = 4002
    BUFSIZ = 12
    ADDR = (HOST, PORT)
    signal_data = []

    def __init__(self, pip):
        multiprocessing.Process.__init__(self)
        self.client = socket(AF_INET, SOCK_STREAM)
        self.client.connect(self.ADDR)
        print 'connected'
        self.pip = pip

    def run(self):
        while True:
            data = self.client.recv(self.BUFSIZ)
            x, y, z = struct.unpack("<iii", data)
            self.signal_data.append(x)
            self.signal_data.append(y)
            self.signal_data.append(z)
            if len(self.signal_data) == 3*20:
                self.pip.send(self.signal_data)
                self.signal_data = []


def inference(pip):
    sample_points = 800
    sample_data = []
    while True:
        data_receive = pip.recv()
        sample_data.append(data_receive)
        if len(sample_data) == sample_points:
            sample_data = np.array(sample_data)
            result = lstm_server_ob.predict(sample_data)
            sample_data = []


if __name__ == '__main__':
    pipe = multiprocessing.Pipe()
    client_p = client_process(pipe[0])
    inference_p = multiprocessing.Process(target=inference, args=(pipe[1],))

    client_p.start()
    inference_p.start()

    client_p.join()
    inference_p.join()
