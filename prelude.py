import itertools
import multiprocessing
import os
import pathlib

import numpy as np
import tensorflow as tf
from mpmath import besselj

from global_settings import *


QPSK_CANDIDATE_SIZE = 2 ** (2 * NUM_ANT)
QPSK_CANDIDATES = np.array([x for x in itertools.product([1, -1], repeat=2 * NUM_ANT)]).T / np.sqrt(2)


def jbtest(data):
    mean_value = np.mean(data)
    s = np.mean((data - mean_value) ** 3) / (np.mean((data - mean_value) ** 2) ** (3 / 2))
    k = np.mean((data - mean_value) ** 4) / (np.mean((data - mean_value) ** 2) ** (4 / 2))
    return s ** 2 / 6 + (k - 3) ** 2 / 24


def get_bits(x):
    return np.where(x < 0, 0, 1)


def mkdir(file_path):
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)


def mkfile(file_path):
    mkdir(file_path)
    filename = pathlib.Path(file_path)
    filename.touch(exist_ok=True)


def concatenate(total, part):
    return part if total is None else np.concatenate((total, part))


def tf_concat(a, b, axis):
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return tf.concat([a, b], axis=axis)


def random_h_batch():
    h_batch = None
    temp = 2 * np.pi * NORMALIZED_DOPPLER_FREQUENCY
    rho_h = float(besselj(0, temp))

    prev_h = None
    for _ in range(PACKETS_PER_BATCH):
        for t in range(TRANSMIT_TIMES_PER_PACKET):
            # 由于tensorflow不支持复数直接运算，所以我们需要分割为实部和虚部的形式
            real = np.random.randn(NUM_ANT, NUM_ANT)
            imag = np.random.randn(NUM_ANT, NUM_ANT)
            h = np.row_stack(
                (
                    np.column_stack((real, -imag)),
                    np.column_stack((imag, real)),
                )
            )
            h = h.reshape([1, 2 * NUM_ANT, 2 * NUM_ANT])        
            if t == 0:
                h_batch = concatenate(h_batch, h)
                prev_h = h
            else:
                doppler_h = rho_h * prev_h + np.sqrt(1 - rho_h * rho_h) * h
                h_batch = concatenate(h_batch, doppler_h)
                prev_h = doppler_h
    
    return h_batch


def random_s_batch():
    s_batch = None
    one_hot_batch = np.zeros([TRANSMIT_TIMES_PER_BATCH, QPSK_CANDIDATE_SIZE])
    random_indexes = np.random.uniform(low=0, high=QPSK_CANDIDATE_SIZE, size=TRANSMIT_TIMES_PER_BATCH)
    for t in range(TRANSMIT_TIMES_PER_BATCH):
        i = int(random_indexes[t])
        one_hot_batch[t, i] = 1
        s = QPSK_CANDIDATES[:, i:i + 1]
        s = s.reshape([1, 2 * NUM_ANT, 1])
        s_batch = concatenate(s_batch, s)
    return s_batch, one_hot_batch


def time_correlated_interference_batch(rho: float):
    w_batch = None
    for t in range(TRANSMIT_TIMES_PER_BATCH):
        u = np.random.randn(1, 2 * NUM_ANT, 1)
        if t == 0:
            w_batch = concatenate(w_batch, u)
        else:
            w_prev = w_batch[t - 1:t, :, :]
            w = np.sqrt(rho) * w_prev + np.sqrt(1 - rho) * u
            w_batch = concatenate(w_batch, w)
    return w_batch


def mld_batch(y, h):
    s_estimated_batch = None
    dst = np.sum(np.square(y - h @ QPSK_CANDIDATES), axis=1)
    min_indexes_on_axis_1 = np.unravel_index(dst.argmin(1), dst.shape)[1]
    for t in range(TRANSMIT_TIMES_PER_BATCH):
        index = min_indexes_on_axis_1[t]
        s_estimated = QPSK_CANDIDATES[:, index]
        s_estimated = s_estimated.reshape([1, 2 * NUM_ANT, 1])
        s_estimated_batch = concatenate(s_estimated_batch, s_estimated)
    return s_estimated_batch
 

def produce_data_batch(rho, sir):
    power = 10 ** (sir / 10)
    h = np.sqrt(power / NUM_ANT) * random_h_batch()
    s, one_hot = random_s_batch()
    w = time_correlated_interference_batch(rho)
    y = h @ s + w

    hat_s = mld_batch(y, h)
    hat_w = y - h @ hat_s

    return y, h, s, one_hot, w, hat_s, hat_w


class DataSet:
    def __init__(self, flag, rho: float, sir: float):
        self.flag = flag
        self.rho = rho
        self.sir = sir

    def __open_file(self, name, mode):
        if self.flag == 0:
            file_name = "savedData/rho{:.1f}_sir{}/train/{}".format(
                self.rho, self.sir, name)
        elif self.flag == 1:
            file_name = "savedData/rho{:.1f}_sir{}/valid/{}".format(
                self.rho, self.sir, name)
        else:
            file_name = "savedData/rho{:.1f}_sir{}/test/{}".format(
                self.rho, self.sir, name)
        mkfile(file_name)
        return open(file_name, mode)

    def produce_func(self, _idx):
        return produce_data_batch(self.rho, self.sir)

    def __open_all(self, mode):
        file_y = self.__open_file("y", mode)
        file_h = self.__open_file("h", mode)
        file_s = self.__open_file("s", mode)
        file_one_hot = self.__open_file("one_hot", mode)
        file_w = self.__open_file("w", mode)
        file_s_mld = self.__open_file("s_mld", mode)
        file_w_mld = self.__open_file("w_mld", mode)
        return file_y, file_h, file_s, file_one_hot, file_w, file_s_mld, file_w_mld

    def __delete_file(self, name):
        if self.flag == 0:
            file_name = "savedData/rho{:.1f}_sir{}/train/{}".format(
                self.rho, self.sir, name)
        elif self.flag == 1:
            file_name = "savedData/rho{:.1f}_sir{}/valid/{}".format(
                self.rho, self.sir, name)
        else:
            file_name = "savedData/rho{:.1f}_sir{}/test/{}".format(
                self.rho, self.sir, name)
        if os.path.exists(file_name):
            os.remove(file_name)

    def delete_all(self):
        self.__delete_file("y")
        self.__delete_file("h")
        self.__delete_file("s")
        self.__delete_file("one_hot")
        self.__delete_file("w")
        self.__delete_file("s_mld")
        self.__delete_file("w_mld")

    def produce_all(self):
        file_y, file_h, file_s, file_one_hot, file_w, file_s_mld, file_w_mld = self.__open_all("wb")

        if self.flag == 0:
            total_batch = TRAIN_TOTAL_BATCH
        elif self.flag == 1:
            total_batch = VALID_TOTAL_BATCH
        else:
            total_batch = TEST_TOTAL_BATCH

        if NUM_WORKERS > 0:
            pool = multiprocessing.pool.Pool(NUM_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD)
        else:
            pool = multiprocessing.pool.Pool(maxtasksperchild=MAX_TASKS_PER_CHILD)
        idx = 0
        for ret_value in pool.imap(self.produce_func, range(total_batch)):
            if self.flag == 0:
                print("Train set，batch {}/{}".format(idx + 1, total_batch), end="\r")
            elif self.flag == 1:
                print("Valid set，batch {}/{}".format(idx + 1, total_batch), end="\r")
            else:
                print("Test set，batch {}/{}".format(idx + 1, total_batch), end="\r")
            
            ret_value[0].astype(np.float32).tofile(file_y)
            ret_value[1].astype(np.float32).tofile(file_h)
            ret_value[2].astype(np.float32).tofile(file_s)
            ret_value[3].astype(np.float32).tofile(file_one_hot)
            ret_value[4].astype(np.float32).tofile(file_w)
            ret_value[5].astype(np.float32).tofile(file_s_mld)
            ret_value[6].astype(np.float32).tofile(file_w_mld)

            file_y.flush()
            file_h.flush()
            file_s.flush()
            file_one_hot.flush()
            file_w.flush()
            file_s_mld.flush()
            file_w_mld.flush()

            idx += 1        
        pool.close()

        file_y.close()
        file_h.close()
        file_s.close()
        file_one_hot.close()
        file_w.close()
        file_s_mld.close()
        file_w_mld.close()

        print()
        print("数据集生成完毕")

    def fetch(self):
        file_y, file_h, file_s, file_one_hot, file_w, file_s_mld, file_w_mld = self.__open_all("rb")
        if self.flag == 0:
            total_batch = TRAIN_TOTAL_BATCH
        elif self.flag == 1:
            total_batch = VALID_TOTAL_BATCH
        else:
            total_batch = TEST_TOTAL_BATCH

        for i in range(total_batch):
            file_y.seek(i * TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1)
            file_h.seek(i * TRANSMIT_TIMES_PER_BATCH *2 * NUM_ANT * 2 * NUM_ANT)
            file_s.seek(i * TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1)
            file_one_hot.seek(i * TRANSMIT_TIMES_PER_BATCH * QPSK_CANDIDATE_SIZE)
            file_w.seek(i * TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1)
            file_s_mld.seek(i * TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1)
            file_w_mld.seek(i * TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1)

            y = np.fromfile(
                file_y,
                dtype=np.float32,
                count=TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            h = np.fromfile(
                file_h,
                dtype=np.float32,
                count=TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 2 * NUM_ANT
            ).reshape([-1, 2 * NUM_ANT, 2 * NUM_ANT])

            s = np.fromfile(
                file_s,
                dtype=np.float32,
                count=TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            one_hot = np.fromfile(
                file_one_hot,
                dtype=np.float32,
                count=TRANSMIT_TIMES_PER_BATCH * QPSK_CANDIDATE_SIZE
            ).reshape([-1, QPSK_CANDIDATE_SIZE])

            w = np.fromfile(
                file_w,
                dtype=np.float32,
                count=TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            hat_s = np.fromfile(
                file_s_mld,
                dtype=np.float32,
                count=TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            hat_w = np.fromfile(
                file_w_mld,
                dtype=np.float32,
                count=TRANSMIT_TIMES_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            yield y, h, s, one_hot, w, hat_s, hat_w

        file_y.close()
        file_h.close()
        file_s.close()
        file_one_hot.close()
        file_w.close()
        file_s_mld.close()
        file_w_mld.close()


def test_mld_batch(rho, sir_db):
    err_mld = 0.0
    total_bits = 0.0
    idx = 0
    while idx < 1000:
        y, h, s, one_hot, w, s_mld, w_mld = produce_data_batch(rho, sir_db)
        bits = get_bits(s)
        bits_mld = get_bits(s_mld)
        err_mld += len(np.argwhere(bits_mld != bits))
        total_bits += bits.size
        ber_mld = err_mld / total_bits
        print("MLD, rho={}, sir={}dB, batch={:04}/1000, BER={:e}({:,.0f}/{:,.0f})".format(rho, sir_db, idx + 1, ber_mld, err_mld, total_bits), end="\r")
        idx += 1  
    print("")


def check_data_set_sir(rho:float, sir_db:float):
    hs_batch = None
    w_batch = None
    for i in range(100):
        y, h, s, one_hot, w, s_mld, w_mld = produce_data_batch(rho, sir_db)
        print("正在生成数据集 {}/100".format(i + 1), end="\r")
        hs_batch = concatenate(hs_batch, h @ s)
        w_batch = concatenate(w_batch, w)
    test_sir = np.sum(hs_batch ** 2) / np.sum(w_batch ** 2)
    test_sir_db = 10 * np.log10(test_sir)
    print("")
    print("数据集rho={}, 目标SIR={}dB, 检验SIR={:.2f}dB".format(rho, sir_db, test_sir_db))


if __name__ == "__main__":
    test_mld_batch(rho=0.5, sir_db=10)
