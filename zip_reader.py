import h5py
import time as t


def read_lzf(filename: str):
    begin = t.time()
    f = h5py.File(filename, 'r')
    end = t.time()


def read_gzip(filename: str):
    begin = t.time()
    end = t.time()


def read_szip(filename: str):
    begin = t.time()
    end = t.time()