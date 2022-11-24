import h5py
import numpy as np
import time as t


f = h5py.File('NEONDSTowerTemperatureData.hdf5', 'r')
arr = np.arange(100)
chunksize = (512, 512)

def chunk_data():
    pass


def write_lzf():
    begin = t.time()
    f.create_dataset("zipped_gzip", data=arr, chunks=chunksize, compression="lzf")
    end = t.time()


def write_gzip():
    begin = t.time()
    f = h5py.File('mytestfile.hdf5', 'r')
    f.create_dataset("zipped_gzip", data=arr, compression="gzip")
    end = t.time()


def write_szip():
    begin = t.time()
    f = h5py.File('mytestfile.hdf5', 'r')
    f.create_dataset("zipped_szip", data=arr, compression="szip")
    end = t.time()


