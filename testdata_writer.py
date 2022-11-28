import h5py
import numpy as np

with h5py.File("testData.hdf5", "w") as f:
    f.create_dataset("dataset", data=np.random.randint(0, 128 + 1, 10000).reshape((100, 100)))
