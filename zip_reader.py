import h5py
import time as t

filename = "testData"

for zipper in ["lzf", "gzip", "szip"]:
    begin = t.monotonic_ns()
    with h5py.File(f"{filename}_{zipper}.hdf5", "r") as f_in:
        print(list(f_in.keys()))
    end = t.monotonic_ns()
    print(f"{zipper} zip time: {(end - begin) / 100000000} s")