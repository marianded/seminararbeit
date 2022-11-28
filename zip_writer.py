import os
import h5py
import numpy as np
import time as t
import matplotlib.pyplot as plt

file_name = "testData"
file_stats = os.stat(f"{file_name}.hdf5")
file_size = file_stats.st_size
chunksize = None

with h5py.File(f"{file_name}.hdf5", "r") as f_in:
    print(list(f_in.keys()))
    arr = f_in['dataset'][:]
    print(arr)

zippers = ["lzf", "gzip"]
compression_rates = []

for zipper in ["lzf", "gzip"]:
    begin = t.monotonic_ns()
    with h5py.File(f"{file_name}_{zipper}.hdf5", "w") as f_zipped:
        f_zipped.create_dataset(f"zipped_{zipper}", data=arr, chunks=chunksize, compression=f"{zipper}")
    end = t.monotonic_ns()
    zipped_file_stats = os.stat(f"{file_name}_{zipper}.hdf5")
    zipped_file_size = zipped_file_stats.st_size
    compression_rate = file_size / zipped_file_size
    compression_rates.append(compression_rate)
    print(f"{zipper} zip time: {(end - begin) / 100000000} s")
    print(f"Compression rate: {compression_rate}")
    print()

# Plot
plt.bar(zippers, compression_rates, width=1, edgecolor="white")
plt.ylabel("Compression rate")
plt.show()

for zipper in ["lzf", "gzip", "szip"]:
    begin = t.monotonic_ns()
    with h5py.File(f"{file_name}_{zipper}.hdf5", "r") as f_in:
        print(list(f_in.keys()))
        data = f_in[f"zipped_{zipper}"]
    end = t.monotonic_ns()
    print(f"{zipper} zip time: {(end - begin) / 100000000} s")

# Plot