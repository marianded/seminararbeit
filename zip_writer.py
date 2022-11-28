import itertools
import os
import h5py
import numpy as np
import time as t
import matplotlib.pyplot as plt

file_name = "BrainEmptyLMP3D-01_00_s2208_PM_Complete_Raw_Tiles_Flat_v002"
# file_name = "PE-2011-00015-H_00_s0199_PM_Complete_Raw_Tiles_Flat_v003"

file_path = "/data/PLI-Group/felix/data/hdf5-test"
file_stats = os.stat(f"{file_path}/{file_name}.h5")
file_size = file_stats.st_size
chunks = [None]
#chunks = [None, (128, 128, 18), (256, 256, 18), (512, 512, 18)]
data = {}
image_attributs = {}
pyramid_attributs = {}

print(file_size)

with h5py.File(f"{file_path}/{file_name}.h5", "r") as f_in:
    image_data = f_in['Image'][:]
    for key in list(f_in["Image"].attrs.keys()):
        image_attributs[key] = f_in["Image"].attrs[key]
    for key in list(f_in["pyramid"].attrs.keys()):
        pyramid_attributs[key] = f_in["pyramid"].attrs[key]
    for key in list(f_in["pyramid"].keys()):
        data[key] = f_in[f"pyramid/{key}"][:]


zippers = ["lzf", "gzip"]
compression_rates = []
run_times = []

for zipper, chunk in itertools.product(zippers, chunks):
    begin = t.monotonic_ns()
    with h5py.File(f"{file_name}_{zipper}_{chunk}.h5", "w") as f_zipped:
        # dset = f_zipped.create_dataset(f"Image_{zipper}", data=image_data, chunks=chunk, compression=f"{zipper}")
        # for key in list(image_attributs.keys()):
        #     dset.attrs[key] = image_attributs[key]
        pyramid = f_zipped.create_group("pyramid")
        for key in pyramid_attributs:
            pyramid.attrs[key] = pyramid_attributs[key]
        for key in data.keys():
            dset2 = pyramid.create_dataset(f"{key}", data=data[key], chunks=chunk, compression=f"{zipper}")
            # print(type(key), key)
            if key == "00":
                for key2 in list(image_attributs.keys()):
                    dset2.attrs[key2] = image_attributs[key2]
                image_ref = f_zipped.create_dataset("Image", (1,), dtype=h5py.ref_dtype)
                image_ref[0] = dset2.ref
                print(image_ref[0])
            #else:
            #    dset2.attrs["scale"] = pyramid_attributs["scales"][int(key)]
    end = t.monotonic_ns()
    zipped_file_stats = os.stat(f"{file_name}_{zipper}_{chunk}.h5")
    zipped_file_size = zipped_file_stats.st_size
    compression_rate = (1 - zipped_file_size / file_size) * 100
    compression_rates.append(compression_rate)
    print(f"{zipper} zip time {chunk}: {(end - begin) / 100000000}s")
    print(f"Compression rate: {compression_rate}")
    print()

# Plot
plt.barh(zippers, compression_rates, edgecolor="white")
plt.xlabel("Memory savings in %")
plt.legend()
plt.xlim(-100, 100)
plt.show()

for zipper, chunk in itertools.product(zippers, chunks):
    begin = t.monotonic_ns()
    with h5py.File(f"{file_name}_{zipper}_{chunk}.h5", "r") as f_in:
        print(list(f_in.keys()))
        image_data = f_in[f"Image_{zipper}"][:]
        # pyramid = f_in[f"pyramid_{zipper}"][:]
    end = t.monotonic_ns()
    run_times.append((end - begin) / 100000000)
    print(f"{zipper} extract time: {(end - begin) / 100000000} s")

# Plot
plt.barh(zippers, run_times)
plt.xlabel("Leselaufzeit in s")
plt.legend()
plt.show()


# TODO SZIP Lizens
# TODO Referenz von orginal Bild
# TODO Chunking
# TODO gzip einstellung (0 - 9)