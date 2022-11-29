import itertools
import os
import h5py
import numpy as np
import time as t
import matplotlib.pyplot as plt

file_name = "BrainEmptyLMP3D-01_00_s2208_PM_Complete_Raw_Tiles_Flat_v002"
# file_name = "PE-2011-00015-H_00_s0199_PM_Complete_Raw_Tiles_Flat_v003"

file_path = "/data/PLI-Group/felix/data/hdf5-test"
# file_path = "E:\MD\Seminararbeit"
file_stats = os.stat(f"{file_path}/{file_name}.h5")
file_size = file_stats.st_size
data = {}
image_attributs = {}
pyramid_attributs = {}
values = []

print(file_size)

with h5py.File(f"{file_path}/{file_name}.h5", "r", rdcc_nbytes=5252880) as f_in:
    #image_data = f_in['Image'][:]
    #image_shape = np.shape(image_data)
    #print(image_shape)
    for key in list(f_in["Image"].attrs.keys()):
        image_attributs[key] = f_in["Image"].attrs[key]
    for key in list(f_in["pyramid"].attrs.keys()):
        pyramid_attributs[key] = f_in["pyramid"].attrs[key]
    for key in list(f_in["pyramid"].keys()):
        data[key] = f_in[f"pyramid/{key}"][:]
        if key == "00":
            image_shape = np.shape(data[key])
        values.append(f_in[f"pyramid/{key}"].attrs["scale"])

chunks = [None]
chunks = [True, None, (256, 256, 1), (256, 256, image_shape[2])]
zippers = ["szip", "lzf", "gzip"]


def compression():
    compression_rates = []
    for zipper, chunk in itertools.product(zippers, chunks):
        begin = t.monotonic_ns()
        with h5py.File(f"{file_name}_{zipper}_{chunk}.h5", "w") as f_zipped:
            pyramid = f_zipped.create_group("pyramid")
            for key in pyramid_attributs:
                pyramid.attrs[key] = pyramid_attributs[key]
            for key in data.keys():
                if chunk is None or chunk is True or np.shape(data[key])[0] > chunk[0]:
                    dset2 = pyramid.create_dataset(f"{key}", data=data[key], chunks=chunk, compression=f"{zipper}")
                else:
                    dset2 = pyramid.create_dataset(f"{key}", data=data[key], chunks=None, compression=f"{zipper}")
                if key == "00":
                    for key2 in list(image_attributs.keys()):
                        dset2.attrs[key2] = image_attributs[key2]
                    f_zipped["Image"] = h5py.SoftLink("/pyramid/00")
                    #image_ref = f_zipped.create_dataset("Image", (1,), dtype=h5py.ref_dtype)
                    #image_ref[0] = dset2.ref
                else:
                    dset2.attrs["scale"] = values[int(key)]
        end = t.monotonic_ns()
        zipped_file_stats = os.stat(f"{file_name}_{zipper}_{chunk}.h5")
        zipped_file_size = zipped_file_stats.st_size
        compression_rate = (1 - zipped_file_size / file_size) * 100
        compression_rates.append(compression_rate)
        print(f"{zipper} zip time {chunk}: {(end - begin) / 1000000000}s")
        print(f"Compression rate: {compression_rate}")
        print()

    # Plot
    compression_rates = np.reshape(compression_rates, (len(zippers), len(chunks))).T

    Labels = zippers
    y_pos = np.arange(len(Labels))
    plt.bar(y_pos + 0, compression_rates[0], width=0.2, color='navy', label='Auto chunk')
    plt.bar(y_pos + 0.2, compression_rates[1], width=0.2, color='skyblue', label='None chunk')
    plt.bar(y_pos + 0.4, compression_rates[2], width=0.2, color='darkcyan', label='256x256x1')
    plt.bar(y_pos + 0.6, compression_rates[3], width=0.2, color='black', label=f'256x26x{image_shape[2]}')

    plt.xticks(y_pos, Labels)
    plt.legend(('Auto chunk', 'None chunk', '256x256x1', f'256x256x{image_shape[2]}'))
    plt.xlabel('Verlustfreie Kompressionsalgorithmen')
    plt.ylabel('Memory savings in %')
    plt.title("Analyse der Kompressionsalgritmen")
    plt.show()
    # plt.savefig('Memory_savings.pdf', dpi=300)

def gzip_compression():
    compression_rates = []
    for chunk, i in itertools.product(chunks, np.arange(10)):
        begin = t.monotonic_ns()
        with h5py.File(f"{file_name}_gzip_{chunk}.h5", "w") as f_zipped:
            pyramid = f_zipped.create_group("pyramid")
            for key in pyramid_attributs:
                pyramid.attrs[key] = pyramid_attributs[key]
            for key in data.keys():
                if chunk is None or chunk is True or np.shape(data[key])[0] > chunk[0]:
                    dset2 = pyramid.create_dataset(f"{key}", data=data[key], chunks=chunk, compression="gzip", compression_opts=i)
                else:
                    dset2 = pyramid.create_dataset(f"{key}", data=data[key], chunks=None, compression="gzip", compression_opts=i)
                if key == "00":
                    for key2 in list(image_attributs.keys()):
                        dset2.attrs[key2] = image_attributs[key2]
                    image_ref = f_zipped.create_dataset("Image", (1,), dtype=h5py.ref_dtype)
                    image_ref[0] = dset2.ref
                else:
                    dset2.attrs["scale"] = values[int(key)]
        end = t.monotonic_ns()
        zipped_file_stats = os.stat(f"{file_name}_gzip_{chunk}.h5")
        zipped_file_size = zipped_file_stats.st_size
        compression_rate = (1 - zipped_file_size / file_size) * 100
        compression_rates.append(compression_rate)
        print(f"gzip zip time chunk {chunk}, compression_opts {i}: {(end - begin) / 1000000000}s")
        print(f"Compression rate: {compression_rate}")
        print()

        # Plot
    compression_rates = np.reshape(compression_rates, (len(chunks), 10)).T

    Labels = chunks
    y_pos = np.arange(len(Labels))
    for i in np.arange(10):
        plt.bar(y_pos + (i / 10.), compression_rates[i], width=0.1, color="", label='Auto chunk')

    plt.xticks(y_pos, Labels)
    plt.legend(('Auto chunk', 'None chunk', '256x256x1', f'256x256x{image_shape[2]}'))
    plt.xlabel('gzip compression_opts')
    plt.ylabel('Kompression in %')
    plt.title("Analyse der Kompressionsalgritmen")
    plt.show()


def runtime() -> np.ndarray:
    read_times = []
    for zipper, chunk in itertools.product(zippers, chunks):
        begin = t.monotonic_ns()
        with h5py.File(f"{file_name}_{zipper}_{chunk}.h5", "r") as f_in:
            for key in list(f_in["Image"].attrs.keys()):
                image_attributs[key] = f_in["Image"].attrs[key]
            for key in list(f_in["pyramid"].attrs.keys()):
                pyramid_attributs[key] = f_in["pyramid"].attrs[key]
            for key in list(f_in["pyramid"].keys()):
                data[key] = f_in[f"pyramid/{key}"][:]
                if key == "00":
                    image_shape = np.shape(data[key])
                values.append(f_in[f"pyramid/{key}"].attrs["scale"])
        end = t.monotonic_ns()
        read_times.append((end - begin) / 1000000000)
        print(f"{zipper} extract time: {(end - begin) / 1000000000} s")

    # Plot
    read_times = np.reshape(read_times, (len(zippers), len(chunks))).T
    title = "Analyse der Kompressionsalgritmen"
    xlabel = "Verlustfreie Kompressionsalgorithmen"
    ylabel = "Leselaufzeit in s"
    legend = ('Auto chunk', 'None chunk', '256x256x1', f'256x256x{image_shape[2]}')
    # plot(read_times, title, zippers, legend, xlabel, ylabel)
    return read_times


def run_x_times(i: int):
    read_times = []
    for _ in range(i):
        read_times.append(runtime())
    print(read_times)
    errors = np.array(0)

    middle_times = np.average(read_times, axis=0)
    print("middle", middle_times)
    # read_times = np.rot90(read_times, 1, (2, 0))
    errors1 = np.max(read_times, axis=0)
    print("e1", errors1)
    errors2 = np.min(read_times, axis=0)
    print("e2", errors2)
    errors = np.concatenate((np.max(read_times, axis=0), np.min(read_times, axis=0))).reshape((middle_times.shape[0], middle_times.shape[1], 2))
    print("error", errors)

    # Plot
    title = "Analyse der Kompressionsalgritmen"
    xlabel = "Verlustfreie Kompressionsalgorithmen"
    ylabel = "Leselaufzeit in s"
    legend = ('Auto chunk', 'None chunk', '256x256x1', f'256x256x{image_shape[2]}')
    plot(middle_times, title, zippers, legend, xlabel, ylabel, errors)


def plot(plot_data: np.ndarray, title: str, labels: np.ndarray, legend: tuple, xlabel: str, ylabel: str, error: np.ndarray=[]):
    y_pos = np.arange(len(labels))
    plt.bar(y_pos + 0, plot_data[0], width=0.2, color='navy', label='Auto chunk')
    plt.bar(y_pos + 0.2, plot_data[1], width=0.2, color='skyblue', label='None chunk')
    plt.bar(y_pos + 0.4, plot_data[2], width=0.2, color='darkcyan', label='256x256x1')
    plt.bar(y_pos + 0.6, plot_data[3], width=0.2, color='black', label=f'256x26x{image_shape[2]}')

    if len(error):
        (_, caps, _) = plt.errorbar(
            , plot_data, yerr=error, fmt='o', markersize=8, capsize=20)

        for cap in caps:
            cap.set_markeredgewidth(1)

    plt.xticks(y_pos, labels)
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


#compression()
#runtime()
#gzip_compression()
run_x_times(5)

# TODO SZIP Lizens
# TODO Referenz von orginal Bild
# TODO Plot mit Laufzeit und Kompression zusammen
