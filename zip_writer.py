import itertools
import os
from typing import Tuple, Any

import h5py
import matplotlib
import numpy as np
import time as t
import matplotlib.pyplot as plt
import gc

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#    "pgf.texsystem": "pdflatex",
#    'font.family': 'serif',
#    'text.usetex': True,
#    'pgf.rcfonts': False,
# })

# file_name = "BrainEmptyLMP3D-01_00_s2208_PM_Complete_Raw_Tiles_Flat_v002"
file_name = "PE-2011-00015-H_00_s0199_PM_Complete_Raw_Tiles_Flat_v003"

# file_path = "/data/PLI-Group/felix/data/hdf5-test" # Insti
# file_path = "E:\MD\Seminararbeit" # home
file_path = "/home/marianded/Desktop/seminararbeit/data"
file_stats = os.stat(f"{file_path}/{file_name[0]}/{file_name}.h5")
file_size = file_stats.st_size
data = {}
image_attributs = {}
pyramid_attributs = {}
values = []
chunks = [True, None, (256, 256, 1), (512, 512, 1), (1024, 1024, 1), (2048, 2048, 1)]
#chunks = [True, None, (256, 256, 1)]

print(file_size)

# rdcc 16-Bit-unsigned Integer = 16 * chunk_x * chunk_y * chunk_z + 1000
rdcc_size = 8389608

def read_unzipped_file():
    with h5py.File(f"{file_path}/{file_name[0]}/{file_name}.h5", "r", rdcc_nbytes=rdcc_size) as f_in:
        # image_data = f_in['Image'][:]
        # image_shape = np.shape(image_data)
        # print(image_shape)
        for key in list(f_in["Image"].attrs.keys()):
            image_attributs[key] = f_in["Image"].attrs[key]
        for key in list(f_in["pyramid"].attrs.keys()):
            pyramid_attributs[key] = f_in["pyramid"].attrs[key]
        for key in list(f_in["pyramid"].keys()):
            data[key] = f_in[f"pyramid/{key}"][:]
            if key == "00":
                image_shape = np.shape(data[key])
            values.append(f_in[f"pyramid/{key}"].attrs["scale"])
        chunks.append((256, 256, image_shape[2]))
        chunks.append((512, 512, image_shape[2]))
        chunks.append((1024, 1024, image_shape[2]))
        return image_shape


zippers = ["lzf", "gzip", "szip"]


def compression() -> tuple[np.ndarray, np.ndarray]:
    compression_rates = []
    compression_times = []
    for zipper, chunk in itertools.product(zippers, chunks):
        begin = t.monotonic_ns()
        with h5py.File(f"{file_path}/{file_name[0]}/{file_name}_{zipper}_{chunk}.h5", "w") as f_zipped:
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
                else:
                    dset2.attrs["scale"] = values[int(key)]
        end = t.monotonic_ns()
        compression_times.append((end - begin) / 1000000000)
        zipped_file_stats = os.stat(f"{file_path}/{file_name[0]}/{file_name}_{zipper}_{chunk}.h5")
        zipped_file_size = zipped_file_stats.st_size
        compression_rate = (1 - zipped_file_size / file_size) * 100
        compression_rates.append(compression_rate)
        print(f"{zipper} zip time {chunk}: {(end - begin) / 1000000000}s")
        print(f"Compression rate: {compression_rate}")
        print()

    # Plot compression rate
    compression_rates = np.reshape(compression_rates, (len(zippers), len(chunks))).T
    title = ""
    safe_title = f"{file_name[0]}_bar_charts_compression_rate"
    xlabel = f"{file_name}"
    ylabel = "compression rate in %"
    plot(compression_rates, title, safe_title, zippers, chunks, xlabel, ylabel)

    # Plot compression time
    compression_times = np.reshape(compression_times, (len(zippers), len(chunks))).T
    title = ""
    safe_title = f"{file_name[0]}_bar_charts_compression_time"
    xlabel = f"{file_name}"
    ylabel = "compression time in s"
    plot(compression_times, title, safe_title, zippers, chunks, xlabel, ylabel)
    save_file_name = f"{file_name[0]}_save_compression_time"
    save_file_name_2 = f"{file_name[0]}_save_compression_rate"
    save_file_path = "./data/saved"
    with open(f"{save_file_path}/{save_file_name}", "a") as f_save:
        f_save.write(str(compression_times))
        f_save.write("\n")
    with open(f"{save_file_path}/{save_file_name_2}", "a") as f_save:
        f_save.write(str(compression_rates))
        f_save.write("\n")
    return compression_rates, compression_times


def gzip_compression():
    compression_rates = []
    for chunk, i in itertools.product(chunks, np.arange(10)):
        begin = t.monotonic_ns()
        with h5py.File(f"{file_path}/{file_name[0]}/{file_name}_gzip_{chunk}.h5", "w") as f_zipped:
            print(t.monotonic_ns() - begin / 1000000000)
            pyramid = f_zipped.create_group("pyramid")
            for key in pyramid_attributs:
                pyramid.attrs[key] = pyramid_attributs[key]
            for key in data.keys():
                if chunk is None or chunk is True or np.shape(data[key])[0] > chunk[0]:
                    dset2 = pyramid.create_dataset(f"{key}", data=data[key], chunks=chunk, compression="gzip",
                                                   compression_opts=i)
                else:
                    dset2 = pyramid.create_dataset(f"{key}", data=data[key], chunks=None, compression="gzip",
                                                   compression_opts=i)
                if key == "00":
                    for key2 in list(image_attributs.keys()):
                        dset2.attrs[key2] = image_attributs[key2]
                    image_ref = f_zipped.create_dataset("Image", (1,), dtype=h5py.ref_dtype)
                    image_ref[0] = dset2.ref
                else:
                    dset2.attrs["scale"] = values[int(key)]
        end = t.monotonic_ns()
        zipped_file_stats = os.stat(f"{file_path}/{file_name[0]}/{file_name}_gzip_{chunk}.h5")
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


def decompression_time() -> np.ndarray:
    read_times = []
    for zipper, chunk in itertools.product(zippers, chunks):
        begin = t.monotonic_ns()
        with h5py.File(f"{file_path}/{file_name[0]}/{file_name}_{zipper}_{chunk}.h5", "r", rdcc_nbytes=rdcc_size) as f_in:
            for key in list(f_in["Image"].attrs.keys()):
                image_attributs[key] = f_in["Image"].attrs[key]
            for key in list(f_in["pyramid"].attrs.keys()):
                pyramid_attributs[key] = f_in["pyramid"].attrs[key]
            for key in list(f_in["pyramid"].keys()):
                data = {key: f_in[f"pyramid/{key}"][:]}
                if key == "00":
                    image_shape = np.shape(data[key])
                values.append(f_in[f"pyramid/{key}"].attrs["scale"])
                del data
                gc.collect()
        end = t.monotonic_ns()
        read_times.append((end - begin) / 1000000000)
        print(f"{zipper} extract time: {(end - begin) / 1000000000} s")

    # Plot
    read_times = np.reshape(read_times, (len(zippers), len(chunks))).T
    title = ""
    safe_title = f"{file_name[0]}_bar_chart_decompression_time"
    xlabel = f"{file_name}"
    ylabel = "decompression time in s"
    #plot(read_times, title, safe_title, zippers, chunks, xlabel, ylabel)
    save_file_name = f"{file_name[0]}_save_decompression_time"
    save_file_path = "./data/saved"
    with open(f"{save_file_path}/{save_file_name}", "a") as f_save:
        f_save.write(str(read_times))
    return read_times


def run_x_times(i: int, compress: bool=False, decompress: bool=False):
    if compress:
        compression_times = []
        compression_rates = []
        for _ in range(i):
            print(_)
            compression_rate, compression_time = compression()
            compression_times.append(compression_time)
            compression_rates.append(compression_rate)
            gc.collect()

        compression_mean_times = np.mean(compression_times, axis=0)
        c_t_errors = np.std(compression_times, axis=0)
        compression_mean_rate = np.mean(compression_rates, axis=0)
        c_r_errors = np.std(compression_rates, axis=0)

    if decompress:
        gc.collect()
        read_times = []
        for _ in range(i):
            print(_)
            read_times.append(decompression_time())

        mean_times = np.mean(read_times, axis=0)
        errors = np.std(read_times, axis=0)

    # Plot
    title = f"mean of {i} times"
    xlabel = f"{file_name}"
    legend = ('Auto chunk', 'None chunk', '256x256x1', f'256x256x{image_shape[2]}')
    if compress:
        ylabel = "compression time in s"
        safe_title = f"{file_name[0]}_bar_chart_compression_time_{i}"
        plot(compression_mean_times, title, safe_title, zippers, legend, xlabel, ylabel, c_t_errors)
        ylabel = "compression rate in %"
        safe_title = f"{file_name[0]}_bar_chart_compression_rate_{i}"
        plot(compression_mean_rate, title, safe_title, zippers, legend, xlabel, ylabel, c_r_errors)
    if decompress:
        ylabel = "decompression time in s"
        safe_title = f"{file_name[0]}_bar_chart_decompression_time_{i}"
        plot(mean_times, title, safe_title, zippers, legend, xlabel, ylabel, errors)

def plot(plot_data: np.ndarray, title: str, safe_title, labels: np.ndarray, legend: tuple, xlabel: str, ylabel: str,
         error=None):
    if error is None:
        error = []
    colors = ["navy", "skyblue", "darkcyan", "steelblue", "royalblue", "dodgerblue", "cornflowerblue", "turquoise",
              "lightseagreen", "8aa7fe"]
    y_pos = np.arange(len(labels))
    n = plot_data.shape[0]
    fig, ax = plt.subplots()
    ax.yaxis.grid(True)
    if len(error):
        ax.bar(y_pos + 0, plot_data[0], width=0.2, color='navy', yerr=error[0].T, ecolor='black', capsize=5,
               label='Auto chunk')
        ax.bar(y_pos + 0.2, plot_data[1], width=0.2, color='skyblue', yerr=error[1].T, ecolor='black', capsize=5,
               label='None chunk')
        ax.bar(y_pos + 0.4, plot_data[2], width=0.2, color='darkcyan', yerr=error[2].T, ecolor='black', capsize=5,
               label='256x256x1')
        ax.bar(y_pos + 0.6, plot_data[3], width=0.2, color='steelblue', yerr=error[3].T, ecolor='black', capsize=5,
               label=f'256x26x{image_shape[2]}')
    else:
        #print(plot_data)
        #print(labels)
        for i in np.arange(n):
            ax.bar(y_pos + i/n, plot_data[i], width=1/n, color=colors[i], label="")

    plt.xticks(y_pos, labels)
    #plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    # plt.savefig(f"{safe_title}.pgf")
    plt.savefig(f"./charts/{safe_title}.svg", format='svg', dpi=300)
   # plt.show()


image_shape = read_unzipped_file()
compression()
# gzip_compression()

del data
gc.collect()
#run_x_times(10, False, True)
#decompression_time()

# TODO SZIP Lizens
# TODO Referenz von orginal Bild
# TODO Plot mit Laufzeit und Kompression zusammen

# Pc plivispc01