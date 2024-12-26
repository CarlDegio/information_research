import os
import numpy as np
import re
import matplotlib.pyplot as plt


def load_npy_files(directory):
    npy_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files


def extract_mnist_ratio(file_path):
    match = re.search(r'mnist_ratio=([\d.]+)', file_path)
    if match:
        return match.group(1)
    raise ValueError(f'No mnist_ratio found in {file_path}')


def plot_entropy():
    entropy_dict = {}
    npy_files = load_npy_files('models/colored')
    for npy_file in npy_files:
        data = np.load(npy_file)
        data = data[data != 0]  # Filter out zeros from the data array
        ratio = extract_mnist_ratio(npy_file)
        entropy_dict[ratio] = data

    plt.figure(figsize=(10, 6))
    for ratio, entropy in entropy_dict.items():
        plt.scatter(float(ratio), entropy.mean(axis=0), label=f'MNIST Ratio: {ratio}')
    plt.xlabel('Bins')
    plt.ylabel('Entropy')
    plt.title('Entropy vs. Bins for Different MNIST Ratios')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_entropy()
