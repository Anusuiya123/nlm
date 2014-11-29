import multiprocessing as mp
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
from skimage import data_dir
import numpy as np
import time
from itertools import product
from functools import reduce, partial

noise_var = np.logspace(-4, -1, 5)

lena = io.ImageCollection(data_dir + "/lena.png")[0].astype(np.float) / 255

# make lena artificial image
lena = np.zeros((100, 100, 3))
lena[:50, :, :] = 1

noisy = []
for sigma2 in noise_var:
    noise = np.random.normal(0, np.sqrt(sigma2), lena.shape)
    n = lena + noise
    # avoid going over bounds
    n[n > 1] = 1
    n[n < 0] = 0
    noisy.append(color.rgb2lab(n))

lena = color.rgb2lab(lena)

n = noisy[2][:, :, 0]
o = lena[:, :, 0]
new_n = np.zeros_like(n)

n_big = 7
n_small = 1
Np = (2 * n_small + 1) ** 2
h2 = 100
n_rows, n_cols = n.shape

# precompute the coordinate difference for the big patch
D = range(-n_big, n_big + 1)
big_diff = [(r, c) for r in D for c in D if not (r == 0 and c == 0)]

# precompute coordinate difference for the small patch
small_rows, small_cols = np.indices((2 * n_small + 1, 2 * n_small + 1)) - n_small

padding = n_big + n_small
n_padded = np.pad(n, padding, mode='reflect')


def make_arguments(n, *args):
    l = [[arg] * n for arg in args]
    return l


def distance(values, r, c, pixel_window, h2):
    patch_window, central_diff = values

    diff = np.sum((pixel_window - patch_window) ** 2)
    # remove the central distance from the computation
    diff -= central_diff

    w = np.exp(-diff / (h2 * Np))

    # return the color of the pixel and the weight associated with the patch
    nr,nc = patch_window.shape
    return w * patch_window[nr/2, nc/2], w


pool = mp.Pool(5)
for r in range(padding, padding + n_rows):
    if (r % 10) == 0:
        print(r)
    sum_time = 0
    for c in range(padding, padding + n_cols):
        start = time.clock()
        pixel_window = n_padded[small_rows + r, small_cols + c]

        # construct a list of patch_windows
        windows = [n_padded[small_rows + r + d[0], small_cols + c + d[1]] for d in big_diff]

        #construct a list of central differences
        central_diffs = [(n_padded[r, c] - n_padded[r + d[0], c + d[1]]) for d in big_diff]

        distance_map = partial(distance, r=r, c=c, pixel_window=pixel_window, h2=h2)

        distances = pool.map(distance_map, zip(windows, central_diffs))
        total_c, total_w = reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]), distances)
        new_n[r - padding, c - padding] = total_c / total_w
        end = time.clock()
        sum_time += end - start

    print("Time per pixel {0}".format(sum_time / n_cols))

plt.subplot(1, 3, 1)
plt.imshow(o)
plt.subplot(1, 3, 2)
plt.imshow(new_n)
plt.subplot(1, 3, 3)
plt.imshow(n)
plt.show()

