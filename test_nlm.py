import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
from skimage import data_dir

from nlm import _nonlocalmeans_naive

noise_var = np.logspace(-4, -1, 5)

lena = io.ImageCollection(data_dir + "/lena.png")[0].astype(np.float) / 255

noisy = []
for sigma2 in noise_var:
    noise = np.random.normal(0, np.sqrt(sigma2), lena.shape)
    n = lena + noise
    # avoid going over bounds
    n[n > 1] = 1
    n[n < 0] = 0
    noisy.append(color.rgb2lab(n))

lena = color.rgb2lab(lena)
n = noisy[3][:, :, 0]

denoised = _nonlocalmeans_naive(n, n_big=7, n_small=1)

plt.subplot(1, 3, 1)
plt.imshow(lena[:, :, 1], cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(denoised, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(n, cmap='gray')
plt.show()


