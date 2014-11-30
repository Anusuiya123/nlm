import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
from skimage import data_dir
from sklearn.decomposition import SparsePCA, PCA
from sklearn.neighbors.ball_tree import BallTree

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
img = noisy[3][:,:,0]
# img = np.zeros((100, 100))
# img[:50, :] = 1
# img += np.random.normal(0, 0.1, img.shape)




# precompute coordinate difference for the small patch



# denoised = nonlocalmeans(n, n_big=7, n_small=1)

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(new_img, cmap='gray')
# plt.subplot(1, 3, 3)
# plt.imshow(n, cmap='gray')
plt.show()


