import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
from skimage import data_dir
from skimage.transform import resize
from nlm import nonlocalmeans


def PSNR(original, noisy, peak=100):
    mse = np.mean((original-noisy)**2)
    return 10*np.log10(peak*peak/mse)

noise_var = np.logspace(-4, -1, 5)

lena = io.ImageCollection(data_dir + "/lena.png")[0].astype(np.float) / 255
lena = resize(lena, (lena.shape[0]/2, lena.shape[1]/2))

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
original = lena[:, :, 0]
img += np.random.normal(0, 0.1, img.shape)

# estimate noise power by fitting an optimal low-pass filter
upper = img[:-2, 1:-1].flatten()
lower = img[2:, 1:-1].flatten()
left = img[1:-1, :-2].flatten()
right = img[1:-1, 2:].flatten()
central = img[1:-1, 1:-1].flatten()
U = np.column_stack((upper, lower, left, right))
c_estimated = np.dot(U, np.dot(np.linalg.pinv(U), central))
error = np.mean((central - c_estimated)**2)
sigma = np.sqrt(error)

print("Estimated noise sigma: {0:.4f}".format(sigma))
new_img = nonlocalmeans(img, algorithm="clustered", h=10*sigma)

print("PSNR of noisy image: {0}dB".format(PSNR(original, img)))
print("PSNR of reconstructed image: {0}dB".format(PSNR(original, new_img)))


plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(new_img, cmap='gray')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(img-original, cmap='gray')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(new_img-original, cmap='gray')
plt.colorbar()
plt.show()


