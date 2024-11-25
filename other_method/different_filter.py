from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as MSE
import cv2

from PIL import Image
import numpy as np
from scipy import signal
from scipy import ndimage
import numpy as np
import random
from scipy.optimize import leastsq
from scipy.signal import wiener
from pywt import Wavelet, dwt2, idwt2
class LogicClass:

    def fun(p,x): 
        f = np.poly1d(p)
        return f(x)

    def error(p,x,y):
        regularization = 0.01
        ret = LogicClass.fun(p,x)-y
        ret = np.append(ret, np.sqrt(regularization) * p)
        return ret


    def deletenoise(image):
        X = np.arange(0, 9) 
        mask = [0,0,0,0,0,0,0,0,0]
        for j in range(1,image.shape[0]-1):
            for i in range(1,image.shape[1]-1):
                for h in range(3):
                    for g in range(3):
                        y = j + h - 1
                        x = i + g - 1
                        mask[h*3+g]=image[x,y]
                p0 = np.random.randn(2)
                para = leastsq(LogicClass.error,p0,args=(X,mask))
                k,b = para[0]
                value = []
                for n in range(9):
                    value.append(mask[n]-(k*n+b))
                if np.argmax(np.absolute(value)) == 4:
                    image[i,j] = k*4+b
        return image
def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
 
 
def ssim_1(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
 
def mssim(img1, img2):

    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim_1(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt

def wavelet(image, wavelet='haar', level=6):
    # Convert image to grayscale if it's not already

    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_thresh = list(coeffs)
    coeffs_thresh[1:] = [tuple(pywt.threshold(c, threshold, mode='soft') for c in detail) for detail in coeffs[1:]]
    
    # Reconstruct the image
    denoised_image = pywt.waverec2(coeffs_thresh, wavelet)
    
    # Clip values to the valid range and convert to uint8 and done
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    
    return denoised_image
def non_local_filtered(image):


    return cv2.fastNlMeansDenoising(image, None, 10,  7, 21)

def apply_mean_filter(image):
    return cv2.blur(image, (3, 3))

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (3, 3), 1.0)

def zuixiaoercheng(image):
    return LogicClass.deletenoise(image)

def bilateral_filtered(image):
    return cv2.bilateralFilter(image, 9, 75, 75)
def boxFilter(image):
    return cv2.boxFilter(image, -1, (5,5), normalize=1)#方框滤波

def apply_median_filter(image):
    return cv2.medianBlur(image, 3)
def filter2D(image):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, ddepth=0, kernel=kernel)

image=cv2.imread('D:\\256_5.0.bmp',cv2.IMREAD_GRAYSCALE)



median = bilateral_filtered(image)
h="D:\\bilateral_filtered.bmp"
cv2.imwrite(h, median)

print(h)
img1 = cv2.imread("D:\\lena.bmp",cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(h,cv2.IMREAD_GRAYSCALE)

print(round(MSE(img1, img3),3))
print(round(psnr(img1, img3),3))
print(round(ssim(img1, img3),3))                            
print(round(mssim(img1, img3),3))
