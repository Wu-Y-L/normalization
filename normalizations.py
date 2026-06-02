from numpy.lib.format import dtype_to_descr
import numpy as np
import skimage 
from scipy.stats import gaussian_kde


def percentile_norm(img, pmin = 2 , pmax = 99.8, dtype = np.float32, eps = 1e-20):
    img = img.astype(dtype)
    min_intensity = np.percentile(img, pmin).astype(dtype)
    max_intensity = np.percentile(img, pmax).astype(dtype)
    return np.clip((img - min_intensity) / (max_intensity - min_intensity + eps), 0, 1)

def min_max_norm(img,dtype = np.float32, eps = 1e-20):
    img = img.astype(dtype)
    return ((img - img.min()) / (img.max() - img.min() + 1e-20))

def z_score_norm(img,dtype = np.float32, eps = 1e-20):
    img = img.astype(dtype)
    mean = np.mean(img)
    sd = np.std(img)
    z = ((img - mean) / (sd + 1e-20))
    return z 

def CLAHE(img, **kwargs):
    """
    kwargs: 
    kernel size (optional) : defines the shape of contextual regions used in the algorithm
    clip_limit  (optional) : default is 0.03 from skimage usage example
    nbins       (optional) : num of gray bins for histogram ( "data range" )
    """
    return skimage.exposure.equalize_adapthist(img, **kwargs)

def global_hist_equal(img):
    return skimage.exposure.equalize_hist(img)

def norm_kde_peak(img, target_peak=100):
    """
    KDE-based peak scaling: use a Gaussian KDE to find the main peak
    and scale so that peak = target_peak.
    """
    vals = img[img > 0]
    if len(vals) < 5:
        return img.astype(np.float32)
    
    kde = gaussian_kde(vals)
    # Evaluate KDE on a fine grid and find the maximum
    search_min = np.percentile(vals, 10)
    x = np.linspace(search_min, vals.max(), 500)
    density = kde(x)
    peak_intensity = x[np.argmax(density)]

    if peak_intensity == 0:
        return img.astype(np.float32)

    return (img.astype(np.float32) / peak_intensity) * target_peak

def norm_whitestripe(img):
    """
    WhiteStripe-like: find the mode of the foreground intensity
    and scale the whole image so that mode maps to target_peak.
    """
    
    vals = img[img > 0]
    hist, edges = np.histogram(vals, bins=256)
    peak_idx = np.argmax(hist)
    peak_intensity = (edges[peak_idx] + edges[peak_idx+1]) / 2.0
    return (img.astype(np.float32) / peak_intensity) * 100

