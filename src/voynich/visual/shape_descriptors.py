"""Method 3: Classical shape descriptors.

Hu moments, Fourier descriptors, and basic geometric features.
These capture rotation/scale-invariant shape properties.
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist


def compute_hu_moments(image_path):
    """7 Hu moment invariants — rotation/scale/translation invariant.

    Log-transformed for better numerical range.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(7)

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu


def compute_fourier_descriptors(image_path, n_descriptors=20):
    """Fourier descriptors of the contour — shape frequency components.

    Low-frequency = overall shape. High-frequency = detail.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(n_descriptors)

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.zeros(n_descriptors)

    contour = max(contours, key=cv2.contourArea)
    contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]

    fft = np.fft.fft(contour_complex)

    # Normalize: translation/rotation/scale invariant
    if abs(fft[0]) > 0:
        fft = fft / abs(fft[0])
    descriptors = np.abs(fft[1:n_descriptors + 1])

    if len(descriptors) < n_descriptors:
        descriptors = np.pad(descriptors, (0, n_descriptors - len(descriptors)))

    return descriptors


def compute_shape_feature_vector(image_path):
    """Combined shape descriptor: Hu (7) + Fourier (20) + geometric (6) = 33 dims."""
    hu = compute_hu_moments(image_path)
    fourier = compute_fourier_descriptors(image_path, n_descriptors=20)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.concatenate([hu, fourier, np.zeros(6)])

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    ink_pixels = np.sum(binary > 0)
    total_pixels = binary.shape[0] * binary.shape[1]
    ink_fraction = ink_pixels / total_pixels

    ys, xs = np.where(binary > 0)
    if len(ys) > 0:
        centroid_y = np.mean(ys) / binary.shape[0]
        centroid_x = np.mean(xs) / binary.shape[1]
        spread_y = np.std(ys) / binary.shape[0]
        spread_x = np.std(xs) / binary.shape[1]
        extent = ((ys.max() - ys.min()) * (xs.max() - xs.min())) / total_pixels
    else:
        centroid_y = centroid_x = spread_y = spread_x = extent = 0

    geometric = [ink_fraction, centroid_y, centroid_x,
                 spread_y, spread_x, extent]

    return np.concatenate([hu, fourier, geometric])


def build_shape_distance_matrix(eva_features, costa_features):
    """Compute pairwise Euclidean distance between shape feature vectors.

    Min-max normalizes each feature before computing distances.
    Returns (n_eva x n_costa) distance matrix.
    """
    all_vecs = np.array(eva_features + costa_features)

    mins = all_vecs.min(axis=0)
    maxs = all_vecs.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    all_vecs = (all_vecs - mins) / ranges

    n_eva = len(eva_features)
    return cdist(all_vecs[:n_eva], all_vecs[n_eva:], metric='euclidean')
