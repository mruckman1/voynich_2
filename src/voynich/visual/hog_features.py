"""Method 5: HOG (Histogram of Oriented Gradients).

HOG captures local stroke orientation — exactly the features
that Costamagna says distinguish signs within a family.
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import hog


def compute_hog_features(image_path, pixels_per_cell=32, cells_per_block=2):
    """Compute HOG descriptor for a glyph image.

    With 224x224 images and 32x32 cells: 7x7 grid = 49 cells.
    2x2 blocks: 6x6 = 36 blocks x 4 cells x 9 orientations = 1296 features.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(1296)

    # Resize to 224x224 for consistent feature length
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        feature_vector=True,
    )
    return features


def build_hog_distance_matrix(eva_hog, costa_hog):
    """Cosine distance between HOG descriptors.

    Returns (n_eva x n_costa) distance matrix.
    """
    eva_array = np.array(eva_hog)
    costa_array = np.array(costa_hog)

    # Normalize rows to sum to 1
    eva_sums = eva_array.sum(axis=1, keepdims=True) + 1e-10
    costa_sums = costa_array.sum(axis=1, keepdims=True) + 1e-10
    eva_array = eva_array / eva_sums
    costa_array = costa_array / costa_sums

    return cdist(eva_array, costa_array, metric='cosine')
