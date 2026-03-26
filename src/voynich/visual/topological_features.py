"""Method 4: Topological features.

Coarse but extremely robust: count structural invariants.
Two signs with the same topology are candidates for matching
regardless of style differences.
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

from voynich.visual.stroke_extraction import extract_graph_features


def compute_topological_features(image_path):
    """Count topological invariants: components, holes, endpoints, junctions.

    Returns dict with n_components, n_holes, n_endpoints, n_junctions,
    euler_number.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            'n_components': 0, 'n_holes': 0,
            'n_endpoints': 0, 'n_junctions': 0, 'euler_number': 0,
        }

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Connected components
    n_components, _ = cv2.connectedComponents(binary)
    n_components -= 1  # subtract background

    # Holes via filled region analysis
    filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((7, 7), np.uint8))
    n_filled_components, _ = cv2.connectedComponents(255 - filled)
    n_holes = max(0, n_filled_components - 2)

    # Skeleton-based features
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    graph_features = extract_graph_features(skeleton)

    n_endpoints = graph_features['n_endpoints'] if graph_features else 0
    n_junctions = graph_features['n_junctions'] if graph_features else 0

    return {
        'n_components': n_components,
        'n_holes': n_holes,
        'n_endpoints': n_endpoints,
        'n_junctions': n_junctions,
        'euler_number': n_components - n_holes,
    }


def topological_distance(topo_a, topo_b):
    """Distance based on topological features.

    Signs with different loop counts cannot be the same sign.
    Signs with different endpoint counts are unlikely matches.
    """
    penalties = {
        'n_holes': 3.0,
        'n_endpoints': 1.5,
        'n_junctions': 1.0,
        'n_components': 2.0,
    }

    dist = 0.0
    for feature, weight in penalties.items():
        dist += weight * abs(topo_a.get(feature, 0) - topo_b.get(feature, 0))
    return dist


def topological_feature_vector(topo):
    """Convert topological features to a 5-dim vector."""
    return np.array([
        topo['n_components'],
        topo['n_holes'],
        topo['n_endpoints'],
        topo['n_junctions'],
        topo['euler_number'],
    ], dtype=float)


def build_topo_distance_matrix(eva_topos, costa_topos):
    """Build distance matrix using weighted topological distance.

    Returns (n_eva x n_costa) matrix.
    """
    n_eva = len(eva_topos)
    n_costa = len(costa_topos)
    matrix = np.zeros((n_eva, n_costa))

    for i, eva_t in enumerate(eva_topos):
        for j, costa_t in enumerate(costa_topos):
            matrix[i, j] = topological_distance(eva_t, costa_t)

    return matrix
