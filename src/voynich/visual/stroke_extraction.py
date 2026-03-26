"""Method 2: Explicit stroke extraction and graph comparison.

Skeletonize each glyph, extract the stroke skeleton as a graph,
compute graph features, compare feature vectors.
"""

import cv2
import numpy as np
from scipy.ndimage import label
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


def extract_skeleton(image_path):
    """Load image, binarize, skeletonize.

    Returns: (skeleton image uint8 0/255, original binary image uint8 0/255)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10,
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    return skeleton, binary


def compute_endpoint_angle(skeleton, x, y, walk_length=8):
    """Walk along skeleton from endpoint, return direction angle in degrees."""
    visited = {(x, y)}
    cx, cy = x, y
    h, w = skeleton.shape

    for _ in range(walk_length):
        found = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if ((nx, ny) not in visited and 0 <= nx < w
                        and 0 <= ny < h and skeleton[ny, nx] > 0):
                    visited.add((nx, ny))
                    cx, cy = nx, ny
                    found = True
                    break
            if found:
                break
        if not found:
            break

    return float(np.arctan2(cy - y, cx - x) * 180 / np.pi)


def compute_direction_histogram(skeleton, n_bins=8):
    """Compute histogram of local stroke directions."""
    gx = cv2.Sobel(skeleton.astype(float), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(skeleton.astype(float), cv2.CV_64F, 0, 1, ksize=3)

    angles = np.arctan2(gy, gx) * 180 / np.pi
    magnitudes = np.sqrt(gx**2 + gy**2)

    mask = skeleton > 0
    angles_masked = angles[mask]
    magnitudes_masked = magnitudes[mask]

    bin_edges = np.linspace(-180, 180, n_bins + 1)
    hist, _ = np.histogram(angles_masked, bins=bin_edges,
                           weights=magnitudes_masked)

    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist.tolist()


def extract_graph_features(skeleton):
    """From a skeleton image, extract structural graph features.

    Returns dict with: n_endpoints, n_junctions, n_loops_topological,
    n_loops_fill, total_length, bbox dims, endpoint_angles, direction_histogram, etc.
    Returns None if skeleton is empty.
    """
    h, w = skeleton.shape
    endpoints = []
    junctions = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0:
                continue
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    if skeleton[y + dy, x + dx] > 0:
                        neighbors += 1
            if neighbors == 1:
                endpoints.append((x, y))
            elif neighbors >= 3:
                junctions.append((x, y))

    total_length = int(np.sum(skeleton > 0))

    ys, xs = np.where(skeleton > 0)
    if len(ys) == 0:
        return None

    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)

    n_endpoints = len(endpoints)
    n_junctions = len(junctions)
    n_loops_est = max(0, 1 - n_endpoints // 2 + n_junctions)

    # Loop count via filled region analysis
    filled = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    inverted = 255 - filled
    _, n_components_fill = label(inverted)
    n_loops_fill = max(0, n_components_fill - 1)

    endpoint_angles = [
        compute_endpoint_angle(skeleton, ex, ey)
        for (ex, ey) in endpoints
    ]

    direction_hist = compute_direction_histogram(skeleton)

    return {
        'n_endpoints': n_endpoints,
        'n_junctions': n_junctions,
        'n_loops_topological': n_loops_est,
        'n_loops_fill': n_loops_fill,
        'total_length': total_length,
        'bbox_height': bbox_h,
        'bbox_width': bbox_w,
        'aspect_ratio': bbox_h / bbox_w if bbox_w > 0 else 0,
        'compactness': total_length / (bbox_h * bbox_w) if bbox_h * bbox_w > 0 else 0,
        'endpoint_angles': endpoint_angles,
        'direction_histogram': direction_hist,
        'center_of_mass': (float(np.mean(xs)), float(np.mean(ys))),
        'has_descender': bool(ys.max() > 0.75 * h),
        'has_ascender': bool(ys.min() < 0.25 * h),
    }


def graph_feature_vector(features):
    """Convert graph features to a fixed-length 22-dim numeric vector."""
    if features is None:
        return np.zeros(22)

    vec = [
        features['n_endpoints'],
        features['n_junctions'],
        features['n_loops_topological'],
        features['n_loops_fill'],
        features['total_length'] / 1000.0,
        features['aspect_ratio'],
        features['compactness'],
        1.0 if features['has_descender'] else 0.0,
        1.0 if features['has_ascender'] else 0.0,
        features['center_of_mass'][0] / 224.0,
        features['center_of_mass'][1] / 224.0,
    ]

    # Direction histogram (8 values)
    hist = features['direction_histogram']
    vec.extend(hist[:8] if len(hist) >= 8 else hist + [0] * (8 - len(hist)))

    # Endpoint angle statistics (3 values)
    angles = features['endpoint_angles']
    if angles:
        vec.extend([float(np.mean(angles)), float(np.std(angles)), len(angles)])
    else:
        vec.extend([0, 0, 0])

    return np.array(vec, dtype=float)


def build_graph_distance_matrix(eva_features, costa_features):
    """Compute pairwise Euclidean distance between graph feature vectors.

    Min-max normalizes each feature before computing distances.
    Returns (n_eva x n_costa) distance matrix.
    """
    all_vecs = np.array(
        [graph_feature_vector(f) for f in eva_features + costa_features]
    )

    mins = all_vecs.min(axis=0)
    maxs = all_vecs.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    all_vecs = (all_vecs - mins) / ranges

    n_eva = len(eva_features)
    eva_vecs = all_vecs[:n_eva]
    costa_vecs = all_vecs[n_eva:]

    return cdist(eva_vecs, costa_vecs, metric='euclidean')
