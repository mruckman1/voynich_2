"""Method 6: Hybrid structural feature vector.

Combine features from Methods 2-5 into a single comprehensive vector.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from voynich.visual.hog_features import compute_hog_features
from voynich.visual.shape_descriptors import compute_shape_feature_vector
from voynich.visual.stroke_extraction import (
    extract_graph_features,
    extract_skeleton,
    graph_feature_vector,
)
from voynich.visual.topological_features import (
    compute_topological_features,
    topological_feature_vector,
)


def compute_hybrid_vector(image_path):
    """Compute all feature types for a single image.

    Returns dict with:
    - graph: 22-dim vector (Method 2)
    - shape: 33-dim vector (Method 3)
    - topo: 5-dim vector (Method 4)
    - hog: 1296-dim vector (Method 5, PCA reduced later)
    """
    skeleton, _ = extract_skeleton(image_path)
    graph = graph_feature_vector(extract_graph_features(skeleton))
    shape = compute_shape_feature_vector(image_path)
    topo_raw = compute_topological_features(image_path)
    topo = topological_feature_vector(topo_raw)
    hog_feat = compute_hog_features(image_path)

    return {
        'graph': graph,
        'shape': shape,
        'topo': topo,
        'hog': hog_feat,
    }


def build_hybrid_distance_matrix(eva_hybrids, costa_hybrids, hog_pca_dims=50):
    """Combine all feature types into one distance matrix.

    1. PCA-reduce HOG from 1296 -> 50 dims
    2. Min-max normalize each feature set independently
    3. Concatenate: 22 + 33 + 5 + 50 = 110 dims
    4. Euclidean distance

    Returns (n_eva x n_costa) distance matrix.
    """
    all_hybrids = eva_hybrids + costa_hybrids
    n_eva = len(eva_hybrids)

    # PCA on HOG
    all_hog = np.array([h['hog'] for h in all_hybrids])
    n_components = min(hog_pca_dims, all_hog.shape[0], all_hog.shape[1])
    pca = PCA(n_components=n_components)
    all_hog_reduced = pca.fit_transform(all_hog)

    # Build full vectors
    full_vecs = []
    for i, h in enumerate(all_hybrids):
        vec = np.concatenate([h['graph'], h['shape'], h['topo'],
                              all_hog_reduced[i]])
        full_vecs.append(vec)

    full_vecs = np.array(full_vecs)

    # Min-max normalize per feature
    mins = full_vecs.min(axis=0)
    maxs = full_vecs.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    full_vecs = (full_vecs - mins) / ranges

    eva_vecs = full_vecs[:n_eva]
    costa_vecs = full_vecs[n_eva:]

    return cdist(eva_vecs, costa_vecs, metric='euclidean')
