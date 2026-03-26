"""Line, word, and character segmentation via projection profiles.

Uses horizontal/vertical projection of binarized manuscript images
with forced alignment from IVTFF transcription data.
"""

import numpy as np
from scipy.signal import argrelmin


def binarize(img_gray, threshold=None):
    """Binarize a grayscale image. Ink pixels = True.

    Uses Otsu's method if no threshold given.
    """
    if threshold is None:
        # Otsu's method: minimize intra-class variance
        hist, bin_edges = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
        total = img_gray.size
        sum_total = np.sum(np.arange(256) * hist)
        sum_bg, weight_bg = 0.0, 0
        max_var, best_t = 0.0, 128
        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var_between > max_var:
                max_var = var_between
                best_t = t
        threshold = best_t

    return img_gray < threshold


def horizontal_projection(binary):
    """Sum ink pixels per row -> 1D profile."""
    return binary.sum(axis=1).astype(float)


def vertical_projection(binary):
    """Sum ink pixels per column -> 1D profile."""
    return binary.sum(axis=0).astype(float)


def find_text_region(binary, margin_frac=0.05, ink_threshold=0.01):
    """Find the bounding box of the main text block, cropping margins.

    Returns (y_start, y_end, x_start, x_end).
    """
    h, w = binary.shape
    h_proj = horizontal_projection(binary)
    v_proj = vertical_projection(binary)

    # Threshold: rows/cols with > ink_threshold fraction of max
    h_thresh = h_proj.max() * ink_threshold
    v_thresh = v_proj.max() * ink_threshold

    h_active = np.where(h_proj > h_thresh)[0]
    v_active = np.where(v_proj > v_thresh)[0]

    if len(h_active) == 0 or len(v_active) == 0:
        return 0, h, 0, w

    # Add small margin
    margin_y = int(h * margin_frac)
    margin_x = int(w * margin_frac)

    y_start = max(0, h_active[0] - margin_y)
    y_end = min(h, h_active[-1] + margin_y)
    x_start = max(0, v_active[0] - margin_x)
    x_end = min(w, v_active[-1] + margin_x)

    return y_start, y_end, x_start, x_end


def find_line_bands(h_proj, n_expected, min_line_height=20):
    """Find text line boundaries using horizontal projection.

    Args:
        h_proj: 1D horizontal projection profile
        n_expected: Expected number of lines (from IVTFF)
        min_line_height: Minimum line height in pixels

    Returns:
        List of (y_start, y_end) tuples for each line.
    """
    n = len(h_proj)
    if n_expected <= 0:
        return []

    # Smooth the projection
    kernel_size = max(3, min_line_height // 3)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(h_proj, kernel, mode='same')

    # Find valleys (local minima with order based on expected spacing)
    expected_spacing = n // (n_expected + 1)
    order = max(3, expected_spacing // 4)

    valley_indices = argrelmin(smoothed, order=order)[0]

    if len(valley_indices) == 0:
        # Fallback: uniform spacing
        boundaries = np.linspace(0, n, n_expected + 1, dtype=int)
        return [(boundaries[i], boundaries[i + 1]) for i in range(n_expected)]

    # Score valleys by depth (lower projection = better gap)
    valley_scores = [(idx, smoothed[idx]) for idx in valley_indices]
    valley_scores.sort(key=lambda x: x[1])

    # Select the best (n_expected - 1) valleys
    n_cuts = n_expected - 1
    selected = sorted([v[0] for v in valley_scores[:n_cuts]])

    # Build bands
    boundaries = [0] + selected + [n]
    bands = []
    for i in range(len(boundaries) - 1):
        y_start = boundaries[i]
        y_end = boundaries[i + 1]
        if y_end - y_start >= min_line_height:
            bands.append((y_start, y_end))

    return bands


def find_word_gaps(v_proj, n_expected):
    """Find word boundaries within a text line using vertical projection.

    Args:
        v_proj: 1D vertical projection profile for one line
        n_expected: Expected number of words

    Returns:
        List of (x_start, x_end) tuples for each word.
    """
    n = len(v_proj)
    if n_expected <= 1:
        # Find ink extent
        active = np.where(v_proj > 0)[0]
        if len(active) == 0:
            return [(0, n)]
        return [(active[0], active[-1] + 1)]

    # Smooth
    kernel_size = max(3, n // (n_expected * 5))
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(v_proj, kernel, mode='same')

    # Find valleys
    order = max(2, n // (n_expected * 3))
    valley_indices = argrelmin(smoothed, order=order)[0]

    if len(valley_indices) == 0:
        # Fallback: uniform spacing
        boundaries = np.linspace(0, n, n_expected + 1, dtype=int)
        return [(boundaries[i], boundaries[i + 1]) for i in range(n_expected)]

    # Score by depth
    valley_scores = [(idx, smoothed[idx]) for idx in valley_indices]
    valley_scores.sort(key=lambda x: x[1])

    n_cuts = n_expected - 1
    selected = sorted([v[0] for v in valley_scores[:n_cuts]])

    boundaries = [0] + selected + [n]
    words = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    return words


def segment_characters_dp(word_binary, n_chars, alpha=0.5):
    """Segment a word image into characters using DP forced alignment.

    Given a binary word image and the number of expected characters,
    finds optimal cut points minimizing:
        cost = width_variance(segments) + alpha * sum(projection_at_cuts)

    Args:
        word_binary: Binary image of one word (ink = True)
        n_chars: Number of expected characters
        alpha: Trade-off between uniform width and low-ink cuts

    Returns:
        List of (x_start, x_end) tuples for each character.
    """
    if n_chars <= 0:
        return []
    if n_chars == 1:
        return [(0, word_binary.shape[1])]

    width = word_binary.shape[1]
    n_cuts = n_chars - 1

    # Vertical projection for cut scoring
    v_proj = vertical_projection(word_binary).astype(float)

    # Find candidate cut points (local minima)
    order = max(2, width // (n_chars * 4))
    candidates = argrelmin(v_proj, order=order)[0]

    # Add extra candidates if too few
    if len(candidates) < n_cuts:
        # Use uniform spacing as fallback candidates
        uniform = np.linspace(0, width, n_chars + 1, dtype=int)[1:-1]
        candidates = np.unique(np.concatenate([candidates, uniform]))
        candidates = np.sort(candidates)

    if len(candidates) < n_cuts:
        # Still not enough — uniform spacing
        boundaries = np.linspace(0, width, n_chars + 1, dtype=int)
        return [(boundaries[i], boundaries[i + 1]) for i in range(n_chars)]

    n_cand = len(candidates)
    target_width = width / n_chars

    # DP: find best n_cuts cuts from candidates
    # dp[i][j] = best cost using j cuts from first i candidates
    INF = float('inf')
    dp = [[INF] * (n_cuts + 1) for _ in range(n_cand + 1)]
    parent = [[(-1, -1)] * (n_cuts + 1) for _ in range(n_cand + 1)]

    # Base: 0 cuts used
    for i in range(n_cand + 1):
        dp[i][0] = 0

    for i in range(1, n_cand + 1):
        for j in range(1, min(i, n_cuts) + 1):
            # Try candidate i-1 as the j-th cut
            cut_pos = candidates[i - 1]
            cut_cost = alpha * v_proj[cut_pos] / (v_proj.max() + 1e-10)

            for prev_i in range(j - 1, i):
                prev_cost = dp[prev_i][j - 1]
                if prev_cost == INF:
                    continue

                # Width of segment between previous cut and this cut
                if j == 1:
                    seg_start = 0
                else:
                    seg_start = candidates[prev_i - 1] if prev_i > 0 else 0
                seg_width = cut_pos - seg_start
                width_penalty = ((seg_width - target_width) / target_width) ** 2

                total = prev_cost + width_penalty + cut_cost
                if total < dp[i][j]:
                    dp[i][j] = total
                    parent[i][j] = (prev_i, j - 1)

    # Find best final state (considering last segment width)
    best_cost = INF
    best_i = -1
    for i in range(n_cuts, n_cand + 1):
        if dp[i][n_cuts] == INF:
            continue
        last_cut = candidates[i - 1]
        last_seg_width = width - last_cut
        width_penalty = ((last_seg_width - target_width) / target_width) ** 2
        total = dp[i][n_cuts] + width_penalty
        if total < best_cost:
            best_cost = total
            best_i = i

    if best_i == -1:
        # DP failed — uniform spacing
        boundaries = np.linspace(0, width, n_chars + 1, dtype=int)
        return [(boundaries[i], boundaries[i + 1]) for i in range(n_chars)]

    # Backtrack to find selected cuts
    selected_cuts = []
    ci, cj = best_i, n_cuts
    while cj > 0:
        selected_cuts.append(candidates[ci - 1])
        ci, cj = parent[ci][cj]

    selected_cuts.sort()

    boundaries = [0] + selected_cuts + [width]
    return [(boundaries[i], boundaries[i + 1]) for i in range(n_chars)]
