"""
socialmapper_utils.py
Helper functions matching mouseEmbedding.m exactly
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def smooth_timeseries(x, medfilt_size=3, smooth_size=3):
    """
    Smooth like MATLAB: smooth(medfilt1(x, 3), 3)
    """
    med = median_filter(x, size=medfilt_size)
    if smooth_size > 1:
        return savgol_filter(med, smooth_size, 0, mode='nearest')
    return med


def compute_velocity(pose, keypoint_idx, cap=5.0):
    """
    Compute velocity for one keypoint.
    MATLAB: [0; medfilt1(sqrt(sum(diff(squeeze(ma(:,:,kp))).^2,2)),10)]
    """
    diff = np.diff(pose[:, keypoint_idx, :], axis=0)
    speed = np.sqrt(np.sum(diff**2, axis=1))
    speed = np.concatenate([[0], speed])
    speed = median_filter(speed, size=10)
    speed[speed > cap] = cap
    return speed


def compute_angle_diff(vec1, vec2):
    """
    Angle between two vector arrays.
    MATLAB: findAngleDiff(vec1, vec2)
    Returns angle in radians [0, pi].
    """
    norm1 = np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8
    norm2 = np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8
    vec1_n = vec1 / norm1
    vec2_n = vec2 / norm2
    dot = np.sum(vec1_n * vec2_n, axis=1)
    dot = np.clip(dot, -1, 1)
    return np.arccos(dot)


def compute_pairwise_distances_upper_triangle(pose_combined):
    """
    All pairwise distances, upper triangle only (like MATLAB Xi~=Yi).
    pose_combined: (n_frames, n_keypoints, 3)
    Returns: (n_frames, n_pairs)
    """
    n_frames, n_kp, _ = pose_combined.shape
    distances = []
    for i in range(n_kp):
        for j in range(n_kp):
            if i != j:
                dist = np.linalg.norm(pose_combined[:, i, :] - pose_combined[:, j, :], axis=1)
                distances.append(dist)
    return np.column_stack(distances)


def smooth_distance_matrix(dist_matrix, medfilt_size=3, smooth_size=3):
    """Smooth each column of distance matrix"""
    smoothed = np.zeros_like(dist_matrix)
    for i in range(dist_matrix.shape[1]):
        smoothed[:, i] = smooth_timeseries(dist_matrix[:, i], medfilt_size, smooth_size)
    return smoothed