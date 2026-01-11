"""
Social features for neuroposelib - extends single-animal embedding to dyadic/social embedding.
Based on SocialMapper methodology (Klibaite et al. 2025).

Add this to your neuroposelib workflow for 2-animal social behavior analysis.
"""

import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d


def smooth_medfilt(x, medfilt_size=3, smooth_size=3):
    """
    Replicate MATLAB's medfilt1 + smooth combination.
    
    Parameters
    ----------
    x : np.ndarray
        1D signal to smooth
    medfilt_size : int
        Median filter kernel size (must be odd)
    smooth_size : int
        Moving average window size
        
    Returns
    -------
    np.ndarray
        Smoothed signal
    """
    x_med = medfilt(x, kernel_size=medfilt_size)
    return uniform_filter1d(x_med, size=smooth_size, mode='nearest')


def preprocess_pose_social(pose, medfilt_size=3, smooth_size=3):
    """
    Apply median filter + smoothing to each coordinate of pose data.
    
    Parameters
    ----------
    pose : np.ndarray
        Shape (frames, 3, n_keypoints) or (frames, n_keypoints, 3)
        
    Returns
    -------
    np.ndarray
        Smoothed pose, same shape as input
    """
    # Handle both (frames, 3, kp) and (frames, kp, 3) formats
    if pose.shape[1] == 3:
        # Shape is (frames, 3, n_keypoints) - SocialMapper format
        smoothed = np.zeros_like(pose)
        for kp in range(pose.shape[2]):
            for dim in range(3):
                smoothed[:, dim, kp] = smooth_medfilt(
                    pose[:, dim, kp], medfilt_size, smooth_size
                )
    else:
        # Shape is (frames, n_keypoints, 3) - neuroposelib format
        smoothed = np.zeros_like(pose)
        for kp in range(pose.shape[1]):
            for dim in range(3):
                smoothed[:, kp, dim] = smooth_medfilt(
                    pose[:, kp, dim], medfilt_size, smooth_size
                )
    return smoothed


def get_pairwise_distances(pose, joint_names=None):
    """
    Compute all pairwise Euclidean distances between keypoints.
    This is the core feature from SocialMapper's individual embedding.
    
    Parameters
    ----------
    pose : np.ndarray
        Shape (frames, n_keypoints, 3) - neuroposelib format
        
    Returns
    -------
    distances : np.ndarray
        Shape (frames, n_pairs) where n_pairs = n_kp * (n_kp - 1)
    labels : list
        Feature labels like "dist_Nose_SpineM"
    """
    n_frames, n_kp, _ = pose.shape
    
    distances = []
    labels = []
    
    for i in range(n_kp):
        for j in range(n_kp):
            if i != j:
                # Euclidean distance between keypoints i and j
                dist = np.sqrt(np.sum((pose[:, i, :] - pose[:, j, :]) ** 2, axis=1))
                distances.append(dist)
                
                if joint_names is not None:
                    labels.append(f"dist_{joint_names[i]}_{joint_names[j]}")
                else:
                    labels.append(f"dist_kp{i}_kp{j}")
    
    return np.array(distances).T, labels


def get_inter_animal_distances(pose1, pose2, joint_names=None, 
                                subset_pairs=None):
    """
    Compute pairwise distances BETWEEN two animals.
    
    Parameters
    ----------
    pose1, pose2 : np.ndarray
        Shape (frames, n_keypoints, 3) each
    joint_names : list, optional
        Names of keypoints
    subset_pairs : list of tuples, optional
        Specific (i, j) pairs to compute. If None, computes all.
        
    Returns
    -------
    distances : np.ndarray
        Shape (frames, n_pairs)
    labels : list
        Feature labels
    """
    n_frames, n_kp, _ = pose1.shape
    
    distances = []
    labels = []
    
    if subset_pairs is None:
        # All pairwise combinations between animals
        pairs = [(i, j) for i in range(n_kp) for j in range(n_kp)]
    else:
        pairs = subset_pairs
    
    for i, j in pairs:
        dist = np.sqrt(np.sum((pose1[:, i, :] - pose2[:, j, :]) ** 2, axis=1))
        distances.append(dist)
        
        if joint_names is not None:
            labels.append(f"inter_dist_{joint_names[i]}_to_{joint_names[j]}")
        else:
            labels.append(f"inter_dist_kp{i}_kp{j}")
    
    return np.array(distances).T, labels


def get_social_angles(pose1, pose2, joint_names, 
                       nose_idx=0, spine_front_idx=3, 
                       center_idx=4, tail_idx=5):
    """
    Compute social orientation angles between two animals.
    Based on SocialMapper's angle features.
    
    Computes angles like:
    - Angle from animal1's spine to animal2's nose
    - Angle from animal1's spine to animal2's center
    - etc.
    
    Parameters
    ----------
    pose1, pose2 : np.ndarray
        Shape (frames, n_keypoints, 3)
    joint_names : list
        Keypoint names
    nose_idx, spine_front_idx, center_idx, tail_idx : int
        Indices of key body parts
        
    Returns
    -------
    angles : np.ndarray
        Shape (frames, n_angles)
    labels : list
        Angle feature labels
    """
    n_frames = pose1.shape[0]
    
    # Key positions for animal 1
    n1 = pose1[:, nose_idx, :]       # nose
    s1 = pose1[:, spine_front_idx, :] # spine front (orientation reference)
    c1 = pose1[:, center_idx, :]      # center
    t1 = pose1[:, tail_idx, :]        # tail
    
    # Key positions for animal 2
    n2 = pose2[:, nose_idx, :]
    s2 = pose2[:, spine_front_idx, :]
    c2 = pose2[:, center_idx, :]
    t2 = pose2[:, tail_idx, :]
    
    def angle_between_vectors(v1, v2):
        """Compute angle between two vectors (in radians)"""
        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
        # Dot product -> angle
        dot = np.sum(v1_norm * v2_norm, axis=1)
        dot = np.clip(dot, -1, 1)
        return np.arccos(dot)
    
    angles = []
    labels = []
    
    # From animal 1's perspective: where is animal 2 relative to my heading?
    heading1 = n1 - s1  # Animal 1's heading direction
    
    # Angle to animal 2's nose
    vec_to_n2 = n2 - s1
    angles.append(angle_between_vectors(heading1, vec_to_n2))
    labels.append("angle_a1_to_a2_nose")
    
    # Angle to animal 2's center
    vec_to_c2 = c2 - s1
    angles.append(angle_between_vectors(heading1, vec_to_c2))
    labels.append("angle_a1_to_a2_center")
    
    # Angle to animal 2's tail
    vec_to_t2 = t2 - s1
    angles.append(angle_between_vectors(heading1, vec_to_t2))
    labels.append("angle_a1_to_a2_tail")
    
    # From animal 2's perspective (symmetric)
    heading2 = n2 - s2
    
    vec_to_n1 = n1 - s2
    angles.append(angle_between_vectors(heading2, vec_to_n1))
    labels.append("angle_a2_to_a1_nose")
    
    vec_to_c1 = c1 - s2
    angles.append(angle_between_vectors(heading2, vec_to_c1))
    labels.append("angle_a2_to_a1_center")
    
    vec_to_t1 = t1 - s2
    angles.append(angle_between_vectors(heading2, vec_to_t1))
    labels.append("angle_a2_to_a1_tail")
    
    return np.array(angles).T, labels


def get_centroid_distance(pose1, pose2):
    """
    Distance between animal centroids.
    
    Returns
    -------
    distance : np.ndarray
        Shape (frames, 1)
    labels : list
    """
    centroid1 = pose1.mean(axis=1)  # (frames, 3)
    centroid2 = pose2.mean(axis=1)
    dist = np.sqrt(np.sum((centroid1 - centroid2) ** 2, axis=1, keepdims=True))
    return dist, ["centroid_distance"]


def get_normalized_distances(distances, centroid_dist):
    """
    Normalize inter-animal distances by centroid distance.
    This makes features scale-invariant to overall proximity.
    
    Parameters
    ----------
    distances : np.ndarray
        Shape (frames, n_features)
    centroid_dist : np.ndarray
        Shape (frames, 1)
        
    Returns
    -------
    normalized : np.ndarray
        Same shape as distances
    """
    return distances / (centroid_dist + 1e-8)


def get_height_features(pose1, pose2, joint_names, 
                         keypoint_indices=None, floor_percentile=10):
    """
    Get height (z-coordinate) features relative to floor.
    
    Parameters
    ----------
    pose1, pose2 : np.ndarray
        Shape (frames, n_keypoints, 3)
    keypoint_indices : list, optional
        Which keypoints to extract height for. Default: all.
    floor_percentile : float
        Percentile to use as floor estimate
        
    Returns
    -------
    heights : np.ndarray
    labels : list
    """
    if keypoint_indices is None:
        keypoint_indices = list(range(pose1.shape[1]))
    
    # Estimate floor from foot/ankle z-coordinates
    # You may want to adjust this based on your skeleton
    all_z = np.concatenate([pose1[:, :, 2].flatten(), pose2[:, :, 2].flatten()])
    floor_z = np.percentile(all_z, floor_percentile)
    
    heights = []
    labels = []
    
    for idx in keypoint_indices:
        heights.append(pose1[:, idx, 2] - floor_z)
        labels.append(f"height_a1_{joint_names[idx]}")
        
    for idx in keypoint_indices:
        heights.append(pose2[:, idx, 2] - floor_z)
        labels.append(f"height_a2_{joint_names[idx]}")
    
    return np.array(heights).T, labels


def build_social_features(pose1, pose2, joint_names, 
                           nose_idx=0, spine_front_idx=3,
                           center_idx=4, tail_idx=5,
                           height_keypoints=None,
                           include_individual_distances=True,
                           include_inter_distances=True,
                           include_angles=True,
                           include_heights=True,
                           scale_factors=None):
    """
    Build complete social feature matrix for two interacting animals.
    
    This replicates the SocialMapper feature construction.
    
    Parameters
    ----------
    pose1, pose2 : np.ndarray
        Shape (frames, n_keypoints, 3) for each animal
    joint_names : list
        Names of keypoints
    nose_idx, spine_front_idx, center_idx, tail_idx : int
        Key body part indices for angle computation
    height_keypoints : list, optional
        Keypoint indices for height features
    include_* : bool
        Which feature types to include
    scale_factors : dict, optional
        Scaling factors for different feature types
        
    Returns
    -------
    features : np.ndarray
        Shape (frames, n_features)
    labels : list
        Feature labels
    """
    if scale_factors is None:
        scale_factors = {
            'individual_dist': 1.0,
            'inter_dist': 1.0,
            'inter_dist_normalized': 2.0,
            'angles': 10.0,
            'heights': 0.1,
        }
    
    all_features = []
    all_labels = []
    
    # Centroid distance (always computed for normalization)
    centroid_dist, _ = get_centroid_distance(pose1, pose2)
    
    # Individual pairwise distances (within each animal)
    if include_individual_distances:
        dist1, labels1 = get_pairwise_distances(pose1, joint_names)
        dist2, labels2 = get_pairwise_distances(pose2, joint_names)
        
        all_features.append(dist1 * scale_factors.get('individual_dist', 1.0))
        all_features.append(dist2 * scale_factors.get('individual_dist', 1.0))
        all_labels.extend([f"a1_{l}" for l in labels1])
        all_labels.extend([f"a2_{l}" for l in labels2])
    
    # Inter-animal distances
    if include_inter_distances:
        # Key body part distances (nose-nose, nose-center, nose-tail, etc.)
        key_pairs = [
            (nose_idx, nose_idx),      # nose to nose
            (nose_idx, center_idx),    # nose to center
            (nose_idx, tail_idx),      # nose to tail
        ]
        
        inter_dist, inter_labels = get_inter_animal_distances(
            pose1, pose2, joint_names, subset_pairs=key_pairs
        )
        
        # Raw distances
        all_features.append(inter_dist * scale_factors.get('inter_dist', 1.0))
        all_labels.extend(inter_labels)
        
        # Normalized by centroid distance
        inter_dist_norm = get_normalized_distances(inter_dist, centroid_dist)
        all_features.append(inter_dist_norm * scale_factors.get('inter_dist_normalized', 2.0))
        all_labels.extend([f"{l}_normalized" for l in inter_labels])
        
        # Symmetric (animal 2 -> animal 1)
        inter_dist_rev, inter_labels_rev = get_inter_animal_distances(
            pose2, pose1, joint_names, subset_pairs=key_pairs
        )
        all_features.append(inter_dist_rev * scale_factors.get('inter_dist', 1.0))
        all_labels.extend([l.replace("_to_", "_rev_to_") for l in inter_labels_rev])
    
    # Social angles
    if include_angles:
        angles, angle_labels = get_social_angles(
            pose1, pose2, joint_names,
            nose_idx=nose_idx,
            spine_front_idx=spine_front_idx,
            center_idx=center_idx,
            tail_idx=tail_idx
        )
        all_features.append(angles * scale_factors.get('angles', 10.0))
        all_labels.extend(angle_labels)
    
    # Height features
    if include_heights:
        if height_keypoints is None:
            height_keypoints = [nose_idx, center_idx, tail_idx]
        heights, height_labels = get_height_features(
            pose1, pose2, joint_names, keypoint_indices=height_keypoints
        )
        all_features.append(heights * scale_factors.get('heights', 0.1))
        all_labels.extend(height_labels)
    
    # Concatenate all features
    features = np.hstack(all_features)
    
    return features, all_labels

def build_social_features_v2(pose1, pose2, joint_names,
                              nose_idx=0, spine_front_idx=3, center_idx=3, tail_idx=4,
                              include_ego=True, include_velocities=True, dt=1/30):
    """
    Enhanced social features with ego-centric alignment and velocities.
    
    Features included:
    - Original: pairwise distances, inter-animal distances, angles, heights
    - New: ego-centric partner position, distance derivatives, angular derivatives
    """
    all_features = []
    all_labels = []
    
    # === ORIGINAL FEATURES ===
    # Pairwise distances within each animal
    pw1, pw1_labels = get_pairwise_distances(pose1, joint_names)
    pw2, pw2_labels = get_pairwise_distances(pose2, joint_names)
    all_features.extend([pw1, pw2])
    all_labels.extend([f"single_{l}" for l in pw1_labels])
    all_labels.extend([f"group_{l}" for l in pw2_labels])
    
    # Inter-animal distances
    inter_dist, inter_labels = get_inter_animal_distances(pose1, pose2, joint_names)
    all_features.append(inter_dist)
    all_labels.extend(inter_labels)
    
    # Centroid distance
    cent_dist, cent_label = get_centroid_distance(pose1, pose2)
    all_features.append(cent_dist)
    all_labels.extend(cent_label)
    
    # Social angles (FIXED: pass joint_names and tail_idx)
    angles, angle_labels = get_social_angles(
        pose1, pose2, joint_names,
        nose_idx=nose_idx, 
        spine_front_idx=spine_front_idx, 
        center_idx=center_idx,
        tail_idx=tail_idx
    )
    all_features.append(angles)
    all_labels.extend(angle_labels)
    
    # === NEW: EGO-CENTRIC FEATURES ===
    if include_ego:
        # Partner position in ego frame
        partner_ego, _ = get_partner_in_egocentric(pose1, pose2, nose_idx, spine_front_idx, center_idx)
        
        # Use key points: nose, center, tail
        for kp_idx, kp_name in [(nose_idx, 'nose'), (center_idx, 'center'), (tail_idx, 'tail')]:
            # X position (left/right of me)
            all_features.append(partner_ego[:, kp_idx, 0:1])
            all_labels.append(f"ego_partner_{kp_name}_x")
            # Y position (front/behind me)
            all_features.append(partner_ego[:, kp_idx, 1:2])
            all_labels.append(f"ego_partner_{kp_name}_y")
        
        # Also from group's perspective
        partner_ego_rev, _ = get_partner_in_egocentric(pose2, pose1, nose_idx, spine_front_idx, center_idx)
        for kp_idx, kp_name in [(nose_idx, 'nose'), (center_idx, 'center'), (tail_idx, 'tail')]:
            all_features.append(partner_ego_rev[:, kp_idx, 0:1])
            all_labels.append(f"ego_rev_partner_{kp_name}_x")
            all_features.append(partner_ego_rev[:, kp_idx, 1:2])
            all_labels.append(f"ego_rev_partner_{kp_name}_y")
    
    # === NEW: VELOCITY FEATURES ===
    if include_velocities:
        # Distance derivatives
        dist_derivs, dist_deriv_labels = get_distance_derivatives(pose1, pose2, joint_names, dt=dt)
        all_features.append(dist_derivs)
        all_labels.extend(dist_deriv_labels)
        
        # Heading derivatives (both animals)
        heading_deriv1, _ = get_heading_derivatives(pose1, nose_idx, spine_front_idx, dt)
        heading_deriv2, _ = get_heading_derivatives(pose2, nose_idx, spine_front_idx, dt)
        all_features.append(heading_deriv1.reshape(-1, 1))
        all_features.append(heading_deriv2.reshape(-1, 1))
        all_labels.extend(["single_angular_vel", "group_angular_vel"])
        
        # Approach angle derivatives
        approach1, _ = get_approach_angle_derivative(pose1, pose2, nose_idx, spine_front_idx, dt)
        approach2, _ = get_approach_angle_derivative(pose2, pose1, nose_idx, spine_front_idx, dt)
        all_features.append(approach1.reshape(-1, 1))
        all_features.append(approach2.reshape(-1, 1))
        all_labels.extend(["single_d_dt_angle_to_partner", "group_d_dt_angle_to_partner"])
    
    # Combine
    features = np.hstack(all_features)
    
    return features, all_labels

# =============================================================================
# Integration with neuroposelib
# =============================================================================

def load_social_poses(meta_path, connectivity, animal1_key, animal2_key):
    """
    Load poses for two animals from metadata file.
    
    You'll need to adapt this based on how your data is organized.
    
    Parameters
    ----------
    meta_path : str
        Path to metadata CSV
    connectivity : object
        neuroposelib connectivity object
    animal1_key, animal2_key : str
        Column names or identifiers for each animal's pose files
        
    Returns
    -------
    pose1, pose2 : np.ndarray
    ids1, ids2 : np.ndarray
    """
    # This is a placeholder - adapt to your data structure
    from neuroposelib import read
    
    # Example: if you have separate columns for each animal
    # pose1, ids1, _, _ = read.pose_from_meta(
    #     path=meta_path, connectivity=connectivity, 
    #     key=animal1_key, file_type="dannce"
    # )
    # pose2, ids2, _, _ = read.pose_from_meta(
    #     path=meta_path, connectivity=connectivity,
    #     key=animal2_key, file_type="dannce"
    # )
    
    raise NotImplementedError(
        "Adapt this function to your data organization. "
        "You need to load pose1 and pose2 as (frames, n_keypoints, 3) arrays."
    )


def run_social_embedding(pose1, pose2, connectivity, config, 
                          downsample=1, embed_method="tsne"):
    """
    Run full social embedding pipeline on two-animal data.
    
    This extends the neuroposelib pipeline to social behavior.
    
    Parameters
    ----------
    pose1, pose2 : np.ndarray
        Shape (frames, n_keypoints, 3) for each animal
    connectivity : object
        neuroposelib connectivity object
    config : dict
        Configuration dictionary
    downsample : int
        Downsampling factor
    embed_method : str
        Embedding method ("tsne" or "umap")
        
    Returns
    -------
    data_obj : DataStruct
        neuroposelib DataStruct with social embedding
    """
    from neuroposelib import features
    from neuroposelib import DataStruct as ds
    from neuroposelib.embed import Embed, Watershed
    
    joint_names = connectivity.joint_names
    
    # Preprocess poses
    pose1_smooth = preprocess_pose_social(pose1)
    pose2_smooth = preprocess_pose_social(pose2)
    
    # Build social features
    # Adjust indices based on your skeleton!
    social_feats, social_labels = build_social_features_v2(
        pose1_smooth, pose2_smooth, 
        joint_names=joint_names,
        nose_idx=0,           # ADJUST THESE
        spine_front_idx=3,    # ADJUST THESE
        center_idx=4,         # ADJUST THESE  
        tail_idx=5,           # ADJUST THESE
    )
    
    print(f"Social features shape: {social_feats.shape}")
    print(f"Number of features: {len(social_labels)}")
    
    # PCA on social features
    pc_feats, pc_labels = features.pca(
        social_feats, social_labels,
        categories=["dist", "angle", "height", "inter"],  # adjust as needed
        n_pcs=15,
        method="fbpca"
    )
    
    # Create dummy IDs (one per frame)
    ids = np.arange(len(pose1))
    
    # Wavelet transform
    wlet_feats, wlet_labels = features.wavelet(
        pc_feats, pc_labels, ids,
        f_s=config.get("fps", 30),
        freq=np.linspace(0.1, 1, 10),
        w0=5
    )
    
    # PCA on wavelets
    pc_wlet, pc_wlet_labels = features.pca(
        wlet_feats, wlet_labels,
        categories=["wlet"],
        n_pcs=5,
        method="fbpca"
    )
    
    # Combine
    final_feats = np.hstack((pc_feats, pc_wlet))
    final_labels = pc_labels + pc_wlet_labels
    
    # Create DataStruct (use pose1 as reference)
    data_obj = ds.DataStruct(
        pose=pose1,
        id=ids,
        connectivity=connectivity,
    )
    data_obj.features = final_feats
    
    # Downsample
    if downsample > 1:
        data_obj = data_obj[::downsample, :]
    
    # Embedding
    embedder = Embed(
        embed_method=embed_method,
        perplexity=config.get("perplexity", 30),
        lr=config.get("lr", 200),
    )
    data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)
    
    # Watershed clustering
    data_obj.ws = Watershed(
        sigma=config.get("sigma", 1.0),
        max_clip=1,
        log_out=True,
        pad_factor=0.05
    )
    data_obj.data["Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)
    
    return data_obj



# ============================================================
# EGO-CENTRIC ALIGNMENT
# ============================================================

def get_heading_angle(pose, nose_idx=0, spine_front_idx=3):
    """
    Get heading angle (yaw) for each frame.
    Heading = direction from spine_front to nose in XY plane.
    
    Returns: (n_frames,) array of angles in radians
    """
    nose = pose[:, nose_idx, :2]  # XY only
    spine = pose[:, spine_front_idx, :2]
    
    direction = nose - spine
    angles = np.arctan2(direction[:, 1], direction[:, 0])
    return angles


def align_to_egocentric(pose, nose_idx=0, spine_front_idx=3, center_idx=3):
    """
    Rotate pose so heading points in +Y direction, centered on spine.
    
    Args:
        pose: (n_frames, n_keypoints, 3)
        nose_idx: index of nose keypoint
        spine_front_idx: index of front spine (for heading direction)
        center_idx: index of keypoint to center on
    
    Returns:
        pose_aligned: (n_frames, n_keypoints, 3) - ego-centric aligned pose
        angles: (n_frames,) - original heading angles (for reference)
    """
    n_frames, n_kp, _ = pose.shape
    
    # Get heading angles
    angles = get_heading_angle(pose, nose_idx, spine_front_idx)
    
    # Target angle: +Y direction = pi/2
    rotation_angles = np.pi/2 - angles
    
    # Center on spine
    center = pose[:, center_idx:center_idx+1, :]  # (n_frames, 1, 3)
    pose_centered = pose - center
    
    # Rotate XY coordinates (keep Z unchanged)
    pose_aligned = np.zeros_like(pose_centered)
    cos_a = np.cos(rotation_angles)
    sin_a = np.sin(rotation_angles)
    
    pose_aligned[:, :, 0] = cos_a[:, None] * pose_centered[:, :, 0] - sin_a[:, None] * pose_centered[:, :, 1]
    pose_aligned[:, :, 1] = sin_a[:, None] * pose_centered[:, :, 0] + cos_a[:, None] * pose_centered[:, :, 1]
    pose_aligned[:, :, 2] = pose_centered[:, :, 2]
    
    return pose_aligned, angles


def get_partner_in_egocentric(pose_self, pose_partner, nose_idx=0, spine_front_idx=3, center_idx=3):
    """
    Transform partner's position into self's egocentric frame.
    
    Returns:
        partner_ego: (n_frames, n_keypoints, 3) - partner in self's reference frame
        self_heading: (n_frames,) - self's heading angle in world frame
    """
    n_frames = pose_self.shape[0]
    
    # Get self's heading
    self_heading = get_heading_angle(pose_self, nose_idx, spine_front_idx)
    rotation_angles = np.pi/2 - self_heading
    
    # Center partner relative to self's center
    self_center = pose_self[:, center_idx:center_idx+1, :]
    partner_centered = pose_partner - self_center
    
    # Rotate partner into self's frame
    partner_ego = np.zeros_like(partner_centered)
    cos_a = np.cos(rotation_angles)
    sin_a = np.sin(rotation_angles)
    
    partner_ego[:, :, 0] = cos_a[:, None] * partner_centered[:, :, 0] - sin_a[:, None] * partner_centered[:, :, 1]
    partner_ego[:, :, 1] = sin_a[:, None] * partner_centered[:, :, 0] + cos_a[:, None] * partner_centered[:, :, 1]
    partner_ego[:, :, 2] = partner_centered[:, :, 2]
    
    return partner_ego, self_heading


# ============================================================
# VELOCITIES / DERIVATIVES
# ============================================================

def compute_derivatives(signal, dt=1/30, smooth_window=5):
    """
    Compute time derivative with optional smoothing.
    
    Args:
        signal: (n_frames,) or (n_frames, n_features)
        dt: time step (1/frame_rate)
        smooth_window: window for moving average smoothing (0 to disable)
    
    Returns:
        derivative: same shape as signal
    """
    deriv = np.gradient(signal, dt, axis=0)
    
    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        deriv = uniform_filter1d(deriv, size=smooth_window, axis=0)
    
    return deriv


def get_distance_derivatives(pose1, pose2, joint_names, key_pairs=None, dt=1/30):
    """
    Compute time derivatives of inter-animal distances.
    Positive = animals separating, Negative = animals approaching.
    
    Args:
        pose1, pose2: (n_frames, n_keypoints, 3)
        joint_names: list of joint names
        key_pairs: list of (idx1, idx2) tuples, or None for default
        dt: time step
    
    Returns:
        derivs: (n_frames, n_pairs) array of distance derivatives
        labels: list of feature labels
    """
    if key_pairs is None:
        # Default: nose-nose, nose-tail, centroid
        key_pairs = [(0, 0), (0, 4), (4, 4)]
    
    # Get distances first
    distances = []
    labels = []
    for i1, i2 in key_pairs:
        dist = np.linalg.norm(pose1[:, i1, :] - pose2[:, i2, :], axis=1)
        distances.append(dist)
        labels.append(f"d_dt_{joint_names[i1]}_{joint_names[i2]}")
    
    # Add centroid distance
    c1 = pose1.mean(axis=1)
    c2 = pose2.mean(axis=1)
    cent_dist = np.linalg.norm(c1 - c2, axis=1)
    distances.append(cent_dist)
    labels.append("d_dt_centroid")
    
    distances = np.column_stack(distances)
    
    # Compute derivatives
    derivs = compute_derivatives(distances, dt=dt, smooth_window=5)
    
    return derivs, labels


def get_heading_derivatives(pose, nose_idx=0, spine_front_idx=3, dt=1/30):
    """
    Compute angular velocity (heading change rate).
    
    Returns:
        angular_vel: (n_frames,) - rad/s
        label: feature label
    """
    angles = get_heading_angle(pose, nose_idx, spine_front_idx)
    
    # Handle angle wrapping
    angles_unwrapped = np.unwrap(angles)
    angular_vel = compute_derivatives(angles_unwrapped, dt=dt, smooth_window=5)
    
    return angular_vel, "angular_velocity"


def get_approach_angle_derivative(pose_self, pose_partner, nose_idx=0, spine_front_idx=3, dt=1/30):
    """
    Compute rate of change of angle to partner.
    Negative = turning toward partner, Positive = turning away.
    
    Returns:
        angle_deriv: (n_frames,)
        label: feature label
    """
    # Vector from self nose to partner center
    self_nose = pose_self[:, nose_idx, :2]
    partner_center = pose_partner.mean(axis=1)[:, :2]
    to_partner = partner_center - self_nose
    
    # Angle to partner (in world frame)
    angle_to_partner = np.arctan2(to_partner[:, 1], to_partner[:, 0])
    
    # Self heading
    self_heading = get_heading_angle(pose_self, nose_idx, spine_front_idx)
    
    # Relative angle (how much do I need to turn to face partner?)
    rel_angle = angle_to_partner - self_heading
    # Wrap to [-pi, pi]
    rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))
    
    # Derivative
    rel_angle_unwrapped = np.unwrap(rel_angle)
    angle_deriv = compute_derivatives(rel_angle_unwrapped, dt=dt, smooth_window=5)
    
    return angle_deriv, "d_dt_angle_to_partner"

if __name__ == "__main__":
    # Example usage
    print("Social features module for neuroposelib")
    print("Import and use build_social_features() or run_social_embedding()")