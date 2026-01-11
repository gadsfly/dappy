# %% [markdown]
# # Full Social Embedding Pipeline with DataStruct Integration
# 
# This notebook loads all recordings, computes social features, embeds, and 
# creates a DataStruct for visualization.

# %%
# ============================================================
# CONFIGURATION
# ============================================================

SKELETON_NAME = "mouse9"
N_KEYPOINTS = 9

# Paths
SKELETON_PATH = "/home/lq53/mir_repos/dappy_24_nov/byws_version/skeletons_social_mir.py"
SOCIAL_CSV_PATH = "/home/lq53/mir_repos/dappy_24_nov/mir_modif_dappy/meta_datas/minji_sorted_pose3d_files.csv"
SOCIAL_FEATURES_PATH = "/home/lq53/mir_repos/dappy_24_nov/byws_version/social_features.py"
OUTPUT_PATH = "/home/lq53/mir_repos/dappy_24_nov/2601_minji/social_embedding_output_with_vid/"

# Social feature indices for mouse9
SOCIAL_INDICES = {
    'nose': 0,
    'spine_front': 3,
    'center': 3,
    'tail': 4,
    'ears': [2, 1],
    'wrists': [5, 6],
    'ankles': [7, 8],
}

JOINT_NAMES = ['Snout', 'EarR', 'EarL', 'SpineF', 'Tail_base', 
               'WristL', 'WristR', 'AnkleL', 'AnkleR']

SEGMENTS = [
    (0, 1), (0, 2), (2, 1), (0, 3), (4, 3),
    (5, 3), (6, 3), (7, 4), (8, 4),
]

# Pipeline parameters
FRAME_RATE = 30
DOWNSAMPLE = 5  # For embedding (set higher for faster testing)
PERPLEXITY = 50
SIGMA = 15  # For watershed

# %%
# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add social features path
sys.path.insert(0, str(Path(SOCIAL_FEATURES_PATH).parent))
from social_features import (
    preprocess_pose_social,
    build_social_features,
    get_inter_animal_distances,
    get_centroid_distance,
)

from neuroposelib import features as npf
from neuroposelib import DataStruct as ds
from neuroposelib import vis
from neuroposelib import read
from neuroposelib import write
from neuroposelib import analysis
from neuroposelib.embed import Embed, Watershed
from fbpca import pca as fbpca_pca

# Create output directory
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

print("Imports complete ✓")

# %%
# ============================================================
# 1. LOAD ALL RECORDINGS
# ============================================================

meta_df = pd.read_csv(SOCIAL_CSV_PATH)
print(f"Found {len(meta_df)} recordings in CSV")

# Storage for all data
all_features = []
all_pose1 = []
all_pose2 = []
all_inter_dist = []
frame_meta = []  # Metadata per frame

t_start = time.time()

for idx, row in meta_df.iterrows():
    try:
        mat_data = sio.loadmat(row['Prediction_path'])
        pose_raw = mat_data['pred']
        
        # Reshape: (frames, 2, 3, 9) -> two arrays of (frames, 9, 3)
        pose1 = pose_raw[:, 0, :, :].transpose(0, 2, 1)
        pose2 = pose_raw[:, 1, :, :].transpose(0, 2, 1)
        
        n_frames = pose1.shape[0]
        
        # Preprocess
        pose1_smooth = preprocess_pose_social(pose1)
        pose2_smooth = preprocess_pose_social(pose2)
        
        # Compute social features
        feats, labels = build_social_features(
            pose1_smooth, pose2_smooth, JOINT_NAMES,
            nose_idx=SOCIAL_INDICES['nose'],
            spine_front_idx=SOCIAL_INDICES['spine_front'],
            center_idx=SOCIAL_INDICES['center'],
            tail_idx=SOCIAL_INDICES['tail'],
        )
        
        # Also compute inter-animal distances for later coloring
        key_pairs = [
            (SOCIAL_INDICES['nose'], SOCIAL_INDICES['nose']),
            (SOCIAL_INDICES['nose'], SOCIAL_INDICES['tail']),
            (SOCIAL_INDICES['tail'], SOCIAL_INDICES['tail']),
        ]
        inter_dist, _ = get_inter_animal_distances(
            pose1_smooth, pose2_smooth, JOINT_NAMES, subset_pairs=key_pairs
        )
        centroid_dist, _ = get_centroid_distance(pose1_smooth, pose2_smooth)
        
        # Store
        all_features.append(feats)
        all_pose1.append(pose1_smooth)
        all_pose2.append(pose2_smooth)
        all_inter_dist.append(np.hstack([inter_dist, centroid_dist]))
        
        # Create per-frame metadata
        for frame_idx in range(n_frames):
            frame_meta.append({
                'recording_idx': idx,
                'frame_in_recording': frame_idx,
                'Animal1': row['Animal1'],
                'Animal2': row['Animal2'],
                'date': row.get('date', 'unknown'),
                'pair': f"{row['Animal1']}_{row['Animal2']}",
            })
        
        print(f"Loaded {idx}: {row['Animal1']}-{row['Animal2']}, {n_frames} frames")
        
    except Exception as e:
        print(f"ERROR loading {idx}: {e}")

# Concatenate all
social_feats = np.vstack(all_features)
pose1_all = np.vstack(all_pose1)
pose2_all = np.vstack(all_pose2)
inter_dist_all = np.vstack(all_inter_dist)
meta_by_frame = pd.DataFrame(frame_meta)

# Feature labels (from last recording)
feat_labels = labels

print(f"\n=== Loaded {len(meta_df)} recordings ===")
print(f"Total frames: {len(social_feats)}")
print(f"Features per frame: {social_feats.shape[1]}")
print(f"Time: {time.time() - t_start:.1f}s")

# %%
# ============================================================
# 2. FEATURE PROCESSING (PCA -> Wavelet -> PCA)
# ============================================================

print("\n=== Feature Processing ===")

# Create recording IDs for wavelet (one ID per recording)
ids = meta_by_frame['recording_idx'].values

# Clean any NaN/Inf
nan_count = np.isnan(social_feats).sum()
inf_count = np.isinf(social_feats).sum()
print(f"NaN: {nan_count}, Inf: {inf_count}")
if nan_count > 0 or inf_count > 0:
    social_feats = np.nan_to_num(social_feats, nan=0.0, posinf=1e6, neginf=-1e6)

# PCA on all features
print("\nPCA on social features...")
n_pcs = min(15, social_feats.shape[1])
U, s, V = fbpca_pca(social_feats.astype(np.float64), k=n_pcs)
pc_feats = social_feats @ V.T
pc_labels = [f"pc_{i}" for i in range(n_pcs)]
print(f"  PCA output: {pc_feats.shape}")

# Wavelet transform
print("\nWavelet transform...")
wlet_feats, wlet_labels = npf.wavelet(
    pc_feats, pc_labels, ids,
    f_s=FRAME_RATE,
    freq=np.linspace(0.1, 1, 10),
    w0=5
)
print(f"  Wavelet output: {wlet_feats.shape}")

# PCA on wavelets
print("\nPCA on wavelets...")
n_wlet_pcs = min(5, wlet_feats.shape[1])
U_w, s_w, V_w = fbpca_pca(wlet_feats.astype(np.float64), k=n_wlet_pcs)
pc_wlet = wlet_feats @ V_w.T
pc_wlet_labels = [f"wlet_pc_{i}" for i in range(n_wlet_pcs)]
print(f"  Wavelet PCA output: {pc_wlet.shape}")

# Combine
final_feats = np.hstack((pc_feats, pc_wlet))
final_labels = pc_labels + pc_wlet_labels
print(f"\nFinal features: {final_feats.shape}")

# Save intermediate results
np.save(OUTPUT_PATH + "final_features.npy", final_feats)
np.save(OUTPUT_PATH + "inter_dist_all.npy", inter_dist_all)
meta_by_frame.to_csv(OUTPUT_PATH + "meta_by_frame.csv", index=False)
print(f"Saved features to {OUTPUT_PATH}")

# %%
# ============================================================
# 3. EMBEDDING
# ============================================================

print("\n=== Embedding ===")

# Subsample for embedding
feats_sub = final_feats[::DOWNSAMPLE]
print(f"Subsampled: {feats_sub.shape[0]} points (from {final_feats.shape[0]})")

# t-SNE
embedder = Embed(
    embed_method="fitsne",
    perplexity=PERPLEXITY,
    lr="auto"
)
embed_vals_sub = embedder.embed(feats_sub, save_self=True)
print(f"Embedding shape: {embed_vals_sub.shape}")

# Watershed clustering
ws = Watershed(sigma=SIGMA, max_clip=1, log_out=True, pad_factor=0.05)
clusters_sub = ws.fit_predict(data=embed_vals_sub)
n_clusters = len(np.unique(clusters_sub))
print(f"Number of clusters: {n_clusters}")

# Save embedding
np.save(OUTPUT_PATH + "embed_vals_sub.npy", embed_vals_sub)
np.save(OUTPUT_PATH + "clusters_sub.npy", clusters_sub)

# %%
# ============================================================
# 4. CREATE DATASTRUCT FOR VISUALIZATION
# ============================================================

print("\n=== Creating DataStruct ===")

# For DataStruct, we need:
# - pose: (n_frames, n_keypoints, 3)
# - id: recording ID per frame
# - meta: metadata DataFrame (one row per recording)
# - meta_by_frame: metadata per frame
# - connectivity: skeleton info

# Create a "combined" pose by stacking both animals
# Shape: (n_frames, 18, 3) = 9 keypoints per animal × 2
pose_combined = np.concatenate([pose1_all, pose2_all], axis=1)
print(f"Combined pose shape: {pose_combined.shape}")

# Create connectivity for combined skeleton (both animals)
class SocialConnectivity:
    """Connectivity object for two-animal visualization"""
    def __init__(self, joint_names, segments, n_keypoints=9):
        self.n_keypoints = n_keypoints * 2
        # Animal 1 names
        self.joint_names = [f"A1_{name}" for name in joint_names]
        # Animal 2 names (offset indices)
        self.joint_names += [f"A2_{name}" for name in joint_names]
        
        # Segments for animal 1
        self.segments = list(segments)
        # Segments for animal 2 (offset by n_keypoints)
        self.segments += [(i + n_keypoints, j + n_keypoints) for i, j in segments]
        
        # Alias for neuroposelib compatibility
        self.links = self.segments
        
        # Colors: blue for animal 1, red for animal 2
        self.colors = np.array([[0, 0, 1, 0.8]] * len(segments) + [[1, 0, 0, 0.8]] * len(segments))
        self.keypt_colors = np.array([[0, 0, 1, 0.8]] * n_keypoints + [[1, 0, 0, 0.8]] * n_keypoints)
        
        # Angles (empty for now - can add later)
        self.angles = []

social_connectivity = SocialConnectivity(JOINT_NAMES, SEGMENTS, N_KEYPOINTS)
print(f"Social connectivity: {social_connectivity.n_keypoints} keypoints, {len(social_connectivity.segments)} segments")

# Create meta DataFrame (one row per recording)
meta = meta_df.copy()
meta['recording_idx'] = meta.index

# Subsample everything to match embedding
pose_sub = pose_combined[::DOWNSAMPLE]
ids_sub = ids[::DOWNSAMPLE]
meta_by_frame_sub = meta_by_frame.iloc[::DOWNSAMPLE].reset_index(drop=True)
inter_dist_sub = inter_dist_all[::DOWNSAMPLE]

# Add cluster and embedding info to meta_by_frame
meta_by_frame_sub['Cluster'] = clusters_sub
meta_by_frame_sub['embed_x'] = embed_vals_sub[:, 0]
meta_by_frame_sub['embed_y'] = embed_vals_sub[:, 1]
meta_by_frame_sub['nose_nose_dist'] = inter_dist_sub[:, 0]
meta_by_frame_sub['centroid_dist'] = inter_dist_sub[:, -1]

# Create DataStruct
data_obj = ds.DataStruct(
    pose=pose_sub,
    id=ids_sub,
    meta=meta,
    meta_by_frame=meta_by_frame_sub,
    connectivity=social_connectivity,
)

# Add features and embedding
data_obj.features = final_feats[::DOWNSAMPLE]
data_obj.embed_vals = embed_vals_sub
data_obj.ws = ws
data_obj.data['Cluster'] = clusters_sub

print(f"\nDataStruct created:")
print(f"  Pose: {data_obj.pose.shape}")
print(f"  Frames: {len(data_obj.data)}")
print(f"  Clusters: {n_clusters}")

# Save DataStruct
data_obj.write_pickle(OUTPUT_PATH)
print(f"\nSaved DataStruct to {OUTPUT_PATH}")

# %%
# ============================================================
# 5. BASIC VISUALIZATIONS
# ============================================================

print("\n=== Visualizations ===")

# 5a. Embedding colored by cluster
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
scatter = ax.scatter(embed_vals_sub[:, 0], embed_vals_sub[:, 1], 
                     c=clusters_sub, s=1, alpha=0.5, cmap='tab20')
ax.set_title(f'Clusters (n={n_clusters})')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

ax = axes[1]
scatter = ax.scatter(embed_vals_sub[:, 0], embed_vals_sub[:, 1], 
                     c=inter_dist_sub[:, 0], s=1, alpha=0.5, cmap='viridis')
ax.set_title('Nose-Nose Distance')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=ax)

ax = axes[2]
scatter = ax.scatter(embed_vals_sub[:, 0], embed_vals_sub[:, 1], 
                     c=inter_dist_sub[:, -1], s=1, alpha=0.5, cmap='viridis')
ax.set_title('Centroid Distance')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + "embedding_overview.png", dpi=150)
plt.show()
print(f"Saved: {OUTPUT_PATH}embedding_overview.png")

# %%
# 5b. Density by animal pair
try:
    vis.plot.density_cat(
        data=data_obj,
        column="pair",
        watershed=data_obj.ws,
        filepath=OUTPUT_PATH + "density_by_pair.png",
        show=True,
    )
    print(f"Saved: {OUTPUT_PATH}density_by_pair.png")
except Exception as e:
    print(f"density_cat error: {e}")

# %%
# 5c. Density by date
try:
    vis.plot.density_cat(
        data=data_obj,
        column="date",
        watershed=data_obj.ws,
        filepath=OUTPUT_PATH + "density_by_date.png",
        show=True,
    )
    print(f"Saved: {OUTPUT_PATH}density_by_date.png")
except Exception as e:
    print(f"density_cat error: {e}")

# %%
# ============================================================
# 6. CLUSTER FREQUENCY ANALYSIS
# ============================================================

print("\n=== Cluster Analysis ===")

# Frequency by recording
freq, combined_keys = analysis.cluster_freq_by_cat(
    data_obj.data["Cluster"].values, cat=data_obj.id
)
freq_df = pd.DataFrame(freq.T, columns=combined_keys)
freq_df.to_csv(OUTPUT_PATH + "cluster_occupancy_by_recording.csv")
print(f"Saved: {OUTPUT_PATH}cluster_occupancy_by_recording.csv")

# Frequency by pair
freq_pair, pair_keys = analysis.cluster_freq_by_cat(
    data_obj.data["Cluster"].values, cat=data_obj.data["pair"].values
)
freq_pair_df = pd.DataFrame(freq_pair.T, columns=pair_keys)
freq_pair_df.to_csv(OUTPUT_PATH + "cluster_occupancy_by_pair.csv")
print(f"Saved: {OUTPUT_PATH}cluster_occupancy_by_pair.csv")

# %%
# ============================================================
# 7. SKELETON VISUALIZATION (SOCIAL VERSION)
# ============================================================

def plot_social_skeleton_3d(pose, frame_idx, segments, n_kp=9, ax=None, 
                            color1='blue', color2='red'):
    """Plot both animals in 3D"""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    pos = pose[frame_idx]  # (18, 3)
    
    # Animal 1
    for i, j in segments[:len(segments)//2]:
        ax.plot([pos[i, 0], pos[j, 0]], 
                [pos[i, 1], pos[j, 1]], 
                [pos[i, 2], pos[j, 2]], 
                c=color1, linewidth=2)
    ax.scatter(pos[:n_kp, 0], pos[:n_kp, 1], pos[:n_kp, 2], c=color1, s=50)
    
    # Animal 2
    for i, j in segments[len(segments)//2:]:
        ax.plot([pos[i, 0], pos[j, 0]], 
                [pos[i, 1], pos[j, 1]], 
                [pos[i, 2], pos[j, 2]], 
                c=color2, linewidth=2)
    ax.scatter(pos[n_kp:, 0], pos[n_kp:, 1], pos[n_kp:, 2], c=color2, s=50)
    
    return ax

# Sample frames from different clusters
def sample_cluster_frames(clusters, n_samples=5):
    """Get sample frame indices from each cluster"""
    unique_clusters = np.unique(clusters)
    samples = {}
    for c in unique_clusters:
        cluster_frames = np.where(clusters == c)[0]
        if len(cluster_frames) >= n_samples:
            samples[c] = np.random.choice(cluster_frames, n_samples, replace=False)
        else:
            samples[c] = cluster_frames
    return samples

# Plot samples from top clusters
print("\n=== Cluster Skeleton Samples ===")

cluster_counts = pd.Series(clusters_sub).value_counts()
top_clusters = cluster_counts.head(9).index.tolist()

fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={'projection': '3d'})
axes = axes.flatten()

for ax_idx, cluster_id in enumerate(top_clusters):
    cluster_frames = np.where(clusters_sub == cluster_id)[0]
    sample_frame = cluster_frames[len(cluster_frames)//2]  # Middle frame
    
    plot_social_skeleton_3d(
        pose_sub, sample_frame, social_connectivity.segments, 
        n_kp=N_KEYPOINTS, ax=axes[ax_idx]
    )
    axes[ax_idx].set_title(f'Cluster {cluster_id} (n={len(cluster_frames)})')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + "cluster_skeleton_samples.png", dpi=150)
plt.show()
print(f"Saved: {OUTPUT_PATH}cluster_skeleton_samples.png")

# %%
# ============================================================
# 8. VIDEO GENERATION (OPTIONAL - CAN BE SLOW)
# ============================================================

# This is adapted from the original script's vis.pose.sample_grid3D
# Uncomment to generate cluster videos


# Center poses for better visualization
pose_centered = pose_sub - pose_sub.mean(axis=-2, keepdims=True)

# FIX: Convert colors to numpy arrays (required by neuroposelib)
social_connectivity.colors = np.array(
    [[0, 0, 1, 0.8]] * 9 +  # Animal 1 segments (blue)
    [[1, 0, 0, 0.8]] * 9    # Animal 2 segments (red)
)
social_connectivity.keypt_colors = np.array(
    [[0, 0, 1, 0.8]] * 9 +  # Animal 1 keypoints
    [[1, 0, 0, 0.8]] * 9    # Animal 2 keypoints
)

vis.pose.sample_grid3D(
    pose_centered,
    connectivity=social_connectivity,
    labels=data_obj.data["Cluster"],
    n_samples=9,
    centered=True,
    N_FRAMES=100,
    fps=30,
    dpi=100,
    watershed=data_obj.ws,
    embed_vals=None,
    VID_NAME="social_cluster",
    filepath=OUTPUT_PATH,
)


print("\n" + "="*50)
print("PIPELINE COMPLETE!")
print("="*50)
print(f"\nOutputs saved to: {OUTPUT_PATH}")
print(f"  - final_features.npy")
print(f"  - embed_vals_sub.npy")
print(f"  - clusters_sub.npy")
print(f"  - meta_by_frame.csv")
print(f"  - DataStruct.pickle")
print(f"  - embedding_overview.png")
print(f"  - cluster_skeleton_samples.png")
print(f"  - cluster_occupancy_*.csv")

# %%
# ============================================================
# BONUS: Quick Analysis - Close vs Far Interactions
# ============================================================

print("\n=== Quick Analysis: Proximity Bins ===")

# Bin by nose-nose distance
nose_dist = meta_by_frame_sub['nose_nose_dist'].values
bins = [0, 100, 200, 400, 800, np.inf]
bin_labels = ['very_close', 'close', 'medium', 'far', 'very_far']
meta_by_frame_sub['proximity_bin'] = pd.cut(nose_dist, bins=bins, labels=bin_labels)

# Plot embedding colored by proximity
fig, ax = plt.subplots(figsize=(10, 8))
for bin_label, color in zip(bin_labels, ['darkblue', 'blue', 'green', 'orange', 'red']):
    mask = meta_by_frame_sub['proximity_bin'] == bin_label
    ax.scatter(embed_vals_sub[mask, 0], embed_vals_sub[mask, 1], 
               s=2, alpha=0.5, label=f'{bin_label} (n={mask.sum()})', c=color)
ax.legend()
ax.set_title('Embedding by Proximity Bin')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.savefig(OUTPUT_PATH + "embedding_by_proximity.png", dpi=150)
plt.show()

# Which clusters are "close interaction" clusters?
close_mask = meta_by_frame_sub['proximity_bin'].isin(['very_close', 'close'])
close_clusters = pd.Series(clusters_sub[close_mask]).value_counts()
print("\nTop 'close interaction' clusters:")
print(close_clusters.head(10))