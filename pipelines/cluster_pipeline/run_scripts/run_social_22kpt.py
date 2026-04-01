"""
sdappy_social_260211_cleaned_final.py
=====================================
Cleaned social embedding pipeline.

- Loads social recordings, computes SocialMapper-style features
- Global embedding + watershed clustering
- Close-cluster re-embedding
- All visualizations: embedding plots, cluster stats, diagnostics,
  per-session density (from sdappy_single), and 9-sample grid videos
- Config loaded from separate config file

Usage:
    python sdappy_social_260211_cleaned_final.py
"""

# ============================================================
# 0. CONFIGURATION (loaded from external config)
# ============================================================
import importlib.util
import os
import sys
import numpy as np

# --- Config path (edit this to switch configs) ---
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "configs",
    "social_22kpt.py",
)

# Load config dynamically
spec = importlib.util.spec_from_file_location("config_module", CONFIG_PATH)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
for attr_name in dir(config_module):
    if not attr_name.startswith("_"):
        globals()[attr_name] = getattr(config_module, attr_name)

# Load skeleton module for SOCIAL_INDICES, joint names, etc.
spec_sk = importlib.util.spec_from_file_location("skeleton_config", SKELETON_PATH)
skeleton_module = importlib.util.module_from_spec(spec_sk)
spec_sk.loader.exec_module(skeleton_module)

SOCIAL_INDICES = skeleton_module.SOCIAL_INDICES[SKELETON_NAME]
JOINT_NAMES = skeleton_module.JOINT_NAME_DICT[SKELETON_NAME]
SEGMENTS = skeleton_module.CONNECTIVITY_DICT[SKELETON_NAME]
N_KEYPOINTS = len(JOINT_NAMES)

# Derived indices from skeleton
CENTER_IDX = SOCIAL_INDICES["center"]
SPINE_FRONT_IDX = SOCIAL_INDICES["spine_front"]
NOSE_IDX = SOCIAL_INDICES["nose"]
TAIL_IDX = SOCIAL_INDICES["tail"]

# SocialMapper-style keypoint subsets
# Joint subset for pairwise distances (12 keypoints, SocialMapper style)
if SKELETON_NAME == "mouse22_notail":
    JOINT_SUBSET_INDICES = [2, 3, 4, 5, 11, 8, 15, 12, 18, 16, 21, 19]
    Z_HEIGHT_IDX = [2, 3, 4, 11, 8, 15, 12]
    VELOCITY_IDX = [2, 8, 15, 12, 16, 19]
elif SKELETON_NAME == "mouse9":
    JOINT_SUBSET_INDICES = list(range(9))
    Z_HEIGHT_IDX = [0, 3, 4]
    VELOCITY_IDX = [0, 5, 6, 7, 8]
else:
    JOINT_SUBSET_INDICES = list(range(N_KEYPOINTS))
    Z_HEIGHT_IDX = list(range(N_KEYPOINTS))
    VELOCITY_IDX = list(range(N_KEYPOINTS))

WAVELET_FREQS = np.geomspace(0.2, 5, 10)

print(f"Config: {SKELETON_NAME}, {N_KEYPOINTS} keypoints")
print(f"  Joint subset ({len(JOINT_SUBSET_INDICES)}): {[JOINT_NAMES[i] for i in JOINT_SUBSET_INDICES]}")
print(f"  Center: {JOINT_NAMES[CENTER_IDX]} ({CENTER_IDX})")
print(f"  Output: {OUTPUT_PATH}")


# ============================================================
# 1. IMPORTS
# ============================================================
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
import time

sys.path.insert(0, str(Path(SOCIAL_FEATURES_PATH).parent))
sys.path.insert(0, str(Path(SOCIAL_VIS_PATH).parent))

from social_features import (
    preprocess_pose_social,
    get_inter_animal_distances,
    get_centroid_distance,
    get_social_angles,
    get_heading_angle,
    compute_derivatives,
)
from socialmapper_utils import compute_velocity, compute_angle_diff, smooth_distance_matrix
from social_4panels import social_video_watershed_4panel
from social_video_grid import social_video_grid
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import label as scipy_label
from fbpca import pca as fbpca_pca

from neuroposelib import features as npf
from neuroposelib import DataStruct as ds
from neuroposelib.embed import Embed, Watershed

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
print("Imports complete")


# ============================================================
# 2. SOCIAL CONNECTIVITY (for 2-animal visualization)
# ============================================================
class SocialConnectivity:
    """Connectivity object for two-animal visualization."""
    def __init__(self, joint_names, segments, n_keypoints):
        self.n_keypoints = n_keypoints * 2
        self.joint_names = (
            [f"A1_{n}" for n in joint_names] +
            [f"A2_{n}" for n in joint_names]
        )
        self.segments = list(segments) + [(i + n_keypoints, j + n_keypoints) for i, j in segments]
        self.links = self.segments
        n_seg = len(segments)
        self.colors = np.array(
            [[0, 0, 1, 0.8]] * n_seg + [[1, 0, 0, 0.8]] * n_seg
        )
        self.keypt_colors = np.array(
            [[0, 0, 1, 0.8]] * n_keypoints + [[1, 0, 0, 0.8]] * n_keypoints
        )
        self.angles = []

social_connectivity = SocialConnectivity(JOINT_NAMES, SEGMENTS, N_KEYPOINTS)
print(f"Social connectivity: {social_connectivity.n_keypoints} kpts, {len(social_connectivity.segments)} segments")


# ============================================================
# 3. LOAD DATA + COMPUTE SOCIALMAPPER-STYLE FEATURES
# ============================================================
print("\n=== Loading Data ===")

meta_df = pd.read_csv(SOCIAL_CSV_PATH)
print(f"Found {len(meta_df)} recordings, loading first {N_SESSIONS}")

# Storage
all_joint_distances = []
all_pose1, all_pose2 = [], []
all_z1, all_z2 = [], []
all_vel1, all_vel2 = [], []
all_ang = []
all_PJ, all_PJN = [], []
all_centroid_dist = []
frame_meta = []

t_start = time.time()

for idx, row in meta_df.iterrows():
    if idx >= N_SESSIONS:
        break
    try:
        mat_data = sio.loadmat(row["Prediction_path"])
        pose_raw = mat_data["pred"]

        pose1 = pose_raw[:, 0, :, :].transpose(0, 2, 1)  # (frames, n_kp, 3)
        pose2 = pose_raw[:, 1, :, :].transpose(0, 2, 1)
        n_frames = pose1.shape[0]

        # Smooth
        pose1_s = preprocess_pose_social(pose1)
        pose2_s = preprocess_pose_social(pose2)

        # --- Floor & scale ---
        all_z_feet = np.concatenate([
            pose1_s[:, SOCIAL_INDICES.get("hindpaws", SOCIAL_INDICES.get("ankles"))[0], 2],
            pose1_s[:, SOCIAL_INDICES.get("hindpaws", SOCIAL_INDICES.get("ankles"))[1], 2],
            pose2_s[:, SOCIAL_INDICES.get("hindpaws", SOCIAL_INDICES.get("ankles"))[0], 2],
            pose2_s[:, SOCIAL_INDICES.get("hindpaws", SOCIAL_INDICES.get("ankles"))[1], 2],
        ])
        floor_z = np.percentile(all_z_feet, 10)

        sn_tail1 = np.linalg.norm(pose1_s[:, NOSE_IDX, :] - pose1_s[:, TAIL_IDX, :], axis=1)
        sn_tail2 = np.linalg.norm(pose2_s[:, NOSE_IDX, :] - pose2_s[:, TAIL_IDX, :], axis=1)
        body_len = np.percentile(np.concatenate([sn_tail1, sn_tail2]), 95)
        scale_val = 90.0 / body_len

        # --- Joint pairwise distances (combined, smoothed) ---
        pose1_sub = pose1_s[:, JOINT_SUBSET_INDICES, :]
        pose2_sub = pose2_s[:, JOINT_SUBSET_INDICES, :]
        pose_combined = np.concatenate([pose1_sub, pose2_sub], axis=1)

        n_kp_comb = pose_combined.shape[1]
        joint_dists = []
        for i in range(n_kp_comb):
            for j in range(n_kp_comb):
                if i != j:
                    d = np.linalg.norm(pose_combined[:, i, :] - pose_combined[:, j, :], axis=1)
                    joint_dists.append(d)
        joint_dist_matrix = smooth_distance_matrix(np.column_stack(joint_dists))

        # --- Z heights ---
        z1 = (pose1_s[:, Z_HEIGHT_IDX, 2] - floor_z) * scale_val
        z2 = (pose2_s[:, Z_HEIGHT_IDX, 2] - floor_z) * scale_val

        # --- Velocities ---
        vel1 = np.column_stack([compute_velocity(pose1_s, kp) for kp in VELOCITY_IDX]) * scale_val
        vel2 = np.column_stack([compute_velocity(pose2_s, kp) for kp in VELOCITY_IDX]) * scale_val

        # --- Key body parts ---
        n1, s1, c1, t1 = (pose1_s[:, NOSE_IDX, :], pose1_s[:, SPINE_FRONT_IDX, :],
                           pose1_s[:, CENTER_IDX, :], pose1_s[:, TAIL_IDX, :])
        n2, s2, c2, t2 = (pose2_s[:, NOSE_IDX, :], pose2_s[:, SPINE_FRONT_IDX, :],
                           pose2_s[:, CENTER_IDX, :], pose2_s[:, TAIL_IDX, :])
        dist_c = np.linalg.norm(c1 - c2, axis=1, keepdims=True) + 1e-8

        # --- Deflection angles (6 total) ---
        ang_info = np.zeros((n_frames, 6))
        h1, h2 = n1 - s1, n2 - s2
        ang_info[:, 0] = compute_angle_diff(n2 - s1, h1)
        ang_info[:, 1] = compute_angle_diff(c2 - s1, h1)
        ang_info[:, 2] = compute_angle_diff(t2 - s1, h1)
        ang_info[:, 3] = compute_angle_diff(n1 - s2, h2)
        ang_info[:, 4] = compute_angle_diff(c1 - s2, h2)
        ang_info[:, 5] = compute_angle_diff(t1 - s2, h2)
        ang_info = np.pi - ang_info  # 0=toward, pi=away

        # --- Inter-animal distances ---
        d1n = np.linalg.norm(n2 - n1, axis=1, keepdims=True)
        d1c = np.linalg.norm(n2 - c1, axis=1, keepdims=True)
        d1t = np.linalg.norm(n2 - t1, axis=1, keepdims=True)
        d2n = np.linalg.norm(n1 - n2, axis=1, keepdims=True)
        d2c = np.linalg.norm(n1 - c2, axis=1, keepdims=True)
        d2t = np.linalg.norm(n1 - t2, axis=1, keepdims=True)
        PJ = np.hstack([d1n, d1c, d1t, d2n, d2c, d2t])
        PJN = PJ / dist_c

        centroid_dist, _ = get_centroid_distance(pose1_s, pose2_s)

        # Store
        all_joint_distances.append(joint_dist_matrix)
        all_pose1.append(pose1_s)
        all_pose2.append(pose2_s)
        all_z1.append(z1)
        all_z2.append(z2)
        all_vel1.append(vel1)
        all_vel2.append(vel2)
        all_ang.append(ang_info)
        all_PJ.append(PJ)
        all_PJN.append(PJN)
        all_centroid_dist.append(centroid_dist)

        for f in range(n_frames):
            frame_meta.append({
                "recording_idx": idx,
                "frame_in_recording": f,
                "Animal1": row["Animal1"],
                "Animal2": row["Animal2"],
            })

        if idx % 10 == 0:
            print(f"  {idx}: {row['Animal1']}-{row['Animal2']}, {n_frames} fr, scale={scale_val:.3f}")

    except Exception as e:
        print(f"  ERROR {idx}: {e}")

# Concatenate all sessions
joint_distances_all = np.vstack(all_joint_distances)
pose1_all = np.vstack(all_pose1)
pose2_all = np.vstack(all_pose2)
z1_all = np.vstack(all_z1)
z2_all = np.vstack(all_z2)
vel1_all = np.vstack(all_vel1)
vel2_all = np.vstack(all_vel2)
ang_all = np.vstack(all_ang)
PJ_all = np.vstack(all_PJ)
PJN_all = np.vstack(all_PJN)
centroid_dist_all = np.vstack(all_centroid_dist)
meta_by_frame = pd.DataFrame(frame_meta)

n_total = len(joint_distances_all)
print(f"\nLoaded {min(N_SESSIONS, len(meta_df))} sessions, {n_total} total frames")
print(f"  Joint dists: {joint_distances_all.shape[1]}, Z: {z1_all.shape[1]*2}, "
      f"Vel: {vel1_all.shape[1]*2}, Ang: {ang_all.shape[1]}, Dist: {PJ_all.shape[1]*2}")
print(f"  Time: {time.time() - t_start:.1f}s")

# --- Save raw poses, meta, centroid_dist for later re-visualization ---
print("\n--- Saving pose/meta for future reload ---")
np.save(OUTPUT_PATH + "pose1_all.npy", pose1_all)
np.save(OUTPUT_PATH + "pose2_all.npy", pose2_all)
np.save(OUTPUT_PATH + "centroid_dist_all.npy", centroid_dist_all)
meta_by_frame.to_csv(OUTPUT_PATH + "meta_by_frame.csv", index=False)
print(f"  Saved pose1_all.npy ({pose1_all.shape}), pose2_all.npy, centroid_dist_all.npy, meta_by_frame.csv")


# ============================================================
# 4. FEATURE ASSEMBLY (SocialMapper-style)
# ============================================================
print("\n=== Feature Assembly ===")

ids = meta_by_frame["recording_idx"].values
joint_distances_all = np.nan_to_num(joint_distances_all, nan=0.0, posinf=1e6, neginf=-1e6)

# PCA on joint distances
scaler = StandardScaler()
joint_centered = scaler.fit_transform(joint_distances_all)
U, s, V = fbpca_pca(joint_centered.astype(np.float64), k=N_PCS_JOINT)
joint_pcs = joint_centered @ V.T
joint_pc_labels = [f"joint_pc_{i}" for i in range(N_PCS_JOINT)]
var_explained = 100 * np.sum(s[:N_PCS_JOINT] ** 2) / np.sum(s ** 2)
print(f"  Joint PCA: {joint_pcs.shape}, var explained: {var_explained:.1f}%")

# Wavelet on joint PCs
wlet_feats, wlet_labels = npf.wavelet(
    joint_pcs, joint_pc_labels, ids,
    f_s=FRAME_RATE, freq=WAVELET_FREQS, w0=5,
)
wlet_log = np.clip(np.log(wlet_feats + 1e-10), -3, None)
print(f"  Wavelet (log): {wlet_log.shape}")

# Weighted feature concatenation
final_feats = np.hstack([
    joint_pcs,
    wlet_log,
    WEIGHT_Z * z1_all,
    WEIGHT_Z * z2_all,
    WEIGHT_VEL * vel1_all,
    WEIGHT_VEL * vel2_all,
    WEIGHT_ANG * ang_all,
    WEIGHT_PJ * PJ_all,
    WEIGHT_PJN * PJN_all,
])
final_feats = np.nan_to_num(final_feats, nan=0.0, posinf=1e6, neginf=-1e6)

print(f"  Final features: {final_feats.shape}")
print(f"    Wavelet: {wlet_log.shape[1]}, Z: {z1_all.shape[1]*2}, "
      f"Vel: {vel1_all.shape[1]*2}, Ang: {ang_all.shape[1]}, Dist: {PJ_all.shape[1]*2}")
np.save(OUTPUT_PATH + "final_features_socialmapper.npy", final_feats)


# ============================================================
# 5. EMBEDDING + CLUSTERING
# ============================================================
print("\n=== Embedding ===")

feats_sub = final_feats[::DOWNSAMPLE]
print(f"Subsampled: {feats_sub.shape[0]} (from {final_feats.shape[0]})")

embedder = Embed(embed_method="fitsne", perplexity=PERPLEXITY, lr="auto")
embed_vals_sub = embedder.embed(feats_sub, save_self=True)
print(f"Embedding: {embed_vals_sub.shape}")

ws = Watershed(sigma=SIGMA, max_clip=1, log_out=True, pad_factor=0.05)
clusters_sub = ws.fit_predict(data=embed_vals_sub)
n_clusters = len(np.unique(clusters_sub))
print(f"Clusters: {n_clusters}")

np.save(OUTPUT_PATH + "embed_vals_sub.npy", embed_vals_sub)
np.save(OUTPUT_PATH + "clusters_sub.npy", clusters_sub)

# Save Watershed object for later reload
import pickle
with open(OUTPUT_PATH + "watershed_global.pkl", "wb") as f:
    pickle.dump(ws, f)
print("  Saved watershed_global.pkl")


# ============================================================
# 6. VISUALIZATION: Embedding Overview
# ============================================================
print("\n=== Embedding Overview ===")

centroid_dist_sub = centroid_dist_all[::DOWNSAMPLE].flatten()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.scatter(embed_vals_sub[:, 0], embed_vals_sub[:, 1],
           c=clusters_sub, cmap="tab20", s=1, alpha=0.5)
ax.set_title(f"Clusters (n={n_clusters})")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

ax = axes[1]
sc = ax.scatter(embed_vals_sub[:, 0], embed_vals_sub[:, 1],
                c=centroid_dist_sub, cmap="viridis", s=1, alpha=0.5,
                vmin=np.percentile(centroid_dist_sub, 5),
                vmax=np.percentile(centroid_dist_sub, 95))
plt.colorbar(sc, ax=ax, label="Centroid dist (mm)")
ax.set_title("Colored by Distance")
ax.set_xlabel("t-SNE 1")

ax = axes[2]
ax.hist2d(embed_vals_sub[:, 0], embed_vals_sub[:, 1], bins=100, cmap="hot")
ax.set_title("Density")
ax.set_xlabel("t-SNE 1")

plt.tight_layout()
plt.savefig(OUTPUT_PATH + "embedding_overview.png", dpi=150)
plt.close()
print("  Saved embedding_overview.png")


# ============================================================
# 7. CLUSTER DISTANCE STATISTICS
# ============================================================
print("\n=== Cluster Distance Stats ===")

cluster_stats = []
for c in np.unique(clusters_sub):
    mask = clusters_sub == c
    cluster_stats.append({
        "cluster": c,
        "mean_dist": centroid_dist_sub[mask].mean(),
        "std_dist": centroid_dist_sub[mask].std(),
        "count": mask.sum(),
    })
stats_df = pd.DataFrame(cluster_stats).sort_values("mean_dist")
stats_df.to_csv(OUTPUT_PATH + "cluster_distance_stats.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(stats_df)), stats_df["mean_dist"].values,
       yerr=stats_df["std_dist"].values, alpha=0.7)
ax.set_xlabel("Cluster (sorted by mean distance)")
ax.set_ylabel("Mean centroid distance (mm)")
ax.set_title("Cluster Distance Homogeneity")
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "cluster_distance_check.png", dpi=150)
plt.close()
print("  Saved cluster_distance_check.png")


# ============================================================
# 8. CLOSE CLUSTER RE-EMBEDDING
# ============================================================
print("\n=== Close Cluster Re-embedding ===")

close_cluster_ids = stats_df[stats_df["mean_dist"] < CLOSE_DIST_THRESHOLD]["cluster"].values
close_mask = np.isin(clusters_sub, close_cluster_ids)
close_indices = np.where(close_mask)[0]
print(f"Close clusters: {len(close_cluster_ids)}, frames: {close_mask.sum()} ({100*close_mask.mean():.1f}%)")

# Temporal bout stats
labeled, n_bouts = scipy_label(close_mask.astype(int))
bout_lengths = [np.sum(labeled == i) for i in range(1, n_bouts + 1)]
print(f"  Bouts: {n_bouts}, median length: {np.median(bout_lengths):.0f} frames")

feats_close = feats_sub[close_mask]
CLOSE_PERPLEXITY = min(50, len(feats_close) // 5)

embedder_close = Embed(embed_method="fitsne", perplexity=CLOSE_PERPLEXITY, lr="auto")
embed_close = embedder_close.embed(feats_close, save_self=True)

ws_close = Watershed(sigma=CLOSE_SIGMA, max_clip=1, log_out=True, pad_factor=0.05)
clusters_close = ws_close.fit_predict(data=embed_close)
n_close_clusters = len(np.unique(clusters_close))
print(f"  Refined close clusters: {n_close_clusters}")

np.save(OUTPUT_PATH + "embed_close.npy", embed_close)
np.save(OUTPUT_PATH + "clusters_close.npy", clusters_close)
np.save(OUTPUT_PATH + "close_indices.npy", close_indices)

with open(OUTPUT_PATH + "watershed_close.pkl", "wb") as f:
    pickle.dump(ws_close, f)
print("  Saved watershed_close.pkl")


# ============================================================
# 9. CLOSE CLUSTER VISUALIZATION
# ============================================================
print("\n=== Close Cluster Visualization ===")

close_dists = centroid_dist_sub[close_mask]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax = axes[0]
ax.scatter(embed_close[:, 0], embed_close[:, 1],
           c=clusters_close, cmap="tab20", s=2, alpha=0.6)
ax.set_title(f"Refined Close Clusters (n={n_close_clusters})")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

ax = axes[1]
sc = ax.scatter(embed_close[:, 0], embed_close[:, 1],
                c=close_dists, cmap="viridis", s=2, alpha=0.6)
plt.colorbar(sc, ax=ax, label="Distance (mm)")
ax.set_title("Colored by Centroid Distance")
ax.set_xlabel("t-SNE 1")

ax = axes[2]
ax.hist2d(embed_close[:, 0], embed_close[:, 1], bins=80, cmap="hot")
ax.set_title("Density")
ax.set_xlabel("t-SNE 1")

plt.tight_layout()
plt.savefig(OUTPUT_PATH + "embedding_close_refined.png", dpi=150)
plt.close()
print("  Saved embedding_close_refined.png")

# Close cluster stats
close_stats = []
for c in np.unique(clusters_close):
    mask = clusters_close == c
    close_stats.append({
        "cluster": c,
        "mean_dist": close_dists[mask].mean(),
        "std_dist": close_dists[mask].std(),
        "count": mask.sum(),
    })
close_stats_df = pd.DataFrame(close_stats).sort_values("mean_dist")
close_stats_df.to_csv(OUTPUT_PATH + "close_cluster_stats.csv", index=False)
print(close_stats_df)


# ============================================================
# 10. EVALUATION DIAGNOSTICS
# ============================================================
print("\n=== Evaluation Diagnostics ===")

ang_sub = ang_all[::DOWNSAMPLE]
PJ_sub = PJ_all[::DOWNSAMPLE]
PJN_sub = PJN_all[::DOWNSAMPLE]
centroid_sub = centroid_dist_all[::DOWNSAMPLE].flatten()
vel1_sub = vel1_all[::DOWNSAMPLE]
vel2_sub = vel2_all[::DOWNSAMPLE]

deflect_to_nose = ang_sub[:, 0]
deflect_to_center = ang_sub[:, 1]
deflect_to_tail = ang_sub[:, 2]
partner_deflect_to_nose = ang_sub[:, 3]
partner_deflect_to_center = ang_sub[:, 4]
partner_deflect_to_tail = ang_sub[:, 5]
nose_to_nose = PJ_sub[:, 0]
nose_to_center = PJ_sub[:, 1]
nose_to_tail = PJ_sub[:, 2]
focal_speed = vel1_sub.mean(axis=1)
partner_speed = vel2_sub.mean(axis=1)

# --- Close Cluster Signatures ---
print("\n--- Close Cluster Signatures ---")
close_sigs = []
for c in close_cluster_ids:
    mask = clusters_sub == c
    sig = {
        "cluster": c, "n_frames": mask.sum(), "mean_dist": centroid_sub[mask].mean(),
        "deflect_nose": deflect_to_nose[mask].mean(),
        "deflect_center": deflect_to_center[mask].mean(),
        "deflect_tail": deflect_to_tail[mask].mean(),
        "partner_deflect_nose": partner_deflect_to_nose[mask].mean(),
        "partner_deflect_center": partner_deflect_to_center[mask].mean(),
        "mutual_nose": (deflect_to_nose[mask].mean() + partner_deflect_to_nose[mask].mean()) / 2,
        "focal_speed": focal_speed[mask].mean(),
        "partner_speed": partner_speed[mask].mean(),
        "nose_to_nose": nose_to_nose[mask].mean(),
        "nose_to_tail": nose_to_tail[mask].mean(),
    }
    close_sigs.append(sig)

sig_df = pd.DataFrame(close_sigs).sort_values("mean_dist")
sig_df.to_csv(OUTPUT_PATH + "close_cluster_signatures.csv", index=False)

for _, row in sig_df.iterrows():
    c = int(row["cluster"])
    interp = []
    if row["mean_dist"] < 50:
        interp.append("VERY CLOSE")
    elif row["mean_dist"] < 70:
        interp.append("CLOSE")
    else:
        interp.append("MODERATE")
    if row["deflect_nose"] > 2.5:
        interp.append("focal->partner_nose")
    elif row["deflect_tail"] > 2.5:
        interp.append("focal->partner_tail")
    elif row["deflect_nose"] < 1.0:
        interp.append("focal_away")
    if row["partner_deflect_nose"] > 2.5:
        interp.append("partner->focal_nose")
    elif row["partner_deflect_nose"] < 1.0:
        interp.append("partner_away")
    if row["mutual_nose"] > 2.5:
        interp.append("MUTUAL_FACING")
    if row["focal_speed"] > 2:
        interp.append("focal_moving")
    if row["partner_speed"] > 2:
        interp.append("partner_moving")
    print(f"  Cluster {c:3d} (n={row['n_frames']:4.0f}, d={row['mean_dist']:5.1f}mm): {' | '.join(interp)}")

# --- Within-cluster distribution histograms ---
top_close = sig_df.nlargest(8, "n_frames")
fig, axes = plt.subplots(len(top_close), 5, figsize=(15, 2.5 * len(top_close)))
for ri, (_, row) in enumerate(top_close.iterrows()):
    c = int(row["cluster"])
    mask = clusters_sub == c
    metrics = [
        ("Deflect->Nose", deflect_to_nose[mask], (0, np.pi)),
        ("Deflect->Tail", deflect_to_tail[mask], (0, np.pi)),
        ("Partner->Nose", partner_deflect_to_nose[mask], (0, np.pi)),
        ("Focal Speed", focal_speed[mask], (0, 5)),
        ("Nose-Nose Dist", nose_to_nose[mask], (0, 150)),
    ]
    for ci, (name, data, xlim) in enumerate(metrics):
        ax = axes[ri, ci]
        ax.hist(data, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(data.mean(), color="blue", linestyle="-", linewidth=2)
        ax.set_xlim(xlim)
        if ri == 0:
            ax.set_title(name, fontsize=10)
        if ci == 0:
            ax.set_ylabel(f"C{c}\n(n={mask.sum()}, d={row['mean_dist']:.0f}mm)", fontsize=8)

plt.suptitle("Within-Cluster Distributions (SocialMapper Features)", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "cluster_distributions.png", dpi=150)
plt.close()
print("  Saved cluster_distributions.png")


# --- Temporal coherence ---
print("\n--- Temporal Coherence ---")

def compute_bout_stats(cluster_labels, frame_rate, downsample):
    bout_lengths = []
    for c in np.unique(cluster_labels):
        mask = (cluster_labels == c).astype(int)
        labeled_b, n_b = scipy_label(mask)
        for bid in range(1, n_b + 1):
            bout_lengths.append((labeled_b == bid).sum() / frame_rate * downsample)
    return np.array(bout_lengths)

bout_durations = compute_bout_stats(clusters_sub, FRAME_RATE, DOWNSAMPLE)
print(f"  Total bouts: {len(bout_durations)}")
print(f"  Mean: {bout_durations.mean():.2f}s, Median: {np.median(bout_durations):.2f}s")
print(f"  >0.5s: {100*(bout_durations > 0.5).mean():.1f}%, >1.0s: {100*(bout_durations > 1.0).mean():.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(bout_durations, bins=50, range=(0, 10), alpha=0.7, edgecolor="black")
ax.axvline(0.5, color="red", linestyle="--", label="0.5s threshold")
ax.axvline(np.median(bout_durations), color="blue", linestyle="-",
           label=f"Median ({np.median(bout_durations):.2f}s)")
ax.set_xlabel("Bout Duration (s)"); ax.set_ylabel("Count")
ax.set_title("Temporal Coherence: Bout Length Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "bout_lengths.png", dpi=150)
plt.close()
print("  Saved bout_lengths.png")


# ============================================================
# 11. PER-SESSION DENSITY PLOTS (from sdappy_single)
# ============================================================
print("\n=== Per-Session Density Plots ===")

EPS = 1e-6

meta_by_frame_sub = meta_by_frame.iloc[::DOWNSAMPLE].reset_index(drop=True)
unique_recordings = meta_by_frame_sub["recording_idx"].unique()

for rec_idx in unique_recordings:
    rec_mask = meta_by_frame_sub["recording_idx"] == rec_idx
    rec_embed = embed_vals_sub[rec_mask]
    rec_a1 = meta_by_frame_sub.loc[rec_mask, "Animal1"].iloc[0]
    rec_a2 = meta_by_frame_sub.loc[rec_mask, "Animal2"].iloc[0]

    density = ws.fit_density(rec_embed, new=False)
    masked = density.copy()
    masked[ws.watershed_map == -1] = 0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(masked, vmin=EPS, cmap="viridis")
    if ws.borders is not None:
        ax.plot(ws.borders[:, 0], ws.borders[:, 1], ".k", markersize=0.1)
    ax.set_aspect(0.9)
    ax.axis("off")
    ax.set_title(f"Rec {rec_idx}: {rec_a1} vs {rec_a2}\nn={rec_mask.sum()}")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + f"density_recording_{rec_idx:03d}.png", dpi=200)
    plt.close()

print(f"  Saved {len(unique_recordings)} density plots")


# ============================================================
# 12. SIGMA SWEEP
# ============================================================
print("\n=== Sigma Sweep ===")
for sigma_test in [15, 25, 35, 50]:
    ws_t = Watershed(sigma=sigma_test, max_clip=1, log_out=True, pad_factor=0.05)
    cl_t = ws_t.fit_predict(data=embed_vals_sub)
    print(f"  Sigma={sigma_test:2d} -> {len(np.unique(cl_t)):3d} clusters")


# ============================================================
# 13. PREPARE POSE FOR VIDEO GENERATION
# ============================================================
print("\n=== Preparing Pose for Videos ===")

pose1_sub = pose1_all[::DOWNSAMPLE]
pose2_sub = pose2_all[::DOWNSAMPLE]
pose_combined_sub = np.concatenate([pose1_sub, pose2_sub], axis=1)

# Center on A1 center
center_point = pose_combined_sub[:, CENTER_IDX:CENTER_IDX + 1, :]
pose_centered = pose_combined_sub - center_point

# Close cluster pose
pose_close = pose_centered[close_mask]
print(f"  General pose: {pose_centered.shape}")
print(f"  Close pose: {pose_close.shape}")


# ============================================================
# 14. GENERAL CLUSTER VIDEOS (9-sample grid, no watershed map)
# ============================================================
print("\n=== General Cluster Videos (Grid) ===")

social_video_grid(
    pose=pose_centered,
    connectivity=social_connectivity,
    labels=clusters_sub,
    n_clusters=None,
    n_samples=VID_N_SAMPLES,
    N_FRAMES=VID_N_FRAMES,
    fps=VID_FPS,
    dpi=VID_DPI,
    n_kp=N_KEYPOINTS,
    filepath=OUTPUT_PATH,
    VID_NAME="general_grid",
    nose_idx=NOSE_IDX,
)


# ============================================================
# 15. CLOSE CLUSTER VIDEOS (9-sample grid, no watershed map)
# ============================================================
print("\n=== Close Cluster Videos (Grid) ===")

social_video_grid(
    pose=pose_close,
    connectivity=social_connectivity,
    labels=clusters_close,
    n_clusters=None,
    n_samples=CLOSE_VID_N_SAMPLES,
    N_FRAMES=CLOSE_VID_N_FRAMES,
    fps=CLOSE_VID_FPS,
    dpi=VID_DPI,
    n_kp=N_KEYPOINTS,
    filepath=OUTPUT_PATH,
    VID_NAME="close_grid",
    nose_idx=NOSE_IDX,
)


# ============================================================
# 16. GENERAL CLUSTER VIDEOS (4-panel with watershed map)
# ============================================================
print("\n=== General Cluster Videos (Watershed 4-panel) ===")

social_video_watershed_4panel(
    pose=pose_centered,
    connectivity=social_connectivity,
    labels=clusters_sub,
    watershed=ws,
    n_clusters=None,
    n_samples=4,
    N_FRAMES=VID_N_FRAMES,
    fps=VID_FPS,
    dpi=VID_DPI,
    n_kp=N_KEYPOINTS,
    filepath=OUTPUT_PATH,
    VID_NAME="general_ws_4panel",
    embed_vals=embed_vals_sub,
    nose_idx=NOSE_IDX,
)


# ============================================================
# 17. CLOSE CLUSTER VIDEOS (4-panel with watershed map)
# ============================================================
print("\n=== Close Cluster Videos (Watershed 4-panel) ===")

social_video_watershed_4panel(
    pose=pose_close,
    connectivity=social_connectivity,
    labels=clusters_close,
    watershed=ws_close,
    n_clusters=None,
    n_samples=4,
    N_FRAMES=CLOSE_VID_N_FRAMES,
    fps=CLOSE_VID_FPS,
    dpi=VID_DPI,
    n_kp=N_KEYPOINTS,
    filepath=OUTPUT_PATH,
    VID_NAME="close_ws_4panel",
    embed_vals=embed_close,
    nose_idx=NOSE_IDX,
)


# ============================================================
# 18. DONE
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"Output: {OUTPUT_PATH}")
print(f"""
Files created:
  Data:
    - final_features_socialmapper.npy
    - embed_vals_sub.npy / clusters_sub.npy
    - embed_close.npy / clusters_close.npy / close_indices.npy

  Plots:
    - embedding_overview.png
    - cluster_distance_check.png
    - embedding_close_refined.png
    - cluster_distributions.png
    - bout_lengths.png
    - density_recording_*.png

  CSVs:
    - cluster_distance_stats.csv
    - close_cluster_stats.csv
    - close_cluster_signatures.csv

  Videos:
    - general_grid/          (9-sample grid, all clusters)
    - close_grid/            (9-sample grid, close clusters)
    - general_ws_4panel/     (watershed + 4-panel, all clusters)
    - close_ws_4panel/       (watershed + 4-panel, close clusters)

Summary:
  Sessions: {min(N_SESSIONS, len(meta_df))}
  Frames: {n_total:,}
  Downsampled: {feats_sub.shape[0]:,}
  Global clusters: {n_clusters}
  Close clusters: {n_close_clusters}
""")
