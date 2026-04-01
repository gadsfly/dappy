# %% [markdown]
#  # Individual Embedding from Social Recordings
# 
# 
# 
#  Loads social recordings, splits into individual animals, and runs
# 
#  the standard single-animal embedding pipeline.

# %%
# ============================================================
# CONFIGURATION
# ============================================================
from neuroposelib import read
import numpy as np
import importlib.util
import os, sys
module_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs'))
sys.path.insert(0, module_path)
s_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'individual_22kpt.py')

# Load config dynamically
spec = importlib.util.spec_from_file_location("config_module", s_config)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# Import all variables from config into current namespace
for attr_name in dir(config_module):
    if not attr_name.startswith('_'):
        globals()[attr_name] = getattr(config_module, attr_name)

# Load skeleton module to get SOCIAL_INDICES
spec = importlib.util.spec_from_file_location("skeleton_config", SKELETON_PATH)
skeleton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skeleton_module)

# Get indices from skeleton file
CENTER_IDX = skeleton_module.SOCIAL_INDICES[SKELETON_NAME]['center']
NOSE_IDX = skeleton_module.SOCIAL_INDICES[SKELETON_NAME]['spine_front'] #bad naming, actually means used for rotation/heading direction (spine_front defines which way the animal faces)
SNOUT_IDX = skeleton_module.SOCIAL_INDICES[SKELETON_NAME]['nose']
WAVELET_FREQS = np.linspace(0.1, 1, 10)



# Load connectivity
connectivity = read.connectivity(path=SKELETON_PATH, skeleton_name=SKELETON_NAME)
connectivity.n_keypoints = len(connectivity.joint_names)
connectivity.segments = connectivity.links
connectivity.keypt_colors = np.array(skeleton_module.COLOR_DICT[SKELETON_NAME])

print(f"Config: {SKELETON_NAME}")
print(f"  Keypoints: {connectivity.n_keypoints}")
print(f"  Center: {connectivity.joint_names[CENTER_IDX]} (idx {CENTER_IDX})")
print(f"  Heading: {connectivity.joint_names[NOSE_IDX]} (idx {NOSE_IDX})")
print(f"  Output: {OUTPUT_PATH}")

# for any loopings in the futer to consider...
# configs = [
#     {"SKELETON_NAME": "mouse9", "OUTPUT_PATH": "/path/to/mouse9_output/"},
#     {"SKELETON_NAME": "mouse22_notail", "OUTPUT_PATH": "/path/to/mouse22_output/"},
# ]

# for cfg in configs:
#     SKELETON_NAME = cfg["SKELETON_NAME"]
#     OUTPUT_PATH = cfg["OUTPUT_PATH"]
    # ... rest of pipeline

# %%
# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
import time

from neuroposelib import features
from neuroposelib import DataStruct as ds
from neuroposelib import vis
from neuroposelib import analysis
from neuroposelib import preprocess
from neuroposelib import write
from neuroposelib.embed import Embed, Watershed

# Create output directory
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

print("Imports complete ✓")


# %%
# ============================================================
# 1. LOAD SOCIAL RECORDINGS → SPLIT INTO INDIVIDUALS
# ============================================================

print("\n=== Loading Data ===")

meta_df = pd.read_csv(SOCIAL_CSV_PATH)
print(f"Found {len(meta_df)} social recordings")

all_pose = []
all_ids = []
frame_meta_rows = []

session_counter = 0
t_start = time.time()

for idx, row in meta_df.iterrows():
    try:
        mat_data = sio.loadmat(row['Prediction_path'])
        pose_raw = mat_data['pred']  # (frames, 2, 3, n_keypoints)
        
        # Split into two animals
        pose_a0 = pose_raw[:, 0, :, :].transpose(0, 2, 1)  # (frames, n_kpts, 3)
        pose_a1 = pose_raw[:, 1, :, :].transpose(0, 2, 1)
        
        n_frames = pose_a0.shape[0]
        
        # === HANDLE DIFFERENT METADATA FORMATS ===
        if 'single_is_index' in row and pd.notna(row['single_is_index']):
            # Original format: single vs group housed
            single_idx = int(row['single_is_index'])
            if single_idx == 0:
                animal_types = ['single', 'group']
                animal_names = [row['Animal1'], row['Animal2']]
            else:
                animal_types = ['group', 'single']
                animal_names = [row['Animal2'], row['Animal1']]
            poses = [pose_a0, pose_a1] if single_idx == 0 else [pose_a1, pose_a0]
        else:
            # No single/group distinction - just use position
            # Assume: index 0 = miniscope animal, index 1 = partner
            # (or change labels as needed)
            animal_types = ['miniscope', 'partner']  # or ['animal_0', 'animal_1']
            animal_names = [row.get('Animal1', 'A0'), row.get('Animal2', 'A1')]
            poses = [pose_a0, pose_a1]
        
        # Cohort key for animal identity across sessions
        date_str = str(row.get('date', 'unknown'))
        parts = date_str.split("_")
        cohort = "_".join(parts[:2]) if len(parts) >= 2 else date_str
        
        # Add both animals as separate "sessions"
        for i, (animal_type, pose_data, animal_name) in enumerate(zip(
            animal_types, poses, animal_names
        )):
            partner_name = animal_names[1-i]  # the other animal
            
            all_pose.append(pose_data)
            all_ids.append(np.full(n_frames, session_counter))
            
            for f in range(n_frames):
                frame_meta_rows.append({
                    'session_id': session_counter,
                    'recording_idx': idx,
                    'frame_in_recording': f,
                    'animal_type': animal_type,
                    'AnimalID': f"{animal_name}__{cohort}",
                    'animal_name': animal_name,
                    'partner_name': partner_name,
                    'date': date_str,
                    'cohort': cohort,
                    'pair': f"{animal_names[0]}_{animal_names[1]}",
                })
            
            session_counter += 1
        
        if idx % 10 == 0:
            print(f"  Loaded {idx+1}/{len(meta_df)}: {n_frames} frames each")
            
    except Exception as e:
        print(f"  ERROR {idx}: {e}")

#below looping is for the indexing for minji stuff... above is a revision that made it optional...
# for idx, row in meta_df.iterrows():
#     try:
#         mat_data = sio.loadmat(row['Prediction_path'])
#         pose_raw = mat_data['pred']  # (frames, 2, 3, 9)
        
#         # Split into two animals
#         pose_a0 = pose_raw[:, 0, :, :].transpose(0, 2, 1)  # (frames, 9, 3)
#         pose_a1 = pose_raw[:, 1, :, :].transpose(0, 2, 1)
        
#         n_frames = pose_a0.shape[0]
#         single_idx = int(row['single_is_index'])
        
#         # Determine which is single-housed vs group-housed
#         if single_idx == 0:
#             pose_single, pose_group = pose_a0, pose_a1
#             single_animal, group_animal = row['Animal1'], row['Animal2']
#         else:
#             pose_single, pose_group = pose_a1, pose_a0
#             single_animal, group_animal = row['Animal2'], row['Animal1']
        
#         # Cohort key for animal identity across sessions
#         date_str = str(row.get('date', 'unknown'))
#         parts = date_str.split("_")
#         cohort = "_".join(parts[:2]) if len(parts) >= 2 else date_str
        
#         # Add both animals as separate "sessions"
#         for animal_type, pose_data, animal_name in [
#             ('single', pose_single, single_animal),
#             ('group', pose_group, group_animal)
#         ]:
#             all_pose.append(pose_data)
#             all_ids.append(np.full(n_frames, session_counter))
            
#             for f in range(n_frames):
#                 frame_meta_rows.append({
#                     'session_id': session_counter,
#                     'recording_idx': idx,
#                     'frame_in_recording': f,
#                     'animal_type': animal_type,
#                     'AnimalID': f"{animal_name}__{cohort}",  # Unique animal ID
#                     'animal_name': animal_name,
#                     'partner_name': group_animal if animal_type == 'single' else single_animal,
#                     'date': date_str,
#                     'cohort': cohort,
#                     'pair': f"{row['Animal1']}_{row['Animal2']}",
#                 })
            
#             session_counter += 1
        
#         if idx % 10 == 0:
#             print(f"  Loaded {idx+1}/{len(meta_df)}: {n_frames} frames each")
            
#     except Exception as e:
#         print(f"  ERROR {idx}: {e}")

# Concatenate
pose = np.vstack(all_pose)
ids = np.concatenate(all_ids)
meta_by_frame = pd.DataFrame(frame_meta_rows)

print(f"\n=== Loaded {len(meta_df)} recordings → {session_counter} animal-sessions ===")
print(f"Total frames: {pose.shape[0]:,}")
# print(f"  Single-housed: {(meta_by_frame['animal_type']=='single').sum():,}")
# print(f"  Group-housed: {(meta_by_frame['animal_type']=='group').sum():,}")
print(f"Unique animals: {meta_by_frame['AnimalID'].nunique()}")
print(f"Time: {time.time() - t_start:.1f}s")


# %%
# ============================================================
# 2. PREPROCESSING (from original single-animal pipeline)
# ============================================================

print("\n=== Preprocessing ===")

# Center on spine and rotate (same as original script)
pose = preprocess.center_spine(pose, keypt_idx=CENTER_IDX)
pose = preprocess.rotate_spine(pose, keypt_idx=[CENTER_IDX, NOSE_IDX])

print("  Centered and rotated ✓")

# Save raw pose
write.pose_h5(pose, ids, OUTPUT_PATH + "pose_aligned.h5")


# %%
# # ============================================================
# # 3. FEATURE EXTRACTION (identical to original script)
# # ============================================================

# print("\n=== Feature Extraction ===")

# # Ego pose features
# ego_pose, labels = features.get_ego_pose(pose, connectivity.joint_names)
# print(f"  Ego pose: {ego_pose.shape}")

# # Optional: angles and velocities (uncomment if needed)
# # angles, angle_labels = features.get_angles(pose, connectivity.angles)
# # rel_vel, vel_labels = features.get_velocities(pose, ids, connectivity.joint_names, 
# #                                                joints=np.delete(np.arange(9), CENTER_IDX),
# #                                                widths=[5, 11, 51], f_s=FRAME_RATE)

# combined_features = ego_pose
# combined_labels = labels

# # Clean NaN/Inf
# nan_count = np.isnan(combined_features).sum()
# inf_count = np.isinf(combined_features).sum()
# if nan_count > 0 or inf_count > 0:
#     print(f"  Cleaning NaN: {nan_count}, Inf: {inf_count}")
#     combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)


# %%
# ============================================================
# 3. FEATURE EXTRACTION (with angles, like original script)
# ============================================================

print("\n=== Feature Extraction ===")

# Ego pose features
ego_pose, ego_labels = features.get_ego_pose(pose, connectivity.joint_names)
print(f"  Ego pose: {ego_pose.shape}")

# Joint angles
angles, angle_labels = features.get_angles(pose, connectivity.angles)
print(f"  Angles: {angles.shape}")

# Combine
combined_features = np.hstack((ego_pose, angles))
combined_labels = ego_labels + angle_labels
print(f"  Combined: {combined_features.shape}")

# Clean NaN/Inf
nan_count = np.isnan(combined_features).sum()
inf_count = np.isinf(combined_features).sum()
if nan_count > 0 or inf_count > 0:
    print(f"  Cleaning NaN: {nan_count}, Inf: {inf_count}")
    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)

# %%
# # ============================================================
# # 4. PCA → WAVELET → PCA (identical to original script)
# # ============================================================

# print("\n=== PCA on Pose Features ===")
# t = time.time()

# pc_feats, pc_labels = features.pca(
#     combined_features,
#     combined_labels,
#     categories=["ego_euc"],
#     n_pcs=N_PCS_POSE,
#     method="fbpca",
# )
# print(f"  PCA output: {pc_feats.shape}")
# print(f"  Time: {time.time() - t:.1f}s")

# write.features_h5(pc_feats, pc_labels, path=OUTPUT_PATH + "pca_feats.h5")


# %%
# ============================================================
# 4. PCA → WAVELET → PCA (with angles)
# ============================================================

print("\n=== PCA on Pose Features ===")
t = time.time()

pc_feats, pc_labels = features.pca(
    combined_features,
    combined_labels,
    categories=["ego_euc", "ang"],  # Added "ang" for angles
    n_pcs=N_PCS_POSE,
    method="fbpca",
)
print(f"  PCA output: {pc_feats.shape}")
print(f"  Time: {time.time() - t:.1f}s")

write.features_h5(pc_feats, pc_labels, path=OUTPUT_PATH + "pca_feats.h5")

# %%
print("\n=== Wavelet Transform ===")
t = time.time()

wlet_feats, wlet_labels = features.wavelet(
    pc_feats, pc_labels, ids,
    f_s=FRAME_RATE,
    freq=WAVELET_FREQS,
    w0=5
)
print(f"  Wavelet output: {wlet_feats.shape}")
print(f"  Time: {time.time() - t:.1f}s")

write.features_h5(wlet_feats, wlet_labels, path=OUTPUT_PATH + "wavelet_feats.h5")


# %%
# print("\n=== PCA on Wavelets ===")

# pc_wlet, pc_wlet_labels = features.pca(
#     wlet_feats,
#     wlet_labels,
#     categories=["wlet_ego_euc"],
#     n_pcs=N_PCS_WAVELET,
#     method="fbpca",
# )
# print(f"  Wavelet PCA output: {pc_wlet.shape}")

# # Combine
# pc_feats = np.hstack((pc_feats, pc_wlet))
# pc_labels = pc_labels + pc_wlet_labels
# print(f"\n  Final features: {pc_feats.shape}")

# write.features_h5(pc_feats, pc_labels, path=OUTPUT_PATH + "pca_on_wavelets.h5")

# # Cleanup
# del wlet_feats, wlet_labels, pc_wlet, pc_wlet_labels
# del combined_features, combined_labels, ego_pose, labels


# %%
print("\n=== PCA on Wavelets ===")

pc_wlet, pc_wlet_labels = features.pca(
    wlet_feats,
    wlet_labels,
    categories=["wlet_ego_euc", "wlet_ang"],
    n_pcs=N_PCS_WAVELET,
    method="fbpca",
)
print(f"  Wavelet PCA output: {pc_wlet.shape}")

# Combine pose PCs with wavelet PCs
pc_feats = np.hstack((pc_feats, pc_wlet))
pc_labels = pc_labels + pc_wlet_labels
print(f"  Final features: {pc_feats.shape}")

write.features_h5(pc_feats, pc_labels, path=OUTPUT_PATH + "pca_on_wavelets.h5")

# Cleanup
del wlet_feats, wlet_labels, pc_wlet, pc_wlet_labels
del combined_features, combined_labels, ego_pose, ego_labels, angles, angle_labels

# %%
# ============================================================
# 5. CREATE DATASTRUCT & EMBED
# ============================================================

print("\n=== Creating DataStruct ===")

data_obj = ds.DataStruct(
    pose=pose,
    id=ids,
    meta=meta_df,
    meta_by_frame=meta_by_frame,
    connectivity=connectivity,
)
data_obj.features = pc_feats

# Downsample
data_obj = data_obj[::DOWNSAMPLE, :]
print(f"  Downsampled: {len(data_obj.data):,} points")


# %%
print("\n=== Embedding ===")

embedder = Embed(
    embed_method="fitsne",
    perplexity=PERPLEXITY,
    lr="auto",
)
data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)
print(f"  Embedding shape: {data_obj.embed_vals.shape}")


# %%
print("\n=== Watershed Clustering ===")

data_obj.ws = Watershed(
    sigma=SIGMA, max_clip=1, log_out=True, pad_factor=0.05
)
data_obj.data["Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)
n_clusters = data_obj.data["Cluster"].nunique()
print(f"  Number of clusters: {n_clusters}")

# Save DataStruct
data_obj.write_pickle(OUTPUT_PATH)
print(f"\n  Saved DataStruct to {OUTPUT_PATH}")


# %%
# %matplotlib inline

# %%
# %%
# ============================================================
# 6. VISUALIZATIONS
# ============================================================

print("\n=== Visualizations ===")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================

print("\n=== Visualizations ===")

# Get animal types dynamically
types = data_obj.data['animal_type'].unique()
type1, type2 = (types[0], types[1]) if len(types) > 1 else (types[0], types[0])

# 6a. Embedding overview
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
scatter = ax.scatter(data_obj.embed_vals[:, 0], data_obj.embed_vals[:, 1], 
                     c=data_obj.data["Cluster"], s=1, alpha=0.5, cmap='tab20')
ax.set_title(f'Clusters (n={n_clusters})')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

ax = axes[1]
mask_type1 = data_obj.data['animal_type'] == type1
ax.scatter(data_obj.embed_vals[mask_type1, 0], data_obj.embed_vals[mask_type1, 1], 
           c='blue', s=1, alpha=0.3, label=type1)
ax.scatter(data_obj.embed_vals[~mask_type1, 0], data_obj.embed_vals[~mask_type1, 1], 
           c='red', s=1, alpha=0.3, label=type2)
ax.legend()
ax.set_title(f'{type1} vs {type2}')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

ax = axes[2]
print(f"  embed_vals: {data_obj.embed_vals.shape[0]}, pose: {data_obj.pose.shape[0]}")

if data_obj.pose.shape[0] == data_obj.embed_vals.shape[0]:
    nose_height = data_obj.pose[:, SNOUT_IDX, 2]  # Snout z-coordinate
else:
    print("  WARNING: pose not downsampled, using manual indexing")
    nose_height = pose[::DOWNSAMPLE, SNOUT_IDX, 2]

scatter = ax.scatter(data_obj.embed_vals[:, 0], data_obj.embed_vals[:, 1], 
                     c=nose_height, s=1, alpha=0.5, cmap='viridis')
ax.set_title('Nose Height (rearing)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + "embedding_overview.png", dpi=150)
plt.show()

# %%
vis.plot.density_cat(
    data=data_obj, column="animal_type", watershed=data_obj.ws,
    filepath=OUTPUT_PATH + "density_by_animal_type.png", show=True
)

# %%
# %%
# ============================================================
# PER-SESSION DENSITY PLOTS (using watershed like density_cat)
# ============================================================

from neuroposelib.embed import Watershed

# Constants from neuroposelib
EPS = 1e-6
DEFAULT_VIRIDIS = "viridis"

def _mask_density(density, watershed_map, eps):
    """Mask density where watershed is -1 (background)"""
    masked = density.copy()
    masked[watershed_map == -1] = 0
    return masked

print("\n=== Per-Session Density Plots (2 animals each) ===")

unique_recordings = data_obj.data['recording_idx'].unique()

for rec_idx in unique_recordings:
    rec_mask = data_obj.data['recording_idx'] == rec_idx
    sessions_in_rec = data_obj.data.loc[rec_mask, 'session_id'].unique()
    
    if len(sessions_in_rec) != 2:
        continue
    
    rec_row = meta_df.iloc[rec_idx]
    date = rec_row.get('date', 'unknown')
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for i, sess_id in enumerate(sessions_in_rec):
        sess_mask = data_obj.data['session_id'] == sess_id
        sess_embed = data_obj.embed_vals[sess_mask]
        
        animal_type = data_obj.data.loc[sess_mask, 'animal_type'].iloc[0]
        animal_name = data_obj.data.loc[sess_mask, 'animal_name'].iloc[0]
        
        ax = axes[i]
        
        # Use watershed fit_density like density_cat does
        density = data_obj.ws.fit_density(sess_embed, new=False)
        
        ax.imshow(
            _mask_density(density, data_obj.ws.watershed_map, EPS * 1.01),
            vmin=EPS,
            cmap=DEFAULT_VIRIDIS,
        )
        
        # Add watershed borders
        if data_obj.ws.borders is not None:
            ax.plot(
                data_obj.ws.borders[:, 0],
                data_obj.ws.borders[:, 1],
                ".k",
                markersize=0.1,
            )
        
        ax.set_aspect(0.9)
        ax.set_title(f"{animal_name} ({animal_type})\nn={sess_mask.sum()}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    
    fig.suptitle(f"Recording {rec_idx}: {rec_row['Animal1']} vs {rec_row['Animal2']} ({date})")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + f"density_recording_{rec_idx:03d}.png", dpi=200)
    plt.close()
    
    print(f"  Saved: density_recording_{rec_idx:03d}.png")

print(f"\nGenerated {len(unique_recordings)} density plot pairs")

# %%
# ============================================================
# 7. CLUSTER ANALYSIS
# ============================================================

print("\n=== Cluster Analysis ===")

# Get animal types dynamically
types = data_obj.data['animal_type'].unique()
type1, type2 = (types[0], types[1]) if len(types) > 1 else (types[0], types[0])

# Frequency by recording
freq, combined_keys = analysis.cluster_freq_by_cat(
    data_obj.data["Cluster"].values, cat=data_obj.id
)
freq_df = pd.DataFrame(freq.T, columns=combined_keys)
freq_df.to_csv(OUTPUT_PATH + "cluster_occupancy.csv")
print(f"Saved: cluster_occupancy.csv")

# Frequency by animal type
freq_type, type_keys = analysis.cluster_freq_by_cat(
    data_obj.data["Cluster"].values, cat=data_obj.data["animal_type"].values
)
freq_type_df = pd.DataFrame(freq_type.T, columns=type_keys)
freq_type_df[f'{type1}_enrichment'] = (freq_type_df[type1] + 0.001) / (freq_type_df[type2] + 0.001)
freq_type_df.to_csv(OUTPUT_PATH + "cluster_freq_by_animal_type.csv")

print(f"\nTop clusters enriched in {type1}:")
print(freq_type_df.sort_values(f'{type1}_enrichment', ascending=False).head(10)[[type1, type2, f'{type1}_enrichment']])

print(f"\nTop clusters enriched in {type2}:")
print(freq_type_df.sort_values(f'{type1}_enrichment', ascending=True).head(10)[[type1, type2, f'{type1}_enrichment']])

# %%
# ============================================================
# 8. CLUSTER VIDEOS
# ============================================================

print("\n=== Generating Cluster Videos ===")

pose_centered = data_obj.pose - data_obj.pose.mean(axis=-2, keepdims=True)
effective_fps = FRAME_RATE / DOWNSAMPLE
vis.pose.sample_grid3D(
    pose_centered,
    connectivity=connectivity,
    labels=data_obj.data["Cluster"],
    n_samples=9,
    centered=True,
    N_FRAMES=100,
    fps=effective_fps,
    dpi=100,
    watershed=data_obj.ws,
    embed_vals=None,
    VID_NAME="cluster",
    filepath=OUTPUT_PATH,
)


# %%
print("\n" + "="*50)
print("PIPELINE COMPLETE!")
print("="*50)
print(f"\nOutputs saved to: {OUTPUT_PATH}")
print(f"""
Files created:
  - pose_aligned.h5
  - pca_feats.h5
  - wavelet_feats.h5  
  - pca_on_wavelets.h5
  - DataStruct.pickle
  - embedding_overview.png
  - density_by_*.png
  - cluster_occupancy.csv
  - cluster_freq_by_animal_type.csv
  - cluster_*.mp4

Summary:
  - Sessions: {session_counter}
  - Frames: {pose.shape[0]:,}
  - Downsampled: {len(data_obj.data):,}
  - Clusters: {n_clusters}
""")


