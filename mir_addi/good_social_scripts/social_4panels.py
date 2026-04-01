import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pathlib import Path
import tqdm
import copy

def social_video_watershed_4panel(
    pose: np.ndarray,
    connectivity,
    labels: np.ndarray,
    watershed,
    n_clusters: int = None,
    n_samples: int = 4,
    N_FRAMES: int = 100,
    fps: int = 10,
    dpi: int = 100,
    n_kp: int = 9,
    filepath: str = "./",
    VID_NAME: str = "social_ws_4panel",
    embed_vals: np.ndarray = None,
    min_sample_distance: int = 200,
    min_zoom_size: float = 80.0,
    individual_padding: float = 30.0,
    animal1_label: str = "Animal 1 (bright)",
    animal2_label: str = "Animal 2 (faded)",
    nose_idx: int = 2,
):
    """
    Social video with:
    - Left: Watershed density map with cluster highlighted (like original)
    - Right: Grid of samples, each sample is 4-panel
    """
    from neuroposelib.visualization.constants import EPS, DEFAULT_BONE
    from neuroposelib.visualization.plot import _mask_density
    
    save_dir = Path(filepath) / VID_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if n_clusters is None or n_clusters >= len(unique_labels):
        top_clusters = unique_labels[np.argsort(-counts)]
    else:
        top_clusters = unique_labels[np.argsort(-counts)[:n_clusters]]
    
    segments = connectivity.segments if hasattr(connectivity, 'segments') else connectivity.links
    n_seg = len(segments) // 2
    
    # Get colorful palette from neuroposelib (like original sample_grid3D)
    from neuroposelib.visualization.constants import PALETTE
    
    # Use PALETTE directly - this gives the colorful skeleton like original
    colors_a1 = np.array(PALETTE[:n_seg])
    
    # Animal 2: same colorful palette but slightly faded to distinguish
    colors_a2 = []
    for c in colors_a1:
        c2 = np.array(c, dtype=float).copy()
        if len(c2) >= 4:
            c2[3] = c2[3] * 0.55  # 55% opacity
        elif len(c2) == 3:
            c2 = np.append(c2, 0.55)
        colors_a2.append(c2)
    colors_a2 = np.array(colors_a2)
    
    print(f"\n=== Generating Social Videos: Watershed + 4-Panel Samples ===")
    print(f"Clusters: {len(top_clusters)}, Samples per cluster: {n_samples}")
    
    for cluster_id in tqdm.tqdm(top_clusters, desc="Clusters"):
        cluster_frames = np.where(labels == cluster_id)[0]
        
        # Sample multiple starting points (spread out)
        sampled_points = []
        permuted = np.random.permutation(cluster_frames)
        
        for idx in permuted:
            if len(sampled_points) >= n_samples:
                break
            if any(abs(idx - s) < min_sample_distance for s in sampled_points):
                continue
            if idx < N_FRAMES // 2 or idx > len(labels) - N_FRAMES // 2:
                continue
            sampled_points.append(idx)
        
        if len(sampled_points) < 1:
            print(f"  Cluster {cluster_id}: not enough samples, skipping")
            continue
        
        n_actual_samples = len(sampled_points)
        
        # === WATERSHED DENSITY (like original) ===
        # Get density for this cluster's points
        if embed_vals is not None:
            cluster_embed = embed_vals[labels == cluster_id]
            density = watershed.fit_density(cluster_embed, new=False)
        else:
            density = watershed.watershed_map
        
        # Create highlighted watershed map (like original sample decorator)
        ws_map_highlighted = np.where(
            watershed.watershed_map == cluster_id, 1, 0.1
        )
        ws_map_highlighted = np.where(
            watershed.watershed_map == 0, 0, ws_map_highlighted
        )
        
        # Get pose data for all samples
        all_sample_poses = []
        for start_frame in sampled_points:
            frame_slice = slice(start_frame - N_FRAMES//2, start_frame + N_FRAMES//2)
            all_sample_poses.append(pose[frame_slice])
        
        # Compute limits for each sample
        sample_limits = []
        sample_a1_sizes = []
        sample_a2_sizes = []
        sample_centroids = []
        
        for sample_pose in all_sample_poses:
            mins = sample_pose.min(axis=(0, 1))
            maxs = sample_pose.max(axis=(0, 1))
            padding = (maxs - mins).max() * 0.15
            combined_lim = np.array([
                [mins[0] - padding, maxs[0] + padding],
                [mins[1] - padding, maxs[1] + padding],
                [max(0, mins[2] - padding), maxs[2] + padding],
            ])
            sample_limits.append(combined_lim)
            
            a1_pose = sample_pose[:, :n_kp, :]
            a2_pose = sample_pose[:, n_kp:, :]
            
            a1_extents = a1_pose.max(axis=1) - a1_pose.min(axis=1)
            a2_extents = a2_pose.max(axis=1) - a2_pose.min(axis=1)
            
            a1_size = max(np.median(a1_extents.max(axis=1)) + individual_padding * 2, min_zoom_size)
            a2_size = max(np.median(a2_extents.max(axis=1)) + individual_padding * 2, min_zoom_size)
            sample_a1_sizes.append(a1_size)
            sample_a2_sizes.append(a2_size)
            
            c1 = a1_pose.mean(axis=1)
            c2 = a2_pose.mean(axis=1)
            sample_centroids.append((c1, c2))
        
        # === FIGURE LAYOUT ===
        sample_rows = int(np.ceil(np.sqrt(n_actual_samples)))
        sample_cols = int(np.ceil(n_actual_samples / sample_rows))
        
        fig_width = 5 + sample_cols * 6
        fig_height = sample_rows * 5
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        outer_gs = fig.add_gridspec(1, 2, width_ratios=[1, sample_cols * 2], wspace=0.1)
        
        ax_ws = fig.add_subplot(outer_gs[0, 0])
        
        inner_gs = outer_gs[0, 1].subgridspec(sample_rows, sample_cols, hspace=0.3, wspace=0.2)
        
        sample_axes_list = []
        for s_i in range(n_actual_samples):
            s_row = s_i // sample_cols
            s_col = s_i % sample_cols
            
            sample_gs = inner_gs[s_row, s_col].subgridspec(2, 2, hspace=0.1, wspace=0.1)
            
            ax_comb = fig.add_subplot(sample_gs[0, 0], projection='3d')
            ax_top = fig.add_subplot(sample_gs[0, 1])
            ax_a1 = fig.add_subplot(sample_gs[1, 0], projection='3d')
            ax_a2 = fig.add_subplot(sample_gs[1, 1], projection='3d')
            
            sample_axes_list.append({
                'combined': ax_comb,
                'topdown': ax_top,
                'animal1': ax_a1,
                'animal2': ax_a2,
            })
        
        output_path = save_dir / f"{VID_NAME}_{cluster_id}.mp4"
        writer = FFMpegWriter(fps=fps, bitrate=6000)
        
        with writer.saving(fig, str(output_path), dpi=dpi):
            for frame_i in range(N_FRAMES):
                ax_ws.cla()
                
                # === WATERSHED MAP (like original) ===
                masked_density = _mask_density(density, ws_map_highlighted, eps=EPS * 1.01)
                ax_ws.imshow(masked_density, vmin=EPS, cmap=DEFAULT_BONE)
                
                if watershed.borders is not None:
                    ax_ws.plot(watershed.borders[:, 0], watershed.borders[:, 1],
                              '.k', markersize=0.1)
                
                ax_ws.set_aspect(0.9)
                ax_ws.axis('off')
                ax_ws.set_title(f'Cluster {cluster_id}\n({len(cluster_frames)} frames)', 
                               fontsize=12, fontweight='bold')
                
                # === EACH SAMPLE: 4 PANELS ===
                for s_i, (sample_pose, axes, limits, a1_size, a2_size, (cent1, cent2)) in enumerate(zip(
                    all_sample_poses, sample_axes_list, sample_limits, 
                    sample_a1_sizes, sample_a2_sizes, sample_centroids
                )):
                    if frame_i >= len(sample_pose):
                        continue
                    
                    for ax in axes.values():
                        ax.cla()
                    
                    current_pose = sample_pose[frame_i]
                    pose_a1 = current_pose[:n_kp]
                    pose_a2 = current_pose[n_kp:]
                    dist = np.linalg.norm(cent1[frame_i] - cent2[frame_i])
                    
                    # --- Combined 3D ---
                    ax = axes['combined']
                    # Animal 1 skeleton (colorful)
                    for seg_i, (i_from, i_to) in enumerate(segments[:n_seg]):
                        ax.plot([current_pose[i_from, 0], current_pose[i_to, 0]],
                               [current_pose[i_from, 1], current_pose[i_to, 1]],
                               [current_pose[i_from, 2], current_pose[i_to, 2]],
                               c=colors_a1[seg_i], linewidth=2, alpha=0.9)
                    # Animal 2 skeleton (same colors)
                    for seg_i, (i_from, i_to) in enumerate(segments[n_seg:]):
                        ax.plot([current_pose[i_from, 0], current_pose[i_to, 0]],
                               [current_pose[i_from, 1], current_pose[i_to, 1]],
                               [current_pose[i_from, 2], current_pose[i_to, 2]],
                               c=colors_a2[seg_i], linewidth=2, alpha=0.5)
                    
                    ax.set_xlim(limits[0]); ax.set_ylim(limits[1]); ax.set_zlim(limits[2])
                    ax.set_box_aspect(limits[:, 1] - limits[:, 0])
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                    ax.set_title(f'Sample {s_i+1} | {dist:.0f}mm', fontsize=8)
                    
                    # --- Top-down 2D ---
                    ax = axes['topdown']
                    for seg_i, (i_from, i_to) in enumerate(segments[:n_seg]):
                        ax.plot([current_pose[i_from, 0], current_pose[i_to, 0]],
                               [current_pose[i_from, 1], current_pose[i_to, 1]],
                               c=colors_a1[seg_i], linewidth=2, alpha=0.9)
                    for seg_i, (i_from, i_to) in enumerate(segments[n_seg:]):
                        ax.plot([current_pose[i_from, 0], current_pose[i_to, 0]],
                               [current_pose[i_from, 1], current_pose[i_to, 1]],
                               c=colors_a2[seg_i], linewidth=2, alpha=0.5)
                    # Nose markers (keep these for orientation)
                    ax.scatter(pose_a1[nose_idx, 0], pose_a1[nose_idx, 1], c='black', s=40, marker='^', 
                              edgecolors='white', linewidths=0.5, zorder=10)
                    ax.scatter(pose_a2[nose_idx, 0], pose_a2[nose_idx, 1], c='black', s=40, marker='v',
                              edgecolors='white', linewidths=0.5, zorder=10)
                    # Distance line
                    ax.plot([cent1[frame_i, 0], cent2[frame_i, 0]],
                           [cent1[frame_i, 1], cent2[frame_i, 1]], 'k--', linewidth=0.5, alpha=0.5)
                    
                    ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
                    ax.set_aspect('equal')
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.grid(True, alpha=0.2, linewidth=0.5)
                    
                    # --- Animal 1 Zoomed ---
                    ax = axes['animal1']
                    a1_center = pose_a1.mean(axis=0)
                    for seg_i, (i_from, i_to) in enumerate(segments[:n_seg]):
                        ax.plot([current_pose[i_from, 0], current_pose[i_to, 0]],
                               [current_pose[i_from, 1], current_pose[i_to, 1]],
                               [current_pose[i_from, 2], current_pose[i_to, 2]],
                               c=colors_a1[seg_i], linewidth=3, alpha=0.95)
                    # Nose marker
                    ax.scatter([pose_a1[nose_idx, 0]], [pose_a1[nose_idx, 1]], [pose_a1[nose_idx, 2]],
                              c='black', s=50, marker='^', edgecolors='white', zorder=10)
                    
                    ax.set_xlim(a1_center[0] - a1_size/2, a1_center[0] + a1_size/2)
                    ax.set_ylim(a1_center[1] - a1_size/2, a1_center[1] + a1_size/2)
                    ax.set_zlim(max(0, a1_center[2] - a1_size/2), a1_center[2] + a1_size/2)
                    ax.set_box_aspect([1, 1, 1])
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                    ax.set_title(animal1_label, fontsize=7)
                    
                    # --- Animal 2 Zoomed ---
                    ax = axes['animal2']
                    a2_center = pose_a2.mean(axis=0)
                    for seg_i, (i_from, i_to) in enumerate(segments[n_seg:]):
                        ax.plot([current_pose[i_from, 0], current_pose[i_to, 0]],
                               [current_pose[i_from, 1], current_pose[i_to, 1]],
                               [current_pose[i_from, 2], current_pose[i_to, 2]],
                               c=colors_a2[seg_i], linewidth=3, alpha=0.5)
                    # Nose marker (different shape to distinguish)
                    ax.scatter([pose_a2[nose_idx, 0]], [pose_a2[nose_idx, 1]], [pose_a2[nose_idx, 2]],
                              c='black', s=50, marker='v', edgecolors='white', zorder=10)
                    
                    ax.set_xlim(a2_center[0] - a2_size/2, a2_center[0] + a2_size/2)
                    ax.set_ylim(a2_center[1] - a2_size/2, a2_center[1] + a2_size/2)
                    ax.set_zlim(max(0, a2_center[2] - a2_size/2), a2_center[2] + a2_size/2)
                    ax.set_box_aspect([1, 1, 1])
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                    ax.set_title(animal2_label, fontsize=7, color='gray')
                
                fig.suptitle(f'Frame {frame_i + 1}/{N_FRAMES}', fontsize=10)
                writer.grab_frame()
        
        plt.close(fig)
    
    print(f"\n✓ Videos saved to: {save_dir}")
    return save_dir

