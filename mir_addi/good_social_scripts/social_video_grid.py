"""
social_video_grid.py
Multi-sample 4-panel grid video (no watershed map), for social pose visualization.
Each sample shows: Combined 3D | Top-down 2D | Animal 1 zoomed | Animal 2 zoomed.
Supports up to 9 samples per cluster.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pathlib import Path
import tqdm


def social_video_grid(
    pose: np.ndarray,
    connectivity,
    labels: np.ndarray,
    n_clusters: int = None,
    n_samples: int = 9,
    N_FRAMES: int = 90,
    fps: int = 30,
    dpi: int = 100,
    n_kp: int = 22,
    filepath: str = "./",
    VID_NAME: str = "social_grid",
    min_sample_distance: int = 200,
    min_zoom_size: float = 80.0,
    individual_padding: float = 30.0,
    animal1_label: str = "Animal 1",
    animal2_label: str = "Animal 2",
    nose_idx: int = 2,
):
    """
    Social video: grid of n_samples, each sample is a 4-panel
    (combined 3D, top-down, A1 zoom, A2 zoom). No watershed map.

    Parameters
    ----------
    pose : (n_frames_total, 2*n_kp, 3)
        Combined pose for both animals.
    connectivity : object
        Must have .segments or .links attribute.
    labels : (n_frames_total,)
        Cluster labels per frame.
    n_clusters : int or None
        How many clusters to generate videos for (by count). None = all.
    n_samples : int
        Number of sample clips per cluster (up to 9).
    N_FRAMES : int
        Frames per clip.
    fps : int
        Output video fps.
    dpi : int
        Output dpi.
    n_kp : int
        Number of keypoints per animal.
    filepath : str
        Parent directory for output.
    VID_NAME : str
        Subdirectory / prefix name.
    """
    from neuroposelib.visualization.constants import PALETTE

    save_dir = Path(filepath) / VID_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    unique_labels, counts = np.unique(labels, return_counts=True)
    if n_clusters is None or n_clusters >= len(unique_labels):
        top_clusters = unique_labels[np.argsort(-counts)]
    else:
        top_clusters = unique_labels[np.argsort(-counts)[:n_clusters]]

    segments = connectivity.segments if hasattr(connectivity, "segments") else connectivity.links
    n_seg = len(segments) // 2

    # Colorful skeleton for A1, faded for A2
    colors_a1 = np.array(PALETTE[:n_seg])
    colors_a2 = []
    for c in colors_a1:
        c2 = np.array(c, dtype=float).copy()
        if len(c2) >= 4:
            c2[3] = c2[3] * 0.55
        elif len(c2) == 3:
            c2 = np.append(c2, 0.55)
        colors_a2.append(c2)
    colors_a2 = np.array(colors_a2)

    print(f"\n=== Generating Social Grid Videos ===")
    print(f"Clusters: {len(top_clusters)}, Samples/cluster: {n_samples}, "
          f"Frames: {N_FRAMES}, fps: {fps}")

    for cluster_id in tqdm.tqdm(top_clusters, desc="Clusters"):
        cluster_frames = np.where(labels == cluster_id)[0]

        # Sample spread-out starting points
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
            continue

        n_actual = len(sampled_points)

        # Collect pose data + limits per sample
        all_sample_poses = []
        sample_limits = []
        sample_a1_sizes, sample_a2_sizes = [], []
        sample_centroids = []

        for sp in sampled_points:
            sl = slice(sp - N_FRAMES // 2, sp + N_FRAMES // 2)
            sample_pose = pose[sl]
            all_sample_poses.append(sample_pose)

            mins = sample_pose.min(axis=(0, 1))
            maxs = sample_pose.max(axis=(0, 1))
            pad = (maxs - mins).max() * 0.15
            lim = np.array([
                [mins[0] - pad, maxs[0] + pad],
                [mins[1] - pad, maxs[1] + pad],
                [max(0, mins[2] - pad), maxs[2] + pad],
            ])
            sample_limits.append(lim)

            a1p = sample_pose[:, :n_kp, :]
            a2p = sample_pose[:, n_kp:, :]
            a1_ext = a1p.max(axis=1) - a1p.min(axis=1)
            a2_ext = a2p.max(axis=1) - a2p.min(axis=1)
            a1_sz = max(np.median(a1_ext.max(axis=1)) + individual_padding * 2, min_zoom_size)
            a2_sz = max(np.median(a2_ext.max(axis=1)) + individual_padding * 2, min_zoom_size)
            sample_a1_sizes.append(a1_sz)
            sample_a2_sizes.append(a2_sz)
            sample_centroids.append((a1p.mean(axis=1), a2p.mean(axis=1)))

        # Layout: grid of samples, each sample = 2x2
        grid_cols = min(3, n_actual)
        grid_rows = int(np.ceil(n_actual / grid_cols))
        fig_w = grid_cols * 7
        fig_h = grid_rows * 6

        fig = plt.figure(figsize=(fig_w, fig_h))
        outer_gs = fig.add_gridspec(grid_rows, grid_cols, hspace=0.3, wspace=0.2)

        sample_axes_list = []
        for si in range(n_actual):
            r, c = divmod(si, grid_cols)
            sg = outer_gs[r, c].subgridspec(2, 2, hspace=0.1, wspace=0.1)
            ax_comb = fig.add_subplot(sg[0, 0], projection="3d")
            ax_top = fig.add_subplot(sg[0, 1])
            ax_a1 = fig.add_subplot(sg[1, 0], projection="3d")
            ax_a2 = fig.add_subplot(sg[1, 1], projection="3d")
            sample_axes_list.append({
                "combined": ax_comb, "topdown": ax_top,
                "animal1": ax_a1, "animal2": ax_a2,
            })

        out_path = save_dir / f"{VID_NAME}_{cluster_id}.mp4"
        writer = FFMpegWriter(fps=fps, bitrate=6000)

        with writer.saving(fig, str(out_path), dpi=dpi):
            for fi in range(N_FRAMES):
                for si, (sp, axes, lim, a1sz, a2sz, (c1, c2)) in enumerate(zip(
                    all_sample_poses, sample_axes_list, sample_limits,
                    sample_a1_sizes, sample_a2_sizes, sample_centroids,
                )):
                    if fi >= len(sp):
                        continue
                    for ax in axes.values():
                        ax.cla()

                    cur = sp[fi]
                    pa1 = cur[:n_kp]
                    pa2 = cur[n_kp:]
                    dist = np.linalg.norm(c1[fi] - c2[fi])

                    # --- Combined 3D ---
                    ax = axes["combined"]
                    for segi, (i_f, i_t) in enumerate(segments[:n_seg]):
                        ax.plot([cur[i_f, 0], cur[i_t, 0]],
                                [cur[i_f, 1], cur[i_t, 1]],
                                [cur[i_f, 2], cur[i_t, 2]],
                                c=colors_a1[segi], linewidth=2, alpha=0.9)
                    for segi, (i_f, i_t) in enumerate(segments[n_seg:]):
                        ax.plot([cur[i_f, 0], cur[i_t, 0]],
                                [cur[i_f, 1], cur[i_t, 1]],
                                [cur[i_f, 2], cur[i_t, 2]],
                                c=colors_a2[segi], linewidth=2, alpha=0.5)
                    ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_zlim(lim[2])
                    ax.set_box_aspect(lim[:, 1] - lim[:, 0])
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                    ax.set_title(f"S{si+1} | {dist:.0f}mm", fontsize=8)

                    # --- Top-down 2D ---
                    ax = axes["topdown"]
                    for segi, (i_f, i_t) in enumerate(segments[:n_seg]):
                        ax.plot([cur[i_f, 0], cur[i_t, 0]],
                                [cur[i_f, 1], cur[i_t, 1]],
                                c=colors_a1[segi], linewidth=2, alpha=0.9)
                    for segi, (i_f, i_t) in enumerate(segments[n_seg:]):
                        ax.plot([cur[i_f, 0], cur[i_t, 0]],
                                [cur[i_f, 1], cur[i_t, 1]],
                                c=colors_a2[segi], linewidth=2, alpha=0.5)
                    ax.scatter(pa1[nose_idx, 0], pa1[nose_idx, 1],
                               c="black", s=40, marker="^", edgecolors="white",
                               linewidths=0.5, zorder=10)
                    ax.scatter(pa2[nose_idx, 0], pa2[nose_idx, 1],
                               c="black", s=40, marker="v", edgecolors="white",
                               linewidths=0.5, zorder=10)
                    ax.plot([c1[fi, 0], c2[fi, 0]], [c1[fi, 1], c2[fi, 1]],
                            "k--", linewidth=0.5, alpha=0.5)
                    ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
                    ax.set_aspect("equal")
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.grid(True, alpha=0.2, linewidth=0.5)

                    # --- Animal 1 zoomed ---
                    ax = axes["animal1"]
                    a1c = pa1.mean(axis=0)
                    for segi, (i_f, i_t) in enumerate(segments[:n_seg]):
                        ax.plot([cur[i_f, 0], cur[i_t, 0]],
                                [cur[i_f, 1], cur[i_t, 1]],
                                [cur[i_f, 2], cur[i_t, 2]],
                                c=colors_a1[segi], linewidth=3, alpha=0.95)
                    ax.scatter([pa1[nose_idx, 0]], [pa1[nose_idx, 1]], [pa1[nose_idx, 2]],
                               c="black", s=50, marker="^", edgecolors="white", zorder=10)
                    ax.set_xlim(a1c[0] - a1sz/2, a1c[0] + a1sz/2)
                    ax.set_ylim(a1c[1] - a1sz/2, a1c[1] + a1sz/2)
                    ax.set_zlim(max(0, a1c[2] - a1sz/2), a1c[2] + a1sz/2)
                    ax.set_box_aspect([1, 1, 1])
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                    ax.set_title(animal1_label, fontsize=7)

                    # --- Animal 2 zoomed ---
                    ax = axes["animal2"]
                    a2c = pa2.mean(axis=0)
                    for segi, (i_f, i_t) in enumerate(segments[n_seg:]):
                        ax.plot([cur[i_f, 0], cur[i_t, 0]],
                                [cur[i_f, 1], cur[i_t, 1]],
                                [cur[i_f, 2], cur[i_t, 2]],
                                c=colors_a2[segi], linewidth=3, alpha=0.5)
                    ax.scatter([pa2[nose_idx, 0]], [pa2[nose_idx, 1]], [pa2[nose_idx, 2]],
                               c="black", s=50, marker="v", edgecolors="white", zorder=10)
                    ax.set_xlim(a2c[0] - a2sz/2, a2c[0] + a2sz/2)
                    ax.set_ylim(a2c[1] - a2sz/2, a2c[1] + a2sz/2)
                    ax.set_zlim(max(0, a2c[2] - a2sz/2), a2c[2] + a2sz/2)
                    ax.set_box_aspect([1, 1, 1])
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                    ax.set_title(animal2_label, fontsize=7, color="gray")

                fig.suptitle(f"Cluster {cluster_id} | Frame {fi+1}/{N_FRAMES}", fontsize=10)
                writer.grab_frame()

        plt.close(fig)

    print(f"\nVideos saved to: {save_dir}")
    return save_dir
