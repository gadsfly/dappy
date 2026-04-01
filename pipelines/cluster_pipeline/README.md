# Sorted Cluster Pipeline (MIR 260330)

Behavioral clustering pipeline for Oct3V1 mouse recordings.
Supports both **22-keypoint** (sDANNCE/DANNCE) and **9-keypoint** (pose3d_mouse9) predictions,
across social, individual-within-social, and single-animal sessions.

## Folder Structure

```
neuroposelib/
├── src/neuroposelib/skeletons.py     # skeleton defs (used by single pipeline)
├── mir_addi/good_social_scripts/     # social dependency modules
│   ├── skeletons_social_mir.py       # skeleton defs + SOCIAL_INDICES
│   ├── social_features.py
│   ├── socialmapper_utils.py
│   ├── social_4panels.py
│   └── social_video_grid.py
└── pipelines/cluster_pipeline/       # <-- this folder
    ├── README.md
    ├── generate_csvs_v2.py           # CSV generator (scans uploaded dataset)
    ├── configs/                      # Pipeline configs (repo-relative paths)
    │   ├── social_22kpt.py
    │   ├── social_9kpt.py
    │   ├── individual_22kpt.py
    │   ├── individual_9kpt.py
    │   ├── single_22kpt.yaml
    │   └── single_9kpt.yaml
    ├── csvs/                         # Session metadata CSVs
    │   ├── social_all_22kpt.csv      # 24 sessions (11 mini + 13 beh_only)
    │   ├── social_all_9kpt.csv       # 33 sessions (11 mini + 22 beh_only)
    │   ├── single_all_22kpt.csv      # 38 sessions (all mini)
    │   └── single_all_9kpt.csv       # 79 sessions (38 mini + 41 beh_only)
    ├── run_scripts/                  # Pipeline run scripts
    │   ├── run_social_22kpt.py
    │   ├── run_social_9kpt.py
    │   ├── run_individual_22kpt.py
    │   ├── run_individual_9kpt.py
    │   ├── run_single_22kpt.py
    │   └── run_single_9kpt.py
    └── output/                       # Created at runtime
```

## Environment

Conda environment: **neuroposelib**

Key packages:
- `neuroposelib` (pip editable install from local repo)
- numpy 1.23.5, pandas 2.2.3, scipy 1.13.1
- matplotlib 3.9.1, scikit-learn 1.6.0
- fbpca 1.0, h5py 3.12.1, tqdm 4.67.1
- faiss-cpu 1.6.5

## How to Run

```bash
conda activate neuroposelib
cd run_scripts/

# 22-keypoint pipelines
python run_social_22kpt.py
python run_individual_22kpt.py
python run_single_22kpt.py

# 9-keypoint pipelines
python run_social_9kpt.py
python run_individual_9kpt.py
python run_single_9kpt.py
```

## Skeleton Definitions

### mouse22_notail (sDANNCE, 22 keypoints)
```
0:EarL  1:EarR  2:Snout  3:SpineF  4:SpineM  5:Tail(base)
6:Tail(mid)  7:Tail(end)  8:ForepawL  9:WristL  10:ElbowL  11:ShoulderL
12:ForepawR  13:WristR  14:ElbowR  15:ShoulderR  16:HindpawL  17:AnkleL
18:KneeL  19:HindpawR  20:AnkleR  21:KneeR
```
- Center: SpineM (4), Heading: SpineF (3) → Snout (2)

### mouse9 (DANNCE collaborator, 9 keypoints)
```
0:Snout  1:EarR  2:EarL  3:SpineF  4:TailBase
5:WristL  6:WristR  7:AnkleL  8:AnkleR
```
- Center: SpineF (3), Heading: SpineF (3) → Snout (0)
- Note: No SpineM in 9kpt. `rotate_spine` uses `[center=3, nose=0]`.

## Pipeline Variants

| Script | Input | Animals | Centering | Embedding |
|--------|-------|---------|-----------|-----------|
| social | social CSVs | 2 (pair features) | SpineM/SpineF | SocialMapper-style |
| individual | social CSVs | 1 (split from pair) | SpineM/SpineF | neuroposelib standard |
| single | single CSVs | 1 | SpineM/SpineF | neuroposelib standard |

## Data Sources

- **22kpt predictions**: Paths from `/data/big_rim/uploade_ssh_mir_dataset/logs/sum_exports.csv`
  - Social: `{Source_Path}/SDANNCE/predict01/save_data_AVG.mat`
  - Single: `{Source_Path}/DANNCE/predict00/save_data_AVG.mat`
- **9kpt predictions**: Scanned from uploaded dataset directories
  - `{UPLOAD_ROOT}/{oct3v1_beh_only|oct3v1_mini}/{social|single}/{session}/annotations/pose3d_mouse9.mat`
- Miniscope animal is **Animal2** (second animal in social recordings)

## 9kpt-Specific Fixes (vs. 22kpt originals)

These run scripts were copied from the working 22kpt versions. The 9kpt variants include:
1. **NaN interpolation** — 9kpt predictions have NaN frames; interpolated before `rotate_spine`
2. **NOSE_IDX** — Changed from `spine_front`(3) to `nose`(0) since center=spine_front=3 in mouse9
3. **Joint array** — `np.delete(np.arange(9), 3)` instead of `np.delete(np.arange(18), 4)`
4. **NaN-safe histogram** — Filter NaN values before `ax.hist()` in social feature plots

## Regenerating CSVs

If the uploaded dataset changes:
```bash
conda activate neuroposelib
python generate_csvs_v2.py
```
This reads `sum_exports.csv` and scans `uploade_ssh_mir_dataset/` to regenerate all 4 CSVs.

## Dependencies (within this repo)

Social/individual scripts use modules from `mir_addi/good_social_scripts/`:
- `skeletons_social_mir.py` — skeleton definitions + SOCIAL_INDICES
- `social_features.py` — SocialMapper-style feature computation
- `socialmapper_utils.py` — velocity, angle_diff, smooth_distance_matrix
- `social_4panels.py` — 4-panel watershed video visualization
- `social_video_grid.py` — grid video visualization

Single scripts use `src/neuroposelib/skeletons.py` (neuroposelib's built-in skeleton definitions).
