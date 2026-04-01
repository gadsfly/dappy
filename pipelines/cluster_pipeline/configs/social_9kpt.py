# config.py - Social embedding pipeline config (26mar all-sessions run, 9kpt)
# Used by: run_social_9kpt.py
import os as _os
_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_DIR)
_REPO = _os.path.dirname(_os.path.dirname(_ROOT))
_SOCIAL_DEPS = _os.path.join(_REPO, "mir_addi", "good_social_scripts")

SKELETON_PATH = _os.path.join(_SOCIAL_DEPS, "skeletons_social_mir.py")
SKELETON_NAME = "mouse9"

SOCIAL_CSV_PATH = _os.path.join(_ROOT, "csvs", "social_all_9kpt.csv")
SOCIAL_FEATURES_PATH = _os.path.join(_SOCIAL_DEPS, "social_features.py")
SOCIAL_VIS_PATH = _os.path.join(_SOCIAL_DEPS, "social_4panels.py")
OUTPUT_PATH = _os.path.join(_ROOT, "output", "social_9kpt") + "/"

# Pipeline params
FRAME_RATE = 30
DOWNSAMPLE = 5
PERPLEXITY = 50
SIGMA = 25           # Watershed sigma for global embedding
CLOSE_SIGMA = 15     # Watershed sigma for close-cluster re-embedding
N_PCS_JOINT = 6      # PCs for joint distance PCA
N_PCS_WAVELET = 5    # (unused currently, kept for reference)

# Close-cluster threshold (mm centroid distance)
CLOSE_DIST_THRESHOLD = 80

# Feature weights (SocialMapper style)
WEIGHT_Z = 0.25
WEIGHT_VEL = 2.0
WEIGHT_ANG = 10.0
WEIGHT_PJ = 0.05
WEIGHT_PJN = 2.0

# Session limit (set high to load all)
N_SESSIONS = 999

# Video params - general clusters
VID_FPS = 30
VID_N_FRAMES = 90
VID_N_SAMPLES = 9
VID_DPI = 100

# Video params - close clusters
CLOSE_VID_FPS = 30
CLOSE_VID_N_FRAMES = 90
CLOSE_VID_N_SAMPLES = 9
