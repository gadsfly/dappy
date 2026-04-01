# config.py - Individual-within-social embedding (26mar all-sessions run, 9kpt)
# Used by: run_individual_9kpt.py
# Splits social dual-animal recordings into individual animals, then embeds.
import os as _os
_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_DIR)
_REPO = _os.path.dirname(_os.path.dirname(_ROOT))
_SOCIAL_DEPS = _os.path.join(_REPO, "mir_addi", "good_social_scripts")

SKELETON_PATH = _os.path.join(_SOCIAL_DEPS, "skeletons_social_mir.py")
SKELETON_NAME = "mouse9"

SOCIAL_CSV_PATH = _os.path.join(_ROOT, "csvs", "social_all_9kpt.csv")
OUTPUT_PATH = _os.path.join(_ROOT, "output", "individual_9kpt") + "/"

# Pipeline params
FRAME_RATE = 30
DOWNSAMPLE = 1
PERPLEXITY = 50
SIGMA = 15
N_PCS_POSE = 10
N_PCS_WAVELET = 5
