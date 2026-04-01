"""
Generate CSVs for ALL Oct3V1 social and single sessions — v2.

Changes from v1:
  - Adds 9kpt prediction CSVs (scanned from uploaded dataset annotations/)
  - Adds miniscope_animal column for social CSVs (= "Animal2" per convention)
  - 9kpt predictions found at: {UPLOAD_ROOT}/{oct3v1_beh_only|oct3v1_mini}/{social|single}/{session}/annotations/pose3d_mouse9.mat

Reads sum_exports.csv from /data/big_rim/uploade_ssh_mir_dataset/logs/
for 22kpt (same as v1). For 9kpt, scans uploaded dataset directories directly.

Outputs:
  - csvs/social_all_22kpt.csv
  - csvs/single_all_22kpt.csv
  - csvs/social_all_9kpt.csv
  - csvs/single_all_9kpt.csv
"""

import os
import re
import glob
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
SUM_EXPORTS_CSV = "/data/big_rim/uploade_ssh_mir_dataset/logs/sum_exports.csv"
UPLOAD_ROOT = "/data/big_rim/uploade_ssh_mir_dataset"

# Existing CSVs for cross-validation
OLD_SOCIAL_CSV = "/home/lq53/mir_repos/dappy_24_nov/2601_dataset/social_predictions_revised.csv"
OLD_SINGLE_CSV = "/home/lq53/mir_repos/dappy_24_nov/2601_dataset/single_predictions.csv"

OUT_DIR = Path(__file__).resolve().parent / "csvs"
OUT_SOCIAL_22 = OUT_DIR / "social_all_22kpt.csv"
OUT_SINGLE_22 = OUT_DIR / "single_all_22kpt.csv"
OUT_SOCIAL_9 = OUT_DIR / "social_all_9kpt.csv"
OUT_SINGLE_9 = OUT_DIR / "single_all_9kpt.csv"


def parse_date_from_dest(dest_rel: str) -> str:
    """Extract date from Dest_Path like 'social/20241031_VC4+UNK1_S1/' -> '2024_10_31'."""
    m = re.search(r'(\d{4})(\d{2})(\d{2})_', dest_rel)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return "unknown"


def parse_date_from_session(session_name: str) -> str:
    """Extract date from session folder like '20241031_VC4+UNK1_S1' -> '2024_10_31'."""
    m = re.match(r'(\d{4})(\d{2})(\d{2})_', session_name)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return "unknown"


def parse_time_from_source(source_path: str) -> str:
    """Extract time from source folder name like '..._14_53' -> '14:53'."""
    folder = source_path.rstrip("/").split("/")[-1]
    m = re.search(r'_(\d{2})_(\d{2})$', folder)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return "unknown"


def parse_session_name(dest_rel: str) -> str:
    """Extract session name from Dest_Path like 'social/20241031_VC4+UNK1_S1/' -> '20241031_VC4+UNK1_S1'."""
    return dest_rel.rstrip("/").split("/")[-1]


def parse_id_from_session(session_name: str) -> str:
    """Extract animal ID from session folder: '20241031_VC4+UNK1_S1' -> 'VC4+UNK1'."""
    m = re.match(r'\d{8}_(.+)_S\d+$', session_name)
    if m:
        return m.group(1)
    return session_name


def is_in_mini(session_type: str, session_name: str) -> bool:
    """Check if session directory exists under oct3v1_mini/."""
    mini_path = os.path.join(UPLOAD_ROOT, "oct3v1_mini", session_type, session_name)
    return os.path.isdir(mini_path)


def has_miniscope(dest_rel: str) -> bool:
    """Check if session directory exists under oct3v1_mini/."""
    mini_path = os.path.join(UPLOAD_ROOT, "oct3v1_mini", dest_rel.rstrip("/"))
    return os.path.isdir(mini_path)


# ──────────────────────────────────────────────────────────────
#  22kpt generation (from sum_exports.csv, same as v1)
# ──────────────────────────────────────────────────────────────

def generate_22kpt_social(social_df):
    """Generate social 22kpt CSV from sum_exports social rows."""
    social_rows = []
    social_skipped = []
    for _, row in social_df.iterrows():
        src = row["Source_Path"].rstrip("/")
        dest = row["Dest_Path"]
        pred_path = f"{src}/SDANNCE/predict01/save_data_AVG.mat"

        if not os.path.exists(pred_path):
            social_skipped.append((row["ID"], dest, pred_path))
            continue

        animals = row["ID"].split("+")
        animal1 = animals[0] if len(animals) > 0 else "UNKNOWN"
        animal2 = animals[1] if len(animals) > 1 else "UNKNOWN"
        session_has_mini = has_miniscope(dest)

        social_rows.append({
            "Animal1": animal1,
            "Animal2": animal2,
            "Sex": "male",
            "Condition": "unknown",
            "date": parse_date_from_dest(dest),
            "time": parse_time_from_source(src),
            "Prediction_path": pred_path,
            "session_name": parse_session_name(dest),
            "has_miniscope": session_has_mini,
            "miniscope_animal": "Animal2" if session_has_mini else "",
        })

    social_out = pd.DataFrame(social_rows)
    social_out.to_csv(OUT_SOCIAL_22, index=False)
    print(f"\n=== SOCIAL 22kpt CSV ===")
    print(f"  Valid (with SDANNCE predictions): {len(social_rows)}")
    print(f"  Skipped (no SDANNCE predictions): {len(social_skipped)}")
    if social_skipped:
        for aid, dest, pp in social_skipped:
            print(f"    SKIP {aid} ({parse_session_name(dest)}): {pp}")
    print(f"  Saved to: {OUT_SOCIAL_22}")
    return social_out


def generate_22kpt_single(single_df):
    """Generate single 22kpt CSV from sum_exports single rows."""
    single_rows = []
    single_skipped = []
    for _, row in single_df.iterrows():
        src = row["Source_Path"].rstrip("/")
        dest = row["Dest_Path"]
        pred_path = f"{src}/DANNCE/predict00/save_data_AVG.mat"

        if not os.path.exists(pred_path):
            single_skipped.append((row["ID"], dest, pred_path))
            continue

        single_rows.append({
            "AnimalID": row["ID"],
            "Sex": "male",
            "Condition": "unknown",
            "date": parse_date_from_dest(dest),
            "time": parse_time_from_source(src),
            "Prediction_path": pred_path,
            "session_name": parse_session_name(dest),
            "has_miniscope": has_miniscope(dest),
            "miniscope": 0,
            "after_oxytocin": 0,
            "before_oxytocin": 0,
            "social": 0,
            "habituation": 0,
            "saline": 0,
            "test": 0,
            "caffeine": 0,
            "cricket": 0,
            "baseline": 0,
            "recording_time": 0,
        })

    single_out = pd.DataFrame(single_rows)
    single_out.to_csv(OUT_SINGLE_22, index=False)
    print(f"\n=== SINGLE 22kpt CSV ===")
    print(f"  Valid (with DANNCE predictions): {len(single_rows)}")
    print(f"  Skipped (no DANNCE predictions): {len(single_skipped)}")
    if single_skipped:
        for aid, dest, pp in single_skipped:
            print(f"    SKIP {aid} ({parse_session_name(dest)}): {pp}")
    print(f"  Saved to: {OUT_SINGLE_22}")
    return single_out


# ──────────────────────────────────────────────────────────────
#  9kpt generation (scan uploaded dataset for pose3d_mouse9.mat)
# ──────────────────────────────────────────────────────────────

def scan_9kpt_predictions(session_type: str):
    """Find all pose3d_mouse9.mat files for a session type (social or single).
    Returns list of (session_name, pred_path, is_mini)."""
    results = []
    for subset in ["oct3v1_beh_only", "oct3v1_mini"]:
        pattern = os.path.join(UPLOAD_ROOT, subset, session_type, "*", "annotations", "pose3d_mouse9.mat")
        for pred_path in sorted(glob.glob(pattern)):
            # path: .../oct3v1_xxx/social/20241031_VC4+UNK1_S1/annotations/pose3d_mouse9.mat
            session_name = pred_path.split(f"/{session_type}/")[1].split("/annotations/")[0]
            is_mini = (subset == "oct3v1_mini")
            results.append((session_name, pred_path, is_mini))
    return results


def generate_9kpt_social():
    """Generate social 9kpt CSV by scanning uploaded dataset."""
    hits = scan_9kpt_predictions("social")
    rows = []
    for session_name, pred_path, is_mini in hits:
        animal_id = parse_id_from_session(session_name)
        animals = animal_id.split("+")
        animal1 = animals[0] if len(animals) > 0 else "UNKNOWN"
        animal2 = animals[1] if len(animals) > 1 else "UNKNOWN"

        rows.append({
            "Animal1": animal1,
            "Animal2": animal2,
            "Sex": "male",
            "Condition": "unknown",
            "date": parse_date_from_session(session_name),
            "time": "unknown",
            "Prediction_path": pred_path,
            "session_name": session_name,
            "has_miniscope": is_mini,
            "miniscope_animal": "Animal2" if is_mini else "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_SOCIAL_9, index=False)
    n_mini = sum(1 for r in rows if r["has_miniscope"])
    print(f"\n=== SOCIAL 9kpt CSV ===")
    print(f"  Sessions found: {len(rows)} ({n_mini} with miniscope, {len(rows)-n_mini} beh_only)")
    print(f"  Saved to: {OUT_SOCIAL_9}")
    return out


def generate_9kpt_single():
    """Generate single 9kpt CSV by scanning uploaded dataset."""
    hits = scan_9kpt_predictions("single")
    rows = []
    for session_name, pred_path, is_mini in hits:
        animal_id = parse_id_from_session(session_name)

        rows.append({
            "AnimalID": animal_id,
            "Sex": "male",
            "Condition": "unknown",
            "date": parse_date_from_session(session_name),
            "time": "unknown",
            "Prediction_path": pred_path,
            "session_name": session_name,
            "has_miniscope": is_mini,
            "miniscope": 0,
            "after_oxytocin": 0,
            "before_oxytocin": 0,
            "social": 0,
            "habituation": 0,
            "saline": 0,
            "test": 0,
            "caffeine": 0,
            "cricket": 0,
            "baseline": 0,
            "recording_time": 0,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_SINGLE_9, index=False)
    n_mini = sum(1 for r in rows if r["has_miniscope"])
    print(f"\n=== SINGLE 9kpt CSV ===")
    print(f"  Sessions found: {len(rows)} ({n_mini} with miniscope, {len(rows)-n_mini} beh_only)")
    print(f"  Saved to: {OUT_SINGLE_9}")
    return out


# ──────────────────────────────────────────────────────────────
#  Cross-checks
# ──────────────────────────────────────────────────────────────

def cross_check(social_22_out, single_22_out):
    print(f"\n=== CROSS-CHECK (22kpt vs old CSVs) ===")

    old_social = pd.read_csv(OLD_SOCIAL_CSV)
    old_social_paths = set(old_social["Prediction_path"].str.strip())
    new_social_paths = set(social_22_out["Prediction_path"].str.strip()) if len(social_22_out) > 0 else set()
    missing_from_new_social = old_social_paths - new_social_paths
    print(f"Old social CSV: {len(old_social)} rows")
    print(f"New social CSV: {len(social_22_out)} rows")
    if missing_from_new_social:
        print(f"  WARNING: {len(missing_from_new_social)} old social paths NOT in new CSV:")
        for p in sorted(missing_from_new_social):
            print(f"    {p}")
    else:
        print(f"  OK — all {len(old_social)} old social paths are covered in new CSV")

    old_single = pd.read_csv(OLD_SINGLE_CSV)
    old_single_paths = set(old_single["Prediction_path"].str.strip())
    new_single_paths = set(single_22_out["Prediction_path"].str.strip()) if len(single_22_out) > 0 else set()
    missing_from_new_single = old_single_paths - new_single_paths
    print(f"\nOld single CSV: {len(old_single)} rows")
    print(f"New single CSV: {len(single_22_out)} rows")
    if missing_from_new_single:
        print(f"  WARNING: {len(missing_from_new_single)} old single paths NOT in new CSV:")
        for p in sorted(missing_from_new_single):
            print(f"    {p}")
    else:
        print(f"  OK — all {len(old_single)} old single paths are covered in new CSV")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load sum_exports.csv ──────────────────────────────────
    df = pd.read_csv(SUM_EXPORTS_CSV)
    print(f"sum_exports.csv rows: {len(df)}")

    social_df = df[df["Session_Type"] == "social"].copy()
    single_df = df[df["Session_Type"] == "single"].copy()
    print(f"  Social sessions: {len(social_df)}")
    print(f"  Single sessions: {len(single_df)}")

    # ── 22kpt CSVs (from sum_exports + rsync source paths) ───
    social_22 = generate_22kpt_social(social_df)
    single_22 = generate_22kpt_single(single_df)

    # ── 9kpt CSVs (scan uploaded dataset annotations) ─────────
    social_9 = generate_9kpt_social()
    single_9 = generate_9kpt_single()

    # ── Cross-check 22kpt against old CSVs ────────────────────
    cross_check(social_22, single_22)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n=== SUMMARY ===")
    for label, df_out in [("Social 22kpt", social_22), ("Single 22kpt", single_22),
                           ("Social 9kpt", social_9), ("Single 9kpt", single_9)]:
        n = len(df_out)
        n_mini = df_out["has_miniscope"].sum() if n > 0 else 0
        print(f"  {label}: {n} total ({n_mini} with miniscope, {n - n_mini} beh_only)")


if __name__ == "__main__":
    main()
