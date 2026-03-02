import re
import pandas as pd
import importlib.util
from pathlib import Path
import numpy as np

# -----------------------
# Paths (edit if needed)
# -----------------------
SEGMENTS_PY = Path("subject_segments_all_cleaned.py")          # your dict file
KW_XLSX     = Path("KW Processing(AutoRecovered).xlsx")        # your workbook
KW_SHEET    = "KW Analysis"
OUT_CSV     = Path("eeg_epochs_dataset.csv")

EPOCH_LEN_SEC = 30.0  # <-- change if your lab used a different epoch length

# -----------------------
# Robust subject ID parser
# -----------------------
def parse_subject_id(sid_raw):
    if pd.isna(sid_raw):
        return None

    # If Excel gave you a number (e.g., 66 or 66.0)
    if isinstance(sid_raw, (int, np.integer)):
        return int(sid_raw)
    if isinstance(sid_raw, (float, np.floating)):
        # e.g. 66.0 -> 66
        if float(sid_raw).is_integer():
            return int(sid_raw)
        return int(round(sid_raw))

    # Otherwise treat as string
    s = str(sid_raw).strip().upper()
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None

# -----------------------
# Load subject_segments from .py file
# -----------------------
def load_subject_segments(py_path: Path):
    spec = importlib.util.spec_from_file_location("subject_segments_all_cleaned", str(py_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.subject_segments

# -----------------------
# Parse a KW stage string like "N1-W-W-W" -> ["N1","W","W","W"]
# -----------------------
ALLOWED = {"W", "N1", "N2", "N3", "R"}  # extend if needed (e.g. "UNK", "MT", etc.)

def parse_stage_seq(x):
    if pd.isna(x) or not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    if "SUBJECT DID NOT COMPLETE" in s.upper():
        return None

    # split on one-or-more hyphens (handles "--" too)
    parts = [p.strip().upper() for p in re.split(r"-+", s) if p.strip()]

    # normalize common variants
    norm = []
    for p in parts:
        p = p.replace("REM", "R")
        p = p.replace("STAGE ", "")  # just in case
        if p in ALLOWED:
            norm.append(p)

    return norm if norm else None

# -----------------------
# Build dataset
# -----------------------
def build_epoch_dataset(subject_segments: dict, kw_df: pd.DataFrame) -> pd.DataFrame:
    clean_cols = ["PreNa Clean 1", "Clean 2", "Clean 3", "Clean 4"]
    rows = []

    for _, r in kw_df.iterrows():
        sid_raw = r.get("Subject ID")
        if pd.isna(sid_raw):
            continue
        #KEEP_SUBJECTS = {42, 46, 91}
        subject_id = parse_subject_id(sid_raw)
        if subject_id is None: #or subject_id not in KEEP_SUBJECTS:
            continue


        # ONLY subject 91
        #if subject_id != 91:
        #    continue

        if subject_id not in subject_segments:
            print(subject_id)
            continue
            #raise KeyError(f"Subject {subject_id} not found in subject_segments")

        clean_segments = subject_segments[subject_id].get("clean", [])

        for seg_idx_1based, col in enumerate(clean_cols, start=1):
            if seg_idx_1based > len(clean_segments):
                continue

            labels = parse_stage_seq(r.get(col))
            if not labels:
                continue

            seg_start, seg_end = clean_segments[seg_idx_1based - 1]

            for i, lab in enumerate(labels):
                ep_start = seg_start + i * EPOCH_LEN_SEC
                if ep_start >= seg_end:
                    break
                ep_end = min(seg_start + (i + 1) * EPOCH_LEN_SEC, seg_end)

                rows.append({
                    "subject_id": subject_id,
                    "segment_id": seg_idx_1based,
                    "epoch_start_time": float(ep_start),
                    "epoch_end_time": float(ep_end),
                    "stage_label": lab
                })

        #break  # stop after subject 91

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows produced. Check that Subject ID 91 exists in the KW sheet.")
    return df.sort_values(["subject_id", "segment_id", "epoch_start_time"]).reset_index(drop=True)


def main():
    subject_segments = load_subject_segments(SEGMENTS_PY)

    kw_df = pd.read_excel(KW_XLSX, sheet_name=KW_SHEET)

    df = build_epoch_dataset(subject_segments, kw_df)

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df):,} rows -> {OUT_CSV.resolve()}")

    # Optional: parquet
    # df.to_parquet(OUT_CSV.with_suffix(".parquet"), index=False)

if __name__ == "__main__":
    main()
