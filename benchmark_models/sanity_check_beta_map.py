from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from termcolor import colored

# ---- Cluster paths ----
PROC_DATA_ROOT = Path("/home/SRastegarnia/hcptrt_decoding_Shima/data/")
RAW_DATA_ROOT = Path(
    "/data/neuromod/DATA/cneuromod/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/"
)


def postproc_beta_map_check(
    subject: str,
    task_label: str,
    region_approach: str,
    resolution: int | str,
    HRFlag_process: str,
) -> None:
    """
    Simple sanity check for generated beta maps and labels.
    Logic preserved; paths mirror the original cluster layout.
    """
    mask_name = "space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

    print(colored((subject, region_approach, resolution), "red", attrs=["bold"]))

    # Example mask file path (for information/visibility only)
    tpl_mask = (
        RAW_DATA_ROOT
        / subject
        / "ses-001"
        / "func"
        / f"{subject}_ses-001_task-{task_label}_run-1_{mask_name}"
    )
    print(str(tpl_mask))

    # Final bold/labels produced by the pipeline
    final_bold_path = (
        PROC_DATA_ROOT
        / "processed_data"
        / "proc_fMRI"
        / str(region_approach)
        / str(resolution)
        / subject
        / f"{subject}_{task_label}_{HRFlag_process}_final_fMRI.npy"
    )
    print(str(final_bold_path))

    final_bold = np.load(final_bold_path)
    print("Shape of api_file:", np.shape(final_bold))

    final_labels_path = (
        PROC_DATA_ROOT
        / "processed_data"
        / "proc_events"
        / str(region_approach)
        / str(resolution)
        / subject
        / f"{subject}_{task_label}_{HRFlag_process}_final_labels.csv"
    )
    print(str(final_labels_path))

    # Read labels, echo basic summaries (same intent as original)
    final_labels = pd.read_csv(final_labels_path, encoding="utf8", header=None)
    print(final_labels)
    print("Number of events:", len(final_labels))

    # Flatten labels (replaces the manual csv.reader loop with equivalent pandas)
    flat_list: list[str] = final_labels.iloc[:, 0].astype(str).tolist()
    print(len(flat_list))
    print(flat_list)
    print(len(flat_list))
    # --------------------------------------------------------------------------------

    # The original GLM sandbox stayed commented-out; leaving it as-is for reference:
    # from nilearn.glm.first_level import FirstLevelModel
    # glm = FirstLevelModel(mask_img=str(tpl_mask), t_r=1.49, high_pass=0.01)
    # glm.fit(run_imgs=str(final_bold_path), events=str(outPath))
    # print('glm is fitted')


def validate_environment() -> None:
    """Light checks: imports OK and whether expected data roots are present."""
    required_dirs = [
        str(RAW_DATA_ROOT),
        str(PROC_DATA_ROOT),
    ]
    missing = [p for p in required_dirs if not os.path.isdir(p)]
    if missing:
        print("Environment OK (imports). Data roots not present on this machine:")
        for m in missing:
            print(" -", m)
    else:
        print("Environment OK: imports and data roots present.")


if __name__ == "__main__":
    # This module is typically imported from other scripts.
    # Keep main minimal; no CLI side effects.
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
