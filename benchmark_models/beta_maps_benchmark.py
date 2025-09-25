# /home/shima/projects/individual-fmri-decoding/benchmark_models/beta_maps_benchmark.py

import csv
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, plotting
from nilearn.decoding import Decoder
from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC

# Repo utils
sys.path.append(os.path.join("../"))
import utils  # noqa: E402

#from load_confounds import Params9  # noqa: E402

# ----------------------------- config ---------------------------------
SEED = 0
TR_SECONDS = 1.49

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO),
    format="[%(levelname)s] %(message)s",
    )
log = logging.getLogger(__name__)


# ------------------------ event relabeling -----------------------------
def _new_conditions(event_path: str, task_label: str) -> pd.DataFrame:
    """Relabel HCPtrt events to clearer condition names for decoding."""
    df = pd.read_table(event_path)

    if task_label == "emotion":
        df.trial_type = df["trial_type"].replace(
            ["response_face", "response_shape"], ["fear", "shape"]
        )

    elif task_label == "wm":
        df.trial_type = df.stim_type.astype(str) + "_" + df.trial_type.astype(str)
        df.trial_type = df["trial_type"].replace(
            [
                "Body_0-Back", "Body_2-Back",
                "Face_0-Back", "Face_2-Back",
                "Place_0-Back", "Place_2-Back",
                "Tools_0-Back", "Tools_2-Back",
                "nan_Cue", "nan_countdown",
            ],
            [
                "body0b", "body2b",
                "face0b", "face2b",
                "place0b", "place2b",
                "tool0b", "tool2b",
                "Cue", "countdown",
            ],
        )

    elif task_label == "language":
        df.trial_type = df["trial_type"].replace(
            [
                "presentation_story", "question_story", "response_story",
                "presentation_math", "question_math", "response_math",
            ],
            ["story", "story", "story", "math", "math", "math"],
        )

    elif task_label == "motor":
        df.trial_type = df["trial_type"].replace(
            [
                "response_left_foot", "response_left_hand",
                "response_right_foot", "response_right_hand",
                "response_tongue",
            ],
            ["footL", "handL", "footR", "handR", "tongue"],
        )

    elif task_label == "relational":
        df.trial_type = df["trial_type"].replace(["Control", "Relational"], ["match", "relational"])

    return df


# ----------------------- beta map generation ---------------------------
def _generate_beta_maps(
    scans,
    confounds,
    events,
    conditions,
    mask,
    fname,
    task_label,
    out_path,
) -> None:
    """Fit first-level GLM per run and save concatenated z-maps, labels, runs."""
    if len(scans) != len(events):
        raise ValueError("Number of event files and BOLD files does not match.")

    glm = FirstLevelModel(
        mask_img=mask[0],
        t_r=TR_SECONDS,
        high_pass=0.01,
        smoothing_fwhm=5,
        standardize=True,
    )

    z_maps, condition_idx, session_idx = [], [], []

    for i, (scan, event, confound) in enumerate(zip(scans, events, confounds, strict=False), start=1):
        log.info("GLM %02d/%02d: %s", i, len(scans), scan)

        ses = scan.split("_task")[0].split("fmriprep-20.2lts/")[1].partition("_")[2]
        run = utils.between(scan.split("run")[1], "-", "_")
        session = f"{ses}_run-{run}"

        glm.fit(run_imgs=scan, events=event, confounds=confound)

        for condition in conditions:
            z_maps.append(glm.compute_contrast(condition))
            condition_idx.append(condition)
            session_idx.append(session)

    sid = fname.split("_")[0]
    out_img = os.path.join(out_path, fname)
    nib.save(image.concat_imgs(z_maps), out_img)
    log.info("Saved z-maps: %s", out_img)

    labels_path = os.path.join(out_path, f"{sid}_{task_label}_add_labels.csv")
    runs_path = os.path.join(out_path, f"{sid}_{task_label}_add_runs.csv")
    np.savetxt(labels_path, condition_idx, fmt="%s")
    np.savetxt(runs_path, session_idx, fmt="%s")
    log.info("Saved labels: %s", labels_path)
    log.info("Saved runs:   %s", runs_path)


# --------------------- baseline decoding (nilearn) ---------------------
def _beta_maps_svm_decoder(out_path: str, fname: str, task_label: str, mask) -> None:
    """Decode saved z-maps using nilearn.Decoder with SVC."""
    sid = fname.split("_")[0]
    z_map = os.path.join(out_path, fname)
    condition = os.path.join(out_path, f"{sid}_{task_label}_add_labels.csv")
    session = os.path.join(out_path, f"{sid}_{task_label}_add_runs.csv")

    condition_idx = pd.read_table(condition, header=None).values.ravel()
    session_idx = pd.read_table(session, header=None).values.ravel()

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    decoder = Decoder(estimator="svc", mask=mask[0], standardize=False, cv=cv, scoring="accuracy")
    decoder.fit(z_map, condition_idx, groups=session_idx)

    scores = {k: float(np.mean(v)) for k, v in decoder.cv_scores_.items()}
    mean_score = float(np.mean(list(scores.values())))
    for k, v in scores.items():
        log.info("Class %s | CV acc: %.2f", k, v)
    log.info("Mean CV accuracy: %.2f", mean_score)

    # Weight maps
    weights = decoder.coef_img_
    for k in weights:
        plotting.plot_stat_map(
            weights[k],
            title=k,
            colorbar=True,
            threshold=0.00005,
            display_mode="ortho",
            black_bg=True,
        )
    plt.show()


# --------------------------- public API --------------------------------
def postproc_task(subject: str, task_label: str) -> None:
    """
    Prepare data, generate beta maps, and run a baseline decoder.
    Paths and behavior preserved.
    """
    raw_data_path = "/data/neuromod/DATA/cneuromod/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/"
    func = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    regr = "desc-confounds_timeseries.tsv"
    mask_name = "space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    pathevents = "/data/neuromod/projects/ml_models_tutorial/data/hcptrt/HCPtrt_events_DATA/"
    out_path = "/data/neuromod/projects/ml_models_tutorial/data/hcptrt/postproc_beta_maps/"

    os.makedirs(out_path, exist_ok=True)

    scans = sorted(Path(raw_data_path, subject).rglob(f"*_task-{task_label}*{func}"))
    scans = [str(s) for s in scans]

    # ---- confounds loader (lazy import, safe across load-confounds versions) ----
    try:
        from load_confounds import Params9 as _Params9  # noqa: E402
        def _load_confounds(tsv_path):
            return pd.DataFrame.from_records(_Params9().load(str(tsv_path)))
    except Exception:
        # Fallback: run without confounds if package/API not available
        def _load_confounds(_):
            return None
    # ---------------------------------------------------------------------------

    regressors = sorted(Path(raw_data_path, subject).rglob(f"*_task-{task_label}*{regr}"))
    # confounds = [pd.DataFrame.from_records(Params9().load(str(r))) for r in regressors]
    confounds = [_load_confounds(r) for r in regressors]

    tpl_mask = sorted(Path(raw_data_path, subject).rglob(f"*_task-{task_label}*{mask_name}"))
    tpl_mask = [str(s) for s in tpl_mask]

    events = sorted(Path(pathevents, subject).rglob(f"*_task-{task_label}*events.tsv"))
    events = [_new_conditions(str(e), task_label) for e in events]

    conditions = list(set(events[0].trial_type))
    log.info("Conditions: %s", conditions)

    # Balance sessions (kept from original)
    scans = scans[:13]
    confounds = confounds[:13]
    events = events[:13]

    fname = f"{subject}_task-{task_label}_{func.replace('preproc_bold', 'postproc_beta_maps_P9')}"

    _generate_beta_maps(
        scans=scans,
        confounds=confounds,
        events=events,
        conditions=conditions,
        mask=tpl_mask,
        fname=fname,
        task_label=task_label,
        out_path=out_path,
    )

    _beta_maps_svm_decoder(out_path=out_path, fname=fname, task_label=task_label, mask=tpl_mask)


def within_subject_decoding(subject: str, task_dir: str, task_label: str, mask_path: str) -> None:
    """
    Alternative decoding path: NiftiMasker + LinearSVC with KFold CV.
    Saves per-run CV accuracies and overall mean to CSV.
    """
    subjects_list = [
        str(d)
        for d in sorted(Path(task_dir).rglob(f"{subject}_task-{task_label}*-postproc_P9.nii.gz"))
    ]
    labels_list = sorted(Path(task_dir).rglob(f"{subject}_{task_label}_add_labels.csv"))
    runs_list = sorted(Path(task_dir).rglob(f"{subject}_{task_label}_add_runs.csv"))

    if not subjects_list or not labels_list or not runs_list:
        log.warning("No decoding inputs found in %s for %s/%s.", task_dir, subject, task_label)
        return

    masker = NiftiMasker(mask_img=mask_path).fit()

    scores = []
    for ims, labs_file, runs_file in zip(subjects_list, labels_list, runs_list, strict=False):
        labs_idx = pd.read_table(labs_file, header=None).values.ravel()
        runs_idx = pd.read_table(runs_file, header=None).values.ravel()

        images = np.asarray(masker.transform(ims))
        decoder = LinearSVC(max_iter=10000)
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

        score = cross_val_score(decoder, images, labs_idx, groups=runs_idx, cv=cv, n_jobs=1)
        scores.append(float(np.mean(score)))

    mean_score = float(np.mean(scores)) if scores else float("nan")
    log.info("Within-subject LinearSVC mean CV accuracy: %.2f", mean_score)

    out_csv = os.path.join(task_dir, f"{subject}_{task_label}_LinearSVC_scores.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mean_cv_accuracy"])
        for s in scores:
            w.writerow([s])
        w.writerow(["overall_mean", mean_score])
    log.info("Saved scores: %s", out_csv)


if __name__ == "__main__":
    # python benchmark_models/beta_maps_benchmark.py --subject sub-01 --task wm
    import argparse

    p = argparse.ArgumentParser(description="Generate beta maps and run baseline decoding.")
    p.add_argument("--subject", required=True, help="Subject ID, e.g. sub-01")
    p.add_argument("--task", required=True,
                   choices=["wm", "motor", "emotion", "language", "relational"])
    args = p.parse_args()
    postproc_task(subject=args.subject, task_label=args.task)
