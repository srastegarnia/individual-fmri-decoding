# benchmark_models/hcptrt_data_prep.py
import glob
import os
import sys
from pathlib import Path

import nilearn.datasets
import numpy as np
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker
from termcolor import colored

# sys.path.append(os.path.join(".."))  # original style
sys.path.append(os.path.join("../"))
import utils  # noqa: E402

"""
Utilities for extracting desired volumes
and labeling/relabeling data.
The outputs are final post-processed data.
"""

# -------- Paths (consistent with other modernized scripts) --------
REPO_ROOT = Path(__file__).resolve().parents[1]
# External data root (defaults keep your original cluster layout)
DATA_ROOT = Path(os.environ.get("HCPTRT_DATA_ROOT", "/home/rastegar/scratch/hcptrt"))
# Processed data root inside the repo (or your original CC path)
PROC_ROOT = Path(
    os.environ.get("HCPTRT_PROC_ROOT", str(REPO_ROOT / "data"))
)

# -------- Confounds helper: strategy -> Params9 -> None --------
def _get_confounds(img_path: str):
    """
    Prefer modern load_confounds_strategy; fall back to Params9 if available;
    last resort is None (keeps code import-safe).
    """
    try:
        conf_tuple = load_confounds_strategy(
            img_path, denoise_strategy="simple", motion="basic", global_signal="basic"
        )
        # nilearn returns (confounds, sample_mask). We need the confounds array.
        return conf_tuple[0]
    except Exception:
        try:
            from load_confounds import Params9  # optional, older API
            return Params9().load(img_path)
        except Exception:
            return None


def _reading_fMRI2(subject, modality, fMRI2_out_path, region_approach, resolution):
    bold_outname = f"{fMRI2_out_path}{subject}_{modality}_fMRI2.npy"
    print("bold_outname:", bold_outname)
    bold_files = np.load(bold_outname, allow_pickle=True)
    return bold_files


def _reading_events2(subject, modality, events2_out_path, region_approach, resolution):
    events_outname = f"{events2_out_path}{subject}_{modality}_events2"
    # pickle_in = open(events_outname, "rb")
    events_files = pd.read_pickle(events_outname)
    return events_files


def _volume_labeling(bold_files, events_files, subject,
                     modality, masker, data_path, TR):  # confounds,
    """
    Generating labels files for each volumes using
    'trial_type' column of events files.

    Parameters
    ----------
    events_files: list
        output of load_events_files function
    """

    expected_volumes = np.shape(bold_files[1])[0]

    conf = _get_confounds(data_path[0])
    sample_fmri = masker.fit_transform(data_path[0], confounds=conf)

    data_lenght = len(bold_files)

    labels_files, session_files = [], []
    for events_file in events_files:
        task_durations = []
        task_modalities = []
        row_counter = 0
        rows_no = len(events_file.axes[0])

        task_modalities.append(events_file.iloc[0]["trial_type"])

        for i in range(1, rows_no):
            if events_file.iloc[i]["trial_type"] != events_file.iloc[i - 1]["trial_type"]:
                task_modalities.append(events_file.iloc[i]["trial_type"])
                duration = (events_file.iloc[i]["onset"]) - (events_file.iloc[row_counter]["onset"])
                task_durations.append(duration)
                row_counter = i

            if i == rows_no - 1:
                duration = (events_file.iloc[i]["onset"]) - (events_file.iloc[row_counter]["onset"]) + (
                    events_file.iloc[i]["duration"] + TR
                )
                task_durations.append(duration)

        if len(task_durations) != len(task_modalities):
            print("error: tasks and durations do not match")

        task_durations = np.array(task_durations)
        task_modalities = np.array(task_modalities)

        # Generate volume No. array for each task condition
        volume_no = []
        for t in task_durations:
            volume_round = np.round((t) / TR).astype(int)
            volume_no.append(volume_round)

            # check that volume labels don't exceed expected volumes of bold file
            if sum(volume_no) > expected_volumes:
                i_eps = 0.1
                while sum(volume_no) > expected_volumes:
                    volume_no = []
                    for t in task_durations:
                        volume_round = np.round((t - i_eps) / TR).astype(int)
                        volume_no.append(volume_round)
                    i_eps = i_eps + 0.1

        # Find the Qty of null ending volumes
        ans_round = utils.sum_(volume_no)
        null_ending = sample_fmri.shape[0] - ans_round

        # Generate timeseries labels considering the volume No.
        final_array = []
        if len(task_modalities) == len(task_durations) == len(volume_no):
            for idx in range(len(task_modalities)):
                f = ((task_modalities[idx],) * volume_no[idx])
                final_array.append(f)

        # Add the null label for the ending volumes
        if null_ending > 0:
            end_volume = (("null",) * null_ending)
            final_array.append(end_volume)

        # Generate a flat list of labels
        flat_list = [item for sublist in final_array for item in sublist]
        volume_labels = np.array(flat_list)
        labels_files.append(volume_labels)

        for _ in range(len(volume_labels)):  # SHSH
            session_files.append(events_file.iloc[-1]["session"])  # SHSH

    flat_labels_files = [item for sublist in labels_files for item in sublist]
    flat_volume_labels = np.array(flat_labels_files)

    flat_session_labels = np.array(session_files)  # SHSH
    flat_session_labels = flat_session_labels.reshape(-1, 1)  # SHSH

    shape = np.shape(bold_files[1])[0]
    flat_volume_labels = np.reshape(flat_volume_labels, ((data_lenght) * shape, 1))  # for 'gambling'

    # Generate a flat list of bold matrices
    flat_bold = [item for sublist in bold_files for item in sublist]
    flat_bold_files = np.array(flat_bold)

    # Checking the same length of the flat bold and label file
    if len(flat_bold_files[:, 0]) != len(flat_volume_labels[:, 0]):
        print("error: labels and bold flat files mismatche")

    if len(flat_bold_files[:, 0]) != len(flat_session_labels[:, 0]):  # SHSH
        print("error: session and bold flat files mismatche")  # SHSH

    print("bold shape:", np.shape(flat_bold_files))
    print("events shape", np.shape(flat_volume_labels))
    print("sessions shape", np.shape(flat_session_labels))  # SHSH

    print("bold type:", type(flat_bold_files))
    print("events type", type(flat_volume_labels))
    print("sessions type", type(flat_session_labels))  # SHSH
    print("### Concatenating fMRI & events files is done!")
    print("-----------------------------------------------")

    return flat_bold_files, flat_volume_labels, flat_session_labels


def _HRFlag_labeling(flat_volume_labels, HRFlag_process):
    if HRFlag_process == "3volumes":
        """
        Labeling the first 3 volumes of stimulus longer than 5 seconds as HRF_lag
        """
        HRFlag_volume_labels = []
        counter = 0
        lenght = len(flat_volume_labels[:, 0])

        while counter < (lenght - 1):
            if flat_volume_labels[counter, 0] != flat_volume_labels[counter + 1, 0]:
                HRFlag_volume_labels.append(flat_volume_labels[counter, 0])

                if counter < (lenght - 4):
                    if (
                        flat_volume_labels[counter + 1, 0]
                        == flat_volume_labels[counter + 2, 0]
                        == flat_volume_labels[counter + 3, 0]
                        == flat_volume_labels[counter + 4, 0]
                    ):
                        for _ in range(1, 4):
                            HRFlag_volume_labels.append("HRF_lag")
                        counter = counter + 4
                    else:
                        counter = counter + 1
                else:
                    # when the last 3 or less volumes should be labeled as HRF
                    for _ii in range(counter, (lenght - 1)):
                        print((lenght - 1) - counter)
                        HRFlag_volume_labels.append("HRF_lag")
                    counter = counter + ((lenght - 1) - counter)
                    if counter == (lenght - 1):
                        print("Exceptional lenght")

            else:
                HRFlag_volume_labels.append(flat_volume_labels[counter, 0])
                counter = counter + 1

        HRFlag_volume_labels.append(flat_volume_labels[lenght - 1, 0])

    elif HRFlag_process == "2-1volumes":
        """
        Labeling the first volume of stimulus longer than 5 seconds for
        previous stimulus(overlap) also the second and third as HRF_lag
        """
        HRFlag_volume_labels = []
        counter = 0
        lenght = len(flat_volume_labels[:, 0])

        while counter < (lenght - 1):
            if flat_volume_labels[counter, 0] != flat_volume_labels[counter + 1, 0]:
                HRFlag_volume_labels.append(flat_volume_labels[counter, 0])

                if (
                    flat_volume_labels[counter + 1, 0]
                    == flat_volume_labels[counter + 2, 0]
                    == flat_volume_labels[counter + 3, 0]
                    == flat_volume_labels[counter + 4, 0]
                ):
                    HRFlag_volume_labels.append(flat_volume_labels[counter, 0])
                    for _ in range(1, 3):
                        HRFlag_volume_labels.append("HRF_lag")
                    counter = counter + 4
                else:
                    counter = counter + 1

            else:
                HRFlag_volume_labels.append(flat_volume_labels[counter, 0])
                counter = counter + 1

        HRFlag_volume_labels.append(flat_volume_labels[lenght - 1, 0])

    else:
        """
        Labeling without considering the HRF lag
        """
        temp = flat_volume_labels.tolist()
        HRFlag_volume_labels = [item for sublist in temp for item in sublist]

    print("### HRF lag labeling is done!")
    print("-----------------------------------------------")

    return HRFlag_volume_labels


def _unwanted_label_removal(
    events_files,
    HRFlag_volume_labels,
    flat_bold_files,
    flat_session_labels,
    final_bold_out_path,
    final_labels_out_path,
    final_session_out_path,
    modality,
    subject,
    region_approach,
    HRFlag_process,  # SHSH
):
    categories = list(events_files[0].trial_type)
    unwanted = {
        "countdown",
        "cross_fixation",
        "Cue",
        "new_bloc_right_hand",
        "new_bloc_right_foot",
        "new_bloc_left_foot",
        "new_bloc_left_hand",
        "new_bloc_tongue",
        "new_bloc_control",
        "new_bloc_relational",
        "new_bloc_shape",
        "new_bloc_face",
        "countdown_nan",
        "Cue_nan",
        "HRF_lag",
        "null",
        "nan_Cue",
        "nan_countdown",
    }

    categories = [c for c in categories if c not in unwanted]
    conditions = list(set(categories))  # noqa: F841 (kept for parity with original)

    final_volume_labels = []
    final_session_labels = []  # SHSH
    parcel_no = np.shape(flat_bold_files[1])[0]
    final_bold_files = np.empty((0, parcel_no), int)

    for i in range(0, len(HRFlag_volume_labels)):
        if HRFlag_volume_labels[i] not in unwanted:
            final_volume_labels.append(HRFlag_volume_labels[i])
            final_session_labels.append(flat_session_labels[i])  # SHSH
            final_bold_files = np.append(
                final_bold_files, np.array([flat_bold_files[i, :]]), axis=0
            )

    final_label_path = (
        f"{final_labels_out_path}{subject}_{modality}_{HRFlag_process}_final_labels.csv"
    )
    df_lable = pd.DataFrame(final_volume_labels)
    df_lable.to_csv(final_label_path, sep=",", index=False, header=None)

    final_session_path = (
        f"{final_session_out_path}{subject}_{modality}_{HRFlag_process}_final_session.csv"
    )  # SHSH
    df_session = pd.DataFrame(final_session_labels)  # SHSH
    df_session.to_csv(final_session_path, sep=",", index=False, header=None)  # SHSH

    final_fMRI_path = (
        f"{final_bold_out_path}{subject}_{modality}_{HRFlag_process}_final_fMRI.npy"
    )
    np.save(final_fMRI_path, final_bold_files)

    print("### Final volume label file is generated from events files!")
    print("File path:", final_label_path)
    print("-----------------------------------------------")
    print("### Final volume session file is generated from sessions files!")  # SHSH
    print("File path:", final_session_path)  # SHSH
    print("-----------------------------------------------")  # SHSH
    print("### Final numpy.ndarray file is generated from fMRI files!")
    print("File path:", final_fMRI_path)

    return final_volume_labels, final_session_labels, final_bold_files  # SHSH


def postproc_data_prep(split_set, subject, modalities, region_approach, HRFlag_process, resolution):  # confounds,
    """
    Outputs are
    """
    TR = 1.49

    # Processed data root per split_set (keeps your original structure)
    proc_data_path = str(PROC_ROOT / split_set) + "/"

    bold_suffix = "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    raw_atlas_dir = os.path.join(proc_data_path, "raw_atlas_dir")

    print(
        colored(
            f"{subject}, {region_approach}, {HRFlag_process}, res={resolution}:",
            attrs=["bold"],
        )
    )

    if region_approach == "MIST":
        masker = NiftiLabelsMasker(
            labels_img=f"{region_approach}_{resolution}.nii.gz",
            standardize=True,
            smoothing_fwhm=5,
        )

    elif region_approach == "difumo":
        atlas = nilearn.datasets.fetch_atlas_difumo(data_dir=raw_atlas_dir, dimension=resolution)
        atlas_filename = atlas["maps"]
        atlas_labels = atlas["labels"]  # noqa: F841 (kept for parity)
        masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, verbose=5)

    elif region_approach == "dypac":
        path_dypac = "/data/cisl/pbellec/models"
        file_mask = os.path.join(
            path_dypac, f"{subject}_space-MNI152NLin2009cAsym_label-GM_mask.nii.gz"
        )
        # file_dypac = os.path.join(
        #     path_dypac,
        #     f"{subject}_space-MNI152NLin2009cAsym_desc-dypac{resolution}_components.nii.gz",
        # )

        masker = NiftiMasker(standardize=True, detrend=False, smoothing_fwhm=5, mask_img=file_mask)

    else:
        masker = NiftiMasker(standardize=True)

    fMRI2_out_path = (
        proc_data_path + f"medial_data/fMRI2/{region_approach}/{resolution}/{subject}/"
    )
    events2_out_path = (
        proc_data_path
        + f"medial_data/events2/{region_approach}/{resolution}/{subject}/"
    )
    final_bold_out_path = (
        proc_data_path
        + f"processed_data/proc_fMRI/{region_approach}/{resolution}/{subject}/"
    )
    final_labels_out_path = (
        proc_data_path
        + f"processed_data/proc_events/{region_approach}/{resolution}/{subject}/"
    )
    final_session_out_path = (
        proc_data_path
        + f"processed_data/proc_sessions/{region_approach}/{resolution}/{subject}/"
    )  # SHSH

    # Ensure directories exist (preserves original trailing-slash semantics)
    os.makedirs(final_bold_out_path, exist_ok=True)
    os.makedirs(final_labels_out_path, exist_ok=True)
    os.makedirs(final_session_out_path, exist_ok=True)

    for modality in modalities:
        print(colored(modality, attrs=["bold"]))

        # Keep the original glob structure; build it from DATA_ROOT
        data_glob = os.path.join(
            str(DATA_ROOT),
            f"{split_set}/derivatives/fmriprep-20.2lts/fmriprep/{subject}/**/*{modality}*{bold_suffix}",
        )
        data_path = sorted(glob.glob(data_glob, recursive=True))

        bold_files = _reading_fMRI2(subject, modality, fMRI2_out_path, region_approach, resolution)
        events_files = _reading_events2(
            subject, modality, events2_out_path, region_approach, resolution
        )

        flat_bold_files, flat_volume_labels, flat_session_labels = _volume_labeling(
            bold_files=bold_files,
            events_files=events_files,
            subject=subject,
            modality=modality,
            masker=masker,
            data_path=data_path,
            TR=TR,
        )  # SHSH

        HRFlag_volume_labels = _HRFlag_labeling(flat_volume_labels, HRFlag_process)

        _ = _unwanted_label_removal(
            events_files,
            HRFlag_volume_labels,
            flat_bold_files,
            flat_session_labels,
            final_bold_out_path,
            final_labels_out_path,
            final_session_out_path,
            modality,
            subject,
            region_approach,
            HRFlag_process,
        )  # SHSH


###############################################################################################################################

def mean_volumes_windows(volume_num, flat_bold_files, flat_volume_labels):
    num_rows, num_cols = flat_bold_files.shape  # noqa: F841

    # Initializing w3_bold_file
    i = 0
    w3_bold_file = np.mean(flat_bold_files[i : i + 3, :], axis=0)
    temp_bold = np.reshape(w3_bold_file, (len(w3_bold_file), 1))
    w3_bold_file = temp_bold
    w3_bold_file = w3_bold_file.T

    for row in range(3, num_rows, 3):
        w3_bold_mean = np.mean(flat_bold_files[row : row + 3, :], axis=0)
        w3_bold_file = np.append(w3_bold_file, [w3_bold_mean], axis=0)

    # Initializing w3_events_file
    w3_events_file = flat_volume_labels[0]
    temp_event = np.reshape(w3_events_file, (len(w3_events_file), 1))
    w3_events_file = temp_event

    for row in range(3, num_rows, 3):
        if (
            flat_volume_labels[row]
            == flat_volume_labels[row + 1]
            == flat_volume_labels[row + 2]
        ):
            w3_events_file = np.append(w3_events_file, [flat_volume_labels[row]], axis=0)
        elif (
            flat_volume_labels[row] == flat_volume_labels[row + 1] != flat_volume_labels[row + 2]
        ):
            w3_events_file = np.append(w3_events_file, [flat_volume_labels[row]], axis=0)
        elif (
            flat_volume_labels[row] != flat_volume_labels[row + 1] == flat_volume_labels[row + 2]
        ):
            w3_events_file = np.append(w3_events_file, [flat_volume_labels[row + 1]], axis=0)
        else:
            w3_events_file = np.append(w3_events_file, [flat_volume_labels[row + 1]], axis=0)

    w3_events_file = [item for sublist in w3_events_file for item in sublist]

    return w3_bold_file, w3_events_file


def same_duration_labels(concat_bold_file, concat_events_file):
    # Placeholder kept as-is
    return concat_bold_file, concat_events_file



def validate_environment() -> None:
    """Light checks: required imports and known data roots exist."""
    import os  # safe local import
    required_dirs = [
        "/data/neuromod/DATA/cneuromod/hcptrt/derivatives/fmriprep-20.2lts/fmriprep",
        "/data/cisl/pbellec/models",  # dypac models (optional)
    ]
    missing = [p for p in required_dirs if not os.path.isdir(p)]
    if missing:
        print("Environment OK (imports). Data roots not present on this machine:")
        for m in missing:
            print(" -", m)
    else:
        print("Environment OK: imports and data roots present.")



if __name__ == "__main__":
    # Script is typically consumed by other modules; no CLI.
    pass






