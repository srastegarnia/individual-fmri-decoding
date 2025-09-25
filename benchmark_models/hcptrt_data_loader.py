import glob
import os
import pickle
from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from termcolor import colored

# dypac is optional; only needed for region_approach == "dypac"
try:
    from dypac.masker import MapsMasker  # type: ignore
    _HAS_DYPAC = True
except Exception:
    MapsMasker = None  # type: ignore
    _HAS_DYPAC = False

# utils lives at repo root; support both "package" and "script" usage.
try:  # package context: individual-fmri-decoding/benchmark_models/...
    from . import utils  # noqa: F401
except Exception:  # script context: run from repo root
    import utils  # type: ignore


"""
Utilities for the first step of reading and processing HCPtrt data.
Run once to produce intermediate "medial_data": fMRI2 (parcellated time series)
and events2 (relabeled events per task/run).
"""


class DataLoader:
    def __init__(
        self,
        TR: float,
        modality: str,
        subject: str,
        bold_suffix: str,
        region_approach: str,
        resolution: int,
        fMRI2_out_path: str | None = None,
        events2_out_path: str | None = None,
        raw_data_path: str | None = None,
        pathevents: str | None = None,
        raw_atlas_dir: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        TR : float
            Repetition time.
        modality : str
            Task label (e.g., "motor", "wm", ...).
        subject : str
            Subject ID (e.g., "sub-01").
        bold_suffix : str
            Suffix of preprocessed BOLD filenames.
        region_approach : str
            Parcellation approach: {"MIST","difumo","schaefer","dypac"} or "voxel" fallback.
        resolution : int
            Parcellation resolution (e.g., 444, 1024, ...).
        fMRI2_out_path : str
            Output directory for parcellated time-series .npy files.
        events2_out_path : str
            Output directory for relabeled events (pickle).
        raw_data_path : str
            Base path to fMRIPrep derivatives.
        pathevents : str
            Base path to events .tsv files.
        raw_atlas_dir : str
            Base path where atlas files live (for difumo / schaefer).
        """
        self.TR = TR
        self.modality = modality
        self.subject = subject
        self.bold_suffix = bold_suffix
        self.region_approach = region_approach
        self.resolution = resolution
        self.fMRI2_out_path = fMRI2_out_path or ""
        self.events2_out_path = events2_out_path or ""
        self.raw_data_path = raw_data_path or ""
        self.pathevents = pathevents or ""
        self.raw_atlas_dir = raw_atlas_dir or ""

        os.makedirs(self.fMRI2_out_path, exist_ok=True)
        os.makedirs(self.events2_out_path, exist_ok=True)
        os.makedirs(self.raw_atlas_dir, exist_ok=True)

    # ------------------------------ fMRI ------------------------------ #
    def _load_fmri_data(self) -> Tuple[List[np.ndarray], NiftiMasker, List[str]]:
        """
        Returns
        -------
        fmri_t : list[np.ndarray]
            List of 2D arrays (n_volumes x n_parcels) per run.
        masker : NiftiMasker | NiftiLabelsMasker | NiftiMapsMasker
            Fitted masker used to extract signals.
        data_path : list[str]
            Sorted list of BOLD file paths used.
        """
        data_path = sorted(
            glob.glob(
                f"{self.raw_data_path}{self.subject}/**/*{self.modality}*{self.bold_suffix}",
                recursive=True,
            )
        )

        print(colored(f"{self.subject}, {self.modality}:", attrs=["bold"]))

        # Keep behavior: cap at 15 runs (historical off-by-one kept intact).
        if len(data_path) > 15:
            data_extra_files = len(data_path) - 15
            print(
                colored(
                    f"Regressed out {data_extra_files} extra following fMRI file(s):",
                    "red",
                    attrs=["bold"],
                )
            )
            for i in range(14, len(data_path)):
                print(colored(data_path[i].split("func/", 1)[1], "red"))
            for _ in range(14, len(data_path)):
                data_path.pop()

        print(f"The number of bold files: {len(data_path)}")

        # Select masker by approach
        if self.region_approach == "MIST":
            masker = NiftiLabelsMasker(
                labels_img=f"{self.region_approach}_{self.resolution}.nii.gz",
                standardize=True,
                smoothing_fwhm=5,
            )
            fmri_t: list[np.ndarray] = []
            t0 = time()
            for dpath in data_path:
                print(dpath.split("func/", 1)[1])
                conf = load_confounds_strategy(
                    dpath, denoise_strategy="simple", motion="basic", global_signal="basic"
                )
                data_fmri = masker.fit_transform(dpath, confounds=conf[0])
                fmri_t.append(data_fmri)
                print("shape:", np.shape(data_fmri))
            print(
                f"Data processing time for {self.subject} using {self.region_approach} "
                f"with {self.resolution} resolution: {round(time()-t0, 3)} s"
            )

        elif self.region_approach == "difumo":
            # Use local difumo atlas layout preserved from original code
            atlas_filename = os.path.join(
                self.raw_atlas_dir,
                f"{self.region_approach}_atlases/{self.resolution}/3mm/maps.nii.gz",
            )
            masker = NiftiMapsMasker(
                maps_img=atlas_filename, standardize=True, verbose=5, smoothing_fwhm=5
            )
            fmri_t = []
            t0 = time()
            for dpath in data_path:
                conf = load_confounds_strategy(
                    dpath, denoise_strategy="simple", motion="basic", global_signal="basic"
                )
                data_fmri = masker.fit_transform(dpath, confounds=conf[0])
                fmri_t.append(data_fmri)
                print("shape:", np.shape(data_fmri))
            print(
                f"Data processing time for {self.subject} using {self.region_approach} "
                f"with {self.resolution} resolution: {round(time()-t0, 3)} s"
            )

        elif self.region_approach == "schaefer":
            atlas_filename = os.path.join(
                self.raw_atlas_dir,
                f"{self.region_approach}_2018/"
                f"Schaefer2018_{self.resolution}Parcels_7Networks_order_FSLMNI152_1mm.nii.gz",
            )
            masker = NiftiLabelsMasker(
                labels_img=atlas_filename, standardize=True, verbose=5, smoothing_fwhm=5
            )
            fmri_t = []
            t0 = time()
            for dpath in data_path:
                conf = load_confounds_strategy(
                    dpath, denoise_strategy="simple", motion="basic", global_signal="basic"
                )
                data_fmri = masker.fit_transform(dpath, confounds=conf[0])
                fmri_t.append(data_fmri)
                print("shape:", np.shape(data_fmri))
            print(
                f"Data processing time for {self.subject} using {self.region_approach} "
                f"with {self.resolution} resolution: {round(time()-t0, 3)} s"
            )

        elif self.region_approach == "dypac":
            
            if not _HAS_DYPAC:
                raise ImportError(
                    'region_approach="dypac" requires the "dypac" package. '
                    "Install it or choose a different region_approach."
                    )
            
            path_dypac = "/data/cisl/pbellec/models"
            file_mask = os.path.join(
                path_dypac, f"{self.subject}_space-MNI152NLin2009cAsym_label-GM_mask.nii.gz"
            )
            file_dypac = os.path.join(
                path_dypac,
                f"{self.subject}_space-MNI152NLin2009cAsym_desc-dypac{self.resolution}_components.nii.gz",
            )
            print("file_mask: ", file_mask)
            print("file_dypac: ", file_dypac, "\n")
            masker = NiftiMasker(standardize=True, detrend=False, smoothing_fwhm=5, mask_img=file_mask)

            fmri_t = []
            for dpath in data_path:
                conf = load_confounds_strategy(dpath, denoise_strategy="simple", global_signal="basic")
                masker.fit(dpath)
                maps_masker = MapsMasker(masker=masker, maps_img=file_dypac)
                data_fmri = maps_masker.transform(img=dpath, confound=conf[0])
                fmri_t.append(data_fmri)
                print("fMRI file:", dpath.split("func/", 1)[1])
                print("shape:", np.shape(data_fmri), "\n")
                print("\n")

        else:
            # Voxel-wise fallback
            masker = NiftiMasker(standardize=True)
            fmri_t = []
            for dpath in data_path:
                conf = load_confounds_strategy(
                    dpath, denoise_strategy="simple", motion="basic", global_signal="basic"
                )
                data_fmri = masker.fit_transform(dpath, confounds=conf[0])
                fmri_t.append(data_fmri)

        print("### Reading Nifiti files is done!")
        print("-----------------------------------------------")
        return fmri_t, masker, data_path

    # ------------------------------ Events ------------------------------ #
    def _load_events_files(self) -> List[pd.DataFrame]:
        """
        Relabel events per task and annotate with session/run index.
        """
        events_path = sorted(
            glob.glob(
                f"{self.pathevents}{self.subject}/**/func/*{self.modality}*_events.tsv",
                recursive=True,
            )
        )

        # Cap at 14 runs (kept as in original).
        if len(events_path) > 14:
            events_extra_files = len(events_path) - 14
            print(
                colored(
                    f"Regressed out {events_extra_files} extra following events file(s):",
                    "red",
                    attrs=["bold"],
                )
            )
            for i in range(14, len(events_path)):
                print(colored(events_path[i].split("func/", 1)[1], "red"))
            for _ in range(14, len(events_path)):
                events_path.pop()

        print(f"The number of events files: {len(events_path)}")

        events_files: list[pd.DataFrame] = []
        count = 1

        for epath in events_path:
            event = pd.read_csv(epath, sep="\t", encoding="utf8")
            print(epath.split("func/", 1)[1])
            print(event.head(5))
            print(np.shape(event))
            print(np.unique(event.trial_type))

            # Relabel per modality
            if self.modality == "emotion":
                event.trial_type = event["trial_type"].replace(
                    ["response_face", "response_shape"], ["fear", "shape"]
                )

            if self.modality == "language":
                event.trial_type = event["trial_type"].replace(
                    [
                        "presentation_story",
                        "question_story",
                        "response_story",
                        "presentation_math",
                        "question_math",
                        "response_math",
                    ],
                    ["story", "story", "story", "math", "math", "math"],
                )

            if self.modality == "motor":
                event.trial_type = event["trial_type"].replace(
                    [
                        "response_left_foot",
                        "response_left_hand",
                        "response_right_foot",
                        "response_right_hand",
                        "response_tongue",
                    ],
                    ["footL", "handL", "footR", "handR", "tongue"],
                )

            if self.modality == "relational":
                event.trial_type = event["trial_type"].replace(
                    ["Control", "Relational"], ["match", "relational"]
                )

            if self.modality == "wm":
                event.trial_type = event.stim_type.astype(str) + "_" + event.trial_type.astype(str)
                event.trial_type = event["trial_type"].replace(
                    [
                        "Body_0-Back",
                        "Body_2-Back",
                        "Face_0-Back",
                        "Face_2-Back",
                        "Place_0-Back",
                        "Place_2-Back",
                        "Tools_0-Back",
                        "Tools_2-Back",
                    ],
                    ["body0b", "body2b", "face0b", "face2b", "place0b", "place2b", "tool0b", "tool2b"],
                )

            # Session/run annotation
            conditions = list(event.trial_type)
            session_idx: list[str] = []
            count_idx: list[int] = []
            for _ in conditions:
                ses = epath.split("_task")[0].split("func/")[1].partition("_")[2]
                temp_run = epath.split("run")[1]
                run = utils.between(temp_run, "-", "_")
                session = f"{ses}_run-{run}"
                session_idx.append(session)
                count_idx.append(count)

            event["session"] = session_idx
            event["count"] = count_idx

            events_files.append(event)
            count += 1

        print("### Reading events files is done!")
        print("-----------------------------------------------")
        return events_files


# ------------------------------ Checks & Saving ------------------------------ #
def _check_input(fmri_t: List[np.ndarray], events_files: List[pd.DataFrame]) -> None:
    """
    - Remove extra ending volumes to match shapes across runs.
    - Check events and fMRI files counts.
    """
    data_length = int(len(fmri_t) or 0)

    # Harmonize run lengths
    for i in range(0, data_length - 1):
        if fmri_t[i].shape != fmri_t[i + 1].shape:
            print("There is mismatch in BOLD file size:")
            if fmri_t[i].shape > fmri_t[i + 1].shape:
                a = np.shape(fmri_t[i])[0] - np.shape(fmri_t[i + 1])[0]
                fmri_t[i] = fmri_t[i][0:-a, 0:]
                print(f"The {a} extra volumes of bold file number {i} is removed.")
            else:
                b = np.shape(fmri_t[i + 1])[0] - np.shape(fmri_t[i])[0]
                fmri_t[i + 1] = fmri_t[i + 1][0:-b, 0:]
                print(f"The {b} extra volumes of bold file number {i+1} is removed.")

    if len(events_files) != len(fmri_t):
        print("Miss-matching between events and fmri files")
        print("Number of Nifti files:", len(fmri_t))
        print("Number of events files:", len(events_files))
    else:
        print("Events and fMRI files are Consistent.")

    print("### Cheking data is done!")
    print("-----------------------------------------------")


def _save_files(
    fmri_t: List[np.ndarray],
    events_files: List[pd.DataFrame],
    subject: str,
    modality: str,
    fMRI2_out_path: str,
    events2_out_path: str,
) -> None:
    """
    - Save parcellated fMRI matrices per task.
    - Save relabeled events (pickle) per task.
    """
    # fMRI
    bold_outname = os.path.join(fMRI2_out_path, f"{subject}_{modality}_fMRI2.npy")
    np.save(bold_outname, fmri_t)

    # sanity round-trip
    temp = np.load(bold_outname, allow_pickle=True)
    fmri_t = temp  # noqa: F841 (kept for parity with original logging)

    print("Bold file:", bold_outname)
    print("### Saving Nifiti files as matrices is done!")
    print("-----------------------------------------------")

    # events
    events_outname = os.path.join(events2_out_path, f"{subject}_{modality}_events2")
    with open(events_outname, "wb") as fh:
        pickle.dump(events_files, fh)

    print("Events pickle file:", events_outname)
    print("### Saving events pickle files is done!")
    print("-----------------------------------------------")


# ------------------------------ Orchestrator ------------------------------ #
def postproc_data_loader(subject: str, modalities: list[str], region_approach: str, resolution: int) -> None:
    """
    Orchestrate loading, checking, and saving for a subject across modalities.
    Paths kept as in the original code (scratch/CC layout).
    """
    TR = 1.49

    # CC paths (kept)
    pathevents = "/home/rastegar/scratch/hcptrt/"
    raw_data_path = "/home/rastegar/scratch/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/"
    proc_data_path = "/home/rastegar/projects/def-pbellec/rastegar/hcptrt_decoding_shima/data/"

    bold_suffix = "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    raw_atlas_dir = os.path.join(proc_data_path, "raw_atlas_dir")

    fMRI2_out_path = os.path.join(proc_data_path, f"medial_data/fMRI2/{region_approach}/{resolution}/{subject}/")
    events2_out_path = os.path.join(proc_data_path, f"medial_data/events2/{region_approach}/{resolution}/{subject}/")

    for modality in modalities:
        print(colored(modality, "red", attrs=["bold"]))

        loader = DataLoader(
            TR=TR,
            modality=modality,
            subject=subject,
            bold_suffix=bold_suffix,
            region_approach=region_approach,
            resolution=resolution,
            fMRI2_out_path=fMRI2_out_path,
            events2_out_path=events2_out_path,
            raw_data_path=raw_data_path,
            pathevents=pathevents,
            raw_atlas_dir=raw_atlas_dir,
        )

        fmri_t, _masker, _paths = loader._load_fmri_data()
        events_files = loader._load_events_files()

        _check_input(fmri_t, events_files)
        _save_files(fmri_t, events_files, subject, modality, fMRI2_out_path, events2_out_path)


