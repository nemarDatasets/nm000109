"""EEG During Mental Arithmetic Tasks


Download command:
```bash
aws s3 sync --no-sign-request s3://physionet-open/eegmat/1.0.0/ source_root/
```
"""

README_CONTENT = """## Introduction

This dataset contains scalp EEG recordings from 36 healthy university students (9 male, 27 female; ages 18-26 years) during mental arithmetic tasks and resting-state periods. The study was designed to investigate EEG correlates of cognitive activity during intensive mental workload involving serial subtraction. The dataset provides brain electrical activity measurements for studying the neural mechanisms of mathematical cognition and cognitive stress responses, with potential applications in cognitive neuroscience research, mental workload assessment, and brain-computer interface development.

## Overview of the experiment

Participants were recorded during two conditions: (1) resting-state with eyes closed, and (2) mental arithmetic task involving serial subtraction. During the resting state, participants sat comfortably in a dark, soundproof chamber and were instructed to relax. After a 3-minute adaptation period, a 3-minute resting-state EEG recording was made with eyes closed. Participants then performed a 4-minute mental arithmetic task during which they were presented with a 4-digit minuend and 2-digit subtrahend (e.g., 3141 - 42) and performed serial subtractions mentally. They were instructed to count accurately and quickly in their self-determined rhythm without speaking or using finger movements. The dataset stores the last 3 minutes of the rest period (180 seconds) and the first minute of mental arithmetic performance (60 seconds) for each participant. EEG was recorded using a Neurocom 23-channel monopolar system sampled at 500 Hz with electrodes placed according to the International 10/20 system and referenced to interconnected ear electrodes. Filters included a high-pass filter (0.5 Hz cut-off), low-pass filter (45 Hz cut-off), and power line notch filter (50 Hz). Participants were divided post-hoc into two performance groups based on the number of completed arithmetic operations: "good counters" (Group G, n=24, mean operations=21, SD=7.4) and "bad counters" (Group B, n=12, mean operations=7, SD=3.6).

## Description of the preprocessing if any

All recordings included only artifact-free EEG segments, with 30 of 66 initially recorded participants excluded due to excessive oculographic and myographic artifacts. Channel names have been standardized to match the International 10-20 nomenclature. The raw EDF files have been converted to BIDS format with proper channel type assignments (EEG for brain signals). Subject birth years were calculated from age and recording year. Recording dates have been set to January 1st of the recording year due to privacy considerations in the original dataset. Impedance checks confirmed all electrodes were below 5 kΩ prior to recording.

## Description of the event values if any

No events.tsv files are provided. The "task" field in the BIDS filenames indicates the experimental condition:
- "rest": resting-state condition
- "mentalArithmetic": mental arithmetic task condition

## Citation

When using this dataset, please cite:

1. Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, Shpenkov O. Electroencephalograms during Mental Arithmetic Task Performance. Data. 2019; 4(1):14. https://doi.org/10.3390/data4010014

2. PhysioNet database: https://doi.org/10.13026/C2JQ1P

3. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

**Data curators:**
Pierre Guetschel (BIDS conversion)

Original data collection team:
- Igor Zyma, PhD (National Technical University of Ukraine)
- Sergii Tukaev (National Technical University of Ukraine)
- Ivan Seleznov (National Technical University of Ukraine)
- Ken Kiyono, PhD
- Anton Popov (National Technical University of Ukraine)
- Mariia Chernykh (National Technical University of Ukraine)
- Oleksii Shpenkov (National Technical University of Ukraine)
"""

DATASET_NAME = "EEG During Mental Arithmetic Tasks"

from pathlib import Path
import re
import datetime
import shutil
import re
import warnings

from mne_bids import BIDSPath, write_raw_bids, make_dataset_description, make_report
import mne
import numpy as np
import pandas as pd


TASK_MAP = {1: "rest", 2: "mentalArithmetic"}
CH_NAME_REGEX = r"(?P<ch_type>[A-Z]+) (?P<ch_name>[a-zA-Z0-9]+)"


def _get_records(source_root: Path):
    records_file = source_root / "RECORDS"
    records = set(records_file.read_text().splitlines())
    record_files = list(source_root.glob("*.edf"))
    files = set([r.name for r in record_files])
    assert records == files, (
        "Records file does not match files in source root. "
        f"Differences:\nIn RECORDS but not in folder: {records - files}.\n"
        f"In folder but not in RECORDS: {files - records}"
    )

    # example record:
    # "Subject00_1.edf"
    record_regex = r"Subject(?P<subject>[0-9]+)_(?P<task_id>[0-9])\.edf"

    for record in record_files:
        match = re.match(record_regex, record.name)
        assert match is not None, f"Record {record} does not match expected format"
        subject = match.group("subject")
        task_id = int(match.group("task_id"))
        task = TASK_MAP[task_id]
        bids_path = BIDSPath(
            subject=subject,
            task=task,
            suffix="eeg",
            datatype="eeg",
        )
        yield record, bids_path


def main(
    source_root: Path,
    bids_root: Path,
    overwrite: bool = False,
    finalize_only: bool = False,
):
    """Convert the dataset to BIDS format.

    Parameters
    ----------
    source_root : Path
        Path to the root folder
    bids_root : Path
        Path to the root of the BIDS dataset to create.
    overwrite : bool
        If True, overwrite existing BIDS files.
    """
    source_root = Path(source_root).expanduser()
    bids_root = Path(bids_root).expanduser()

    if finalize_only:
        _finalize_dataset(bids_root, overwrite=overwrite)
        return

    records = list(_get_records(source_root))

    subjects_file = source_root / "subject-info.csv"
    subjects_df = pd.read_csv(subjects_file)
    subjects_df = subjects_df.set_index("Subject")
    subjects_df["birth_year"] = (
        subjects_df["Recording year"] - subjects_df["Age"]
    ).apply(lambda i: datetime.date(year=i, month=1, day=1))
    subjects_df["sex"] = subjects_df["Gender"].map({"M": 1, "F": 2})

    # Add bids root:
    bids_root.mkdir(parents=True, exist_ok=True)
    for _, bids_path in records:
        bids_path = bids_path.update(root=bids_root)

    # sanity check: no duplicate bids paths
    bids_paths = [bids_path.fpath for _, bids_path in records]
    assert len(bids_paths) == len(set(bids_paths)), "Duplicate BIDS paths found"

    std_list = []
    for source_path, bids_path in records:
        raw = mne.io.read_raw_edf(source_path, preload=True, verbose=False)

        subject_row = subjects_df.loc[f"Subject{bids_path.subject}"]
        raw.info["subject_info"] = {
            "his_id": bids_path.subject,
            "birthday": subject_row["birth_year"],
            "sex": int(subject_row["sex"]),
        }
        raw.set_meas_date(
            datetime.datetime(
                year=int(subject_row["Recording year"]),
                month=1,
                day=1,
                tzinfo=datetime.timezone.utc,
            )
        )
        # rename channels
        ch_names_mapping = {}
        ch_types_mapping = {}
        for ch_name in raw.ch_names:
            match = re.match(CH_NAME_REGEX, ch_name)
            assert (
                match is not None
            ), f"Channel name '{ch_name}' does not match expected format"
            new_ch_name = match.group("ch_name")
            ch_names_mapping[ch_name] = new_ch_name
            ch_types_mapping[new_ch_name] = match.group("ch_type").lower()

        raw.rename_channels(ch_names_mapping)
        raw.set_channel_types(ch_types_mapping)

        write_raw_bids(
            raw,
            bids_path,
            overwrite=overwrite,
            verbose=False,
            allow_preload=True,
            format="EDF",
        )

    _finalize_dataset(bids_root, overwrite=overwrite)


def _finalize_dataset(bids_root: Path, overwrite: bool = False):
    # save script
    script_path = Path(__file__)
    script_dest = bids_root / "code" / script_path.name
    script_dest.parent.mkdir(exist_ok=True)
    shutil.copy2(script_path, script_dest)
    description_file = bids_root / "dataset_description.json"
    if description_file.exists() and overwrite:
        description_file.unlink()
    make_dataset_description(
        path=bids_root,
        name=DATASET_NAME,
        dataset_type="derivative",
        references_and_links=[
            "https://doi.org/10.3390/data4010014",
        ],
        source_datasets=[
            {"URL": "https://physionet.org/content/eegmat/1.0.0/"},
        ],
        authors=[
            "Igor Zyma",
            "Sergii Tukaev",
            "Ivan Seleznov",
            "Ken Kiyono",
            "Anton Popov",
            "Mariia Chernykh",
            "Oleksii Shpenkov",
        ],
        acknowledgements="Pierre Guetschel updated the data to BIDS format.",
        overwrite=overwrite,
        data_license="ODC-By-1.0",
    )

    # Remove macOS resource fork files that can break make_report
    for dotfile in bids_root.rglob("._*"):
        dotfile.unlink()

    try:
        report_str = make_report(bids_root)
        print(report_str)
    except Exception as e:
        warnings.warn(f"make_report failed: {e}")
        report_str = str(e)

    # overwrite README (include automatic report)
    readme_path = bids_root / "README.md"
    readme_path.write_text(
        f"# {DATASET_NAME}\n\n{README_CONTENT}\n\n---\n\n"
        f"## Automatic report\n\n*Report automatically generated by `mne_bids.make_report()`.*\n\n> {report_str}"
    )

    # Remove participants.json if it exists
    participants_json = bids_root / "participants.json"
    if participants_json.exists():
        participants_json.unlink()
        print(f"Removed {participants_json}")

    # Clean up participants.tsv by removing columns where all values are "n/a"
    participants_tsv = bids_root / "participants.tsv"
    if participants_tsv.exists():
        df = pd.read_csv(participants_tsv, sep="\t")
        # Find columns where all non-participant_id values are "n/a"
        cols_to_drop = []
        for col in df.columns:
            if col != "participant_id" and (df[col] == "n/a").all():
                cols_to_drop.append(col)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            df.to_csv(participants_tsv, sep="\t", index=False)
            print(
                f"Removed columns with all 'n/a' values from {participants_tsv}: {cols_to_drop}"
            )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
    # python bids_maker/datasets/zyma2019.py --source_root ~/data/arithmetic_zyma2019/ --bids_root ~/data/bids/arithmetic_zyma2019/ --overwrite=True
