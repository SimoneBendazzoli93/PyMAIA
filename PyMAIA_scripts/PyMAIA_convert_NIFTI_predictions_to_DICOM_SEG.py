#!/usr/bin/env python

import datetime
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent

from tqdm import tqdm

from PyMAIA.utils.file_utils import convert_nifti_pred_to_dicom_seg
from PyMAIA.utils.file_utils import subfolders
from PyMAIA.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args

if "N_THREADS" not in os.environ:
    os.environ["N_THREADS"] = "1"

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to convert NIFTI Predictions ( as output from the model inference ), back into the original DICOM context ( as SEG modalities ).
    The NIFTI Predictions are converted into DICOM SEG and assigned to the corresponding original DICOM Study. 
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --data-folder <NIFTI_DATA_FOLDER> --dicom-folder <DICOM_DATA_FOLDER> --output-folder <DICOM_SEG_FOLDER> --pred-suffix _seg.nii.gz --template-file <DCMQI_TEMPLATE> --study-id-summary STUDY_ID_DICT.json
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="Data Folder containing the NIFTI Predictions.",
    )

    pars.add_argument(
        "--dicom-folder",
        type=str,
        required=True,
        help="DICOM Folder containing the original Data.",
    )

    pars.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Folder where to store the generated DICOM SEG.",
    )

    pars.add_argument(
        "--pred-suffix",
        type=str,
        required=True,
        help="Suffix to append to the Patient folder name to identify the NIFTI prediction.",
    )

    pars.add_argument(
        "--n-workers",
        type=str,
        required=False,
        default=os.environ["N_THREADS"],
        help="Number of Parallel Threads.",
    )

    pars.add_argument(
        "--template-file",
        type=str,
        required=True,
        help="DCMQI DICOM SEG Template, generated from http://qiicr.org/dcmqi/#/seg",
    )

    pars.add_argument(
        "--study-id-summary",
        type=str,
        required=True,
        help="Study ID Summary, generated when converting the DICOM Dataset to a NIFTI Dataset.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )

    with open(arguments["study_id_summary"], "r") as f:
        study_id_dict = json.load(f)

    subjects = subfolders(arguments["data_folder"], join=False)

    pool = Pool(int(arguments["n_workers"]))
    Path(arguments["output_folder"]).mkdir(parents=True, exist_ok=True)
    NIFTI_to_DICOM_conversions = []
    for subject in subjects:
        dicom_subject = subject[:-2]
        subject_dicom_folder = str(Path(arguments["dicom_folder"]).joinpath(dicom_subject))
        subject_nifti_pred = str(Path(arguments["data_folder"]).joinpath(subject, subject + arguments["pred_suffix"]))

        NIFTI_to_DICOM_conversions.append(
            pool.starmap_async(
                convert_nifti_pred_to_dicom_seg,
                ((
                     subject_nifti_pred,
                     subject_dicom_folder,
                     arguments["template_file"],
                     str(Path(arguments["output_folder"]).joinpath(subject + ".dcm")),
                     study_id_dict[dicom_subject][subject[-1]]

                 ),),
            )
        )

    patients_map = {}
    for i in tqdm(NIFTI_to_DICOM_conversions):
        _ = i.get()


if __name__ == "__main__":
    main()
