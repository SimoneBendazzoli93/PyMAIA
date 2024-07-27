#!/usr/bin/env python

import datetime
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from PyMAIA.utils.file_utils import subfolders
from PyMAIA.utils.log_utils import add_verbosity_options_to_argparser
from PyMAIA.utils.volume_utils import resample_image

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to downsample the modality of a given NIFTI dataset, using another modality as reference.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename}  --data-folder /PATH/TO/NIFTI_DATASET --input-modality _CT.nii.gz --ref-modality _PET.nii.gz --output-modality _downsampled_CT.nii.gz
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
        help="Patient NIFTI dataset folder.",
    )

    pars.add_argument(
        "--input-modality",
        type=str,
        required=True,
        help="Input modality suffix.",
    )

    pars.add_argument(
        "--ref-modality",
        type=str,
        required=True,
        help="Reference modality suffix.",
    )

    pars.add_argument(
        "--output-modality",
        type=str,
        required=True,
        help="Output modality suffix.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()
    arguments = vars(parser.parse_args())

    subjects = subfolders(arguments["data_folder"], join=False)
    input_modality = arguments["input_modality"]
    ref_modality = arguments["ref_modality"]
    output_modality = arguments["output_modality"]

    for subject in subjects:
        input_filename = os.path.join(arguments["data_folder"], subject, str(subject + input_modality))
        ref_filename = os.path.join(arguments["data_folder"], subject, str(subject + ref_modality))
        output_filename = os.path.join(arguments["data_folder"], subject, str(subject + output_modality))
        resample_image(input_filename, ref_filename, output_filename)
        print("Subject: ", subject, " downsampling done.")

if __name__ == "__main__":
    main()
