#!/usr/bin/env python

import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

from k8s_DP.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

DESC = dedent(
    """
    Run nnUNet command to create experiment plan, preprocessing and ( optionally ) verify the dataset integrity.

    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
     {filename} --config-file ../LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json
     {filename} --config-file ../LungLobeSeg_3d_fullres_100_LungLobeSeg_3D_Single_Modality.json --verify_dataset_integrity
    """.format(  # noqa: E501
        filename=os.path.basename(__file__)
    )
)

if __name__ == "__main__":
    parser = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiments variables (Task_ID) ",
    )

    add_verbosity_options_to_argparser(parser)
    arguments, unknown_arguments = parser.parse_known_args()
    args = vars(arguments)

    logger = get_logger(
        name=os.path.basename(__file__),
        level=log_lvl_from_verbosity_args(args),
    )

    config_file = args["config_file"]

    with open(config_file) as json_file:
        data = json.load(json_file)

        arguments = [
            "-t",
            data["Task_ID"],
        ]

        arguments.extend(unknown_arguments)
        os.system("nnUNet_plan_and_preprocess " + " ".join(arguments))