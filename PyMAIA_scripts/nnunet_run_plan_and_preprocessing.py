#!/usr/bin/env python

import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from PyMAIA.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)

DESC = dedent(
    """
    Run nnUNet command to extract dataset fingerprints, create experiment plan, preprocessing and ( optionally ) verify 
    the dataset integrity.
    The CL script called is  ``nnUNetv2_plan_and_preprocess``, with the arguments extracted from the given configuration file.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file /PATH/TO/CONFIG_FILE.json
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiments variables (Task_ID) ",
    )

    pars.add_argument(
        "--n-workers",
        type=str,
        default=None,
        help="Number of parallel processes used when pre-processing and unpacking the image data (Default: ``N_THREADS``)",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()
    arguments, unknown_arguments = parser.parse_known_args()
    args = vars(arguments)

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(args),
    )
    config_file = args["config_file"]

    if not "N_THREADS" in os.environ:
        os.environ["N_THREADS"] = str(os.cpu_count())
    n_workers = "1"
    if args["n_workers"] is None:
        if "N_THREADS" in os.environ is not None:
            n_workers = str(os.environ["N_THREADS"])
    else:
        n_workers = str(args["n_workers"])

    with open(config_file) as json_file:
        data = json.load(json_file)

        arguments = [
            "-d",
            data["Task_ID"],
            "-c",
            "3d_fullres",
            "-np",
            n_workers,
            "-npfp",
            n_workers
        ]

        os.environ["nnUNet_raw"] = str(Path(data["base_folder"]).joinpath("nnUNet_raw_data"))
        os.environ["nnUNet_preprocessed"] = data["preprocessing_folder"]
        os.environ["nnUNet_def_n_proc"] = n_workers
        os.environ["nnUNet_results"] = data["results_folder"]
        arguments.extend(unknown_arguments)
        os.system("nnUNetv2_plan_and_preprocess " + " ".join(arguments))


if __name__ == "__main__":
    main()
