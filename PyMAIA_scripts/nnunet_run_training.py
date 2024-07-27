#!/usr/bin/env python

import json
import os
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from PyMAIA.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, str2bool

DESC = dedent(
    """
        Run ``nnUNetv2_train`` command to start nnUNet training for the specified fold.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file ../CONFIG_FILE.json --run-fold 0
        {filename} --config-file ../CONFIG_FILE.json --run-fold 0 --resume-training y
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
        help="File path for the configuration dictionary, used to retrieve experiments variables (Task_ID)",
    )

    pars.add_argument(
        "--run-fold",
        type=int,
        choices=range(-1, 5),
        metavar="[-1-4]",
        default=0,
        help="int value indicating which fold (in the range 0-4) to run",
    )

    pars.add_argument(
        "--run-validation-only",
        type=str2bool,
        default="no",
        help="Flag to run only the Validation step ( after the Training step is completed). Default ``no``.",
    )

    pars.add_argument(
        "--post-processing-folds",
        type=str,
        nargs="+",
        required=False,
        default="-1",
        help="Trained Folds to include in the post-processing and model export. Default ``-1`` (All Folds are used).",
    )

    pars.add_argument(
        "--output-model-file",
        type=str,
        required=False,
        default=None,
        help="File Path where to save the zipped Model File.",
    )

    pars.add_argument(
        "--resume-training",
        type=str2bool,
        default="no",
        help="Flag to indicate training resume after stopping it. Default ``no``.",
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

    with open(config_file) as json_file:
        data = json.load(json_file)

        arguments = [
            "nnUNetv2_train",
            data["Task_ID"],
            "3d_fullres",
            str(args["run_fold"])
        ]
        if args["run_validation_only"]:
            arguments.append("--val")

        if args["resume_training"]:
            arguments.append("--c")

        arguments.extend(unknown_arguments)

        if not "N_THREADS" in os.environ:
            os.environ["N_THREADS"] = str(os.cpu_count())
        n_workers = "1"
        if args["n_workers"] is None:
            if "N_THREADS" in os.environ is not None:
                n_workers = str(os.environ["N_THREADS"])
        else:
            n_workers = str(args["n_workers"])

        os.environ["nnUNet_raw"] = str(Path(data["base_folder"]).joinpath("nnUNet_raw_data"))
        os.environ["nnUNet_preprocessed"] = data["preprocessing_folder"]
        os.environ["nnUNet_results"] = data["results_folder"]
        os.environ["nnUNet_def_n_proc"] = n_workers
        os.environ["nnUNet_n_proc_DA"] = n_workers

        if args["output_model_file"] is None:
            args["output_model_file"] = str(
                Path(data["results_folder"]).joinpath(data["Experiment Name"] + "_nnUNet_3d_fullres.zip"))

        if str(args["run_fold"]) == "-1" and "output_model_file" in args:
            if args["post_processing_folds"] != "-1":
                cmd = ["nnUNetv2_find_best_configuration", data["Task_ID"], "-c", "3d_fullres", "-f",
                       *args["post_processing_folds"]]
                cmd.extend(unknown_arguments)
                subprocess.run(cmd)
                cmd = ["nnUNetv2_export_model_to_zip", "-d", data["Task_ID"], "--exp_cv_preds", "-f",
                       *args["post_processing_folds"], "-c", "3d_fullres", "-o", args["output_model_file"]
                       ]
                cmd.extend(unknown_arguments)
                subprocess.run(cmd)
            else:
                cmd = ["nnUNetv2_find_best_configuration", data["Task_ID"], "-c", "3d_fullres"]
                cmd.extend(unknown_arguments)
                subprocess.run(cmd)
                cmd = ["nnUNetv2_export_model_to_zip", "-d", data["Task_ID"], "--exp_cv_preds", "-c", "3d_fullres",
                       "-o",
                       args["output_model_file"]]
                cmd.extend(unknown_arguments)
                subprocess.run(
                    cmd)
        else:
            subprocess.run(arguments)


if __name__ == "__main__":
    main()
