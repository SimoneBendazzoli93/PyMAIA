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
    Run ``nndet_train`` command to start nnDetection training for the specified fold.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file ../CONFIG_FILE.json --run-fold 0
        {filename} --config-file ../CONFIG_FILE.json --run-fold 0 --resume-training y
    """.format(  # noqa: E501
        filename=Path(__file__).stem
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
        "--train-config",
        type=str,
        required=True,
        help="File path for the nnDetection Training YAML configuration dictionary, used to configure the nnDetection Experiment.",
    )

    pars.add_argument(
        "--run-fold",
        type=str,
        default="0",
        required=False,
        help="Index indicating which fold to run. Default: ``0``",
    )

    pars.add_argument(
        "--n-folds",
        type=str,
        default="5",
        required=False,
        help="Number of Folds for final consolidation. Default: ``5``",
    )

    pars.add_argument(
        "--model",
        type=str,
        default="RetinaUNetV001_D3V001_3d",
        required=False,
        help="nnDetection Model used for sweeping and consolidation. Default: ``RetinaUNetV001_D3V001_3d``",
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
            "nndet_train",
            data["Task_ID"],
            "-o",
            "exp.fold={}".format(args["run_fold"]),
            "--train-config",
            args["train_config"]
        ]

        if args["resume_training"]:
            arguments.append("-o")
            arguments.append("train.mode=resume")

        arguments.extend(unknown_arguments)

        if not "N_THREADS" in os.environ:
            os.environ["N_THREADS"] = str(os.cpu_count())
        n_workers = "1"
        if args["n_workers"] is None:
            if "N_THREADS" in os.environ is not None:
                n_workers = str(os.environ["N_THREADS"])
        else:
            n_workers = str(args["n_workers"])

        os.environ["det_data"] = data["base_folder"]
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["det_num_threads"] = n_workers
        os.environ["nnUNet_def_n_proc"] = n_workers
        os.environ["det_models"] = data["results_folder"]
        os.environ["global_preprocessing_folder"] = data["preprocessing_folder"]

        if int(args["run_fold"]) >= 0:
            subprocess.run(arguments)
            subprocess.run(["nndet_sweep", data["Task_ID"], args["model"], str(args["run_fold"])])

        else:
            subprocess.run(
                ["nndet_consolidate", data["Task_ID"], args["model"], "--sweep_boxes", "--num_folds", args["n_folds"]])
            subprocess.run(
                ["nndet_seg2nii", data["Task_ID"], args["model"], "--fold", str(args["run_fold"])])
            subprocess.run(
                ["nndet_boxes2nii", data["Task_ID"], args["model"], "--fold", str(args["run_fold"])])

            subprocess.run(
                ["nndet_eval", data["Task_ID"], args["model"], str(args["run_fold"]), "--boxes", "--seg",
                 "--analyze_boxes"])


if __name__ == "__main__":
    main()
