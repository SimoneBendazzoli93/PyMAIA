#!/usr/bin/env python

import hashlib
import importlib.resources
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import OrderedDict
from datetime import datetime
from textwrap import dedent

import k8s_DP.configs
from k8s_DP.evaluation.evaluator import compute_metrics_for_folder, order_scores_with_means
from k8s_DP.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args

DESC = dedent(
    """
    Run segmentation metrics evaluation for a set of volumes and store the results in a JSON file.
    ``Labels`` and ``evaluation metrics`` are specified in the configuration file. If ``evaluation metrics`` are not specified,
    default metrics are evaluated.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --ground-truth-folder /path/to/gt_folder --prediction-folder /path/to/pred_folder --config-file /path/to/config_file.json
    """.format(  # noqa: E501
        filename=os.path.basename(__file__)
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--ground-truth-folder",
        type=str,
        required=True,
        help="Folder path containing the Ground Truth volumes",
    )

    pars.add_argument(
        "--prediction-folder",
        type=str,
        required=True,
        help="Folder path containing the prediction volumes",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="File path for the configuration dictionary, used to retrieve experiment settings ",
    )

    add_verbosity_options_to_argparser(pars)
    return pars


if __name__ == "__main__":
    parser = get_arg_parser()
    args = vars(parser.parse_args())

    logger = get_logger(
        name=os.path.basename(__file__),
        level=log_lvl_from_verbosity_args(args),
    )

    try:
        with open(args["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(k8s_DP.configs, args["config_file"]) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    file_suffix = config_dict["FileExtension"]
    label_dict = config_dict["label_dict"]
    label_dict.pop("0", None)

    labels = list(label_dict.keys())

    all_res = compute_metrics_for_folder(args["ground_truth_folder"], args["prediction_folder"], labels, file_suffix)
    all_scores = order_scores_with_means(all_res)

    json_dict = OrderedDict()
    json_dict["name"] = config_dict["DatasetName"]
    timestamp = datetime.today()
    json_dict["timestamp"] = str(timestamp)
    json_dict["task"] = "Task" + config_dict["Task_ID"] + "_" + config_dict["Task_Name"]
    json_dict["results"] = all_scores
    json_dict["id"] = hashlib.md5(json.dumps(json_dict).encode("utf-8")).hexdigest()[:12]
    with open(os.path.join(args["prediction_folder"], "summary.json"), "w") as outfile:
        json.dump(json_dict, outfile)