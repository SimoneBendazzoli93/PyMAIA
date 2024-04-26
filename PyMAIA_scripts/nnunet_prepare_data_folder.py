#!/usr/bin/env python

import csv
import datetime
import importlib.resources
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import numpy as np
from sklearn.model_selection import KFold

import PyMAIA.configs
from PyMAIA.utils.file_utils import (
    create_nnunet_data_folder_tree,
    split_dataset,
    copy_data_to_dataset_folder,
    copy_data_from_dict_to_dataset_folder,
    save_config_json,
    generate_dataset_json,
)
from PyMAIA.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Prepare Dataset folder according to the nnUNet specifications, creating and populating the subfolders ``imagesTr``,
    ``labelsTr``, ``imagesTs`` and ``labelsTs``. In addition, a JSON instance configuration file (as required by nnUNet)
    for each label mask is generated, alongside a summary of the train/test split of the dataset.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} -i /PATH/TO/DATA_FOLDER --task-ID 000 --task-name Example --config-file Example_config.json
        {filename} -i /PATH/TO/DATA_FOLDER --task-ID 000 --task-name Example --config-file Example_config.json --test-split 30
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )

    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(PyMAIA.configs, arguments["config_file"]) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    os.environ["raw_data_base"] = str(
        Path(os.environ["ROOT_FOLDER"]).joinpath(
            config_dict["Experiment Name"], config_dict["Experiment Name"] + "_base"
        )
    )
    os.environ["preprocessed_folder"] = str(
        Path(os.environ["ROOT_FOLDER"]).joinpath(
            config_dict["Experiment Name"], config_dict["Experiment Name"] + "_preprocess"
        )
    )
    os.environ["RESULTS_FOLDER"] = str(
        Path(os.environ["ROOT_FOLDER"]).joinpath(
            config_dict["Experiment Name"], config_dict["Experiment Name"] + "_results"
        )
    )
    try:
        dataset_path = str(
            Path(os.environ["raw_data_base"]).joinpath(
                "nnUNet_raw_data",
                "Dataset" + arguments["task_ID"] + "_" + arguments["task_name"],
            )
        )

    except KeyError:
        logger.error("raw_data_base is not set as environment variable")
        return 1

    create_nnunet_data_folder_tree(
        os.environ["raw_data_base"],
        arguments["task_name"],
        arguments["task_ID"],
    )

    if Path(arguments["input_data_folder"]).is_file() and arguments["input_data_folder"].endswith(".json"):
        with open(arguments["input_data_folder"]) as json_file:
            dataset_dict = json.load(json_file)
            train_dataset = dataset_dict["train"]
            train_dataset = [Path(train_id[list(config_dict["Modalities"].values())[0]]).name[
                             :-len(list(config_dict["Modalities"].keys())[0])] for train_id in train_dataset]
            if "test" in dataset_dict:
                test_dataset = dataset_dict["test"]
                test_dataset = [Path(test_id[list(config_dict["Modalities"].values())[0]]).name[
                                :-len(list(config_dict["Modalities"].keys())[0])] for test_id in test_dataset]

            else:
                test_dataset = []
    else:
        train_dataset, test_dataset = split_dataset(arguments["input_data_folder"], arguments["test_split"],
                                                    config_dict["Seed"])

    dataset_split = []
    for test_subject in test_dataset:
        dataset_split_dict = {"Subject": test_subject, "Split": "Testing"}
        dataset_split.append(dataset_split_dict)

    train_dataset_sorted = np.sort(train_dataset)
    kfold = KFold(n_splits=config_dict["n_folds"], shuffle=True, random_state=config_dict["Seed"])
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_dataset_sorted)):
        for test in test_idx:
            dataset_split_dict = {"Subject": train_dataset_sorted[test], "Split": "Validation_fold_{}".format(i)}
            dataset_split.append(dataset_split_dict)

    dataset_split_summary = Path(os.environ["ROOT_FOLDER"]).joinpath(
        config_dict["Experiment Name"], "dataset_split.csv"
    )

    with open(dataset_split_summary, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Subject", "Split"])
        writer.writeheader()
        for data in dataset_split:
            writer.writerow(data)

    if Path(arguments["input_data_folder"]).is_file() and arguments["input_data_folder"].endswith(".json"):
        config_dict["dataset_folder"] = str(Path(arguments["input_data_folder"]))
    else:
        config_dict["dataset_folder"] = str(Path(arguments["input_data_folder"]))

    if Path(arguments["input_data_folder"]).is_file() and arguments["input_data_folder"].endswith(".json"):
        copy_data_from_dict_to_dataset_folder(
            arguments["input_data_folder"],
            "train",
            Path(dataset_path).joinpath("imagesTr"),
            config_dict,
            Path(dataset_path).joinpath("labelsTr"),

        )
        copy_data_from_dict_to_dataset_folder(
            arguments["input_data_folder"],
            "test",
            Path(dataset_path).joinpath("imagesTs"),
            config_dict,
            Path(dataset_path).joinpath("labelsTs"),
        )
    else:
        copy_data_to_dataset_folder(
            arguments["input_data_folder"],
            train_dataset,
            Path(dataset_path).joinpath("imagesTr"),
            config_dict,
            Path(dataset_path).joinpath("labelsTr")
        )
        copy_data_to_dataset_folder(
            arguments["input_data_folder"],
            test_dataset,
            Path(dataset_path).joinpath("imagesTs"),
            config_dict,
            Path(dataset_path).joinpath("labelsTs")
        )

    if "RegionClassOrder" in config_dict:
        region_class_order = config_dict["RegionClassOrder"]
    else:
        region_class_order = None
    generate_dataset_json(
        str(
            Path(dataset_path).joinpath("dataset.json"
                                        )
        ),
        train_dataset,
        test_dataset,
        list(config_dict["Modalities"].values()),
        config_dict["label_dict"],
        task_name="Dataset{}_{}".format(arguments["task_ID"], arguments["task_name"]),
        file_extension=config_dict["FileExtension"],
        nnunet_format=True,
        region_class_order=region_class_order

    )

    config_dict["Task_ID"] = arguments["task_ID"]
    config_dict["Task_Name"] = arguments["task_name"]
    config_dict["train_test_split"] = arguments["test_split"]
    config_dict["base_folder"] = os.environ["raw_data_base"]

    output_json_basename = "Dataset" + arguments["task_ID"] + "_" + config_dict["Experiment Name"] + ".json"

    try:
        config_dict["results_folder"] = os.environ["RESULTS_FOLDER"]
        Path(config_dict["results_folder"]).mkdir(parents=True, exist_ok=True)
    except KeyError:
        logger.warning(
            "RESULTS_FOLDER is not set as environment variable, {} is not saved".format(output_json_basename))
        return 1
    try:
        config_dict["preprocessing_folder"] = os.environ["preprocessed_folder"]
        Path(config_dict["preprocessing_folder"]).mkdir(parents=True, exist_ok=True)

    except KeyError:
        logger.warning(
            "preprocessed_folder is not set as environment variable, not saved in {}".format(output_json_basename)
            # noqa E501
        )
    save_config_json(config_dict, str(Path(config_dict["results_folder"]).joinpath(output_json_basename)))


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "-i",
        "--input-data-folder",
        type=str,
        required=True,
        help="Input Dataset folder",
    )

    pars.add_argument(
        "--task-ID",
        type=str,
        default="100",
        help="Task ID used in the folder path tree creation (Default: 100)",
    )

    pars.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task Name used in the folder path tree creation.",  # noqa E501
    )

    pars.add_argument(
        "--test-split",
        type=int,
        choices=range(0, 101),
        metavar="[0-100]",
        default=20,
        help="Split value ( in %% ) to create Test set from Dataset (Default: 20)",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration JSON file with experiment and dataset parameters.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


if __name__ == "__main__":
    main()
