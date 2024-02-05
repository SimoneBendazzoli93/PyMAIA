import json
import os
from pathlib import Path
from typing import Union, Optional

import torch
from torch.backends import cudnn


def get_nnunet_trainer(dataset_name_or_id: Union[str, int],
                       configuration: str, fold: Union[int, str],
                       hive_config_file: str,  # To set env variables
                       trainer_class_name: str = 'nnUNetTrainer',
                       plans_identifier: str = 'nnUNetPlans',
                       pretrained_weights: Optional[str] = None,
                       num_gpus: int = 1,
                       use_compressed_data: bool = False,
                       export_validation_probabilities: bool = False,
                       continue_training: bool = False,
                       only_run_validation: bool = False,
                       disable_checkpointing: bool = False,
                       val_with_best: bool = False,
                       device: torch.device = torch.device(
                           'cuda')):  # From nnUNet/nnunetv2/run/run_training.py#run_training

    ## Block Added
    with open(hive_config_file, "r") as f:
        hive_config_dict = json.load(f)

    os.environ["nnUNet_raw"] = str(Path(hive_config_dict["base_folder"]).joinpath("nnUNet_raw_data"))
    os.environ["nnUNet_preprocessed"] = hive_config_dict["preprocessing_folder"]
    os.environ["nnUNet_results"] = hive_config_dict["results_folder"]

    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
    ##

    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if int(num_gpus) > 1:
        ...  # Disable for now
    else:
        nnunet_trainer = get_trainer_from_args(str(dataset_name_or_id), configuration, fold, trainer_class_name,
                                               plans_identifier, use_compressed_data, device=device)

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (
                continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)
        nnunet_trainer.on_train_start()  # To Initialize Trainer
        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        # Skip Training and Validation Phase

        return nnunet_trainer


def prepare_nnunet_batch(batch, device, non_blocking):
    data = batch["data"].to(device, non_blocking=non_blocking)
    if isinstance(batch["target"], list):
        target = [i.to(device, non_blocking=non_blocking) for i in batch["target"]]
    else:
        target = batch["target"].to(device, non_blocking=non_blocking)
    return data, target
