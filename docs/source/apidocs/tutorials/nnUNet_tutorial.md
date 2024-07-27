# nnUNet Tutorial

In this tutorial, we are going through the basic steps to run an nnUNet V2 experiment.

## 1. Data folder preparation
Your dataset folder should be formatted in such a way that the nnUNet scripts can successfully detect all the dataset subjects and the corresponding NIFTI files.

NOTE 1: nnUNet requires all the volumes and segmentation masks to be in the NIFTI format. Use the script `PyMAIA_convert_DICOM_dataset_to_NIFTI_dataset` to convert a DICOM dataset into a NIFTI format.


The standard dataset folder structure is the following:

    [Dataset_folder]
        [Subject_0]
            - Subject_0_image0.nii.gz    # Subject_0 modality 0
            - Subject_0_image1.nii.gz    # Subject_0 modality 1
            - Subject_0_mask.nii.gz      # Subject_0 semantic segmentation mask
        [Subject_1]
            - Subject_1_image0.nii.gz    # Subject_1 modality 0
            - Subject_1_image1.nii.gz    # Subject_1 modality 1
            - Subject_1_mask.nii.gz      # Subject_1 semantic segmentation mask
        ...

## 2. Create Pipeline File

After organizing the Dataset folder according to the standard format, you are ready to generate a *Pipeline file*, that
will be later used to run all the experiment steps (Data Preparation, Preprocessing and Training).
In order to do so, you first need to prepare a JSON configuration file, specifying all the parameters and attributes for
the experiment, as described in the section `Configs -> nnUNet config`. Some default config files are made
available with the **PyMAIA** package, in the **configs** section.
In addition, *ROOT_FOLDER* should be set as an environment variable (this will be the base folder for all the
experiments that will be created). Check it with:
```
echo $ROOT_FOLDER
```
and, if not present, run:
```
export ROOT_FOLDER=/YOUR/PATH/TO/Experiments
```
To generate the *Pipeline file*, run:
```
nnunet_create_pipeline --input-data-folder /PATH/TO/Dataset_folder --config-file /YOUR/CONFIG_FILE.json --task-ID 000
```
with `task_id` representing an unique identifier number for the experiment. 

Optionally, you can set the split ratio (in %, set a value between 0-100) between train and test data:
```
nnunet_create_pipeline --input-data-folder /PATH/TO/Dataset_folder --config-file /YOUR/CONFIG_FILE.json --task-ID 000 --test-split 20
```

By default, 80% of the available data will be dedicated for cross-fold validation, while 20% will be reserved as testing
set.

The *Pipeline file* will be available, as a *txt* file, in `ROOT_FOLDER/experiment_folder`, with *experiment_folder* as
indicated in the config file with the  `"Experiment Name"` attribute.

The argument ``--extra_training_config`` in the ``nnunet_create_pipeline`` command is used to provide extra arguments for
the ``nnunet_run_training`` step in the pipeline.
For example, the file **extra_training_config** ( see below ), is passed as argument to set the pretrained weights in the ``nnUNetv2_train`` script:

```json
{
  "-pretrained_weights": "<path_to_pretrained_weights>"
}
```

## 3. Run Pipeline
Once the *Pipeline file* is created, you are ready to run your nnUNet experiment, either with the available script, or by just copying/pasting the single commands from the *txt* file into your shell.
To run the full pipeline with the **PyMAIA** script:
```
PyMAIA_run_pipeline_from_file --file /YOUR/PIPELINE_FILE.txt
```
