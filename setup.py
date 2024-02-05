import os

import setuptools
from setuptools import setup

import versioneer


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


setup(
    version=versioneer.get_version(),
    packages=setuptools.find_packages(),
    package_data={
        "": ["configs/*.yml", "configs/*.json"],
    },
    zip_safe=False,
    data_files=[('', ["requirements.txt"]), ],
    # package_dir={"": "src"},
    # install_requires=resolve_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt")),

    cmdclass=versioneer.get_cmdclass(),
    entry_points={
        "console_scripts": [
            "Hive_convert_DICOM_dataset_to_NIFTI_dataset = scripts.Hive_convert_DICOM_dataset_to_NIFTI_dataset:main",
            "Hive_run_pipeline_from_file = scripts.Hive_run_pipeline_from_file:main",
            "Hive_convert_NIFTI_predictions_to_DICOM_SEG = scripts.Hive_convert_NIFTI_predictions_to_DICOM_SEG:main",
            "Hive_create_subset = scripts.Hive_create_subset:main",
            "nndet_create_pipeline = scripts.nndet_create_pipeline:main",
            "nndet_prepare_data_folder = scripts.nndet_prepare_data_folder:main",
            "nndet_run_preprocessing = scripts.nndet_run_preprocessing:main",
            "nndet_run_training = scripts.nndet_run_training:main",
            "Hive_convert_semantic_to_instance_segmentation = scripts.Hive_convert_semantic_to_instance_segmentation:main",
            "Hive_extract_experiment_predictions = scripts.Hive_extract_experiment_predictions:main",
            "nndet_compute_metric_results = scripts.nndet_compute_metric_results:main",
            "Hive_order_data_folder = scripts.Hive_order_data_folder:main",
            "nnunet_prepare_data_folder = scripts.nnunet_prepare_data_folder:main",
            "nnunet_run_preprocessing = scripts.nnunet_run_preprocessing:main",
            "nnunet_run_training = scripts.nnunet_run_training:main",
        ],
    },
    keywords=["deep learning", "image segmentation", "medical image analysis", "medical image segmentation",
              "object detection"],
    # scripts=glob.glob("scripts/*"),
)
