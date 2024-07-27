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
            "PyMAIA_convert_DICOM_dataset_to_NIFTI_dataset = PyMAIA_scripts.PyMAIA_convert_DICOM_dataset_to_NIFTI_dataset:main",
            "PyMAIA_run_pipeline_from_file = PyMAIA_scripts.PyMAIA_run_pipeline_from_file:main",
            "PyMAIA_convert_NIFTI_predictions_to_DICOM_SEG = PyMAIA_scripts.PyMAIA_convert_NIFTI_predictions_to_DICOM_SEG:main",
            "PyMAIA_convert_semantic_to_instance_segmentation = PyMAIA_scripts.PyMAIA_convert_semantic_to_instance_segmentation:main",
            "PyMAIA_create_subset = PyMAIA_scripts.PyMAIA_create_subset:main",
            "PyMAIA_order_data_folder = PyMAIA_scripts.PyMAIA_order_data_folder:main",
            "PyMAIA_downsample_modality = PyMAIA_scripts.PyMAIA_downsample_modality:main",

            "nndet_create_pipeline = PyMAIA_scripts.nndet_create_pipeline:main",
            "nndet_prepare_data_folder = PyMAIA_scripts.nndet_prepare_data_folder:main",
            "nndet_run_preprocessing = PyMAIA_scripts.nndet_run_preprocessing:main",
            "nndet_run_training = PyMAIA_scripts.nndet_run_training:main",
            "nndet_extract_experiment_predictions = PyMAIA_scripts.nndet_extract_experiment_predictions:main",
            "nndet_compute_metric_results = PyMAIA_scripts.nndet_compute_metric_results:main",

            "nnunet_prepare_data_folder = PyMAIA_scripts.nnunet_prepare_data_folder:main",
            "nnunet_run_preprocessing = PyMAIA_scripts.nnunet_run_preprocessing:main",
            "nnunet_run_plan_and_preprocessing = PyMAIA_scripts.nnunet_run_plan_and_preprocessing:main",
            "nnunet_run_training = PyMAIA_scripts.nnunet_run_training:main",
            "nnunet_create_pipeline = PyMAIA_scripts.nnunet_create_pipeline:main",
        ],
    },
    keywords=["deep learning", "image segmentation", "medical image analysis", "medical image segmentation",
              "object detection"],
    # PyMAIA_scripts=glob.glob("PyMAIA_scripts/*"),
)
