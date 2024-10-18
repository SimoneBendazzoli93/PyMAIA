# PyMAIA

<p align="center">
<img src="https://raw.githubusercontent.com/SimoneBendazzoli93/PyMAIA/main/images/MAI_A_logo.png" width="50%" alt='PyMAIA'>
</p>

[![Documentation Status](https://readthedocs.org/projects/pymaia/badge/?version=latest)](https://pymaia.readthedocs.io/en/latest/?badge=latest)
![Version](https://img.shields.io/badge/PyMAIA-v1.1-blue)
[![License](https://img.shields.io/badge/license-GPL%203.0-green.svg)](https://opensource.org/licenses/GPL-3.0)
![Python](https://img.shields.io/badge/python-3.8+-orange)
![CUDA](https://img.shields.io/badge/CUDA-10.1%2F10.2%2F11.0-green)

![GitHub Release Date - Published_At](https://img.shields.io/github/release-date/simonebendazzoli93/PyMAIA?logo=github)
![GitHub contributors](https://img.shields.io/github/contributors/simonebendazzoli93/PyMAIA?logo=github)
![GitHub top language](https://img.shields.io/github/languages/top/simonebendazzoli93/PyMAIA?logo=github)
![GitHub language count](https://img.shields.io/github/languages/count/simonebendazzoli93/PyMAIA?logo=github)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/simonebendazzoli93/PyMAIA/publish_release.yaml?logo=github)
![GitHub all releases](https://img.shields.io/github/downloads/simonebendazzoli93/PyMAIA/total?logo=github)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pymaia-learn?logo=pypi)
![GitHub](https://img.shields.io/github/license/simonebendazzoli93/PyMAIA?logo=github)
![PyPI - License](https://img.shields.io/pypi/l/pymaia-learn?logo=pypi)

![Conda](https://img.shields.io/conda/pn/MAIA-KTH/pymaia-learn?logo=anaconda)
![Conda](https://img.shields.io/conda/v/MAIA-KTH/pymaia-learn?logo=anaconda)

![GitHub repo size](https://img.shields.io/github/repo-size/simonebendazzoli93/PyMAIA?logo=github)
![GitHub release (with filter)](https://img.shields.io/github/v/release/simonebendazzoli93/PyMAIA?logo=github)
![PyPI](https://img.shields.io/pypi/v/pymaia-learn?logo=pypi)

## What is PyMAIA?

Hive is a Python package to support Deep Learning data preparation, pre-processing. training, result visualization and
model deployment across different frameworks ([nnUNet](https://github.com/MIC-DKFZ/nnUNet)
, [nnDetection](https://github.com/MIC-DKFZ/nnDetection), [MONAI](https://monai.io/) ).

## Local Environment Setup

To install the package, run the following commands:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
or conda install cudatoolkit cuda-version=11
pip install nnunetv2/nndetection
pip install pymaia-learn
```

More information can be found in the [documentation](https://pymaia.readthedocs.io/en/latest/).

## Tutorials

- [nnUNet Tutorial](https://pymaia.readthedocs.io/en/latest/tutorials/nnUNet_tutorial.html)
- [nnDetection Tutorial](https://pymaia.readthedocs.io/en/latest/tutorials/nnDetection_tutorial.html)

## Docker and Singularity
PyMAIA can be run in a containerized environment using Docker or Singularity. To
create the PyMAIA image, you can use [HPPCM](https://github.com/NVIDIA/hpc-container-maker), a tool to create container
images for HPC applications from given recipes.

```bash
pip install hpccm

hpccm --recipe recipe.py --format singularity > PyMAIA.def
singularity build PyMAIA.sif PyMAIA.def

hpccm --recipe recipe.py --format docker > Dockerfile
docker build -t pymaia .
```
