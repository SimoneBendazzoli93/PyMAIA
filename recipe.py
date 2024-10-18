#!/usr/bin/env python

#rocm/pytorch:rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1
#rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_1.13.1

Stage0 += baseimage(image='nvcr.io/nvidia/pytorch:23.04-py3')
Stage0 += gnu()
Stage0 += pip(ospackages=[""], packages=["monai[nibabel, skimage, scipy, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, pandas, mlflow, matplotlib, pydicom]",
                                         "fire",
                                         "jupyter",
                                         "einops",
                                         "jupyterlab",
                                         "TotalSegmentator",
                                         "pymaia-learn",
                                         #"lightning==2.1.4"
                                         ])

Stage0 += runscript(commands=[''])
