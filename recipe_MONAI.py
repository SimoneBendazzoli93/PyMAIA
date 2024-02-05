#!/usr/bin/env python

Stage0 += baseimage(image='rocm/pytorch:rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1')
Stage0 += gnu()
Stage0 += pip(ospackages=[""], packages=["monai", "fire"])
Stage0 += runscript(commands=[''])
