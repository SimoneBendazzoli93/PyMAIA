#!/usr/bin/env python
import hpccm

hpccm.config.set_working_directory('/opt/code/PyMAIA')

Stage0 += baseimage(image='')
Stage0 += gnu()
Stage0 += apt_get(ospackages=['libgl1'])
Stage0 += copy(src=[
    'requirements.txt',
    'setup.cfg',
    'versioneer.py',
    '.gitattributes',
    'setup.py',
    'MANIFEST.in',
    'README.md'
], dest='/opt/code/PyMAIA', _mkdir=True)
Stage0 += copy(src=[
    './PyMAIA/'], dest='/opt/code/PyMAIA/PyMAIA/')
Stage0 += copy(src=[
    './PyMAIA_scripts/'], dest='/opt/code/PyMAIA/PyMAIA_scripts/')
Stage0 += workdir(directory="/opt/code/PyMAIA")
Stage0 += pip(ospackages=[""], packages=["/opt/code/PyMAIA"])
Stage0 += runscript(commands=[''])
