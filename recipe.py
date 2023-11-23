#!/usr/bin/env python
import hpccm

hpccm.config.set_working_directory('/opt/code/Hive')

Stage0 += baseimage(image='')
Stage0 += gnu()

Stage0 += copy(src=[
    'requirements.txt',
    # 'MLproject',
    'setup.cfg',
    'versioneer.py',
    '.gitattributes',
    'setup.py',
    # 'main.py',
    # 'ensemble_main.py',
    'MANIFEST.in',
    'README.md'
], dest='/opt/code/Hive', _mkdir=True)
Stage0 += copy(src=[
    './docs/source/apidocs/configs/nnDet_config_template.json'],
    dest='/opt/code/Hive/docs/source/apidocs/configs/nnDet_config_template.json', _mkdir=True)
Stage0 += copy(src=[
    './Hive/'], dest='/opt/code/Hive/Hive/')
Stage0 += copy(src=[
    './scripts/'], dest='/opt/code/Hive/scripts/')
Stage0 += workdir(directory="/opt/code/Hive")
Stage0 += pip(ospackages=[""], packages=["/opt/code/Hive"])
Stage0 += runscript(commands=[''])
