#!/usr/bin/env bash

export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

pip install opencv-python-headless
pip install opencv-contrib-python-headless
pip install numpy==1.19.5
pip install attrs==21.2.0
pip install wheel==0.35

python setup.py install --single-version-externally-managed --record=record.txt
