#!/usr/bin/env bash

export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

python -m pip install opencv-python-headless
python -m pip install opencv-contrib-python-headless
python setup.py install --single-version-externally-managed --record=record.txt
