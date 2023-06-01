@echo off

set PIP_NO_INDEX=False
set PIP_NO_DEPENDENCIES=False
set PIP_IGNORE_INSTALLED=False

python -m pip install opencv-python-headless
python -m pip install opencv-contrib-python-headless
python setup.py install --single-version-externally-managed --record=record.txt
