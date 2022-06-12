#!/bin/bash
set -x;
python3 setup.py sdist;
python3 -m pip install dist/deep-uncertainty-0.0.0.tar.gz;