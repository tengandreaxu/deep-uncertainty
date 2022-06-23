#!/bin/bash

module load python/3.7.7

python -m venv bert_venv

source bert_venv/bin/activate
pip install -r requirements.txt
