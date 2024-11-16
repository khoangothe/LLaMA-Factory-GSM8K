#!/bin/bash

python interview/utils/preprocess_gsm8k.py interview/raw/gsm8k/train-00000-of-00001.parquet data/gsm8k.json
python interview/utils/preprocess_gsm8k_cot.py interview/raw/gsm8k/train-00000-of-00001.parquet data/gsm8k_cot.json
