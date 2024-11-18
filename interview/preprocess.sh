#!/bin/bash

python interview/utils/preprocess_gsm8k.py interview/raw/gsm8k/train-00000-of-00001.parquet data/gsm8k.json
python interview/utils/preprocess_gsm8k_cot.py interview/raw/gsm8k/train-00000-of-00001.parquet data/gsm8k_cot.json
python interview/utils/preprocess_gsmic.py interview/raw/gsmic/GSM-IC_2step.json  --source data/gsm8k.json  --output data/gsmic2.json
python interview/utils/preprocess_gsmic.py interview/raw/gsmic/GSM-IC_mstep.json  --source data/gsm8k.json  --output data/gsmicm.json
#Boxed
python interview/utils/preprocess_gsm8k_boxed.py interview/raw/gsm8k/train-00000-of-00001.parquet data/gsm8k_boxed.json
