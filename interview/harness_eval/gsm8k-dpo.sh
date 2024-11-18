#!/bin/bash
#
lm_eval --model vllm --model_args pretrained=models/gsm8k-dpo \
  --tasks gsm8k,gsm8k_cot,gsm8k_cot_zeroshot \
    --device cuda:0 \
    --wandb_args project=gsm8k_dashboard,name=gsm8k_dpo\
    --log_samples \
    --output_path output/gsm8k-dpo \
    --batch_size auto

