#!/bin/bash
#
lm_eval --model vllm --model_args pretrained=models/gsmic \
  --tasks gsm8k,gsm8k_cot \
    --device cuda:0 \
    --wandb_args project=gsm8k_dashboard,name=gsmic\
    --log_samples \
    --output_path output/gsmic \
    --batch_size auto
