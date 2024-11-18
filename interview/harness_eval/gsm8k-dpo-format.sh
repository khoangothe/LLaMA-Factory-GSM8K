#!/bin/bash
#
lm_eval --model vllm --model_args pretrained=models/gsm8k-dpo-format \
  --tasks gsm8k,gsm8k_cot \
    --device cuda:0 \
    --wandb_args project=gsm8k_dashboard,name=gsm8k_dpo_format \
    --log_samples \
    --output_path output/gsm8k-dpo-format \
    --batch_size auto


