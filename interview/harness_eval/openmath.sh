#!/bin/bash
#
lm_eval --model vllm --model_args pretrained=models/openmath \
  --tasks gsm8k,gsm8k_cot,gsm8k_cot_zeroshot \
    --device cuda:0 \
    --wandb_args project=gsm8k_dashboard_2,name=openmath \
    --log_samples \
    --output_path output/openmath \
    --batch_size auto
