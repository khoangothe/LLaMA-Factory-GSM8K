#!/bin/bash
#
lm_eval --model vllm --model_args pretrained=models/gsmic \
  --tasks gsm8k,gsm8k_cot,gsm8k_cot_self_consistency,gsm8k_cot_zeroshot \
    --device cuda:0 \
    --wandb_args project=gsm8k_dashboard_2,name=gsmic\
    --log_samples \
    --output_path output/gsmic \
    --batch_size auto
