#!/bin/bash
#
lm_eval --model vllm --model_args pretrained=models/gsm8k-qwen \
  --tasks gsm8k,gsm8k_cot \
    --device cuda:0 \
    --wandb_args project=gsm8k_dashboard,name=gsm8k_sft\
    --log_samples \
    --output_path output/gsm8k-qwen \
    --batch_size auto