#!/bin/bash
#
lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-Coder-7B \
  --tasks gsm8k,gsm8k_cot \
    --device cuda:0 \
    --wandb_args project=gsm8k_dashboard,name=gsm8k_sft\
    --batch_size auto

