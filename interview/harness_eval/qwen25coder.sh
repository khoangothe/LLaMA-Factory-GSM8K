#!/bin/bash
#
lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-Coder-7B \
  --tasks gsm8k,gsm8k_cot,gsm8k_cot_self_consistency,gsm8k_cot_zeroshot \
    --device cuda:0 \
    --wandb_args project=gsm8k_dashboard_2,name=qwen25coder\
    --log_samples \
    --output_path output/qwen25coder \
    --batch_size auto

