#!/bin/bash
#
lm_eval --model vllm --model_args pretrained=models/gsm8k-cot-combine-qwen \
  --tasks gsm8k,gsm8k_cot,gsm8k_cot_zeroshot \
    --device cuda:0 \
    --wandb_args project=gsm8k_dashboard_2,name=gsm8k_cot_combine_sft\
    --log_samples \
    --output_path output/gsm8k-cot-combine-qwen \
    --batch_size auto
