#!/bin/bash

# Set the PYTHONPATH to ensure the workspace folder is included
export PYTHONPATH=".:src/"

# Run the Python script with the specified arguments
python3 src/run_language_modeling.py \
  --output_dir eval_vanilla_t5 \
  --do_only_eval \
  --config_name t5-base \
  --per_device_eval_batch_size 32 \
  --seed 42 \
  --remove_unused_columns False \
  --number_encoding rt \
  --batch_eval_metrics \
  --report_to none \
  --model_name_or_path train_run_1/checkpoint-2500 \
