#!/bin/bash

# Set the PYTHONPATH to ensure the workspace folder is included
export PYTHONPATH=".:src/"

# Run the Python script with the specified arguments
python3 src/run_language_modeling.py \
  --output_dir train_run_1 \
  --config_name t5-base \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --save_total_limit 2 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --per_device_train_batch_size 16 \
  --seed 42 \
  --logging_steps 5 \
  --remove_unused_columns False \
  --number_encoding rt \
  --load_best_model_at_end True \
  --eval_strategy steps \
  --metric_for_best_model loss \
  --greater_is_better False \
  --report_to wandb
