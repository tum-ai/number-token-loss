#!/bin/bash

# Set the PYTHONPATH to ensure the workspace folder is included
export PYTHONPATH=".:src/"

# Run the Python script with the specified arguments
python3 src/run_language_modeling.py \
  --output_dir train_run_vanilla_t5 \
  --config_name t5-base \
  --learning_rate 5e-6 \
  --lr_scheduler_type reduce_lr_on_plateau \
  --lr_scheduler_kwargs "{\"factor\": 0.5, \"patience\": 5}" \
  --num_train_epochs 2000 \
  --save_total_limit 2 \
  --save_steps 2000 \
  --eval_steps 2000 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --seed 42 \
  --logging_steps 100 \
  --remove_unused_columns False \
  --number_encoding none \
  --load_best_model_at_end True \
  --eval_strategy steps \
  --batch_eval_metrics \
  --metric_for_best_model loss \
  --greater_is_better False \
  --report_to wandb \
# --model_name_or_path train_run_vanilla_t5/checkpoint-36000 \
#  --eval_on_start \
