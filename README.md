# Regress, Don’t Guess – A Regression-like Loss on Number Tokens for Language Models
![ntl-image.jpg](resources%2Fntl-image.jpg)

Introducing "Number Token Loss" (NTL) for language models to improve numerical reasoning by using regression-based loss functions that account for the proximity of numbers, achieving better performance on math tasks without increasing computational overhead.
## Setup

### Via Python
- Requires Python 3.9 or higher
- Install the required packages
    ```bash
    pip install -r requirements.txt
    ```
- Log into wandb in the terminal
    ```
    wandb login
    ```
  Enter you username and auth token (wandb.ai/auth)

### Via Docker

- Start a docker container with the transformers image
    ```bash
    docker run --name container_name --gpus <device_number> -v /home/students/code/<name>/path_to_code:/app/data -it huggingface/transformers-pytorch-gpu
  ```
- Inside the container, interactively set the transformers library to version  4.42.4 and install wandb and hydra
    ```bash
    pip install transformers==4.42.4
    pip install wandb
    pip install hydra-core
    ```
- Log into wandb in the terminal 
    ```
    wandb login
    ```
    Enter you username and auth token (wandb.ai/auth)

## Training
- The main script is [src.run_language_modeling.py](src%2Frun_language_modeling.py).
  - The Arguments are configured via Hydra (Yadan, Omry. *Hydra - A framework for elegantly configuring complex applications*. Github, 2019. Available at: [https://github.com/facebookresearch/hydra](https://github.com/facebookresearch/hydra).)
  - Therefore the script can be called via 
    ```bash
    export PYTHONPATH=".:src/"
    python src/run_language_modeling.py dataset_args=<gsm8k or mathematics_dataset, default mathematics_dataset>
                                        model_args=<rt, rt_ntl, vanilla_t5, vanilla_t5_ntl, xval>
                                        training_args=<eval or train>
    ```
  - You can override the default config via the command line, e.g. 
    ```bash
    python src/run_language_modeling.py model_args=vanilla_t5 training_args=train training_args.per_device_train_batch_size=8
    ```
    or override them in the [config/run_specific_config/config.yaml](config%2Frun_specific_config%2Fconfig.yaml) file.
  - For debugging, you can use the [config/run_specific_config/debug_config.yaml](config%2Frun_specific_config%2Fdebug_config.yaml) file via
    ```bash
    python src/run_language_modeling.py model_args=vanilla_t5 training_args=train run_specific_config@_global_=debug_config
    ```
  - For running in nohup mode, use
    ```bash
    nohup python src/run_language_modeling.py dataset_args=mathematics_dataset model_args=vanilla_t5 training_args=train >logs/log_<run_name>.txt &
    ```
## Reproduce our results
1. Get the data from https://console.cloud.google.com/storage/browser/mathematics-dataset;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false
2. Execute [create_data_splits.py](data%2Fmathematics_dataset-v1.0%2Fcreate_data_splits.py)
3. Put the .txt files under data/mathematics_dataset-v1.0/
4. Execute the run_language_modeling.py script with the following arguments:
- Standard T5: 
  ```
  python src/run_language_modeling.py model_args=vanilla_t5 +training_args.max_steps=1050000
  ```
- Standard T5 + **NTL-MSE**:
  ```
  python src/run_language_modeling.py model_args=vanilla_t5_ntl +training_args.max_steps=1050000
  ```
- Standard T5 + **NTL-WAS**: 
  ```
  python src/run_language_modeling.py model_args=vanilla_t5_ntl  model_args.number_token_loss_with_wasserstein=true +training_args.max_steps=1050000
  ```
- RT: 
  ```
  python src/run_language_modeling.py model_args=rt +training_args.max_steps=1050000
  ```
- RT + **NTL-MSE**: 
  ```
  python src/run_language_modeling.py model_args=rt_ntl +training_args.max_steps=1050000
  ```
- xVal: 
  ```
  python src/xval/train.py
  ```

For evaluating instead of training a model, add those two parameters to the respective python command: ```training_args=eval model_args.model_name_or_path=<path to checkpoint file>``` 
e.g for Standard T5 + **NTL-WAS**: 
```
python src/run_language_modeling.py model_args=vanilla_t5_ntl  model_args.number_token_loss_with_wasserstein=true training_args=eval model_args.model_name_or_path=<path to checkpoint file>
```

