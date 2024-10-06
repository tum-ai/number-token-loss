# IBM Impact Project

## Training

- ssh to the server
- Check which GPUs are available with `nvidia-smi`
- Either attach to the docker container IBM_project (docker attach IBM_project) or run you own new container with
    ```bash
    docker run --name container_name --gpus <device_number> -v /home/students/code/<name>/path_to_code:/app/data -it huggingface/transformers-pytorch-gpu
  ```

- In container, interactively set the transformers library to version  4.42.4 and install wandb and hydra
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

