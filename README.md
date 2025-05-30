# Regress, Don’t Guess – A Regression-like Loss on Number Tokens for Language Models
![ntl-image.jpg](resources%2Fntl-image.jpg)

Introducing "Number Token Loss" (NTL) for language models to improve numerical reasoning by using regression-based loss functions that account for the proximity of numbers, achieving better performance on math tasks without increasing computational overhead.

## Resources
Find our paper [here](https://arxiv.org/abs/2411.02083) and the poster of the NeurIPS 2024 MathAI workshop [here](https://github.com/tum-ai/number-token-loss/blob/main/resources/neurips_mathai_poster.pdf "Poster")

## Setup

### Via Python
- Requires Python 3.9 or higher
- Install the required packages
    ```bash
    conda create -n ntl python=3.9
    conda activate ntl
    pip install -r requirements.txt
    pip install -e .
    ```
- Log into wandb in the terminal
    ```
    wandb login
    ```
  Enter you username and auth token (wandb.ai/auth). To specify the wandb entity and project for logging the experiment, set the following environment variables
    ```
    export WANDB_ENTITY='<your_entity>'
    export WANDB_PROJECT='<your_project_name>'  
    ```

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
    python src/ntl/run_language_modeling.py dataset_args=<gsm8k or mathematics_dataset, default mathematics_dataset>
                                        model_args=<rt, rt_ntl, vanilla_t5, vanilla_t5_ntl, xval>
                                        training_args=<eval or train>
    ```
  - You can override the default config via the command line, e.g. 
    ```bash
    python src/ntl/run_language_modeling.py model_args=vanilla_t5 training_args=train training_args.per_device_train_batch_size=8
    ```
    or override them in the [config/run_specific_config/config.yaml](config%2Frun_specific_config%2Fconfig.yaml) file.
  - For debugging, you can use the [config/run_specific_config/debug_config.yaml](config%2Frun_specific_config%2Fdebug_config.yaml) file via
    ```bash
    python src/ntl/run_language_modeling.py model_args=vanilla_t5 training_args=train run_specific_config@_global_=debug_config
    ```
  - For running in nohup mode, use
    ```bash
    nohup python src/ntl/run_language_modeling.py dataset_args=mathematics_dataset model_args=vanilla_t5 training_args=train >logs/log_<run_name>.txt &
    ```
## Reproduce our results
### Mathematics Dataset
1. Get the data from https://console.cloud.google.com/storage/browser/mathematics-dataset;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false
2. Execute [create_data_splits.py](data%2Fmathematics_dataset-v1.0%2Fcreate_data_splits.py)
3. Put the .txt files under data/mathematics_dataset-v1.0/
4. Execute the run_language_modeling.py script with the following arguments:
- Standard T5: 
  ```
  python src/ntl/run_language_modeling.py run_specific_config@_global_=mathematics_dataset_run model_args=vanilla_t5 dataset_args=mathematcis_dataset
  ```
- Standard T5 + **NTL-MSE**:
  ```
  python src/ntl/run_language_modeling.py run_specific_config@_global_=mathematics_dataset_run model_args=vanilla_t5_ntl dataset_args=mathematcis_dataset
  ```
- Standard T5 + **NTL-WAS**: 
  ```
  python src/ntl/run_language_modeling.py run_specific_config@_global_=mathematics_dataset_run model_args=vanilla_t5_ntl  model_args.number_token_loss_with_wasserstein=true dataset_args=mathematcis_dataset
  ```
- RT: 
  ```
  python src/ntl/run_language_modeling.py run_specific_config@_global_=mathematics_dataset_run model_args=rt dataset_args=mathematcis_dataset
  ```
- RT + **NTL-MSE**: 
  ```
  python src/ntl/run_language_modeling.py run_specific_config@_global_=mathematics_dataset_run model_args=rt_ntl dataset_args=mathematcis_dataset
  ```
- xVal: 
  ```
  python src/nlt/xval/train.py
  ```
### Ablation Studies on part of the Mathematics Dataset
1. Execute [arith_create_splits.py](data%2Fmathematics_dataset-v1.0%2Farith_create_splits.py)
2. The resulting files (arithmetic_train.txt, arithmetic_val.txt, arithmetic_test_interpolate.txt, arithmetic_test_extrapolate.txt) should be under data/mathematics_dataset-v1.0/
3. Execute the run_language_modeling.py script with the following arguments:
- T5:
  ```
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5 training_args.special_name=default_CE training_args.seed=<NUMBER>
  ```

- T5 + **NTL-MSE**:
  ```
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=false model_args.number_token_loss_weight=0.3 training_args.special_name=NTL-MSE_Lambda0.3 training_args.seed=<NUMBER>
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=false model_args.number_token_loss_weight=0.8 training_args.special_name=NTL-MSE_Lambda0.8 training_args.seed=<NUMBER>
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=false model_args.number_token_loss_weight=2.0 training_args.special_name=NTL-MSE_Lambda2.0 training_args.seed=<NUMBER>
  ```

- T5 + **NTL-WAS**:
  ```
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=true model_args.number_token_loss_weight=0.3 training_args.special_name=NTL-WAS_Lambda0.3 training_args.seed=<NUMBER>
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=true model_args.number_token_loss_weight=0.8 training_args.special_name=NTL-WAS_Lambda0.8 training_args.seed=<NUMBER>
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=true model_args.number_token_loss_weight=2.0 training_args.special_name=NTL-WAS_Lambda2.0 training_args.seed=<NUMBER>
  ```

- T5 + **NTL-MAE**:
  ```
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=false +model_args.number_token_loss_function=mae training_args.special_name=NTL-MAE_Lambda0.3 training_args.seed=<NUMBER>
  ```
- T5 + **NTL-Huber**:
  ```
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=false +model_args.number_token_loss_function=huber training_args.special_name=NTL-Huber_Lambda0.3 training_args.seed=<NUMBER>
  ```

- T5 + Gaussian-CE
  ```
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5 +model_args.gaussian_label_smoother=true +model_args.label_smoother_sigma=1.0 training_args.special_name=gaussian_ce_sigma1 training_args.seed=<NUMBER>
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5 +model_args.gaussian_label_smoother=true +model_args.label_smoother_sigma=2.0 training_args.special_name=gaussian_ce_sigma2 training_args.seed=<NUMBER>
  ```

- T5 + Gaussian-CE +**NTL-WAS**:
  ```
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=true model_args.number_token_loss_weight=0.3 +model_args.gaussian_label_smoother=true +model_args.label_smoother_sigma=1.0 training_args.special_name=GaussianCE_sigma1_NTL-WAS_Lambda0.3 training_args.seed=<NUMBER>
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl model_args.number_token_loss_with_wasserstein=true model_args.number_token_loss_weight=0.3 +model_args.gaussian_label_smoother=true +model_args.label_smoother_sigma=2.0 training_args.special_name=GaussianCE_sigma2_NTL-WAS_Lambda0.3 training_args.seed=<NUMBER>
  ```
- Tests on different tokenizers:
  ```
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_custom_tokenizer training_args.seed=<NUMBER> 
  python src/ntl/run_language_modeling.py dataset_args=arithmetic model_args=vanilla_t5_ntl_default_tokenizer training_args.seed=<NUMBER>  
  ```

### NTL on a regression task (rjokes dataset)
1. Download data from https://github.com/orionw/rJokesData
2. Put train.tsv, dev.tsv and test.tsv under data/rjokes-dataset/data
3. Execute [generate_dataset.py](data%2Frjokes-dataset%2Fgenerate_dataset.py)
4. Execute the run_language_modeling.py script with the following arguments:
- T5:
  ```
  python src/ntl/run_language_modeling.py model_args=vanilla_t5 dataset_args=rjokes training_args.seed=<NUMBER>
  ```
- T5 + **NTL-WAS**:
  ```
  python src/ntl/run_language_modeling.py model_args=vanilla_t5_ntl dataset_args=rjokes model_args.number_token_loss_weight=2.0 training_args.special_name=lambda2 training_args.seed=<NUMBER>
  ```
- T5 + **Regression Head**:
  ```
  python src/ntl/run_language_modeling.py model_args=vanilla_t5_regression_head dataset_args=rjokes training_args.language_modelling="mlm" training_args.seed=<NUMBER>
  ```
- T5 + **Custom Tokenizer**:
  ```
  python src/ntl/run_language_modeling.py model_args=vanilla_t5_custom_tokenizer dataset_args=rjokes training_args.seed=<NUMBER>
  ```
- T5 + **NTL with Default Tokenizer**:
  ```
  python src/ntl/run_language_modeling.py model_args=vanilla_t5_ntl_default_tokenizer dataset_args=rjokes model_args.number_token_loss_weight=2.0 training_args.seed=<NUMBER>
  ```
  
### NTL does not hamper text learning on the transformed MultiRC dataset
1. Download the MultiRC dataset from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip
2. Put the train.jsonl, val.jsonl and test.jsonl files under data/multirc/data
3. Execute [generate_dataset.py](data%2Frjokes-dataset%2Fgenerate_dataset.py)
4. The generated files should be under data/multirc/data/preprocessed
5. Execute the run_language_modeling.py script with the following arguments:
- T5:
  ```
  python src/ntl/run_language_modeling.py model_args=vanilla_t5 dataset_args=multirc training_args.trial=nlp_task_run training_args.seed=<NUMBER>
  ```
- T5 + **NTL-WAS**:
  ```
  python src/ntl/run_language_modeling.py model_args=vanilla_t5_ntl dataset_args=multirc training_args.special_name=lambda2 model_args.number_token_loss_weight=2.0 training_args.trial=nlp_task training_args.seed=<NUMBER>
  ```

###  NTL scales well to LLM-size on GSM8k
Execute the run_language_modeling.py script with the following arguments:
- T5:
  ```
  python src/ntl/run_language_modeling.py run_specific_config@_global_=gsm8k_runs model_args=vanilla_t5 dataset_args=gsm8k training_args.seed=<NUMBER>
  ```
- T5 + **NTL-WAS**:
  ```
    python src/ntl/run_language_modeling.py run_specific_config@_global_=gsm8k_runs model_args=vanilla_t5_ntl dataset_args=gsm8k model_args.number_token_loss_weight=0.3 training_args.seed=<NUMBER>
  ```

---
For evaluating instead of training a model, add those two parameters to the respective python command: ```training_args=eval model_args.model_name_or_path=<path to checkpoint file>``` 
e.g for Standard T5 + **NTL-WAS**: 
```
python src/ntl/run_language_modeling.py model_args=vanilla_t5_ntl  model_args.number_token_loss_with_wasserstein=true training_args=eval model_args.model_name_or_path=<path to checkpoint file>
```
---
The repository also includes a generic version of the NTL (`/src/ntl/loss_functions/base_number_token_loss.py`), which is compatible with a broader range of Hugging Face models and is based on the NTL-WAS.

## Citation
If you use this work, please cite:
```bib
@inproceedings{zausinger24regress,
  title={Regress, Don't Guess--A Regression-like Loss on Number Tokens for Language Models},
  author={Zausinger, Jonas and Pennig, Lars and Chlodny, Kacper and Limbach, Vincent and Ketteler, Anna and Prein, Thorben and Singh, Vishwa Mohan and Danziger, Michael and Born, Jannis},
  booktitle={The 4th Workshop on Mathematical Reasoning and AI at NeurIPS'24},
  year={2024}
}
