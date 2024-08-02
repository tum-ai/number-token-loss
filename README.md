# ibm_impact_project

ssh to the server

Check which GPUs are available with 

nvidia-smi

Either attach to the docker container IBM_project (docker attach IBM_project) or run you own new container.

docker run --name container_name --gpus <device_number> -v /home/students/code/<name>/path_to_code:/app/data -it huggingface/transformers-pytorch-gpu

In container, interactively set the transformers library to version  4.42.4

pip install transformers=4.42.4

Log into wandb in the terminal 

wandb login

Enter you username and auth token (wandb.ai/auth, you can also find the api key on the wandb invitation to our team)

Set the train arguments in run_train.sh and run script in the backgroud and log outputs
(for me a learning rate of 5e-6 worked perfectly)

nohup bash run_train.sh >logs/log_<run_name>.txt & 


Feel free to update this readme to better versions :)
