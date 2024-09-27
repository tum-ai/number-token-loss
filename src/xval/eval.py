import os

import torch
from torch import optim
from datasets import DatasetDict
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation import CustomMetrics
# Where the model and collator is defined
from src.xval import numformer

from src.data import load_txt_dataset
from src.tokenizer.xval_tokenizer import XvalTokenizer
from src.collators.xval_mask_question_collator import XvalQuestionAnswerCLMCollator

import wandb


def inverse_signed_log(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


train_data_path = '../data/mathematics_dataset-v1.0/train.txt'
eval_data_path = '../data/mathematics_dataset-v1.0/val.txt'
test_interpolate_data_path = '../data/mathematics_dataset-v1.0/test_interpolate.txt'
test_extrapolate_data_path = '../data/mathematics_dataset-v1.0/test_extrapolate.txt'

train_dataset = load_txt_dataset(train_data_path)
eval_dataset = load_txt_dataset(eval_data_path)
test_interpolate_dataset = load_txt_dataset(test_interpolate_data_path)
test_extrapolate_dataset = load_txt_dataset(test_extrapolate_data_path)

tokenizer = XvalTokenizer.from_pretrained("t5-small")

### Define model
# The vocab_size is the number of different tokens in the tokenizer.
# context length is the maximum sequence size.
model = numformer.Numformer(vocab_size=len(tokenizer), nhead=3, num_layers=3, d_model=384, dim_feedforward=1536,
                            context_length=955).cuda()

### Load the tokenizer
pad_token_id = tokenizer.pad_token_id
num_token_id = tokenizer.get_num_token_ids()[0]
mask_token_id = tokenizer.additional_special_tokens_ids[0]
epochs = 10000

# Define the masked xVal collator which takes samples of unequal length and masks out both the token_ids and the numbers.
collator = XvalQuestionAnswerCLMCollator(tokenizer)

val_loader = DataLoader(
    eval_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collator,
)

test_interpolate_loader = DataLoader(
    test_interpolate_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collator,
)

test_extrapolate_loader = DataLoader(
    test_extrapolate_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collator,
)

metrik = CustomMetrics(
    tokenizer=tokenizer,
    number_encoding="xval",
    output_dir="./train_1",
    save_all_output=True,
)

if not os.path.exists("./train_1"):
    os.makedirs("./train_1")

checkpoint_path = "./ckpt.pt"
checkpoint = torch.load(checkpoint_path)
# Load model state
model.load_state_dict(checkpoint["model"])

### Run eval loop

try:
    # evaluation
    model.eval()
    with torch.autocast(device_type="cuda"):
        with torch.no_grad():
            for data_loader, dataset_name in zip([val_loader, test_interpolate_loader, test_extrapolate_loader], ["val", "test_interpolate", "test_extrapolate"]):
                eval_loss = []
                eval_loss_mlm = []
                eval_loss_num = []
                for batch_idx, val_batch in enumerate(tqdm(data_loader, desc=f"Validation {dataset_name}")):
                    logit_preds, num_preds = model(val_batch["x"].cuda(), val_batch["x_num"].cuda())
                    loss_mlm = F.cross_entropy(
                        logit_preds.view(-1, logit_preds.size(-1)),
                        val_batch["y"].cuda().view(-1),
                        ignore_index=-100,
                        reduction="mean",
                    )
                    num_mask = val_batch['y'] == num_token_id
                    loss_num = F.mse_loss(
                        num_preds[num_mask],
                        val_batch["y_num"][num_mask].view(-1, 1).cuda(),
                        reduction="mean",
                    )
                    loss = loss_mlm + loss_num

                    eval_loss.append(loss.item())
                    eval_loss_mlm.append(loss_mlm.item())
                    eval_loss_num.append(loss_num.item())

                    is_last_batch = batch_idx == len(data_loader) - 1

                    mask = val_batch["mask"]

                    predictions = (
                    logit_preds.argmax(-1)[mask].reshape(-1, 1), inverse_signed_log(num_preds[mask]))
                    model_output = (
                        logit_preds[mask].reshape(logit_preds.shape[0], 1, logit_preds.shape[-1]), predictions)
                    labels = (
                        val_batch["y"][mask].reshape(-1, 1).cuda(),
                        inverse_signed_log(val_batch["y_num"][mask].reshape(-1, 1).cuda()))
                    pred = (model_output, labels)

                    computed_result = metrik(pred, compute_result=is_last_batch)

                avg_eval_loss = sum(eval_loss) / len(eval_loss)
                avg_eval_loss_mlm = sum(eval_loss_mlm) / len(eval_loss_mlm)
                avg_eval_loss_num = sum(eval_loss_num) / len(eval_loss_num)

                print(
                    f"Validation results: {dataset_name};"
                    f"Validation loss: {avg_eval_loss}; "
                    f"Validation loss_mlm: {avg_eval_loss_mlm}; "
                    f"Validation loss_num: {avg_eval_loss_num}; "
                    f"Validation loss_total: {avg_eval_loss}; "
                    f"Validation results: {computed_result}"
                )
except KeyboardInterrupt:
    print('Interrupted')
