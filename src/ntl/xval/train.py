import os

import torch
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ntl.xval.xval_mask_question_collator import XvalMaskedQuestionAnswerCollator
from ntl.data.data import load_txt_dataset
from ntl.evaluation import CustomMetrics
from ntl.tokenizer.xval_tokenizer import XvalTokenizer
from ntl.utils.numerical_operations import inverse_signed_log
# Where the model and collator is defined
from ntl.xval import numformer

train_data_path = 'data/mathematics_dataset-v1.0/train.txt'
eval_data_path = 'data/mathematics_dataset-v1.0/val.txt'
test_interpolate_data_path = 'data/mathematics_dataset-v1.0/test_interpolate.txt'
test_extrapolate_data_path = 'data/mathematics_dataset-v1.0/test_extrapolate.txt'


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
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

lr = 1e-4
weight_decay = 0.01
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

### Load the tokenizer
pad_token_id = tokenizer.pad_token_id
num_token_id = tokenizer.get_num_token_ids()[0]
mask_token_id = tokenizer.additional_special_tokens_ids[0]
epochs = 10000

# Define the masked xVal collator which takes samples of unequal length and masks out both the token_ids and the numbers.
collator = XvalMaskedQuestionAnswerCollator(tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collator,
)

val_loader = DataLoader(
    eval_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collator,
)

OUTPUT_DIR = "./outptus/mathematics_dataset/xval/model_small"

metrik = CustomMetrics(
    tokenizer=tokenizer,
    number_encoding="xval",
    output_dir=OUTPUT_DIR,
    save_all_output=True,
)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

### Run training loop


# Initialize Weights and Biases
config = {
    'epochs': epochs,
    'batch_size': 32,
    'learning_rate': lr,
    'weight_decay': weight_decay,
    'max_steps': 10000,
    'eval_steps': 500,
    'log_steps': 5,
    'model_name': 'numformer',
    'nhead': 3,
    'num_layers': 3,
    'd_model': 384,
    'dim_feedforward': 1536,
    'context_length': 955,
    # Add other hyperparameters as needed
}

wandb.init(project='xval', config=config, name='model_smal')


loss_hist = []
loss_mlm_hist = []
loss_num_hist = []

max_steps = 1_050_000
eval_steps = 10_000
log_steps = 1000

steps = 0
best_validation_loss = float('inf')

try:
    for e in range(epochs):
        for batch in train_loader:
            if steps > max_steps:
                break
            logit_preds, num_preds = model(batch["x"].cuda(), batch["x_num"].cuda())
            with torch.autocast(device_type="cuda"):
                loss_mlm = F.cross_entropy(
                    logit_preds.view(-1, logit_preds.size(-1)),
                    batch["y"].cuda().view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
                num_mask = batch['y'] == num_token_id
                loss_num = F.mse_loss(
                    num_preds[num_mask],
                    batch["y_num"][num_mask].view(-1, 1).cuda(),
                    reduction="mean",
                )
            loss = loss_mlm + loss_num
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
            loss_mlm_hist.append(loss_mlm.item())
            loss_num_hist.append(loss_num.item())
            steps += 1

            # calculate the running average of the losses
            try:
                loss_avg = 0.99 * loss_avg + 0.01 * loss.item()
                loss_mlm_avg = 0.99 * loss_mlm_avg + 0.01 * loss_mlm.item()
                loss_num_avg = 0.99 * loss_num_avg + 0.01 * loss_num.item()
            except:
                loss_avg = loss.item()
                loss_mlm_avg = loss_mlm.item()
                loss_num_avg = loss_num.item()

            if steps % log_steps == 0:
                print(
                    f"Step #{steps}: loss_mlm = {loss_mlm_avg:.3f}; loss_num = {loss_num_avg:.3f}; loss_total = {loss_avg:.3f}")
                # Log training metrics to wandb
                wandb.log({
                    'train/loss': loss_avg,
                    'train/token_loss': loss_mlm_avg,
                    'train/number_loss': loss_num_avg,
                    'train/global_step': steps,
                })

            if steps % eval_steps == 0:
                # evaluation
                model.eval()
                with torch.autocast(device_type="cuda"):
                    with torch.no_grad():
                        eval_loss = []
                        eval_loss_mlm = []
                        eval_loss_num = []
                        for batch_idx, val_batch in enumerate(tqdm(val_loader, desc="Validation")):
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

                            is_last_batch = batch_idx == len(val_loader) - 1

                            mask = val_batch["mask"]

                            predictions = (logit_preds.argmax(-1)[mask].reshape(-1, 1), inverse_signed_log(num_preds[mask]))
                            model_output = (
                            logit_preds[mask].reshape(logit_preds.shape[0], 1, logit_preds.shape[-1]), predictions)
                            labels = (
                            val_batch["y"][mask].reshape(-1, 1).cuda(), inverse_signed_log(val_batch["y_num"][mask].reshape(-1, 1).cuda()))
                            pred = (model_output, labels)

                            computed_result = metrik(pred, compute_result=is_last_batch)

                        avg_eval_loss = sum(eval_loss) / len(eval_loss)
                        avg_eval_loss_mlm = sum(eval_loss_mlm) / len(eval_loss_mlm)
                        avg_eval_loss_num = sum(eval_loss_num) / len(eval_loss_num)

                        print(
                            f"Validation loss: {avg_eval_loss}; "
                            f"Validation loss_mlm: {avg_eval_loss_mlm}; "
                            f"Validation loss_num: {avg_eval_loss_num}; "
                            f"Validation loss_total: {avg_eval_loss}; "
                            f"Validation results: {computed_result}"
                        )

                        wandb.log({
                            'eval/loss': avg_eval_loss,
                            'eval/token_loss': avg_eval_loss_mlm,
                            'eval/number_loss': avg_eval_loss_num,
                            'train/global_step': steps,
                            **{f'eval/{k}': v for k, v in computed_result.items()}
                        })

                        checkpoint = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "loss": loss_avg,
                            "loss_hist": loss_hist,
                            "loss_mlm_hist": loss_mlm_hist,
                            "loss_num_hist": loss_num_hist,
                        }
                        torch.save(checkpoint, "./ckpt_latest.pt")
                        print("Latest checkpoint saved")

                        if avg_eval_loss < best_validation_loss:
                            best_validation_loss = avg_eval_loss
                            torch.save(checkpoint, "./ckpt_best.pt")
                            print("Best checkpoint saved")
                            wandb.run.summary["best_validation_loss"] = best_validation_loss  # Update wandb summary

                model.train()

except KeyboardInterrupt:
    print('Interrupted')
