import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ntl.collators.era5_mlm.xval_era5_mlm_collator import XvalEra5MLMCollator
from ntl.data.data import load_txt_dataset, load_json_dataset
from ntl.evaluation import CustomMetrics
from ntl.tokenizer.xval_tokenizer import XvalTokenizer
from ntl.utils.numerical_operations import inverse_signed_log
# Where the model and collator is defined
from ntl.xval import numformer

train_data_path = 'data/era5/train_samples.jsonl'
eval_data_path = 'data/era5/val_samples.jsonl'
test_data_path = 'data/era5/test_samples.jsonl'

train_dataset = load_json_dataset(train_data_path)
eval_dataset = load_json_dataset(eval_data_path)
test_dataset = load_json_dataset(eval_data_path)

tokenizer = XvalTokenizer.from_pretrained("t5-small")

LOG_SCALE = False
MODEL_NAME = "era5_xval"
OUTPUT_DIR = f"./outputs/era5/xval/{MODEL_NAME}"

### Define model
# The vocab_size is the number of different tokens in the tokenizer.
# context length is the maximum sequence size.
model = numformer.Numformer(vocab_size=len(tokenizer), nhead=3, num_layers=3, d_model=384, dim_feedforward=1536,
                            context_length=955).cuda()

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

### Load the tokenizer
pad_token_id = tokenizer.pad_token_id
num_token_id = tokenizer.get_num_token_ids()[0]
mask_token_id = tokenizer.additional_special_tokens_ids[0]
epochs = 10000

# Define the masked xVal collator which takes samples of unequal length and masks out both the token_ids and the numbers.
collator = XvalEra5MLMCollator(tokenizer, log_scale=LOG_SCALE)

val_loader = DataLoader(
    eval_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collator,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collator,
)


metrik = CustomMetrics(
    tokenizer=tokenizer,
    number_encoding="xval",
    output_dir=OUTPUT_DIR,
    save_all_output=True,
    log_scale=LOG_SCALE,
)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

checkpoint_path = f"{OUTPUT_DIR}/ckpt_latest.pt"
checkpoint = torch.load(checkpoint_path)
# Load model state
model.load_state_dict(checkpoint["model"])

### Run eval loop

try:
    # evaluation
    model.eval()
    with torch.autocast(device_type="cuda"):
        with torch.no_grad():
            for data_loader, dataset_name in zip([val_loader, test_loader], ["val", "test"]):
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

                     # Get the predicted and label numbers
                    raw_predictions = inverse_signed_log(num_preds[mask]) if LOG_SCALE else num_preds[mask]
                    raw_labels = inverse_signed_log(val_batch["y_num"][mask].reshape(-1, 1).cuda()) if LOG_SCALE else val_batch["y_num"][mask].reshape(-1, 1).cuda()


                # Function to round predictions to match label precision
                    def round_to_precision(predictions, labels):
                        rounded_preds = predictions.clone()
                        for i in range(len(predictions)):
                            try:
                                # Convert to Python float first to avoid overflow
                                label_val = float(labels[i].cpu().item())
                                label_str = str(label_val)

                                # Determine decimal places
                                decimal_places = 0
                                if '.' in label_str:
                                    decimal_places = len(label_str.split('.')[1].rstrip('0'))

                                # Apply rounding with safe calculations
                                if decimal_places > 0:
                                    factor = 10 ** decimal_places
                                    pred_val = float(predictions[i].cpu().item())
                                    rounded_val = round(pred_val * factor) / factor
                                    rounded_preds[i] = torch.tensor(rounded_val,
                                                                    device=predictions.device,
                                                                    dtype=predictions.dtype)
                                else:
                                    # For integers, just round to nearest integer
                                    rounded_preds[i] = torch.round(predictions[i])
                            except (OverflowError, RuntimeError):
                                # Fallback for numbers that are too large: keep as is
                                pass
                        return rounded_preds

                    # Round predictions to match label precision
                    rounded_pred_nums = round_to_precision(raw_predictions, raw_labels)


                    predictions = (
                    logit_preds.argmax(-1)[mask].reshape(-1, 1), rounded_pred_nums)
                    model_output = (
                        logit_preds[mask].reshape(logit_preds.shape[0], 1, logit_preds.shape[-1]), predictions)
                    labels = (
                        val_batch["y"][mask].reshape(-1, 1).cuda(),
                        raw_labels)
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
