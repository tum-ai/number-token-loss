import time
import random
import string

import torch
from torch.optim import AdamW

from transformers import T5Tokenizer

from src.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from src.transformer_backbone.t5.t5_vanilla_for_number_token_loss import T5VanillaForNumberTokenLoss
from src.loss_functions.number_token_loss import NumberTokenLoss
from src.loss_functions.wasserstein_distance_number_token_loss import WassersteinNumberTokenLoss
from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


# Regular Cross Entropy (CE) Loss (Baseline)
# CE with NTL-MSE
# CE + NTL-Wasserstein

# 1. Measure standalone loss computation (mock tensor with fixed batch size)
# 2. Measure speed of forward pass
# 3. Measure speed of training step (including gradient update)
# Run each configuration in each setup 100 times (on GPU) and make a barplot showing the differences

class CustomCrossEntropyLoss:
    def forward(self, logits, labels):
        # Automatisch die Shapes bestimmen und dann umformen
        batch_size, seq_len, num_classes = logits.shape  # Die Form der Logits erfassen

        # Vergewissern, dass die Shapes übereinstimmen, bevor wir den Verlust berechnen
        assert labels.shape == (batch_size, seq_len), "Die Shapes von Logits und Targets müssen übereinstimmen."

        return torch.nn.functional.cross_entropy(logits.view(-1, num_classes), labels.view(-1))

def generate_inputs(batch_size, sequence_length, vocab_size, device):
    ntl_logits = torch.randn(batch_size, sequence_length, vocab_size, requires_grad=True, device=device)
    ntl_labels = torch.randint(vocab_size, (batch_size, sequence_length), device=device)

    return ntl_logits, ntl_labels

def random_letter():
    letters_and_digits = string.ascii_letters + string.digits + " "
    return random.choice(letters_and_digits)

def generate_text_inputs(length):
    text_inputs = [random_letter() for _ in range(length)]
    return text_inputs

def standalone_benchmark(ce_loss, mse_ntl_loss, wasserstein_ntl_loss, batch_size, sequence_length, steps, vocab_size, device):
    ntl_logits, ntl_labels = generate_inputs(batch_size, sequence_length, vocab_size, device)

    print("\nStarting benchmarking...")

    times = {}

    start = time.time()
    for _ in range(steps):
        ce_loss.forward(ntl_logits, ntl_labels)
    times["CE Loss"] = time.time() - start
    print(f"CE Loss: {times['CE Loss']}s")

    start = time.time()
    for _ in range(steps):
        mse_ntl_loss.forward(ntl_logits, ntl_labels)
    times["MSE NTL Loss"] = time.time() - start
    print(f"MSE NTL Loss: {times['MSE NTL Loss']}s")

    start = time.time()
    for _ in range(steps):
        wasserstein_ntl_loss.forward(ntl_logits, ntl_labels)
    times["Wasserstein NTL Loss"] = time.time() - start
    print(f"Wasserstein NTL Loss: {times['Wasserstein NTL Loss']}s")

    return times

def forward_pass(model, loss_func, steps, tokenizer,  batch_size, sequence_length, device, gradient_update = False):

    # text_inputs = generate_text_inputs(sequence_length)
    # text_outputs = generate_text_inputs(sequence_length)

    input_sentence = ["translate English to German: Hello, it is 7 o'clock?"] * batch_size
    target_sentence = ["Hallo, ist es 7 Uhr?"] * batch_size

    start_time = time.time()

    for _ in range(steps):
        # Tokenize input
        input_encoding = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)
        input_ids = input_encoding["input_ids"].to(device)
        attention_mask = input_encoding["attention_mask"].to(device)

        # Tokenize target
        target_encoding = tokenizer(target_sentence, return_tensors="pt", padding=True, truncation=True)
        labels = target_encoding["input_ids"].to(device)

        # Replace padding token ID in labels with -100 (ignored in loss computation)
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits              

        #loss = loss_func.forward(logits, labels)
        loss = loss_func.forward(logits, labels)
        loss.backward()

        if(gradient_update):
            # optimizer
            optimizer = AdamW(model.parameters(), lr=5e-5)

            # Perform one optimization step
            optimizer.step()
            optimizer.zero_grad()

    duration = time.time() - start_time

    print(f"Loss: {loss.item()}")
    print(f"Time: {duration}")

    return duration
    
def main():
    batch_size = 32
    sequence_length = 128
    steps = 100
    wassertein_order_numbers = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Tokenizer
    tokenizer = T5Custom_Tokenizer.from_pretrained("t5-small")

    # Model 
    model = T5VanillaForNumberTokenLoss.from_pretrained("t5-small")
    model.to(device)
    
    # Vocab size
    pad_to_multiple_of = 64 if torch.cuda.is_available() else 1
    pad_to_multiple_of = 64 # for model predictions 
    vocab_size = len(tokenizer) + pad_to_multiple_of - (len(tokenizer) % pad_to_multiple_of)

    # Loss
    ce_loss = CustomCrossEntropyLoss()
    mse_ntl_loss = NumberTokenLoss(tokenizer, vocab_size=vocab_size, device=device)
    wasserstein_ntl_loss = WassersteinNumberTokenLoss(tokenizer, 
                                                      vocab_size=vocab_size,
                                                      order_numbers = wassertein_order_numbers,
                                                      device=device)
    # Stnadalone benchmark
    standalone_benchmark(ce_loss, 
                         mse_ntl_loss, 
                         wasserstein_ntl_loss, 
                         batch_size, sequence_length, 
                         steps, vocab_size, device)

    # Forward pass
    print("\nForward pass")
    for name,loss in {"CE Loss": ce_loss, "MSE Loss:": mse_ntl_loss, "Wassserstein Loss:": wasserstein_ntl_loss}.items():
        print(f"{name}")
        forward_pass(model, loss, steps, tokenizer, batch_size, sequence_length, device, gradient_update = False)
    
    # Full training step
    for name,loss in {"CE Loss": ce_loss, "MSE Loss:": mse_ntl_loss, "Wassserstein Loss:": wasserstein_ntl_loss}.items():
        print(f"{name}")
        forward_pass(model, loss, steps, tokenizer, batch_size, sequence_length, device, gradient_update = True)


if __name__ == "__main__":
    main()
