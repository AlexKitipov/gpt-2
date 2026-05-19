# 🧠 AetherOS GPT‑2 — Training Pipeline Documentation

This document describes the full training workflow for the AetherOS GPT‑2 model,
including dataset streaming, checkpointing, Google Drive integration, and the
Colab training environment.

The goal is to ensure that training can be resumed, repeated, or continued at
any time with minimal setup.

---

# 1. 📌 Training Environment

The model is trained in Google Colab using CPU‑only execution.

**Colab Notebook URL (main training notebook):**  
https://colab.research.google.com/github/AlexKitipov/gpt-2/blob/master/AetherOS_Coder_TinyStories.ipynb

This notebook contains:

- dataset streaming setup  
- tokenizer loading  
- model loading  
- training loop  
- checkpoint saving  
- Google Drive integration  

---

# 2. 📌 Checkpoint Storage (Google Drive)

All model checkpoints are stored in Google Drive.

**Drive Folder URL (official checkpoint directory):**  
https://drive.google.com/drive/folders/1meBQVjvaS7CEjhR-ljMDzoN4ZrmWdfyy?lfhs=2

**Colab path to the same folder:**  
`/content/drive/MyDrive/AetherOS_GPT2_Checkpoints`

Inside this directory, the training script automatically creates:

- `checkpoint-<step>` folders  
- `epoch-<n>` folders  
- `final_model/`  
- `latest/` (optional)  

GitHub does **not** store model weights — only code and documentation.

---

# 3. 📌 Mounting Google Drive in Colab

Add this at the top of the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

# 4. 📌 Creating the Checkpoint Directory

This cell ensures the checkpoint folder exists:

```python
import os

ckpt_dir = "/content/drive/MyDrive/AetherOS_GPT2_Checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

print(f"Checkpoint directory created at: {ckpt_dir}")
```

This is correct and safe — it will not overwrite anything.

# 5. 📌 Loading the Model and Tokenizer

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

checkpoint_path = "/content/drive/MyDrive/AetherOS_GPT2_Checkpoints/latest"

model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)

device = "cpu"
model.to(device)
```

If the folder does not exist, training starts from scratch.

# 6. 📌 Dataset Streaming (RAM‑safe)

The training uses HuggingFace streaming mode:

- does not load the dataset into RAM
- tokenizes on the fly
- works safely on Colab CPU
- prevents session crashes

Example:

```python
from datasets import load_dataset

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True
)
```

# 7. 📌 Training Loop (Full Code)

This is the exact training cell used in the project, documented line‑by‑line:

```python
import torch
from torch.optim import AdamW
from tqdm.notebook import tqdm
import os

# Ensure the model and tokenizer are defined and on the correct device (CPU)
# 'model' and 'tokenizer' were loaded earlier
# 'device' was set to 'cpu'

# --- 1. Define Training Parameters ---
learning_rate = 5e-5
num_epochs = 3  # CPU-friendly

# --- 2. Define Optimizer ---
optimizer = AdamW(model.parameters(), lr=learning_rate)

# --- 3. Training Loop ---
model.train()

# Checkpoint directory
ckpt_dir = "/content/drive/MyDrive/AetherOS_GPT2_Checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

print(f"Starting training for {num_epochs} epochs on CPU...")
print("Each step processes a single example to keep RAM usage low.")

global_step = 0
save_steps = 100  # Save checkpoint every 100 steps

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    total_loss = 0

    progress_bar = tqdm(streaming_dataloader, desc=f"Training Epoch {epoch + 1}")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # --- Save checkpoint every N steps ---
        if global_step % save_steps == 0:
            current_save_path = os.path.join(ckpt_dir, f"checkpoint-{global_step}")
            os.makedirs(current_save_path, exist_ok=True)
            model.save_pretrained(current_save_path)
            tokenizer.save_pretrained(current_save_path)
            print(f"\nSaved checkpoint at step {global_step}: {current_save_path}")

    avg_train_loss = total_loss / len(streaming_dataloader)
    print(f"Epoch {epoch + 1} finished. Avg loss: {avg_train_loss:.4f}")

    # Save checkpoint after each epoch
    epoch_save_path = os.path.join(ckpt_dir, f"epoch-{epoch + 1}")
    os.makedirs(epoch_save_path, exist_ok=True)
    model.save_pretrained(epoch_save_path)
    tokenizer.save_pretrained(epoch_save_path)
    print(f"Saved epoch checkpoint: {epoch_save_path}")

# Final model save
final_save_path = os.path.join(ckpt_dir, "final_model")
os.makedirs(final_save_path, exist_ok=True)
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print(f"Training complete! Final model saved to: {final_save_path}")
```

# 8. 📌 Checkpoint Structure

Google Drive will contain:

```text
AetherOS_GPT2_Checkpoints/
│
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
│
├── epoch-1/
├── epoch-2/
├── epoch-3/
│
└── final_model/
```

Each folder contains:

- `pytorch_model.bin`
- `config.json`
- `tokenizer.json`
- `vocab.json`
- `merges.txt`

# 9. 📌 Resuming Training

To resume from the latest checkpoint:

```python
model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/AetherOS_GPT2_Checkpoints/final_model")
tokenizer = GPT2Tokenizer.from_pretrained("/content/drive/MyDrive/AetherOS_GPT2_Checkpoints/final_model")
```

Or choose any checkpoint folder.

# 10. 📌 What GitHub Stores

GitHub stores:

- training code
- dataset streaming code
- documentation (TRAINING.md, CHECKPOINTS.md)
- Colab links
- project structure

GitHub does not store:

- model weights
- checkpoints
- large files

✅ DONE