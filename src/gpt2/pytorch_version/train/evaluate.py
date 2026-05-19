import math
import torch
from torch.utils.data import DataLoader


def compute_perplexity(model, dataloader: DataLoader, device: str = "cuda") -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs[:, :-1, :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )

            total_loss += loss.item()
            count += 1

    model.train()
    if count == 0:
        return float("inf")
    return math.exp(total_loss / count)
