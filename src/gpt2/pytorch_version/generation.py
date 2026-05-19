import torch
import torch.nn.functional as F


def top_k_filter(logits, k):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_val = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_val, torch.full_like(logits, -float("inf")), logits)


def top_p_filter(logits, p):
    if p <= 0 or p >= 1:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    cutoff = cumulative_probs > p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_logits = torch.where(
        cutoff, torch.full_like(sorted_logits, -float("inf")), sorted_logits
    )
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)


def apply_repetition_penalty(logits, input_ids, penalty):
    if penalty == 1.0:
        return logits
    for token in set(input_ids.tolist()):
        logits[:, token] /= penalty
    return logits


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    repetition_penalty=1.0,
):
    model.eval()

    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs[:, -1, :]

        logits = logits / temperature
        logits = apply_repetition_penalty(logits, input_ids[0], repetition_penalty)
        logits = top_k_filter(logits, top_k)
        logits = top_p_filter(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
