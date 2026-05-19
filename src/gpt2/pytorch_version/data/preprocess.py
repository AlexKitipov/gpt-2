import hashlib
import json
import os

import torch


def _tokenizer_cache_id(tokenizer):
    """Return a stable tokenizer identifier for cache fingerprinting."""
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if name_or_path:
        return str(name_or_path)
    name = getattr(tokenizer, "name", None)
    if name:
        return str(name)
    return tokenizer.__class__.__name__


def compute_fingerprint(file_path, tokenizer_name, block_size):
    """Compute a stable fingerprint from tokenizer, block size, and file bytes."""
    h = hashlib.sha256()
    h.update(tokenizer_name.encode("utf-8"))
    h.update(str(block_size).encode("utf-8"))

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def cache_path(base_dir, fingerprint):
    return os.path.join(base_dir, f"{fingerprint}.pt")


def load_cached_tokens(base_dir, fingerprint):
    path = cache_path(base_dir, fingerprint)
    if os.path.isfile(path):
        return torch.load(path)
    return None


def save_cached_tokens(base_dir, fingerprint, tokens):
    os.makedirs(base_dir, exist_ok=True)
    torch.save(tokens, cache_path(base_dir, fingerprint))


def tokenize_text_file(tokenizer, file_path, block_size, cache_dir):
    """Tokenize a text file and cache the flattened token tensor."""
    fingerprint = compute_fingerprint(
        file_path=file_path,
        tokenizer_name=_tokenizer_cache_id(tokenizer),
        block_size=block_size,
    )
    cached = load_cached_tokens(cache_dir, fingerprint)
    if cached is not None:
        return cached

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    save_cached_tokens(cache_dir, fingerprint, token_tensor)
    return token_tensor


def tokenize_jsonl_file(tokenizer, file_path, block_size, cache_dir):
    """Tokenize a JSONL file and cache the flattened token tensor."""
    fingerprint = compute_fingerprint(
        file_path=file_path,
        tokenizer_name=_tokenizer_cache_id(tokenizer),
        block_size=block_size,
    )
    cached = load_cached_tokens(cache_dir, fingerprint)
    if cached is not None:
        return cached

    all_tokens = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if text is None:
                text = f"{obj.get('input', '')}{obj.get('output', '')}"
            all_tokens.extend(tokenizer.encode(text))

    token_tensor = torch.tensor(all_tokens, dtype=torch.long)
    save_cached_tokens(cache_dir, fingerprint, token_tensor)
    return token_tensor
