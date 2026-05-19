import json
import random
import torch
from torch.utils.data import IterableDataset


class StreamingDataset(IterableDataset):
    """
    Streams JSONL shards from disk or web without loading everything into memory.
    """

    def __init__(self, tokenizer, shard_paths, shuffle=True, block_size=1024):
        self.tokenizer = tokenizer
        self.shard_paths = shard_paths
        self.shuffle = shuffle
        self.block_size = block_size

    def __iter__(self):
        shard_paths = self.shard_paths.copy()
        if self.shuffle:
            random.shuffle(shard_paths)

        for path in shard_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("input", "") + obj.get(
                        "output", ""
                    )
                    tokens = self.tokenizer.encode(text)[: self.block_size]
                    tokens = torch.tensor(tokens, dtype=torch.long)
                    yield {
                        "input_ids": tokens.clone(),
                        "labels": tokens.clone(),
                    }
