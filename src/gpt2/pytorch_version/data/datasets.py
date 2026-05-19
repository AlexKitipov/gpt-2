import json
import os
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Loads a plain text file and splits it into fixed-length token blocks.
    """

    def __init__(self, tokenizer, file_path, block_size=1024):
        assert os.path.isfile(file_path), f"File not found: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text)
        self.block_size = block_size

        # drop remainder
        total_len = (len(tokens) // block_size) * block_size
        tokens = tokens[:total_len]

        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        block = self.tokens[start:end]
        return {
            "input_ids": block.clone(),
            "labels": block.clone(),
        }


class JsonlDataset(Dataset):
    """
    Loads JSONL files with {"text": "..."} or {"input": "...", "output": "..."}.
    """

    def __init__(self, tokenizer, file_path, block_size=1024):
        assert os.path.isfile(file_path), f"File not found: {file_path}"

        self.samples = []
        self.tokenizer = tokenizer
        self.block_size = block_size

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "text" in obj:
                    text = obj["text"]
                elif "input" in obj and "output" in obj:
                    text = obj["input"] + "\n" + obj["output"]
                else:
                    continue
                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text)[: self.block_size]
        tokens = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": tokens.clone(),
            "labels": tokens.clone(),
        }
