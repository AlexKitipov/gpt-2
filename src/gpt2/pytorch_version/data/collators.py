from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


class DynamicPaddingCollator:
    """Pad variable-length tokenized samples to the longest sequence in a batch."""

    def __init__(self, pad_token_id: int = 0, label_pad_token_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        input_ids = [sample["input_ids"] for sample in batch]
        labels = [sample["labels"] for sample in batch]

        padded_inputs = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        padded_labels = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_token_id
        )
        attention_mask = (padded_inputs != self.pad_token_id).long()

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }


class CausalLMBlockCollator:
    """Stack fixed-length blocks for causal language modeling."""

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([sample["input_ids"] for sample in batch])
        labels = torch.stack([sample["labels"] for sample in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
