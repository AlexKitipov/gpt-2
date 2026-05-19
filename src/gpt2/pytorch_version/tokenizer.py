"""Lightweight local tokenizer utilities for the PyTorch GPT-2 package.

This module intentionally avoids heavyweight optional dependencies so that
notebooks and local scripts can import ``SimpleTokenizer`` out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

try:
    import regex as re  # type: ignore[import-not-found]

    _TOKEN_PATTERN = re.compile(r"\s+|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]", re.VERSION1)
except ModuleNotFoundError:
    import re

    _TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z]+|\d+|[^\sA-Za-z\d]")


@dataclass
class SimpleTokenizer:
    """A tiny dynamic tokenizer with GPT-style special token support.

    The vocabulary is grown as new tokens are observed. This keeps the class
    dependency-free while providing a familiar ``encode``/``decode`` interface
    for local experiments.
    """

    unk_token: str = "<|unk|>"
    eos_token: str = "<|endoftext|>"
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_to_id:
            self._add_token(self.unk_token)
            self._add_token(self.eos_token)

    def _add_token(self, token: str) -> int:
        existing = self.token_to_id.get(token)
        if existing is not None:
            return existing
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        pieces = _TOKEN_PATTERN.findall(text)
        return [self._add_token(piece) for piece in pieces]

    def decode(self, token_ids: List[int]) -> str:
        pieces = [self.id_to_token.get(idx, self.unk_token) for idx in token_ids]
        return "".join(pieces)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
