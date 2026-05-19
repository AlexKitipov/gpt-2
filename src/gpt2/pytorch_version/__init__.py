"""PyTorch GPT-2 package.

Imports are intentionally lazy/optional so lightweight utilities (for example
``tokenizer.SimpleTokenizer``) can be used without requiring all training
runtime dependencies to be installed.
"""

from typing import Any

from .config import GPT2Config as GPT2Config

_GPT2Model: Any
try:
    from .modeling_gpt2 import GPT2Model as _GPT2Model
except ModuleNotFoundError:  # optional runtime dependency (e.g., torch)
    _GPT2Model = None
GPT2Model: Any = _GPT2Model

_generate: Any
try:
    from .generation import generate as _generate
except ModuleNotFoundError:  # optional runtime dependency (e.g., torch)
    _generate = None
generate: Any = _generate

__all__ = ["GPT2Config", "GPT2Model", "generate"]
