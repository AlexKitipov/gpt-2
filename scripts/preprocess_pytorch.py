import argparse

import torch
from transformers import GPT2TokenizerFast

from src.gpt2.pytorch_version.data.block_packing import pack_tokens
from src.gpt2.pytorch_version.data.preprocess import (
    tokenize_jsonl_file,
    tokenize_text_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--cache_dir", type=str, default="cache_tokens")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)

    if args.jsonl:
        tokens = tokenize_jsonl_file(
            tokenizer=tokenizer,
            file_path=args.input,
            block_size=args.block_size,
            cache_dir=args.cache_dir,
        )
    else:
        tokens = tokenize_text_file(
            tokenizer=tokenizer,
            file_path=args.input,
            block_size=args.block_size,
            cache_dir=args.cache_dir,
        )

    blocks = pack_tokens(tokens, args.block_size)
    torch.save(blocks, args.output)
    print(f"Saved {blocks.size(0)} blocks to {args.output}")


if __name__ == "__main__":
    main()
