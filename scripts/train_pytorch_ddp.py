import argparse

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from src.gpt2.pytorch_version.config import GPT2Config
from src.gpt2.pytorch_version.modeling_gpt2 import GPT2Model
from src.gpt2.pytorch_version.train.distributed import init_distributed, wrap_ddp
from src.gpt2.pytorch_version.train.loop import TrainConfig, Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--output_dir", type=str, default="checkpoints_pytorch_ddp")
    return parser.parse_args()


def main():
    args = parse_args()

    rank, _, local_rank = init_distributed()
    device = f"cuda:{local_rank}" if local_rank is not None else "cuda"

    config = GPT2Config()
    model = GPT2Model(config)

    if local_rank is not None:
        model = wrap_ddp(model.to(device), local_rank)

    # Dummy dataset (replace in PR #9)
    dummy_tokens = torch.randint(0, config.vocab_size, (128, 64))
    dataset = [{"input_ids": x, "labels": x} for x in dummy_tokens]

    sampler = DistributedSampler(dataset) if local_rank is not None else None

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
        output_dir=args.output_dir,
    )
    train_cfg.rank = rank or 0

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=None,
        config=train_cfg,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
