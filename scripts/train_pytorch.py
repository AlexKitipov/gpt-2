import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.gpt2.pytorch_version.config import GPT2Config
from src.gpt2.pytorch_version.modeling_gpt2 import GPT2Model
from src.gpt2.pytorch_version.train.loop import Trainer, TrainConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--output_dir", type=str, default="checkpoints_pytorch")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPT2Config()
    model = GPT2Model(config)

    # TODO: replace with real dataset in dataset pipeline PR
    dummy_tokens = torch.randint(0, config.vocab_size, (128, 64))
    dataset = TensorDataset(dummy_tokens, dummy_tokens)
    train_loader = DataLoader(
        [{"input_ids": x, "labels": y} for x, y in dataset],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        [{"input_ids": x, "labels": y} for x, y in dataset[:16]],
        batch_size=args.batch_size,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
        output_dir=args.output_dir,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
