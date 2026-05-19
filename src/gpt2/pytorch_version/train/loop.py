import os
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 1
    grad_accum_steps: int = 4
    use_amp: bool = True
    output_dir: str = "checkpoints_pytorch"
    checkpoint_interval: int = 0
    max_grad_norm: float = 1.0


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.rank = getattr(config, "rank", 0)

        self.scaler = GradScaler(enabled=config.use_amp)
        self.global_step = 0
        self.best_val_loss = math.inf

        os.makedirs(config.output_dir, exist_ok=True)

    def save_checkpoint(self, name: str = "last.pt") -> None:
        if self.rank != 0:
            return

        path = os.path.join(self.config.output_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "scaler": self.scaler.state_dict(),
                "step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt["scheduler"] is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["step"]

    def evaluate(self) -> float:
        if self.val_loader is None:
            return math.inf

        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids)
                logits = outputs[:, :-1, :]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                )

                total_loss += loss.item()
                count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def train(self) -> None:
        self.model.train()
        checkpoints_root = os.path.join(self.config.output_dir, "checkpoints")

        for epoch in range(self.config.epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(input_ids)
                    logits = outputs[:, :-1, :]
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1),
                    )
                    loss = loss / self.config.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (self.global_step + 1) % self.config.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None:
                        self.scheduler.step()

                self.global_step += 1
                pbar.set_postfix({"loss": loss.item()})

                if (
                    self.rank == 0
                    and self.config.checkpoint_interval > 0
                    and self.global_step % self.config.checkpoint_interval == 0
                ):
                    step_dir = os.path.join(
                        checkpoints_root, f"step_{self.global_step:04d}"
                    )
                    os.makedirs(step_dir, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(step_dir, "pytorch_model.bin"),
                    )
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(step_dir, "optimizer.pt"),
                    )
                    torch.save(
                        (
                            self.scheduler.state_dict()
                            if self.scheduler is not None
                            else None
                        ),
                        os.path.join(step_dir, "scheduler.pt"),
                    )

            val_loss = self.evaluate()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")

            self.save_checkpoint("last.pt")

        if self.rank == 0 and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(self.config.output_dir)

        if dist.is_initialized():
            dist.barrier()
