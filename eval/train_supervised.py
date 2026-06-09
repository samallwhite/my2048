"""Train a supervised 2048 policy network from expert data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.supervised import DEFAULT_NORMALIZE_BY, build_model, resolve_device

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
except ImportError as exc:  # pragma: no cover - depends on local ML env
    raise SystemExit(
        "PyTorch is required for training. Install torch with CUDA support "
        "before running this script."
    ) from exc


class Supervised2048Dataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        valid_masks: np.ndarray,
        model_type: str,
        normalize_by: float,
    ) -> None:
        self.states = states.astype(np.float32, copy=False) / float(normalize_by)
        self.actions = actions.astype(np.int64, copy=False)
        self.valid_masks = valid_masks.astype(np.bool_, copy=False)
        self.model_type = model_type

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.states[index]
        if self.model_type == "cnn":
            state = state.reshape(1, 4, 4)
        action = self.actions[index]
        valid_mask = self.valid_masks[index]
        return (
            torch.from_numpy(state),
            torch.tensor(action, dtype=torch.long),
            torch.from_numpy(valid_mask),
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def split_indices(
    n_samples: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_samples < 3:
        raise ValueError("At least 3 samples are required for train/val/test split")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    test_size = max(1, int(round(n_samples * test_ratio)))
    val_size = max(1, int(round(n_samples * val_ratio)))
    train_size = n_samples - val_size - test_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested split ratios")
    train = indices[:train_size]
    val = indices[train_size:train_size + val_size]
    test = indices[train_size + val_size:]
    return train, val, test


def make_loader(
    dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    workers: int,
    use_cuda: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=workers > 0,
    )


def masked_accuracy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    valid_masks: torch.Tensor,
) -> tuple[int, int, int]:
    raw_pred = torch.argmax(logits, dim=1)
    masked_logits = logits.masked_fill(~valid_masks.to(torch.bool), -1.0e9)
    masked_pred = torch.argmax(masked_logits, dim=1)
    raw_correct = int((raw_pred == actions).sum().item())
    masked_correct = int((masked_pred == actions).sum().item())
    legal = int(valid_masks.gather(1, raw_pred.unsqueeze(1)).sum().item())
    return raw_correct, masked_correct, legal


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    raw_correct = 0
    masked_correct = 0
    legal_count = 0
    use_amp = device.type == "cuda"

    for states, actions, valid_masks in loader:
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        valid_masks = valid_masks.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(states)
                loss = criterion(logits, actions)

            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = int(actions.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        batch_raw, batch_masked, batch_legal = masked_accuracy(
            logits.detach(),
            actions,
            valid_masks,
        )
        raw_correct += batch_raw
        masked_correct += batch_masked
        legal_count += batch_legal

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": raw_correct / max(1, total_samples),
        "masked_accuracy": masked_correct / max(1, total_samples),
        "legal_action_rate": legal_count / max(1, total_samples),
    }


def load_npz_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    states = data["states"]
    actions = data["actions"]
    valid_masks = data["valid_masks"]
    metadata = {}
    if "metadata" in data:
        metadata = json.loads(str(data["metadata"]))
    if states.ndim != 2 or states.shape[1] != 16:
        raise ValueError(f"Expected states shape (N, 16), got {states.shape}")
    if len(actions) != len(states) or len(valid_masks) != len(states):
        raise ValueError("states, actions, and valid_masks lengths do not match")
    if len(actions) == 0:
        raise ValueError("Dataset is empty")
    return states, actions, valid_masks, metadata


def train(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    states, actions, valid_masks, data_metadata = load_npz_dataset(args.data)
    dataset = Supervised2048Dataset(
        states=states,
        actions=actions,
        valid_masks=valid_masks,
        model_type=args.model,
        normalize_by=args.normalize_by,
    )
    train_idx, val_idx, test_idx = split_indices(
        len(dataset),
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_loader = make_loader(
        dataset,
        train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
        use_cuda=use_cuda,
    )
    val_loader = make_loader(
        dataset,
        val_idx,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        use_cuda=use_cuda,
    )
    test_loader = make_loader(
        dataset,
        test_idx,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        use_cuda=use_cuda,
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.csv"
    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_model.pt"

    best_val_acc = -1.0
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_masked_accuracy": train_metrics["masked_accuracy"],
            "train_legal_action_rate": train_metrics["legal_action_rate"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_masked_accuracy": val_metrics["masked_accuracy"],
            "val_legal_action_rate": val_metrics["legal_action_rate"],
        }
        rows.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"val_masked_acc={row['val_masked_accuracy']:.4f}"
        )

        if val_metrics["masked_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["masked_accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_type": args.model,
                    "model_state": model.state_dict(),
                    "metadata": {
                        "model_type": args.model,
                        "normalize_by": args.normalize_by,
                        "data_path": str(args.data),
                        "data_metadata": data_metadata,
                        "train_samples": int(len(train_idx)),
                        "val_samples": int(len(val_idx)),
                        "test_samples": int(len(test_idx)),
                        "best_epoch": best_epoch,
                        "best_val_masked_accuracy": best_val_acc,
                    },
                },
                best_path,
            )
        else:
            stale_epochs += 1

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    torch.save(
        {
            "model_type": args.model,
            "model_state": model.state_dict(),
            "metadata": {
                "model_type": args.model,
                "normalize_by": args.normalize_by,
                "data_path": str(args.data),
                "best_epoch": best_epoch,
                "best_val_masked_accuracy": best_val_acc,
            },
        },
        last_path,
    )

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, criterion, device)

    summary = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if use_cuda else None,
        "model": args.model,
        "data": str(args.data),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "best_epoch": int(best_epoch),
        "best_val_masked_accuracy": float(best_val_acc),
        "test_metrics": test_metrics,
        "best_model": str(best_path),
        "last_model": str(last_path),
        "training_log": str(log_path),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a supervised neural policy for 2048."
    )
    parser.add_argument("--data", type=Path, default=Path("data/supervised_data.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/supervised"))
    parser.add_argument("--model", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--batch-size", type=positive_int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=non_negative_int, default=5)
    parser.add_argument("--workers", type=non_negative_int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--normalize-by", type=float, default=DEFAULT_NORMALIZE_BY)
    args = parser.parse_args()

    if args.val_ratio <= 0 or args.test_ratio <= 0:
        raise SystemExit("val-ratio and test-ratio must be positive")
    if args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("val-ratio + test-ratio must be < 1")
    if args.normalize_by <= 0:
        raise SystemExit("normalize-by must be positive")

    train(args)


if __name__ == "__main__":
    main()
