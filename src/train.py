"""
src/train.py
Full training loop with validation, checkpointing, and logging.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch_size 8 --lr 1e-4
"""

import os
import argparse
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LEVIRDataset
from model import SiameseUNet, build_loss
from metrics import confusion_counts, scores_from_counts


def write_json_atomic(data, path):
    """Replace a JSON file only after its complete replacement is written."""
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temporary_path, path)


def save_state_dict_atomic(state_dict, path):
    """Prevent an interrupted save from corrupting the previous checkpoint."""
    temporary_path = f"{path}.tmp"
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, path)


def load_existing_best(save_dir, validation_signature):
    """Return the best IoU already stored in save_dir, if one exists.

    New runs write a small metadata file next to the checkpoint. For checkpoints
    created by older versions of this script, history.json is used once to
    recover the score and bootstrap the metadata file.
    """
    checkpoint_path = os.path.join(save_dir, "best_model.pth")
    metadata_path = os.path.join(save_dir, "best_model_metadata.json")
    history_path = os.path.join(save_dir, "history.json")

    if not os.path.exists(checkpoint_path):
        return 0.0, None

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        recorded_signature = metadata.get("validation_signature")
        if recorded_signature != validation_signature:
            return float("inf"), "incomparable validation configuration"
        return float(metadata["val_iou"]), "best_model_metadata.json"

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            previous_history = json.load(f)

        if previous_history:
            previous_best = max(previous_history, key=lambda row: row["val_iou"])
            metadata = {
                "epoch": int(previous_best["epoch"]),
                "val_iou": float(previous_best["val_iou"]),
                "val_f1": float(previous_best["val_f1"]),
                "val_loss": float(previous_best["val_loss"]),
                "source": "recovered from history.json",
            }
            write_json_atomic(metadata, metadata_path)
            return metadata["val_iou"], "history.json"

    # The checkpoint is valuable, but there is no trustworthy score with which
    # to compare it. Refuse to overwrite it until its metric is supplied.
    return float("inf"), "unscored checkpoint"


def save_best_metadata(save_dir, epoch, val_loss, metrics, args, validation_signature):
    """Persist the score associated with best_model.pth for future runs."""
    metadata = {
        "epoch": epoch,
        "val_iou": metrics["iou"],
        "val_f1": metrics["f1"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_loss": val_loss,
        "encoder": args.encoder,
        "fusion_mode": args.fusion_mode,
        "loss": args.loss,
        "img_size": args.img_size,
        "validation_signature": validation_signature,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source": "training validation",
    }
    write_json_atomic(
        metadata, os.path.join(save_dir, "best_model_metadata.json")
    )


# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #

def get_args():
    # Keep the CLI small and explicit so training runs are easy to reproduce.
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/raw",   type=str)
    p.add_argument("--save_dir",   default="models",     type=str)
    p.add_argument("--img_size",   default=256,          type=int)
    p.add_argument("--batch_size", default=8,            type=int)
    p.add_argument("--epochs",     default=100,          type=int)
    p.add_argument("--lr",         default=1e-4,         type=float)
    p.add_argument("--encoder",    default="resnet34",   type=str)
    p.add_argument("--num_workers",default=4,            type=int)
    p.add_argument(
        "--fusion_mode",
        choices=["early_fusion", "siamese_diff"],
        default="siamese_diff",
    )
    p.add_argument(
        "--loss",
        choices=["dice_bce", "focal_dice", "tversky"],
        default="focal_dice",
    )
    p.add_argument("--positive_crop_probability", default=0.70, type=float)
    p.add_argument("--plus_val_fraction", default=0.10, type=float)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Train / Val loop                                                            #
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for img_a, img_b, mask in tqdm(loader, desc="  train", leave=False):
        # Move the full batch to the target device before the forward pass.
        img_a, img_b, mask = img_a.to(device), img_b.to(device), mask.to(device)
        optimizer.zero_grad()
        # Forward, loss, backward, step: the standard supervised training cycle.
        logits = model(img_a, img_b)
        loss   = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        batch_size = img_a.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    tp = fp = fn = tn = 0
    for img_a, img_b, mask in tqdm(loader, desc="  val  ", leave=False):
        # Validation mirrors training data flow but skips gradient tracking.
        img_a, img_b, mask = img_a.to(device), img_b.to(device), mask.to(device)
        logits = model(img_a, img_b)
        loss   = criterion(logits, mask)
        batch_size = img_a.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        batch_tp, batch_fp, batch_fn, batch_tn = confusion_counts(logits, mask)
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn
        tn += batch_tn
    return total_loss / total_samples, scores_from_counts(tp, fp, fn)


def seed_worker(_worker_id):
    """Give each data-loading worker a deterministic independent RNG."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    # Resolve configuration first so every downstream component uses the same settings.
    args   = get_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create the output directory up front so checkpoints and logs can be saved.
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Data ----------------------------------------------------------------
    # Train and validation splits are loaded independently to keep evaluation honest.
    dataset_options = {
        "img_size": args.img_size,
        "positive_crop_probability": args.positive_crop_probability,
        "plus_val_fraction": args.plus_val_fraction,
        "split_seed": args.seed,
    }
    train_ds = LEVIRDataset(args.data_dir, split="train", **dataset_options)
    val_ds = LEVIRDataset(args.data_dir, split="val", **dataset_options)

    validation_signature = (
        "global-v1|levir-val+levir-plus-train-holdout|"
        f"fraction={args.plus_val_fraction}|seed={args.seed}|"
        f"center-crop={args.img_size}"
    )

    # Shuffle training data, but keep validation deterministic.
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ---- Model ---------------------------------------------------------------
    # The model, loss, optimizer, and scheduler are initialized together so the
    # full optimization state is defined in one place.
    model = SiameseUNet(
        encoder_name=args.encoder,
        fusion_mode=args.fusion_mode,
    ).to(device)
    criterion = build_loss(args.loss)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop -------------------------------------------------------
    # Start from the best score achieved across all earlier runs in this
    # directory. This prevents a weaker new run from overwriting the checkpoint.
    best_iou, best_source = load_existing_best(
        args.save_dir, validation_signature
    )
    if best_source in {
        "unscored checkpoint",
        "incomparable validation configuration",
    }:
        print(
            f"WARNING: Existing best_model.pth has {best_source}. "
            "It will be preserved; use a new --save_dir for this experiment."
        )
    elif best_source:
        print(f"Preserving existing global best IoU: {best_iou:.4f} ({best_source})")

    history  = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Run one full optimization pass, then evaluate on the held-out split.
        train_loss              = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_iou = val_metrics["iou"]
        val_f1 = val_metrics["f1"]
        scheduler.step()

        # Keep a compact per-epoch record for later analysis or plotting.
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_f1": val_f1,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        })

        print(f"  Train loss : {train_loss:.4f}")
        print(
            f"  Val   loss : {val_loss:.4f}  |  IoU : {val_iou:.4f}  | "
            f"F1 : {val_f1:.4f}  |  Precision : {val_metrics['precision']:.4f} "
            f"|  Recall : {val_metrics['recall']:.4f}"
        )

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            save_state_dict_atomic(
                model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
            )
            save_best_metadata(
                args.save_dir,
                epoch=epoch,
                val_loss=val_loss,
                metrics=val_metrics,
                args=args,
                validation_signature=validation_signature,
            )
            print(f"  Saved new global best model (IoU={best_iou:.4f})")

    # Always save final model
    save_state_dict_atomic(
        model.state_dict(), os.path.join(args.save_dir, "last_model.pth")
    )
    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")

    # Save training history
    # Persist the epoch metrics as JSON so downstream scripts can inspect them
    # without needing to rerun training.
    write_json_atomic(history, os.path.join(args.save_dir, "history.json"))
    history_archive = os.path.join(
        args.save_dir,
        f"history_{run_id}_{args.fusion_mode}_{args.loss}.json",
    )
    write_json_atomic(history, history_archive)
    print(f"Saved run history: {history_archive}")


if __name__ == "__main__":
    main()
