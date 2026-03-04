"""
Train ResNet-50 for DR grading (0-4) on CROP images with train_1/valid/test CSVs.
Metrics: QWK, Accuracy, Macro F1, One-off accuracy, confusion matrix.

Run (from classify/DR):
  CUDA_VISIBLE_DEVICES=0 python train.py
  CUDA_VISIBLE_DEVICES=0 python train.py --eval_test
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR,
    NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY, IMAGE_SIZE,
    CHECKPOINT_DIR, LOG_DIR,
)
from dataset import AptosCropDataset, get_transforms
from metrics import compute_metrics, print_metrics


def get_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_pred, all_label = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_label.append(y.cpu().numpy())
    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_label, axis=0)
    return compute_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on APTOS CROP (0-4)")
    parser.add_argument("--train_csv", default=TRAIN_CSV, help="Train label CSV")
    parser.add_argument("--val_csv", default=VAL_CSV, help="Val label CSV")
    parser.add_argument("--test_csv", default=TEST_CSV, help="Test label CSV")
    parser.add_argument("--train_img", default=TRAIN_IMG_DIR, help="Train image dir")
    parser.add_argument("--val_img", default=VAL_IMG_DIR, help="Val image dir")
    parser.add_argument("--test_img", default=TEST_IMG_DIR, help="Test image dir")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--eval_test", action="store_true", help="Evaluate on test set after training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = AptosCropDataset(
        args.train_csv,
        args.train_img,
        transform=get_transforms(args.image_size, is_train=True),
        is_train=True,
    )
    val_ds = AptosCropDataset(
        args.val_csv,
        args.val_img,
        transform=get_transforms(args.image_size, is_train=False),
        is_train=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_qwk = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        train_loss = running_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader, device)
        val_qwk = val_metrics["qwk"]
        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_qwk={val_qwk:.4f}  val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            ckpt_path = os.path.join(args.save_dir, "best_resnet50_aptos.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": val_qwk,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint to {ckpt_path}")

    print("\n--- Best validation ---")
    ckpt = torch.load(
        os.path.join(args.save_dir, "best_resnet50_aptos.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    val_metrics = evaluate(model, val_loader, device)
    print_metrics(val_metrics, prefix="Val ")

    if args.eval_test:
        test_ds = AptosCropDataset(
            args.test_csv,
            args.test_img,
            transform=get_transforms(args.image_size, is_train=False),
            is_train=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        test_metrics = evaluate(model, test_loader, device)
        print("\n--- Test ---")
        print_metrics(test_metrics, prefix="Test ")


if __name__ == "__main__":
    main()
