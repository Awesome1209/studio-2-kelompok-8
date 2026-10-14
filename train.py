import argparse
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from evaluate import metrics
from submission.model import get_model, get_transform


def split_data(src, frac, seed):
    tr, va = Path("my_train"), Path("my_val")
    if va.exists():
        return tr, va
    rng = random.Random(seed)
    for cls in sorted(Path(src).iterdir()):
        if not cls.is_dir():
            continue
        files = sorted(cls.glob("*.jpg"))
        rng.shuffle(files)
        n_val = int(len(files) * frac)
        for i, f in enumerate(files):
            out = (va if i < n_val else tr) / cls.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./train")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_dir, va_dir = split_data(a.data, 0.15, a.seed)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr_ds = datasets.ImageFolder(tr_dir, train_tf)
    va_ds = datasets.ImageFolder(va_dir, get_transform())
    print(tr_ds.classes, len(tr_ds), len(va_ds))
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, num_workers=2)
    va_dl = DataLoader(va_ds, a.batch_size, num_workers=2)

    model = get_model()
    if not a.scratch:
        pre = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
        pre = {k: v for k, v in pre.items() if not k.startswith("fc.")}
        model.load_state_dict(pre, strict=False)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in va_dl:
                y_pred += model(x.to(device)).argmax(1).cpu().tolist()
                y_true += y.tolist()
        m = metrics(y_true, y_pred)
        print(f"epoch {ep} macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(model.state_dict(), "submission/model.pth")
    print("best macro_f1", round(best, 4))


if __name__ == "__main__":
    main()
