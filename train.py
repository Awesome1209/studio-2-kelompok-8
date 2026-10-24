import argparse
import copy
import math
import time

import timm
import torch
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evaluate import metrics
from submission2.model import IMG, NAME, get_model, get_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="gtrain")
    ap.add_argument("--val", default="gval")
    ap.add_argument("--out", default="submission2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.25),
    ])
    tr_ds = datasets.ImageFolder(a.train, train_tf)
    va_ds = datasets.ImageFolder(a.val, get_transform())
    print(NAME, IMG, tr_ds.classes, len(tr_ds), len(va_ds), device)
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers)
    va_dl = DataLoader(va_ds, a.batch_size, num_workers=a.workers)

    model = get_model()
    own = model.state_dict()
    pre = timm.create_model(NAME, pretrained=True).state_dict()
    pre = {k: v for k, v in pre.items() if k in own and own[k].shape == v.shape}
    print("bobot pretrained dimuat:", len(pre), "dari", len(own))
    model.load_state_dict(pre, strict=False)
    model.to(device)
    ema = copy.deepcopy(model).eval()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(tr_dl)
    warm = len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    mix = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5,
                label_smoothing=0.1, num_classes=5)
    loss_fn = SoftTargetCrossEntropy()

    step, best = 0, 0.0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, y in tr_dl:
            x, y = mix(x.to(device), y.to(device))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            total += loss.item()

            d = min(0.995, (1 + step) / (10 + step))
            with torch.no_grad():
                for e, p in zip(ema.state_dict().values(), model.state_dict().values()):
                    if e.dtype.is_floating_point:
                        e.mul_(d).add_(p.detach(), alpha=1 - d)
                    else:
                        e.copy_(p)

        y_true, y_pred = [], []
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            for x, y in va_dl:
                y_pred += ema(x.to(device)).argmax(1).cpu().tolist()
                y_true += y.tolist()
        m = metrics(y_true, y_pred)
        print(f"epoch {ep} loss={total / len(tr_dl):.4f} macro_f1={m['macro_f1']:.4f} "
              f"acc={m['accuracy']:.4f} ({time.time() - t0:.0f}s)")
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(ema.state_dict(), f"{a.out}/model.pth")
    print("best macro_f1", round(best, 4))


if __name__ == "__main__":
    main()
