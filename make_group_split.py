import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

SRC = Path("train")
FRAC = 0.15
SEED = 42


def domain(url):
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


if Path("gval").exists():
    raise SystemExit("gval/ sudah ada, hapus gtrain/ dan gval/ kalau mau split ulang")

groups = defaultdict(list)
with open("train_sources.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if (SRC / r["file"]).exists():
            key = domain(r["page_url"]) or domain(r["image_url"]) or r["file"]
            groups[key].append(r["file"])

total = sum(len(v) for v in groups.values())
target = int(total * FRAC)
keys = sorted(groups)
random.Random(SEED).shuffle(keys)

val, train = [], []
for k in keys:
    if len(val) + len(groups[k]) <= target:
        val += groups[k]
    else:
        train += groups[k]

for files, dst in ((train, Path("gtrain")), (val, Path("gval"))):
    for rel in files:
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / rel, out)

print("domain:", len(groups), "| train:", len(train), "| val:", len(val))
print("val per kelas  :", dict(Counter(p.split("/")[0] for p in val)))
print("train per kelas:", dict(Counter(p.split("/")[0] for p in train)))
