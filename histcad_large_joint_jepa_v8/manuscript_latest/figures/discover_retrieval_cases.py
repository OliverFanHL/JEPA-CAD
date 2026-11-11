"""Locate contrasting held-out examples within the matched retrieval cohort."""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "histcad_large_joint_jepa_v8/benchmarking"
sys.path.insert(0, str(BENCH))
from src.cad_benchmark.config import load_config  # noqa: E402
from src.cad_benchmark.registry import Registry  # noqa: E402

torch.set_num_threads(4)
r = Registry(load_config(BENCH / "config.yaml"), apply_shared_coverage=True)
names = ["jepa_joint", "graph_autoencoder_joint", "brepnet_official_histcad"]
reps = [r.load(x) for x in names]
test = reps[0].indices("test")
subset = test[::10]
targets = reps[0].targets["dimensions"]
source = torch.load(ROOT / "histcad_large_joint_jepa_v8/outputs/full_dataset_4096/representations.pt",
                    map_location="cpu", weights_only=False, mmap=True)
meta = {row["uid"]: row for row in source["metadata"]}
results = {}
for rep in reps:
    bank = F.normalize(rep.values[test].float(), dim=1)
    query = F.normalize(rep.values[subset].float(), dim=1)
    sim = query @ bank.T
    sim[torch.arange(len(subset)), torch.searchsorted(test, subset)] = -torch.inf
    picked = test[sim.argmax(dim=1)]
    error = (targets[subset] - targets[picked]).abs().mean(dim=1)
    results[rep.name] = (picked, error)
ranked = []
for i, index in enumerate(subset):
    js = float(results[names[0]][1][i])
    ae = float(results[names[1]][1][i])
    bn = float(results[names[2]][1][i])
    if js + .10 < min(ae, bn) and js < .17:
        uid = reps[0].uids[int(index)]
        ranked.append((min(ae, bn) - js, uid, js, ae, bn,
                       *(reps[0].uids[int(results[n][0][i])] for n in names),
                       meta[uid].get("functional_type", "")))
for item in sorted(ranked, reverse=True)[:90]:
    print(item)
