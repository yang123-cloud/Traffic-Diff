"""Train FIGR-enhanced Traffic-Diff using a stratified 70%/30% train/test split.
The test set is not used for checkpoint selection or early stopping.
"""
import os
import random
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    adjusted_mutual_info_score, classification_report, confusion_matrix)

from config import cfg
from model import TrafficDiff, pretrain_loss, finetune_loss


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(cfg.SEED)


class TrafficDataset(Dataset):
    def __init__(self, data, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.raw = torch.from_numpy(np.ascontiguousarray(data["raw_bytes"])).float()
        self.wave = torch.from_numpy(np.ascontiguousarray(data["wavelet"])).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(data["label"])).long()
        self.src = np.asarray(data["src_id"], dtype=np.int64)
        self.dst = np.asarray(data["dst_id"], dtype=np.int64)
        self.start = np.asarray(data["start_time"], dtype=np.float32)
        self.end = np.asarray(data["end_time"], dtype=np.float32)
        self.capture = np.asarray(data["capture_id"], dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, pos):
        i = int(self.indices[pos])
        return (self.raw[i], self.wave[i], self.labels[i],
                torch.tensor(self.src[i], dtype=torch.long),
                torch.tensor(self.dst[i], dtype=torch.long),
                torch.tensor(self.start[i], dtype=torch.float32),
                torch.tensor(self.end[i], dtype=torch.float32),
                torch.tensor(self.capture[i], dtype=torch.long))


def make_loaders(data):
    required = {"raw_bytes", "wavelet", "label", "src_id", "dst_id", "start_time", "end_time", "capture_id"}
    missing = sorted(required - set(data))
    if missing: raise KeyError(f"Dataset missing {missing}; run build_dataset.py again.")
    labels = np.asarray(data["label"], dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2: raise ValueError("Need at least two classes.")
    if np.any(counts < 2):
        raise ValueError("Stratified 70/30 split requires at least 2 samples per class. Increase MAX_FLOWS_PER_CLASS or inspect the PCAPs.")
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=cfg.TEST_SIZE,
        stratify=labels, random_state=cfg.SEED)
    train_set, test_set = TrafficDataset(data, train_idx), TrafficDataset(data, test_idx)
    n_classes = int(labels.max()) + 1
    train_counts = np.bincount(labels[train_idx], minlength=n_classes)
    active_counts = train_counts[train_counts > 0]
    mean_count = active_counts.mean() if len(active_counts) else 1.0
    weights = np.sqrt(mean_count / np.maximum(train_counts, 1))
    weights[train_counts == 0] = 0.0
    weights[weights > 0] /= weights[weights > 0].mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)
    common = dict(batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                  pin_memory=torch.cuda.is_available(), drop_last=False)
    pretrain_loader = DataLoader(train_set, shuffle=True, **common)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return pretrain_loader, train_loader, test_loader, train_counts, class_weights, n_classes


def unpack(batch, device):
    raw, wave, y, src, dst, start, end, capture = batch
    raw, wave, y = raw.to(device, non_blocking=True), wave.to(device, non_blocking=True), y.to(device, non_blocking=True)
    meta = (src.to(device, non_blocking=True), dst.to(device, non_blocking=True),
            start.to(device, non_blocking=True), end.to(device, non_blocking=True),
            capture.to(device, non_blocking=True))
    return raw, wave, y, meta


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval(); ys, ps = [], []
    for batch in loader:
        raw, wave, y, meta = unpack(batch, device)
        logits, _, _, _ = model(raw, wave, mode="eval", graph_meta=meta)
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(logits.argmax(1).cpu().numpy().tolist())
    labels = list(range(n_classes))
    return {"ACC": accuracy_score(ys, ps),
            "PRE": precision_score(ys, ps, labels=labels, average="macro", zero_division=0),
            "REC": recall_score(ys, ps, labels=labels, average="macro", zero_division=0),
            "MacroF1": f1_score(ys, ps, labels=labels, average="macro", zero_division=0),
            "WeightedF1": f1_score(ys, ps, labels=labels, average="weighted", zero_division=0),
            "AMI": adjusted_mutual_info_score(ys, ps), "y_true": ys, "y_pred": ps}


def run_pretrain(model, loader, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR * 0.5, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    path = os.path.join(cfg.DATA_DIR, "pretrained_figr.pt")
    best, stale = float("inf"), 0
    for epoch in range(cfg.PRETRAIN_EPOCHS):
        model.train(); losses = []
        progress = epoch / max(cfg.PRETRAIN_EPOCHS - 1, 1)
        for batch in loader:
            raw, wave, _, meta = unpack(batch, device)
            output = model(raw, wave, mode="pretrain", graph_meta=meta, progress=progress)
            loss = pretrain_loss(output, raw, wave, cfg)
            if not torch.isfinite(loss):
                warnings.warn("Non-finite pretraining loss; skipped batch."); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        avg = float(np.mean(losses)) if losses else float("inf")
        scheduler.step(avg)
        ratio = cfg.MASK_RATIO_START + progress * (cfg.MASK_RATIO_END - cfg.MASK_RATIO_START)
        print(f"Pretrain {epoch+1:3d}/{cfg.PRETRAIN_EPOCHS} | mask={ratio:.2f} | train loss={avg:.5f}")
        if np.isfinite(avg) and avg < best - 1e-6:
            best, stale = avg, 0; torch.save(model.state_dict(), path)
        else: stale += 1
        if stale >= cfg.PATIENCE:
            print(f"Pretraining early stop: training loss did not improve for {cfg.PATIENCE} epochs."); break
    if not os.path.exists(path): raise RuntimeError("Pretraining checkpoint not created.")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    print(f"Loaded best pretraining checkpoint (train loss={best:.5f}).")
    return model


def run_finetune(model, loader, device, class_weights, n_classes):
    head_keys = ("classifier", "aux_temporal", "aux_wavelet", "aux_graph", "feature_norm")
    backbone, heads = [], []
    for name, p in model.named_parameters():
        (heads if any(k in name for k in head_keys) else backbone).append(p)
    opt = torch.optim.AdamW([{"params": backbone, "lr": cfg.LR * 0.3},
                             {"params": heads, "lr": cfg.LR}], weight_decay=cfg.WEIGHT_DECAY)
    path = os.path.join(cfg.DATA_DIR, "best_model_figr.pt")
    class_weights = class_weights.to(device)
    best_acc, best_f1, best_loss, stale = -1.0, -1.0, float("inf"), 0
    for epoch in range(cfg.FINETUNE_EPOCHS):
        model.train(); losses = []
        for batch in loader:
            raw, wave, y, meta = unpack(batch, device)
            logits, aux_t, aux_w, aux_g = model(raw, wave, mode="finetune", graph_meta=meta)
            loss = finetune_loss(logits, aux_t, aux_w, aux_g, y, class_weights=class_weights)
            if not torch.isfinite(loss):
                warnings.warn("Non-finite fine-tuning loss; skipped batch."); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        avg_loss = float(np.mean(losses)) if losses else float("inf")
        # Training metrics are used only for checkpoint selection because user requested
        # exactly 70/30 and no validation split. Final test set is never consulted here.
        train_metrics = evaluate(model, loader, device, n_classes)
        acc, macro = train_metrics["ACC"], train_metrics["MacroF1"]
        print(f"Finetune {epoch+1:3d}/{cfg.FINETUNE_EPOCHS} | loss={avg_loss:.5f} | train ACC={acc:.4f} | train MacroF1={macro:.4f}")
        improved = (acc > best_acc + 1e-6 or
                    (abs(acc-best_acc) <= 1e-6 and macro > best_f1 + 1e-6) or
                    (abs(acc-best_acc) <= 1e-6 and abs(macro-best_f1) <= 1e-6 and avg_loss < best_loss))
        if improved:
            best_acc, best_f1, best_loss, stale = acc, macro, avg_loss, 0
            torch.save(model.state_dict(), path)
        else: stale += 1
        if epoch >= 4 and acc >= cfg.EARLY_STOP_ACC:
            print(f"Early stop: training ACC reached {acc:.4f} >= {cfg.EARLY_STOP_ACC:.2f}."); break
        if stale >= cfg.PATIENCE:
            print(f"Fine-tuning early stop: training metrics did not improve for {cfg.PATIENCE} epochs."); break
    if not os.path.exists(path): raise RuntimeError("Fine-tuning checkpoint not created.")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    print(f"Loaded best training checkpoint: ACC={best_acc:.4f}, MacroF1={best_f1:.4f}.")
    return model


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    path = os.path.join(cfg.DATA_DIR, cfg.DATA_FILE)
    if not os.path.exists(path): raise FileNotFoundError(f"Dataset not found: {path}. Run build_dataset.py first.")
    with np.load(path, allow_pickle=False) as archive: data = {k: archive[k] for k in archive.files}
    pretrain_loader, train_loader, test_loader, counts, weights, n_classes = make_loaders(data)
    print(f"Classes: {n_classes}")
    print(f"Train/Test: {len(train_loader.dataset)}/{len(test_loader.dataset)} ({1-cfg.TEST_SIZE:.0%}/{cfg.TEST_SIZE:.0%})")
    print(f"Train class counts: {counts.tolist()}")
    print(f"Class weights: {weights.numpy().round(3).tolist()}")
    model = TrafficDiff(d_model=cfg.D_MODEL, n_classes=n_classes, packet_bytes=cfg.PACKET_BYTES).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")
    for name in ("pretrained_figr.pt", "best_model_figr.pt"):
        p = os.path.join(cfg.DATA_DIR, name)
        if os.path.exists(p): os.remove(p); print(f"Removed old checkpoint: {p}")
    print("\n========== Stage 1: Curriculum Graph-Conditioned Denoising ==========")
    model = run_pretrain(model, pretrain_loader, device)
    print("\n========== Stage 2: Graph-aware Fine-tuning ==========")
    model = run_finetune(model, train_loader, device, weights, n_classes)
    # The held-out 30% test split is evaluated once, after checkpoint selection.
    test = evaluate(model, test_loader, device, n_classes)
    print("\n========== FINAL TEST (held-out 30%) ==========")
    for key in ("ACC", "PRE", "REC", "MacroF1", "WeightedF1", "AMI"): print(f"{key:12s}: {test[key]:.4f}")
    classes_path = os.path.join(cfg.DATA_DIR, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f: names = [x.strip() for x in f if x.strip()]
    else: names = []
    names += [f"class_{i}" for i in range(len(names), n_classes)]
    names = names[:n_classes]
    print("\n========== Classification Report ==========")
    print(classification_report(test["y_true"], test["y_pred"], labels=list(range(n_classes)), target_names=names, zero_division=0, digits=4))
    print("========== Confusion Matrix ==========")
    print(confusion_matrix(test["y_true"], test["y_pred"], labels=list(range(n_classes))))
    print(f"Checkpoints saved under: {cfg.DATA_DIR}")

if __name__ == "__main__": main()

