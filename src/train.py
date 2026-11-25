import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import random

from config_v6 import Config
from model_v3 import BFactorPredictor
from loss_v6 import CombinedLoss
from dataset_v6 import get_dataloaders


class EMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep this False for speed; set True only if you need exact reproducibility.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def compute_metrics(pred, target):
    import numpy as _np
    from scipy import stats as _stats

    if hasattr(pred, "detach"):
        pred_np = pred.detach().float().cpu().numpy().reshape(-1)
    else:
        pred_np = _np.asarray(pred, dtype=_np.float32).reshape(-1)

    if hasattr(target, "detach"):
        target_np = target.detach().float().cpu().numpy().reshape(-1)
    else:
        target_np = _np.asarray(target, dtype=_np.float32).reshape(-1)

    if pred_np.shape[0] != target_np.shape[0]:
        raise RuntimeError(f"compute_metrics length mismatch: pred={pred_np.shape}, target={target_np.shape}")

    mask = _np.isfinite(pred_np) & _np.isfinite(target_np)
    pred_np = pred_np[mask]
    target_np = target_np[mask]

    n = pred_np.shape[0]
    if n == 0:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "pcc": 0.0, "scc": 0.0, "spearman": 0.0, "r2": 0.0}

    diff = pred_np - target_np
    mae = float(_np.mean(_np.abs(diff)))
    mse = float(_np.mean(diff ** 2))
    rmse = float(_np.sqrt(mse))

    if n >= 2 and _np.std(pred_np) > 1e-12 and _np.std(target_np) > 1e-12:
        pcc = float(_stats.pearsonr(pred_np, target_np)[0])
        scc = float(_stats.spearmanr(pred_np, target_np)[0])
        if not _np.isfinite(pcc):
            pcc = 0.0
        if not _np.isfinite(scc):
            scc = 0.0
    else:
        pcc = 0.0
        scc = 0.0

    ss_res = float(_np.sum((target_np - pred_np) ** 2))
    ss_tot = float(_np.sum((target_np - _np.mean(target_np)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    if not _np.isfinite(r2):
        r2 = 0.0

    return {"mae": mae, "mse": mse, "rmse": rmse, "pcc": pcc, "scc": scc, "spearman": scc, "r2": r2}


def move_to_device(data, device):
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def get_scheduler(optimizer, config, steps_per_epoch):
    """Cosine with warmup, stepped per optimizer update"""
    total_steps = config.max_epochs * max(steps_per_epoch, 1)
    warmup_steps = config.warmup_epochs * max(steps_per_epoch, 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, config, ema=None):
    model.train()
    total_loss = 0.0
    total_metrics = {'mse': 0.0, 'mae': 0.0, 'pcc': 0.0, 'scc': 0.0}
    count = 0

    optimizer.zero_grad(set_to_none=True)

    for i, data in enumerate(loader):
        data = move_to_device(data, config.device)
        target = data['bfactors']
        target_flat = target.view(-1)

        if config.label_noise_std > 0:
            noise = torch.randn_like(target) * config.label_noise_std
            target_noisy = (target + noise).view(-1)
        else:
            target_noisy = target_flat

        if config.mixed_precision:
            with torch.cuda.amp.autocast():
                pred = model(data).view(-1)
                if pred.numel() != target_noisy.numel():
                    raise RuntimeError(f"pred/target_noisy mismatch: pred={tuple(pred.shape)}, target_noisy={tuple(target_noisy.shape)}")
                loss, loss_dict = criterion(pred, target_noisy)
                loss = loss / config.accumulation_steps
            scaler.scale(loss).backward()
        else:
            pred = model(data).view(-1)
            if pred.numel() != target_noisy.numel():
                raise RuntimeError(f"pred/target_noisy mismatch: pred={tuple(pred.shape)}, target_noisy={tuple(target_noisy.shape)}")
            loss, loss_dict = criterion(pred, target_noisy)
            loss = loss / config.accumulation_steps
            loss.backward()

        do_update = ((i + 1) % config.accumulation_steps == 0) or ((i + 1) == len(loader))
        if do_update:
            if config.mixed_precision:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            m = compute_metrics(pred, target_flat)
        total_loss += float(loss_dict['total'])
        for k in total_metrics:
            total_metrics[k] += float(m[k])
        count += 1

    avg = {k: v / max(count, 1) for k, v in total_metrics.items()}
    avg['loss'] = total_loss / max(count, 1)
    return avg


@torch.no_grad()
def evaluate(model, loader, criterion, config):
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_metrics = {'mse': 0.0, 'mae': 0.0, 'pcc': 0.0, 'scc': 0.0}
    count = 0
    per_protein = []

    for data in loader:
        data = move_to_device(data, config.device)
        target = data['bfactors'].to(device).view(-1)

        if config.mixed_precision:
            with torch.cuda.amp.autocast():
                pred = model(data).view(-1)
                if pred.numel() != target.numel():
                    raise RuntimeError(f"pred/target mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
                loss, loss_dict = criterion(pred, target)
        else:
            pred = model(data).view(-1)
            if pred.numel() != target.numel():
                raise RuntimeError(f"pred/target mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
            loss, loss_dict = criterion(pred, target)

        m = compute_metrics(pred, target)
        total_loss += float(loss_dict['total'])
        for k in total_metrics:
            total_metrics[k] += float(m[k])
        count += 1

        pdb_id = data.get('pdb_id', 'unknown')
        if isinstance(pdb_id, (list, tuple)) and len(pdb_id) > 0:
            pdb_id = pdb_id[0]
        num_res = data.get('num_residues', pred.shape[0])
        if isinstance(num_res, (list, tuple)) and len(num_res) > 0:
            num_res = num_res[0]
        if torch.is_tensor(num_res):
            num_res = int(num_res.item())

        per_protein.append({
            'pdb_id': str(pdb_id),
            'pcc': m['pcc'],
            'scc': m['scc'],
            'mse': m['mse'],
            'mae': m['mae'],
            'num_residues': int(num_res)
        })

    avg = {k: v / max(count, 1) for k, v in total_metrics.items()}
    avg['loss'] = total_loss / max(count, 1)
    return avg, per_protein


@torch.no_grad()
def evaluate_ensemble(model, ckpt_paths, loader, criterion, config):
    """Load top-k checkpoints and average predictions."""
    device = next(model.parameters()).device
    model.eval()

    state_dicts = []
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        state_dicts.append(ckpt['model_state_dict'])

    total_loss = 0.0
    total_metrics = {'mse': 0.0, 'mae': 0.0, 'pcc': 0.0, 'scc': 0.0}
    count = 0
    per_protein = []

    for data in loader:
        data = move_to_device(data, config.device)
        target = data['bfactors'].to(device).view(-1)

        preds = []
        for sd in state_dicts:
            model.load_state_dict(sd, strict=True)
            if config.mixed_precision:
                with torch.cuda.amp.autocast():
                    pred_i = model(data).view(-1)
            else:
                pred_i = model(data).view(-1)
            preds.append(pred_i)

        pred = torch.stack(preds, dim=0).mean(dim=0)

        if pred.numel() != target.numel():
            raise RuntimeError(f"ensemble pred/target mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

        loss, loss_dict = criterion(pred, target)
        m = compute_metrics(pred, target)

        total_loss += float(loss_dict['total'])
        for k in total_metrics:
            total_metrics[k] += float(m[k])
        count += 1

        pdb_id = data.get('pdb_id', 'unknown')
        if isinstance(pdb_id, (list, tuple)) and len(pdb_id) > 0:
            pdb_id = pdb_id[0]
        num_res = data.get('num_residues', pred.shape[0])
        if isinstance(num_res, (list, tuple)) and len(num_res) > 0:
            num_res = num_res[0]
        if torch.is_tensor(num_res):
            num_res = int(num_res.item())

        per_protein.append({
            'pdb_id': str(pdb_id),
            'pcc': m['pcc'],
            'scc': m['scc'],
            'mse': m['mse'],
            'mae': m['mae'],
            'num_residues': int(num_res)
        })

    avg = {k: v / max(count, 1) for k, v in total_metrics.items()}
    avg['loss'] = total_loss / max(count, 1)
    return avg, per_protein


def update_topk_checkpoints(model, optimizer, config, epoch, val_pcc, ema, topk_list, topk=3):
    save_dir = config.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)

    should_save = (len(topk_list) < topk) or (val_pcc > min(x["pcc"] for x in topk_list))
    if not should_save:
        return topk_list

    if ema is not None:
        ema.apply_shadow(model)

    save_path = os.path.join(save_dir, f"top_epoch{epoch:03d}_pcc{val_pcc:.4f}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_pcc': val_pcc,
        'config': config.__dict__,
    }, save_path)

    if ema is not None:
        ema.restore(model)

    topk_list.append({'path': save_path, 'pcc': float(val_pcc), 'epoch': int(epoch)})
    topk_list = sorted(topk_list, key=lambda x: x['pcc'], reverse=True)

    while len(topk_list) > topk:
        rm = topk_list.pop(-1)
        if os.path.exists(rm['path']):
            os.remove(rm['path'])

    return topk_list


def main():
    config = Config()
    set_seed(config.seed)
    config.mixed_precision = False

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    print("=" * 60)
    print("B-Factor Prediction V3")
    print("=" * 60)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = str(device)
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(config)

    model = BFactorPredictor(config).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    criterion = CombinedLoss(config)
    esm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("esm_adapter."):
            esm_params.append(param)
        else:
            other_params.append(param)

    esm_lr = config.learning_rate * getattr(config, "esm_adapter_lr_scale", 0.2)
    print(f"Optimizer: base_lr={config.learning_rate:.2e}, esm_adapter_lr={esm_lr:.2e}")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": other_params,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": esm_params,
                "lr": esm_lr,
                "weight_decay": config.weight_decay,
            },
        ]
    )

    updates_per_epoch = (len(train_loader) + config.accumulation_steps - 1) // config.accumulation_steps
    scheduler = get_scheduler(optimizer, config, updates_per_epoch)

    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    ema = EMA(model, config.ema_decay) if config.use_ema else None

    best_val_pcc = -float('inf')
    patience_counter = 0
    min_delta = getattr(config, "early_stop_min_delta", 1e-5)
    best_ckpt_path = os.path.join(config.checkpoint_dir, 'best_model_v3.pt')
    history = []

    topk = 3
    topk_ckpts = []

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, config, ema)

        if ema is not None:
            ema.apply_shadow(model)
        val_metrics, _ = evaluate(model, val_loader, criterion, config)
        if ema is not None:
            ema.restore(model)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d}/{config.max_epochs} ({elapsed:.1f}s) | "
            f"lr={lr_now:.2e} | "
            f"Train loss={train_metrics['loss']:.4f} PCC={train_metrics['pcc']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} PCC={val_metrics['pcc']:.4f} SCC={val_metrics['scc']:.4f}"
        )

        history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics, 'lr': lr_now})

        topk_ckpts = update_topk_checkpoints(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            val_pcc=val_metrics['pcc'],
            ema=ema,
            topk_list=topk_ckpts,
            topk=topk
        )

        improved = val_metrics['pcc'] > (best_val_pcc + min_delta)
        if improved:
            best_val_pcc = val_metrics['pcc']
            patience_counter = 0

            if ema is not None:
                ema.apply_shadow(model)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_pcc': best_val_pcc,
                'config': config.__dict__,
            }, best_ckpt_path)
            print(f"  >> Saved best model to {best_ckpt_path} (PCC={best_val_pcc:.4f})")

            if ema is not None:
                ema.restore(model)
        else:
            patience_counter += 1
            print(f"  >> No improvement. patience {patience_counter}/{config.patience}")
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\n" + "=" * 60)
    print("Testing model (top-k ensemble if available)...")
    print("=" * 60)

    if len(topk_ckpts) > 0:
        topk_ckpts = sorted(topk_ckpts, key=lambda x: x['pcc'], reverse=True)[:topk]
        ckpt_paths = [x['path'] for x in topk_ckpts]
        print("Using ensemble checkpoints:")
        for x in topk_ckpts:
            print(f"  - epoch {x['epoch']:3d} | pcc={x['pcc']:.4f} | {x['path']}")
        test_metrics, per_protein = evaluate_ensemble(model, ckpt_paths, test_loader, criterion, config)
        best_epoch_for_log = topk_ckpts[0]['epoch']
        best_val_for_log = topk_ckpts[0]['pcc']
    else:
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        test_metrics, per_protein = evaluate(model, test_loader, criterion, config)
        best_epoch_for_log = ckpt['epoch']
        best_val_for_log = ckpt.get('val_pcc', best_val_pcc)

    print("\nTest Results:")
    print(f"  MSE:  {test_metrics['mse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  PCC:  {test_metrics['pcc']:.4f}")
    print(f"  SCC:  {test_metrics['scc']:.4f}")
    print(f"  Loss: {test_metrics['loss']:.4f}")

    print(f"\nPer-protein results ({len(per_protein)} proteins):")
    for p in sorted(per_protein, key=lambda x: x['pcc'], reverse=True):
        print(f"  {p['pdb_id']:6s} | L={p['num_residues']:4d} | PCC={p['pcc']:.4f} | SCC={p['scc']:.4f} | MAE={p['mae']:.4f}")

    results = {
        'test_metrics': test_metrics,
        'per_protein': per_protein,
        'history': history,
        'best_epoch': int(best_epoch_for_log),
        'best_val_pcc': float(best_val_for_log),
        'topk_ckpts': topk_ckpts
    }
    results_path = os.path.join(config.log_dir, 'results_v3.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
