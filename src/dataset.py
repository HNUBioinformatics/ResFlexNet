"""Dataset V3 - 直接加载V2预处理的.pt文件"""
import os
import glob
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_bfactor_arrays(data):
    if 'bfactors_raw' in data:
        raw = tensor_to_numpy(data['bfactors_raw']).reshape(-1)
    elif 'bfactors' in data and 'bfactors_norm' in data:
        raw = tensor_to_numpy(data['bfactors']).reshape(-1)
    else:
        raw = None

    if 'bfactors_norm' in data:
        norm = tensor_to_numpy(data['bfactors_norm']).reshape(-1)
    elif 'bfactors' in data and 'bf_mean' in data and 'bf_std' in data:
        raw_for_norm = tensor_to_numpy(data['bfactors']).reshape(-1)
        std = float(data['bf_std'])
        mean = float(data['bf_mean'])
        norm = (raw_for_norm - mean) / std if abs(std) > 1e-12 else None
    else:
        norm = None

    return raw, norm


def passes_label_filter(pt_path, config):
    if not getattr(config, 'filter_abnormal_labels', False):
        return True, ''

    try:
        data = torch.load(pt_path, map_location='cpu')
        raw, norm = get_bfactor_arrays(data)
    except Exception as exc:
        return False, f'load_error={exc}'

    if raw is None or norm is None or len(raw) < 2 or len(norm) < 2:
        return False, 'missing_or_short_bfactor'

    raw = raw[np.isfinite(raw)]
    norm = norm[np.isfinite(norm)]
    if len(raw) < 2 or len(norm) < 2:
        return False, 'nonfinite_bfactor'

    raw_std = float(np.std(raw))
    raw_range = float(np.max(raw) - np.min(raw))
    norm_range = float(np.max(norm) - np.min(norm))

    min_raw_std = getattr(config, 'min_target_raw_std', 0.0)
    max_norm_range = getattr(config, 'max_target_norm_range', float('inf'))
    max_raw_range = getattr(config, 'max_target_raw_range', float('inf'))

    if raw_std < min_raw_std:
        return False, f'raw_std={raw_std:.4g}<{min_raw_std}'
    if norm_range > max_norm_range:
        return False, f'norm_range={norm_range:.4g}>{max_norm_range}'
    if raw_range > max_raw_range:
        return False, f'raw_range={raw_range:.4g}>{max_raw_range}'

    if getattr(config, 'filter_experiment_quality', False):
        resolution = data.get('resolution', None)
        r_value = data.get('r_value', None)
        keep_missing = getattr(config, 'keep_missing_quality', True)

        if torch.is_tensor(resolution):
            resolution = float(resolution.view(-1)[0].item())
        if torch.is_tensor(r_value):
            r_value = float(r_value.view(-1)[0].item())

        if resolution is None and not keep_missing:
            return False, 'missing_resolution'
        if r_value is None and not keep_missing:
            return False, 'missing_r_value'

        max_resolution = getattr(config, 'max_resolution', float('inf'))
        max_r_value = getattr(config, 'max_r_value', float('inf'))

        if resolution is not None and np.isfinite(float(resolution)) and float(resolution) > max_resolution:
            return False, f'resolution={float(resolution):.4g}>{max_resolution}'
        if r_value is not None and np.isfinite(float(r_value)) and float(r_value) > max_r_value:
            return False, f'r_value={float(r_value):.4g}>{max_r_value}'

    return True, ''


class BFactorDataset(Dataset):
    def __init__(self, pt_files):
        self.files = pt_files
        print(f"  Dataset: {len(self.files)} proteins")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu')

        bfactors_norm = data['bfactors_norm'] if 'bfactors_norm' in data else data['bfactors']

        if 'bfactors_raw' in data:
            bfactors_raw = data['bfactors_raw']
        else:
            bfactors_raw = data['bfactors']

        length = data.get('length', data.get('num_residues', len(bfactors_norm)))
        pdb_id = data.get('pdb_id', data.get('name', 'unknown'))

        # 确保必要字段存在
        result = {
            'node_feats': data['node_feats'],       # [N, 72]
            'coords': data['coords'],               # [N, 3]
            'edge_index': data['edge_index'],       # [2, E]
            'edge_feats': data['edge_feats'],       # [E, 68]
            'edge_dist': data['edge_dist'],         # [E]
            'bfactors': bfactors_norm,              # [N] normalized
            'bfactors_raw': bfactors_raw,           # [N] raw
            'num_residues': length,
            'pdb_id': pdb_id,
        }

        # ESM embedding
        if 'esm_embedding' in data and data['esm_embedding'] is not None:
            result['esm_embedding'] = data['esm_embedding']  # [N, 1536]
        elif 'esm3_embed' in data and data['esm3_embed'] is not None:
            result['esm_embedding'] = data['esm3_embed']
        elif 'esm3_embedding' in data and data['esm3_embedding'] is not None:
            result['esm_embedding'] = data['esm3_embedding']

        # norm_params for denormalization
        if 'norm_params' in data:
            result['norm_params'] = data['norm_params']

        return result


def get_dataloaders(config):
    """加载V2预处理的数据并划分train/val/test"""
    pt_files = sorted(glob.glob(os.path.join(config.processed_dir, '*.pt')))
    print(f"Found {len(pt_files)} processed .pt files")
    
    if len(pt_files) == 0:
        raise RuntimeError(f"No .pt files found in {config.processed_dir}")

    if getattr(config, 'filter_abnormal_labels', False):
        kept_files = []
        dropped = []
        for path in pt_files:
            keep, reason = passes_label_filter(path, config)
            if keep:
                kept_files.append(path)
            else:
                dropped.append((path, reason))
        print(f"Label filter: kept={len(kept_files)}, dropped={len(dropped)}")
        if dropped:
            for path, reason in dropped[:20]:
                print(f"  [DROP] {os.path.basename(path)}: {reason}")
            if len(dropped) > 20:
                print(f"  ... {len(dropped) - 20} more dropped")
        pt_files = kept_files

        if len(pt_files) == 0:
            raise RuntimeError("No .pt files left after label filtering")
    
    # 固定随机种子划分
    split_file = getattr(config, 'split_file', None)
    if split_file is None:
        split_file = os.path.join(os.path.dirname(config.processed_dir), 'splits_v3.json')

    file_by_name = {os.path.basename(path): path for path in pt_files}

    if os.path.exists(split_file):
        print(f"Loading fixed split from: {split_file}")
        with open(split_file, 'r') as f:
            splits = json.load(f)

        missing = []
        train_files = []
        val_files = []
        test_files = []

        for name in splits.get('train', []):
            if name in file_by_name:
                train_files.append(file_by_name[name])
            else:
                missing.append(name)
        for name in splits.get('val', []):
            if name in file_by_name:
                val_files.append(file_by_name[name])
            else:
                missing.append(name)
        for name in splits.get('test', []):
            if name in file_by_name:
                test_files.append(file_by_name[name])
            else:
                missing.append(name)

        if missing and os.environ.get('RESFLEXNET_ALLOW_PARTIAL_SPLIT') != '1':
            raise FileNotFoundError(f'{len(missing)} split inputs missing or filtered: {", ".join(missing[:10])}')
        if missing:
            print(f"  [WARN] {len(missing)} files in split are missing from processed_dir/filter")
            for name in missing[:20]:
                print(f"    missing: {name}")
            if len(missing) > 20:
                print(f"    ... {len(missing) - 20} more")

        if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
            raise RuntimeError(
                f"Invalid fixed split loaded from {split_file}: "
                f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
            )
    else:
        if os.environ.get('RESFLEXNET_ALLOW_NEW_SPLIT') != '1':
            raise FileNotFoundError(f'Original fixed split not found: {split_file}. For new experiments only, set RESFLEXNET_ALLOW_NEW_SPLIT=1 and use a separate output path.')
        print(f"Creating a NEW split: {split_file}")
        rng = np.random.default_rng(int(os.environ.get("RESFLEXNET_SPLIT_SEED", "42")))
        indices = rng.permutation(len(pt_files))

        n_total = len(pt_files)
        n_train = int(n_total * config.train_ratio)
        n_val = int(n_total * config.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_files = [pt_files[i] for i in train_idx]
        val_files = [pt_files[i] for i in val_idx]
        test_files = [pt_files[i] for i in test_idx]

        splits = {
            'seed': int(os.environ.get("RESFLEXNET_SPLIT_SEED", "42")),
            'processed_dir': config.processed_dir,
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
            'test_ratio': config.test_ratio,
            'train': [os.path.basename(p) for p in train_files],
            'val': [os.path.basename(p) for p in val_files],
            'test': [os.path.basename(p) for p in test_files],
        }
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)
    
    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    train_dataset = BFactorDataset(train_files)
    val_dataset = BFactorDataset(val_files)
    test_dataset = BFactorDataset(test_files)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
