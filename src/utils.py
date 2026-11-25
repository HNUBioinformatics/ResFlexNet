import os
import sys
import random
import logging
import numpy as np
import torch
from typing import Optional, Dict
from typing import List, Tuple, Dict, Optional
def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers: return logger
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 氨基酸数据
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
THREE_TO_ONE = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y','MSE':'M'}
AA_PROPERTIES = {'A': [0.62,0.13,0.47,0.17,0.36,0,0,0,0,1,0.00,0.77,0.53,0.58,0.25,0.44,0,0.00,0.00,0.00], 'C': [0.29,0.18,0.37,0.23,0.35,0,0,1,0,0,0.58,0.51,0.70,0.57,0.31,0.39,1,0.25,0.25,0.17], 'D': [0.03,0.20,0.19,0.24,0.51,0,1,1,0,0,0.24,0.58,0.37,0.77,0.35,0.54,0,0.00,0.75,0.33], 'E': [0.07,0.23,0.23,0.33,0.50,0,1,1,0,0,0.27,0.84,0.27,0.44,0.45,0.59,0,0.00,0.75,0.50], 'F': [1.00,0.27,0.39,0.42,0.31,0,0,0,1,0,0.00,0.64,0.73,0.47,0.50,0.42,0,0.00,0.00,0.33], 'G': [0.48,0.08,0.46,0.00,0.54,0,0,0,0,0,0.00,0.37,0.43,0.88,0.00,0.58,0,0.00,0.00,0.00], 'H': [0.40,0.24,0.53,0.35,0.32,1,0,1,1,0,0.40,0.63,0.60,0.63,0.43,0.52,0,0.50,0.50,0.33], 'I': [0.88,0.20,0.45,0.37,0.22,0,0,0,0,1,0.00,0.63,0.87,0.31,0.40,0.36,0,0.00,0.00,0.33], 'K': [0.28,0.22,0.72,0.40,0.47,1,0,1,0,0,0.71,0.68,0.50,0.58,0.52,0.56,0,0.75,0.25,0.67], 'L': [0.85,0.20,0.46,0.37,0.40,0,0,0,0,1,0.00,0.78,0.57,0.36,0.43,0.40,0,0.00,0.00,0.33], 'M': [0.74,0.24,0.42,0.37,0.30,0,0,0,0,0,0.00,0.77,0.60,0.40,0.47,0.44,0,0.00,0.25,0.50], 'N': [0.09,0.20,0.39,0.27,0.46,0,0,1,0,0,0.00,0.53,0.37,0.78,0.36,0.56,0,0.50,0.50,0.33], 'P': [0.38,0.17,0.48,0.24,0.51,0,0,0,0,0,0.00,0.28,0.30,0.96,0.30,0.62,0,0.00,0.25,0.17], 'Q': [0.11,0.23,0.40,0.35,0.49,0,0,1,0,0,0.00,0.66,0.47,0.56,0.47,0.55,0,0.50,0.50,0.50], 'R': [0.00,0.29,0.77,0.42,0.53,1,0,1,0,0,0.85,0.64,0.53,0.56,0.58,0.56,0,1.00,0.25,0.83], 'S': [0.36,0.15,0.42,0.14,0.51,0,0,1,0,0,0.00,0.50,0.53,0.73,0.24,0.53,0,0.25,0.50,0.17], 'T': [0.40,0.18,0.42,0.22,0.44,0,0,1,0,0,0.00,0.50,0.67,0.58,0.33,0.49,0,0.25,0.50,0.17], 'V': [0.76,0.17,0.46,0.29,0.23,0,0,0,0,1,0.00,0.60,0.90,0.34,0.33,0.36,0,0.00,0.00,0.17], 'W': [0.81,0.35,0.43,0.55,0.27,0,0,0,1,0,0.00,0.63,0.73,0.47,0.61,0.42,0,0.25,0.25,0.33], 'Y': [0.76,0.31,0.42,0.47,0.34,0,0,1,1,0,0.69,0.51,0.77,0.53,0.54,0.44,0,0.25,0.25,0.33]}

def get_node_features(aa: str) -> np.ndarray:
    one_hot = np.zeros(20, dtype=np.float32)
    if aa in AA_TO_IDX: one_hot[AA_TO_IDX[aa]] = 1.0
    props = np.array(AA_PROPERTIES.get(aa, [0.5]*20), dtype=np.float32)
    return np.concatenate([one_hot, props])

def rbf_encode(distances: np.ndarray, num_rbf: int = 64, max_dist: float = 15.0) -> np.ndarray:
    centers = np.linspace(0, max_dist, num_rbf)
    gamma = 1.0 / (2.0 * ((max_dist / num_rbf) ** 2))
    return np.exp(-gamma * (distances.reshape(-1, 1) - centers.reshape(1, -1)) ** 2).astype(np.float32)

def positional_encoding(L: int, d_model: int = 32) -> np.ndarray:
    pe = np.zeros((L, d_model), dtype=np.float32)
    pos = np.arange(L, dtype=np.float32).reshape(-1, 1)
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div); pe[:, 1::2] = np.cos(pos * div)
    return pe

def normalize_bfactors(bf: np.ndarray, method: str = "zscore") -> Tuple[np.ndarray, Dict]:
    mu, sigma = float(np.mean(bf)), float(np.std(bf)) + 1e-8
    normed = (bf - mu) / sigma
    return normed.astype(np.float32), {"method": "zscore", "mean": mu, "std": sigma}

def denormalize_bfactors(normed: np.ndarray, params: Dict) -> np.ndarray:
    return normed * params['std'] + params['mean']

def pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 3: return 0.0
    p_m, t_m = pred - pred.mean(), target - target.mean()
    num = (p_m * t_m).sum()
    den = np.sqrt((p_m**2).sum() * (t_m**2).sum()) + 1e-8
    return float(num / den)
