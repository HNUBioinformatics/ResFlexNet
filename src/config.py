"""
V3 配置 - 精准改进，不盲目加大
核心策略: 
  1. 回到合理模型大小(~15M参数)
  2. 更强的正则化(dropout, weight_decay, stochastic depth)
  3. 改进归一化策略(per-protein + global alignment)
  4. 多尺度结构感知
  5. 改进训练策略(EMA, gradient accumulation, warmup)
"""
import os
from dataclasses import dataclass, field
from typing import List
from release_paths import DATA, RUN, path_setting, env_flag


@dataclass
class Config:
    # ==================== 路径 ====================
    pdb_dir: str = field(default_factory=lambda: path_setting("RESFLEXNET_PDB_DIR", RUN / "pdb_clean"))
    raw_pdb_dir: str = field(default_factory=lambda: path_setting("RESFLEXNET_RAW_PDB_DIR", RUN / "pdb_raw"))
    processed_dir: str = field(default_factory=lambda: path_setting("RESFLEXNET_PROCESSED_DIR", RUN / "processed_data_v6"))
    checkpoint_dir: str = field(default_factory=lambda: str(RUN / "checkpoints" / ("seed" + os.environ.get("SEED", "42"))))
    log_dir: str = field(default_factory=lambda: str(RUN / "logs" / ("seed" + os.environ.get("SEED", "42"))))
    split_file: str = field(default_factory=lambda: path_setting("RESFLEXNET_SPLIT", DATA / "internal/splits_v6.json"))
    
    # ==================== ESM3配置 ====================
    esm3_embed_dim: int = 1536
    
    # ==================== 特征维度 ====================
    node_feat_dim: int = 72
    edge_feat_dim: int = 68          # 20 AA + 20 props
    pos_embed_dim: int = 32          # positional encoding
    rbf_num: int = 64
    extra_struct_dim: int = 6        # contact_num, depth, sasa, density, local_std, seq_sep
    
    # ==================== 模型 (V3: 精简但更有效) ====================
    hidden_dim: int = 256            # 回到256，避免过拟合
    
    # EGNN
    egnn_layers: int = 4
    
    # ESM Adapter  
    esm_adapter_layers: int = 2
    esm_adapter_heads: int = 8
    esm_adapter_lr_scale: float = 1.0
    
    # Cross Attention
    cross_attn_heads: int = 8
    cross_attn_rounds: int = 2
    
    # Graph Transformer
    gt_heads: int = 8
    gt_layers: int = 4
    gt_ffn_dim: int = 512
    
    # Dropout (V3: 分层dropout)
    dropout: float = 0.2
    attn_dropout: float = 0.2
    path_dropout: float = 0.15        # Stochastic depth
    
    # Prediction head
    pred_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # ===== Multi-shell edge tagging =====
    shell_bounds: tuple = (3.0, 5.0, 8.0, 12.0)   # [3,5), [5,8), [8,12]
    shell_embed_dim: int = 12                      # shell one-hot -> learned embedding

    # ===== GNM attention bias =====
    use_gnm_bias: bool = True
    gnm_gamma: float = 1.0
    gnm_eps: float = 1e-5

    # ===== Decoder =====
    decoder_type: str = "swiglu"
    # ==================== 训练 ====================
    batch_size: int = 1              # 单蛋白训练，避免collate问题
    accumulation_steps: int = 8      # effective batch = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.03
    max_epochs: int = 100
    patience: int = 15
    warmup_epochs: int = 2
    
    # 损失权重
    loss_mse_weight: float = 0.6
    loss_pcc_weight: float = 1.4
    loss_rank_weight: float = 0.3
    loss_smooth_weight: float = 0.02
    loss_var_weight: float = 0.08
    loss_diff_weight: float = 0.1

    # Label quality filtering. These remove proteins with nearly constant or
    # extremely spiky B-factor labels before train/val/test splitting.
    filter_abnormal_labels: bool = True
    min_target_raw_std: float = 2.0
    max_target_norm_range: float = 8.0
    max_target_raw_range: float = 200.0

    # Experimental quality filtering. preprocess_v6 stores resolution/R-value
    # when raw PDB metadata are available.
    filter_experiment_quality: bool = True
    max_resolution: float = 2.5
    max_r_value: float = 0.25
    keep_missing_quality: bool = True
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Label noise
    label_noise_std: float = 0.0
    
    # ==================== 数据 ====================
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = field(default_factory=lambda: int(os.environ.get("SEED", "42")))
    random_seed: int = field(default_factory=lambda: int(os.environ.get("SEED", "42")))
    num_workers: int = field(default_factory=lambda: int(os.environ.get("RESFLEXNET_WORKERS", "0" if os.name == "nt" else "4")))
    
    # ==================== 设备 ====================
    device: str = field(default_factory=lambda: os.environ.get("RESFLEXNET_DEVICE", "cuda"))
    mixed_precision: bool = field(default_factory=lambda: env_flag("RESFLEXNET_MIXED_PRECISION"))
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
