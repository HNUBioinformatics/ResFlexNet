# ResFlexNet：蛋白质残基相对柔性预测

*Multiscale Equivariant Graph Transformer with Evolutionary and Topological Priors for Protein Residue Flexibility Prediction*。ResFlexNet 根据静态蛋白结构和序列预测标准化的 Cα B 因子，用作残基相对柔性的晶体学代理指标。标准化在每个模型输入的保留残基内进行；多链输入共享同一输入尺度。

本目录包含 代码、固定划分、内外部逐输入评估、配体分析表与结构示例。三组主模型权重和消融权重分别作为独立 ZIP 提供。使用权重重跑完整实验仍需原实验图张量和 ESM3 嵌入。

## 目录与逐文件说明

| 位置 | 内容 |
| --- | --- |
| [`src/`](src/) | 下载、预处理、网络、损失、训练、评估、消融、空间分析和作图代码。查阅[每个脚本的用途](docs/source-code-index.md)。 |
| [`data/source/`](data/source/) | CullPDB 原始候选表和 UniProt/GO 注释。 |
| [`data/internal/`](data/internal/) | 固定划分、原始评估、残基预测、补充表 S10 映射与复算统计。 |
| [`data/external/`](data/external/) | 外部候选 PDB ID、三个子集的实际评估名单、完整逐输入评估、指标和残基预测。 |
| [`data/ligand/`](data/ligand/) | 参考组分、残基距离、代表案例、Fig. 6 原始分析表与独立复算结果。 |
| [`data/ablation/`](data/ablation/) | 找到的原始消融运行结果。 |
| [`data/reported/`](data/reported/) | 论文和作者作图用汇总值、权重参数与 SHA-256。 |
| [`data/examples/`](data/examples/) | 3 个代表案例的结构坐标、PyMOL 会话和图片。 |
| [`figures/`](figures/) | 论文相关作图输出。数据与图件的[逐文件索引](docs/data-file-index.md)也标注了历史版本。 |
| [`requirements.txt`](requirements.txt) | 模型、预处理和统计依赖。 |
| [`requirements-esm3.txt`](requirements-esm3.txt) | 生成 ESM3 嵌入的另选依赖。 |
| [`.gitignore`](.gitignore) | 排除运行目录、权重、归档和缓存。 |

## 用已有预测复算结果

在仓库根目录运行。统计复算只需 NumPy、Pandas 和 SciPy。图形需要 Matplotlib。

```bash
python -m pip install numpy pandas scipy matplotlib
python src/reproduce_saved_results.py
python src/plot_external_validation_v6.py
python src/plot_external_lowhomology_v6.py
```

复算输出在 `runs/saved_result_checks/`，图形在 `runs/figures/`。内部数据的 249,580 条残基记录应按原始输入分成 1,067 个块，再做宏平均。使用现存预测复算得到 PCC **0.7811**、SCC **0.7469**、RMSE **0.6060**。1,108 个结构链是导出后的结构标识数，旧版按链作图源表使用另一种统计单位。

空间分析从匹配表选取存在参考组分且与结构映射成功的残基，从同链的 8 Å 邻域外抽取等量背景。Direct、Neighborhood、Combined 分别纳入 **102、99、99** 条链。后两种标签使用同一批目标残基，分别抽样造成少量数值波动。图 6 的原始表在 `data/ligand/original/`。内外部的 `*.json.gz` 和外部的 `*.csv.gz` 解压后保留原始结果字节。

## 准备结构和特征

建议使用 Python 3.10 或更新版本。根据系统和 CUDA 配置安装 [PyTorch](https://pytorch.org/get-started/locally/)，然后安装其余依赖。ESM3 模型请从 [EvolutionaryScale 的官方发布页](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) 获取。已经有匹配的嵌入或图张量时，可跳过对应步骤。

```bash
python -m pip install -r requirements.txt
python src/fast_download_clean_pdb_from_csv.py --csv data/source/cullpdb_pc25.0_res0.0-3.0_noBrks_len40-10000_R0.3_Xray_d2026_01_26_chains10856.csv
python -m pip install -r requirements-esm3.txt
python src/esm3_embedder.py --pdb_dir runs/pdb_clean --output_dir runs/esm_embeddings --device cuda
python src/preprocess_v6.py
```

下载器按链组织结构；原实验还包含按整条 PDB 命名的输入。应优先使用原实验的清洗结构与特征，或按固定划分准备对应名称和内容。预处理默认要求序列嵌入，避免缺失时误用零张量。外部候选可用 `download_external_2026_pdb.py` 下载，参数 `--ids` 可指定候选清单。将清洗后的结构放在 `runs/external/pdb_clean/`，对应嵌入放在 `runs/external/esm_embeddings/`，再运行 `preprocess_external_2026_v6.py`。CIF 下载需要先转换为预处理支持的 PDB 格式。

## 训练、集成评估和消融

默认划分是 [`data/internal/splits_v6.json`](data/internal/splits_v6.json)，含 8,524 个训练输入、1,065 个验证输入和 1,067 个测试输入。图张量默认位于 `runs/processed_data_v6/`，其他位置可用 `RESFLEXNET_PROCESSED_DIR` 指定。若要使用新划分，将 `RESFLEXNET_SPLIT` 指向独立的新路径，并设置 `RESFLEXNET_ALLOW_NEW_SPLIT=1`。主实验使用 FP32 与三个训练种子。

```powershell
foreach ($seed in 42, 43, 44) {
  $env:SEED = "$seed"
  python src/train_v6.py
}
```

将 `ResFlexNet_model_weights.zip` 解压到仓库根目录，三个检查点自动进入下列路径。归档保留训练得到的参数张量、轮次与验证 PCC；原优化器状态未包含在推理归档中。

```bash
python src/evaluate_ensemble_v6.py --checkpoints runs/checkpoints/seed42/best_model_v3.pt runs/checkpoints/seed43/best_model_v3.pt runs/checkpoints/seed44/best_model_v3.pt --output_dir runs/ensemble
python src/export_residue_bfactor_predictions_v6.py --result_json runs/ensemble --pdb_dir runs/pdb_clean --output_csv runs/ensemble/residue_predictions.csv
```

外部实验用同一集成入口。将 `RESFLEXNET_PROCESSED_DIR` 指向外部图，分别以 `--input-list` 传入 `data/external/all/inputs.txt`、`data/external/low30/inputs.txt` 和 `data/external/low25/inputs.txt`，并指定各自的 `--output_dir`。这些名单来自原始逐输入评估。重新计算低序列一致性筛选可使用 `make_external_low_homology_v6.py --train-pdb-dir <训练结构目录>`，覆盖阈值 0.8、模式 1。重新查询 RCSB 或改变训练参考集合会产生新样本名单。

Linux 或 Git Bash 运行 `bash src/run_ablation_v6.sh`，默认依次训练七个组件消融。也可在 PowerShell 中设置 `$env:ABLATION_NAME = 'no_cross_attention'` 后运行 `python src/train_v6_ablation.py`。将 `ResFlexNet_ablation_weights.zip` 解压到仓库根目录后，评估时用对应的 `ABLATION_NAME` 并执行 `python src/evaluate_v6_ablation.py --checkpoint <权重路径>`。

可选环境变量包括 `RESFLEXNET_RUN_DIR`、`RESFLEXNET_PDB_DIR`、`RESFLEXNET_RAW_PDB_DIR`、`RESFLEXNET_ESM_DIR`、`RESFLEXNET_DEVICE` 和 `RESFLEXNET_WORKERS`。恢复了历史训练文件名 `best_model_v3.pt`，主入口实际使用 v6 数据与参数。

## GitHub 发布

仓库上传使用此目录中的源码和数据。两个权重 ZIP 及可选的 `ResFlexNet_available_structures.zip` 可作为 GitHub Release 或数据仓库的独立附件。权重文件名、实际运行设置和 SHA-256 见 [`weights_manifest.json`](data/reported/weights_manifest.json)。结构包保存 2,667 个通过初筛的本地 PDB 副本，完整数据仍以原实验档案为准。桌面和输出目录中的本地归档尚未发布至 GitHub。

CullPDB 候选表来自 [PISCES](https://dunbrack.fccc.edu/pisces/)，结构示例来自 [RCSB PDB](https://www.rcsb.org/)。代码发布许可待作者确定，ESM3 等第三方资源适用各自的许可。
