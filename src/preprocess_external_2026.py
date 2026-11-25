import os
import sys
import glob
import torch
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
try:
    from Bio.PDB.SASA import ShrakeRupley
except Exception:
    ShrakeRupley = None
from scipy.spatial.distance import cdist
from config_v6 import Config
from release_paths import RUN, path_setting


AA3TO1= {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
# Amino acid properties (hydrophobicity, charge, size, flexibility, polarity)
AA_PROPS = {
    'A': [1.8,  0, 0.27, 0.36, 0], 'C': [2.5,  0, 0.44, 0.35, 0],
    'D': [-3.5,-1, 0.50, 0.51, 1], 'E': [-3.5,-1, 0.62, 0.50, 1],
    'F': [2.8,  0, 0.77, 0.31, 0], 'G': [-0.4, 0, 0.00, 0.54, 0],
    'H': [-3.2, 0, 0.60, 0.32, 1], 'I': [4.5,  0, 0.56, 0.46, 0],
    'K': [-3.9, 1, 0.60, 0.47, 1], 'L': [3.8,  0, 0.56, 0.40, 0],
    'M': [1.9,  0, 0.62, 0.30, 0], 'N': [-3.5, 0, 0.49, 0.46, 1],
    'P': [-1.6, 0, 0.36, 0.51, 0], 'Q': [-3.5, 0, 0.62, 0.49, 1],
    'R': [-4.5, 1, 0.72, 0.47, 1], 'S': [-0.8, 0, 0.32, 0.51, 1],
    'T': [-0.7, 0, 0.45, 0.44, 1], 'V': [4.2,  0, 0.50, 0.39, 0],
    'W': [-0.9, 0, 0.90, 0.31, 0], 'Y': [-1.3, 0, 0.80, 0.42, 1]
}
def get_node_features(residues):
    """20-dim one-hot + 20-dim extra features"""
    feats = []
    L = len(residues)
    for i, res in enumerate(residues):
        aa1 = AA3TO1.get(res.get_resname(), 'G')
        # One-hot
        oh = [0.0] * 20
        idx = AA_TO_IDX.get(aa1, AA_TO_IDX['G'])
        oh[idx] = 1.0
        # Properties (5 dim)
        props = AA_PROPS.get(aa1, AA_PROPS['G'])
        # Positional encoding (8 dim)
        rel_pos = i / max(L - 1, 1)
        pos_enc = []
        for freq in [1, 2, 4, 8]:
            pos_enc.append(math.sin(2 * math.pi * freq * rel_pos))
            pos_enc.append(math.cos(2 * math.pi * freq * rel_pos))
        # Window features (7 dim): distance to N/C terminus, local AA composition
        d_n = i / max(L - 1, 1)
        d_c = (L - 1 - i) / max(L - 1, 1)
        win = min(3, i, L - 1 - i)
        local_charged = 0
        local_hydro = 0
        local_polar = 0
        for j in range(max(0, i - 3), min(L, i + 4)):
            aa_j = AA3TO1.get(residues[j].get_resname(), 'G')
            p = AA_PROPS.get(aa_j, AA_PROPS['G'])
            local_charged += abs(p[1])
            local_hydro += p[0]
            local_polar += p[4]
        n_win = min(L, i + 4) - max(0, i - 3)
        local_charged /= max(n_win, 1)
        local_hydro /= max(n_win, 1)
        local_polar /= max(n_win, 1)
        window = [d_n, d_c, float(win) / 3.0, local_charged, local_hydro / 4.5, local_polar, float(n_win) / 7.0]
        feat = oh + props + pos_enc + window  # 20 + 5 + 8 + 7 = 40
        feats.append(feat)
    return torch.tensor(feats, dtype=torch.float32)
def get_positional_embedding(L, dim=32):
    """Sinusoidal positional embedding"""
    pos = torch.arange(L, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    pe = torch.zeros(L, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def parse_raw_pdb_quality(pdb_id, config):
    raw_dir = getattr(config, 'raw_pdb_dir', None)
    if not raw_dir:
        return None, None

    base_id = pdb_id.split('_')[0].lower()
    candidates = [
        os.path.join(raw_dir, f"{base_id}.pdb"),
        os.path.join(raw_dir, f"{base_id.upper()}.pdb"),
    ]

    raw_path = None
    for path in candidates:
        if os.path.exists(path):
            raw_path = path
            break
    if raw_path is None:
        return None, None

    resolution = None
    r_value = None
    with open(raw_path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('REMARK   2 RESOLUTION.'):
                parts = line.replace('ANGSTROMS.', '').split()
                for token in parts:
                    try:
                        resolution = float(token)
                        break
                    except ValueError:
                        pass
            elif line.startswith('REMARK   3   R VALUE') and 'FREE' not in line:
                parts = line.replace(':', ' ').split()
                for token in reversed(parts):
                    try:
                        value = float(token)
                        if 0.0 < value < 1.0:
                            r_value = value
                            break
                    except ValueError:
                        pass
            if resolution is not None and r_value is not None:
                break

    return resolution, r_value


def safe_angle_features(angle):
    if angle is None:
        return 0.0, 0.0
    return math.sin(float(angle)), math.cos(float(angle))


def classify_secondary(phi, psi):
    if phi is None or psi is None:
        return [0.0, 0.0, 1.0]

    phi_deg = math.degrees(float(phi))
    psi_deg = math.degrees(float(psi))

    if -100.0 <= phi_deg <= -30.0 and -90.0 <= psi_deg <= 20.0:
        return [1.0, 0.0, 0.0]  # helix-like
    if -180.0 <= phi_deg <= -45.0 and (45.0 <= psi_deg <= 180.0 or -180.0 <= psi_deg <= -120.0):
        return [0.0, 1.0, 0.0]  # beta-like
    return [0.0, 0.0, 1.0]      # coil/other


def get_phi_psi_features(residues):
    phis = [None] * len(residues)
    psis = [None] * len(residues)

    for i, res in enumerate(residues):
        try:
            if i > 0 and all(atom in residues[i - 1] for atom in ('C',)) and all(atom in res for atom in ('N', 'CA', 'C')):
                phis[i] = calc_dihedral(
                    residues[i - 1]['C'].get_vector(),
                    res['N'].get_vector(),
                    res['CA'].get_vector(),
                    res['C'].get_vector()
                )
            if i < len(residues) - 1 and all(atom in res for atom in ('N', 'CA', 'C')) and 'N' in residues[i + 1]:
                psis[i] = calc_dihedral(
                    res['N'].get_vector(),
                    res['CA'].get_vector(),
                    res['C'].get_vector(),
                    residues[i + 1]['N'].get_vector()
                )
        except Exception:
            phis[i] = None
            psis[i] = None

    feats = []
    for phi, psi in zip(phis, psis):
        phi_sin, phi_cos = safe_angle_features(phi)
        psi_sin, psi_cos = safe_angle_features(psi)
        feats.append([phi_sin, phi_cos, psi_sin, psi_cos] + classify_secondary(phi, psi))
    return np.asarray(feats, dtype=np.float32)


def get_sasa_features(model, residues):
    sasa = np.zeros(len(residues), dtype=np.float32)
    if ShrakeRupley is None:
        return sasa.reshape(-1, 1)

    try:
        sr = ShrakeRupley(n_points=100)
        sr.compute(model, level='R')
        for i, res in enumerate(residues):
            value = getattr(res, 'sasa', None)
            if value is None:
                value = sum(float(getattr(atom, 'sasa', 0.0)) for atom in res.get_atoms())
            sasa[i] = float(value)
        max_sasa = float(np.max(sasa))
        if max_sasa > 1e-8:
            sasa = sasa / max_sasa
    except Exception:
        sasa[:] = 0.0
    return sasa.reshape(-1, 1)


def get_packing_features(coords_np):
    if len(coords_np) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    dist = cdist(coords_np, coords_np)
    feats = []
    for cutoff in (8.0, 12.0, 16.0):
        counts = ((dist < cutoff) & (dist > 0.0)).sum(axis=1).astype(np.float32)
        max_count = float(np.max(counts))
        if max_count > 1e-8:
            counts = counts / max_count
        feats.append(counts)
    return np.stack(feats, axis=1).astype(np.float32)


def get_advanced_structure_features(model, residues, coords):
    coords_np = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
    sasa = get_sasa_features(model, residues)                # 1
    torsion_ss = get_phi_psi_features(residues)              # 4 torsion + 3 ss = 7
    packing = get_packing_features(coords_np)                # 3
    return torch.tensor(np.concatenate([sasa, torsion_ss, packing], axis=1), dtype=torch.float32)


def rbf_encode(distances, num_rbf=64, d_min=0.0, d_max=20.0):
    """Radial basis function encoding"""
    mu = torch.linspace(d_min, d_max, num_rbf)
    sigma = (d_max - d_min) / num_rbf
    diff = distances.unsqueeze(-1) - mu.to(distances.device)
    return torch.exp(-0.5 * (diff / sigma) ** 2)
def build_graph(coords, radius=12.0, max_knn=30):
    """V3: Hybrid graph — radius + KNN"""
    dist_mat = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    L = coords.shape[0]
    # Radius edges
    radius_mask = (dist_mat < radius) & (dist_mat > 0)
    # KNN edges
    knn_mask = torch.zeros_like(dist_mat, dtype=torch.bool)
    _, knn_idx = dist_mat.topk(min(max_knn + 1, L), largest=False, dim=1)
    for i in range(L):
        for j in knn_idx[i]:
            if i != j:
                knn_mask[i, j] = True
    # Union
    combined = radius_mask | knn_mask
    src, dst = combined.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    edge_dist = dist_mat[src, dst]
    # Edge features: RBF + direction + distance
    direction = coords[dst] - coords[src]
    direction = direction / (edge_dist.unsqueeze(-1) + 1e-8)
    rbf = rbf_encode(edge_dist)
    edge_feats = torch.cat([rbf, direction, edge_dist.unsqueeze(-1)], dim=-1)
    return edge_index, edge_feats, edge_dist
import math
def process_pdb(pdb_path, config, esm_embed_dir=None):
    """Process a single PDB file into training data"""
    parser = PDBParser(QUIET=True)
    pdb_id = os.path.basename(pdb_path).replace('.pdb', '')
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as e:
        print(f"  [SKIP] {pdb_id}: parse error — {e}")
        return None
    try:
        model = next(structure.get_models())
    except StopIteration:
        print(f"  [SKIP] {pdb_id}: no model found")
        return None
    residues = []
    coords_list = []
    bfactors = []
    for chain in model:
        for res in chain:
            if res.id[0] != ' ':
                continue
            if 'CA' not in res:
                continue
            residues.append(res)
            ca = res['CA']
            coords_list.append(ca.get_vector().get_array())
            bfactors.append(ca.get_bfactor())
    if len(residues) < 10:
        print(f"  [SKIP] {pdb_id}: only {len(residues)} residues")
        return None
    coords = torch.tensor(np.array(coords_list), dtype=torch.float32)
    bfactors = torch.tensor(bfactors, dtype=torch.float32)
    # Normalize B-factors (z-score per protein)
    bf_mean = bfactors.mean()
    bf_std = bfactors.std()
    if bf_std < 1e-6:
        print(f"  [SKIP] {pdb_id}: constant B-factors")
        return None
    bfactors_norm = (bfactors - bf_mean) / bf_std
    L = len(residues)
    # Node features
    node_feats = get_node_features(residues)
    pos_embed = get_positional_embedding(L, config.pos_embed_dim)
    adv_struct_feats = get_advanced_structure_features(model, residues, coords)
    node_feats = torch.cat([node_feats, pos_embed, adv_struct_feats], dim=-1)  # 40 + 32 + 11 = 83
    # Graph
    edge_index, edge_feats, edge_dist = build_graph(coords)
    # ESM3 embeddings
        # ESM3 embeddings
    esm_embed = None
    if esm_embed_dir:
        esm_candidates = [
            os.path.join(esm_embed_dir, f"{pdb_id}.pt"),
            os.path.join(esm_embed_dir, f"{pdb_id.lower()}.pt"),
            os.path.join(esm_embed_dir, f"{pdb_id.upper()}.pt"),
        ]

        esm_path = None
        for candidate in esm_candidates:
            if os.path.exists(candidate):
                esm_path = candidate
                break

        if esm_path is not None:
            esm_data = torch.load(esm_path, map_location="cpu", weights_only=False)

            if isinstance(esm_data, dict):
                esm_embed = esm_data.get(
                    "embeddings",
                    esm_data.get(
                        "esm_embedding",
                        esm_data.get(
                            "esm3_embed",
                            esm_data.get("representations", None)
                        )
                    )
                )
            elif isinstance(esm_data, torch.Tensor):
                esm_embed = esm_data

            if esm_embed is not None and esm_embed.dim() == 3:
                esm_embed = esm_embed.squeeze(0)

            # Match length
            if esm_embed is not None and esm_embed.shape[0] != L:
                if esm_embed.shape[0] > L:
                    esm_embed = esm_embed[:L]
                else:
                    pad_len = L - esm_embed.shape[0]
                    pad = torch.zeros(
                        pad_len,
                        esm_embed.shape[1],
                        dtype=esm_embed.dtype
                    )
                    esm_embed = torch.cat([esm_embed, pad], dim=0)

        else:
            print(f"[WARN] ESM not found for {pdb_id}")
    if esm_embed is None:
        if os.environ.get('RESFLEXNET_ALLOW_ZERO_ESM3') != '1':
            raise FileNotFoundError(f'No ESM3 embedding for {pdb_id}.')
        esm_embed = torch.zeros(L, config.esm3_embed_dim)
    resolution, r_value = parse_raw_pdb_quality(pdb_id, config)
    data = {
        'pdb_id': pdb_id,
        'coords': coords,
        'node_feats': node_feats,
        'edge_index': edge_index,
        'edge_feats': edge_feats,
        'edge_dist': edge_dist,
        'esm3_embed': esm_embed,
        'bfactors': bfactors_norm,
        'bfactors_raw': bfactors,
        'bf_mean': bf_mean,
        'bf_std': bf_std,
        'num_residues': L,
        'resolution': resolution,
        'r_value': r_value,
        'advanced_struct_dim': adv_struct_feats.shape[-1],
        'node_feat_dim_actual': node_feats.shape[-1],
    }
    return data
def main():
    config = Config()
    config.pdb_dir = path_setting('RESFLEXNET_PDB_DIR', RUN / 'external/pdb_clean')
    config.raw_pdb_dir = path_setting('RESFLEXNET_RAW_PDB_DIR', RUN / 'external/pdb_raw')
    config.processed_dir = path_setting('RESFLEXNET_PROCESSED_DIR', RUN / 'external/processed_data_v6')
    os.makedirs(config.processed_dir, exist_ok=True)
    pdb_files = sorted(glob.glob(os.path.join(config.pdb_dir, '*.pdb')))
    print(f"Found {len(pdb_files)} PDB files")
    esm_dir = path_setting('RESFLEXNET_ESM_DIR', RUN / 'external/esm_embeddings')
    if not os.path.isdir(esm_dir):
        esm_dir = None
        print("No ESM embeddings directory found, using zeros")
    success = 0
    for i, pdb_path in enumerate(pdb_files):
        pdb_id = os.path.basename(pdb_path).replace('.pdb', '')
        out_path = os.path.join(config.processed_dir, f"{pdb_id}.pt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            success += 1
            continue
        data = process_pdb(pdb_path, config, esm_dir)
        if data is not None:
            torch.save(data, out_path)
            success += 1
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(pdb_files)}, success={success}")
    print(f"\nDone: {success}/{len(pdb_files)} proteins processed")
    print(f"Saved to: {config.processed_dir}")
if __name__ == '__main__':
    main()
