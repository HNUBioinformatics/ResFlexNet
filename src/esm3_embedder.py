"""
ESM3 嵌入提取独立模块
支持:
- 本地模型 / HuggingFace 加载
- 批量序列处理
- 磁盘缓存
- 自动分块处理超长序列
"""
import os
import sys
import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from utils import setup_logger
from release_paths import RUN


class ESM3Embedder:
    """
    ESM3 嵌入提取器
    
    使用方式:
        embedder = ESM3Embedder(device='cuda')
        embedding = embedder.get_embedding("MKLLV...")   # (L, D)
    """
    
    def __init__(
        self,
        model_name: str = "esm3_sm_open_v1",
        device: str = "cuda",
        cache_dir: str = "./data/esm3_cache",
        max_len: int = 1024,
        layer: int = -1,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.layer = layer
        self.model_name = model_name
        self.logger = setup_logger("esm3_embedder")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.embed_dim = None
        self._backend = None  # 'esm3_official' or 'huggingface' or 'esm2_fallback'
    
    def _load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return
        
        self.logger.info(f"Loading ESM3 model: {self.model_name}")
        
        # 尝试方式1: ESM3 官方 API
        try:
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, GenerationConfig
            
            self.model = ESM3.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self._backend = 'esm3_official'
            self.embed_dim = self._detect_dim_official()
            self.logger.info(f"ESM3 loaded via official API. dim={self.embed_dim}")
            return
        except (ImportError, Exception) as e:
            self.logger.info(f"ESM3 official API not available: {e}")
        
        # 尝试方式2: HuggingFace transformers
        try:
            from transformers import AutoModel, AutoTokenizer
            
            hf_name = f"facebook/{self.model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
            self.model = AutoModel.from_pretrained(hf_name).to(self.device)
            self.model.eval()
            self._backend = 'huggingface'
            self.embed_dim = self.model.config.hidden_size
            self.logger.info(f"ESM3 loaded via HuggingFace. dim={self.embed_dim}")
            return
        except (ImportError, Exception) as e:
            self.logger.info(f"HuggingFace ESM3 not available: {e}")
        
        # 尝试方式3: 回退到 ESM2
        try:
            import esm as esm_module
            
            self.model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = alphabet.get_batch_converter()
            self._backend = 'esm2_fallback'
            self.embed_dim = 1280
            self.logger.warning("Using ESM2 (650M) as fallback. dim=1280")
            return
        except (ImportError, Exception) as e:
            self.logger.info(f"ESM2 fallback not available: {e}")
        
        # 方式4: 随机嵌入（调试用）
        raise RuntimeError("No real ESM model available. Refusing to generate random embeddings.")
    
    def _detect_dim_official(self) -> int:
        """检测ESM3官方模型的输出维度"""
        try:
            from esm.sdk.api import ESMProtein
            
            test_protein = ESMProtein(sequence="ACDEF")
            with torch.no_grad():
                output = self.model.encode(test_protein)
            
            if hasattr(output, 'embeddings'):
                return output.embeddings.shape[-1]
            elif hasattr(output, 'last_hidden_state'):
                return output.last_hidden_state.shape[-1]
            else:
                # 尝试从模型配置获取
                if hasattr(self.model, 'config'):
                    return getattr(self.model.config, 'hidden_size', 1536)
                return 1536
        except Exception:
            return 1536
    
    def _cache_path(self, sequence: str) -> str:
        """生成缓存文件路径"""
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"esm3_{seq_hash}_{len(sequence)}.npy")
    
    def _extract_official(self, sequence: str) -> np.ndarray:
        """Extract per-residue embeddings with official ESM3 API."""
        from esm.sdk.api import ESMProtein, LogitsConfig

        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            protein_tensor = self.model.encode(protein)
            protein_tensor = protein_tensor.to(self.device)

            output = self.model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )

        emb = output.embeddings

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()

        if emb.ndim == 3:
            emb = emb[0]

        if emb.shape[0] == len(sequence) + 2:
            emb = emb[1:-1]
        elif emb.shape[0] == len(sequence) + 1:
            emb = emb[1:]

        if emb.shape[0] != len(sequence):
            raise RuntimeError(
                f"Embedding length {emb.shape[0]} != sequence length {len(sequence)}"
            )

        return emb.astype(np.float32)
    
    def _extract_huggingface(self, sequence: str) -> np.ndarray:
        """HuggingFace 方式提取"""
        inputs = self.tokenizer(sequence, return_tensors="pt",
                                padding=False, truncation=True,
                                max_length=self.max_len + 2)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        emb = outputs.last_hidden_state[0].cpu().numpy()  # (L+2, D)
        
        # 去除特殊token
        if emb.shape[0] == len(sequence) + 2:
            emb = emb[1:-1]
        elif emb.shape[0] == len(sequence) + 1:
            emb = emb[1:]
        
        return emb.astype(np.float32)
    
    def _extract_esm2(self, sequence: str) -> np.ndarray:
        """ESM2 回退提取"""
        batch_converter = self.tokenizer
        data = [("protein", sequence)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        emb = results["representations"][33][0].cpu().numpy()  # (L+2, 1280)
        emb = emb[1:len(sequence)+1]  # 去掉特殊token
        
        return emb.astype(np.float32)
    
    def _extract_random(self, sequence: str) -> np.ndarray:
        """随机嵌入（调试用）"""
        np.random.seed(hash(sequence) % (2**32))
        return np.random.randn(len(sequence), self.embed_dim).astype(np.float32) * 0.1
    
    def _extract_chunks(self, sequence: str) -> np.ndarray:
        """分块处理超长序列"""
        L = len(sequence)
        if L <= self.max_len:
            return self._extract_single(sequence)
        
        self.logger.info(f"Splitting sequence (L={L}) into chunks of {self.max_len}")
        
        overlap = 64
        chunks = []
        starts = []
        
        pos = 0
        while pos < L:
            end = min(pos + self.max_len, L)
            chunks.append(sequence[pos:end])
            starts.append(pos)
            if end >= L:
                break
            pos = end - overlap  # 重叠
        
        embeddings = np.zeros((L, self.embed_dim), dtype=np.float32)
        counts = np.zeros(L, dtype=np.float32)
        
        for chunk_seq, start in zip(chunks, starts):
            chunk_emb = self._extract_single(chunk_seq)
            end = start + len(chunk_seq)
            embeddings[start:end] += chunk_emb
            counts[start:end] += 1
        
        # 平均重叠区域
        embeddings /= np.maximum(counts[:, None], 1)
        
        return embeddings
    
    def _extract_single(self, sequence: str) -> np.ndarray:
        """提取单条序列嵌入"""
        if self._backend == 'esm3_official':
            return self._extract_official(sequence)
        elif self._backend == 'huggingface':
            return self._extract_huggingface(sequence)
        elif self._backend == 'esm2_fallback':
            return self._extract_esm2(sequence)
        elif self._backend == 'random':
            return self._extract_random(sequence)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")
    
    def get_embedding(self, sequence: str, use_cache: bool = True) -> np.ndarray:
        """
        获取序列嵌入（带缓存）
        
        Args:
            sequence: 氨基酸序列
            use_cache: 是否使用磁盘缓存
            
        Returns:
            embedding: (L, D) numpy array
        """
        # 确保模型已加载
        self._load_model()
        
        # 清理序列
        sequence = sequence.strip().upper()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = ''.join(c if c in valid_aa else 'X' for c in sequence)
        
        if len(sequence) == 0:
            raise ValueError("Empty sequence")
        
        # 检查缓存
        cache_path = self._cache_path(sequence)
        if use_cache and os.path.exists(cache_path):
            try:
                emb = np.load(cache_path)
                if emb.shape[0] == len(sequence):
                    return emb
            except Exception:
                pass
        
        # 提取
        if len(sequence) > self.max_len:
            emb = self._extract_chunks(sequence)
        else:
            emb = self._extract_single(sequence)
        
        # 缓存
        if use_cache:
            try:
                np.save(cache_path, emb)
            except Exception as e:
                self.logger.warning(f"Cache save failed: {e}")
        
        return emb
    
    def get_batch_embeddings(self, sequences: List[str], 
                             use_cache: bool = True) -> List[np.ndarray]:
        """批量提取嵌入"""
        results = []
        for i, seq in enumerate(sequences):
            try:
                emb = self.get_embedding(seq, use_cache=use_cache)
                results.append(emb)
            except Exception as e:
                self.logger.warning(f"Sequence {i} failed: {e}")
                # 返回零嵌入作为fallback
                self._load_model()
                results.append(np.zeros((len(seq), self.embed_dim), dtype=np.float32))
        return results
    
    def get_dim(self) -> int:
        """获取嵌入维度"""
        self._load_model()
        return self.embed_dim
    
    def clear_cache(self):
        """清除缓存"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            self.logger.info("Cache cleared")


# =====================================================================
# 命令行工具：批量提取 ESM3 嵌入
# =====================================================================

def batch_extract_from_pdb_dir(pdb_dir: str, output_dir: str, device: str = 'cuda'):
    """从PDB目录批量提取ESM3嵌入"""
    from Bio.PDB import PDBParser
    
    logger = setup_logger("batch_esm3")
    embedder = ESM3Embedder(device=device, cache_dir=output_dir)
    
    embedder._load_model()
    if embedder._backend != 'esm3_official':
        raise RuntimeError('Use the official ESM3 backbone for this release. Check the esm package and Hugging Face model access.')
    pdb_files = sorted(Path(pdb_dir).glob("*.pdb"))
    logger.info(f"Found {len(pdb_files)} PDB files")
    
    parser = PDBParser(QUIET=True)
    
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    
    success = 0
    fail = 0
    
    for i, pdb_path in enumerate(pdb_files):
        try:
            # 提取序列
            structure = parser.get_structure(pdb_path.stem, str(pdb_path))
            model = structure[0]
            
            sequence = ""
            for residue in model.get_residues():
                if residue.id[0] != ' ' or 'CA' not in residue:
                    continue
                resname = residue.get_resname()
                if resname in three_to_one:
                    sequence += three_to_one[resname]
                else:
                    sequence += 'X'
            
            if len(sequence) < 10:
                logger.warning(f"Skipping {pdb_path.stem}: too short ({len(sequence)})")
                fail += 1
                continue
            
            # 提取嵌入
            emb = embedder.get_embedding(sequence, use_cache=True)
            
            # 保存成 preprocess_v3.py 能读取的 pdb_id.pt
            out_path = os.path.join(output_dir, f"{pdb_path.stem.lower()}.pt")
            torch.save(
                {"embeddings": torch.tensor(emb, dtype=torch.float32)},
                out_path
            )

            logger.info(
                f"[{i+1}/{len(pdb_files)}] {pdb_path.stem}: "
                f"L={len(sequence)}, emb={emb.shape}"
            )
            success += 1
            
        except Exception as e:
            logger.warning(f"[{i+1}/{len(pdb_files)}] {pdb_path.stem} failed: {e}")
            fail += 1
            continue
    
    logger.info(f"Done. Success: {success}, Failed: {fail}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESM3 Embedding Extraction")
    parser.add_argument('--pdb_dir', type=str, required=True, help="PDB directory")
    parser.add_argument('--output_dir', type=str, default=str(RUN / "esm_embeddings"))
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    batch_extract_from_pdb_dir(args.pdb_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
