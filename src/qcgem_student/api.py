"""Public representation & calibrated QC API for qcGEM-Student."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from rdkit import Chem

from .features import aim_features, device_batch, input_batch_many, make_record, missing_aim
from .head import (
    EDGE_NAMES,
    NODE_NAMES,
    STARRED_ELEMENT_NPA_CHARGE,
    STARRED_UNRELIABLE_CHANNELS,
    CalibratedQCHead,
)
from .model import Student, checkpoint_warm

RELEASE_URL = "https://github.com/GHUSER-haoyu/qcGEM/releases/tag/v0.1.0"


def kabsch_align(
    pred_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, float]:
    """Align pred_xyz onto target_xyz via optimal rigid transform (SO(3) by default) and return (aligned, RMSD)."""
    p = np.asarray(pred_xyz, dtype=np.float64)
    t = np.asarray(target_xyz, dtype=np.float64)
    if p.shape != t.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected identical (N, 3) coordinate shapes, got {p.shape} and {t.shape}")
    pc = p - p.mean(axis=0, keepdims=True)
    tc = t - t.mean(axis=0, keepdims=True)
    u, _, vh = np.linalg.svd(pc.T @ tc)
    fix = np.eye(3, dtype=np.float64)
    if not allow_reflection:
        fix[2, 2] = np.linalg.det(u @ vh)
    rot = u @ fix @ vh
    aligned = pc @ rot + t.mean(axis=0, keepdims=True)
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - t) ** 2, axis=-1))))
    return aligned.astype(np.float32), rmsd


class QCGEMStudent:
    """High-level wrapper around frozen refined qcGEM-Student (+ optional CalibratedQCHead).

    Weights are NEVER bundled with the package and NEVER downloaded automatically.
    Download ``qcgem_student_v1.pt`` from GitHub Releases and pass ``checkpoint_path``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        mode: str = "cpu",
        aim_python: str | Path | None = None,
        cache_aim_script: str | Path | None = None,
        stats_path: str | Path | None = None,
        head_checkpoint_path: str | Path | None = None,
        priors_path: str | Path | None = None,
    ):
        if not checkpoint_path:
            raise FileNotFoundError(
                "Weights are not bundled with qcgem-student and are never downloaded automatically.\n"
                f"Download 'qcgem_student_v1.pt' from {RELEASE_URL} and pass checkpoint_path='/path/to/qcgem_student_v1.pt'."
            )
        ckpt = Path(checkpoint_path).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found: {ckpt}\n"
                "Weights are not bundled with qcgem-student and are never downloaded automatically.\n"
                f"Download 'qcgem_student_v1.pt' from {RELEASE_URL} and pass checkpoint_path='/path/to/qcgem_student_v1.pt'."
            )
        if mode not in ("cpu", "gpu"):
            raise ValueError(f"mode must be 'cpu' or 'gpu', got {mode!r}")

        self.device = str(device)
        self.mode = mode
        self.aim_python = str(Path(aim_python).expanduser()) if aim_python else None
        script = cache_aim_script or os.environ.get("QCGEM_CACHE_AIM_SCRIPT")
        self.cache_aim_script = str(Path(script).expanduser()) if script else None
        if self.mode == "gpu":
            if not self.aim_python or not Path(self.aim_python).exists():
                raise ValueError(
                    "mode='gpu' requires an external AIMNet2 interpreter via aim_python=... "
                    "For CUDA execution with standard RDKit/MMFF/EHT features, set device='cuda:0' and mode='cpu'."
                )
            if not self.cache_aim_script or not Path(self.cache_aim_script).exists():
                raise ValueError(
                    "mode='gpu' requires cache_aim_script='/path/to/cache_aim.py' "
                    "(or QCGEM_CACHE_AIM_SCRIPT env var). Otherwise use mode='cpu' (with device='cpu' or device='cuda:0')."
                )

        st = torch.load(ckpt, weights_only=False, map_location="cpu")
        if st.get("diagnostic", False):
            raise ValueError("Refusing to load a diagnostic checkpoint.")

        if "stats" in st:
            self.stats = st["stats"]
        elif stats_path is not None:
            self.stats = json.loads(Path(stats_path).expanduser().read_text())
        else:
            raise FileNotFoundError(
                "Raw backbone checkpoint does not embed 'stats'. Pass stats_path='/path/to/stats.json' "
                f"or use the unified 'qcgem_student_v1.pt' release asset from {RELEASE_URL}."
            )

        self.student = Student(self.stats, device=self.device, initialize_official=False)
        self.student.load_state_dict(st["model"])
        self.student.set_stage(checkpoint_warm(st))
        self.student.eval()
        self.student.requires_grad_(False)

        self.head: CalibratedQCHead | None = None
        meta = st.get("channel_metadata", {})
        if head_checkpoint_path is not None:
            hckpt = Path(head_checkpoint_path).expanduser()
            if not hckpt.is_file():
                raise FileNotFoundError(f"Head checkpoint not found: {hckpt}")
            hst = torch.load(hckpt, weights_only=False, map_location="cpu")
            if "priors" in hst:
                priors = hst["priors"]
            elif priors_path is not None:
                priors = torch.load(Path(priors_path).expanduser(), weights_only=False, map_location="cpu")
            else:
                raise FileNotFoundError("Split head checkpoint requires priors_path=... or use calibrated_head_v1.pt.")
            self.head = CalibratedQCHead(self.stats, priors).to(self.device)
            self.head.load_state_dict(hst["head_state_dict"])
            self.head.eval()
            self.head.requires_grad_(False)
            meta = hst.get("channel_metadata", meta)
        elif "head_state_dict" in st and "priors" in st:
            self.head = CalibratedQCHead(self.stats, st["priors"]).to(self.device)
            self.head.load_state_dict(st["head_state_dict"])
            self.head.eval()
            self.head.requires_grad_(False)

        self.node_names = list(meta.get("node_names", NODE_NAMES))
        self.edge_names = list(meta.get("edge_names", EDGE_NAMES))
        self.starred_unreliable_channels = list(meta.get("starred_unreliable_channels", STARRED_UNRELIABLE_CHANNELS))
        self.starred_element_npa_charge = list(meta.get("starred_element_npa_charge", STARRED_ELEMENT_NPA_CHARGE))
        self._aim_dim = len(self.stats["features"]["aim_node"]["std"]) - 7

    def encode_mol(
        self,
        mols: Chem.Mol | Sequence[Chem.Mol],
        *,
        generate_conformer: bool = False,
        return_qc: bool = False,
        return_decoded_xyz: bool = False,
        batch_size: int = 16,
    ) -> dict | list[dict]:
        single_input = isinstance(mols, Chem.Mol)
        mol_list = [mols] if single_input else list(mols)
        if not mol_list:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if return_qc and self.head is None:
            raise RuntimeError(
                "CalibratedQCHead weights are not loaded. Use 'qcgem_student_v1.pt' or pass head_checkpoint_path=..."
            )

        records = [make_record(m, cid=str(idx), generate=generate_conformer) for idx, m in enumerate(mol_list)]
        need_decode = bool(return_qc or return_decoded_xyz)
        outputs: list[dict] = []

        with torch.no_grad():
            for lo in range(0, len(records), batch_size):
                chunk = records[lo : lo + batch_size]
                aa = (
                    aim_features(chunk, self.aim_python, self.cache_aim_script)
                    if self.mode == "gpu"
                    else [missing_aim(r, self._aim_dim) for r in chunk]
                )
                b = device_batch(input_batch_many(chunk, aa), self.device)
                out = self.student(b, mode=self.mode, decode=need_decode)
                if return_qc:
                    qc_n, qc_e = self.head(out, b)

                ptr = b["ptr"].cpu().tolist()
                eptr = b["eptr"].cpu().tolist()
                for g, r in enumerate(chunk):
                    n0, n1 = ptr[g], ptr[g + 1]
                    e0, e1 = eptr[g], eptr[g + 1]
                    item = {
                        "id": r["cid"],
                        "readout": out["readout"][g].detach().cpu().numpy(),
                        "global_emb": out["global_"][g].detach().cpu().numpy(),
                        "node_emb": out["node"][n0:n1].detach().cpu().numpy(),
                        "edge_emb": out["edge"][e0:e1].detach().cpu().numpy(),
                        "atomic_numbers": np.asarray(r["top"]["z"], dtype=np.int64),
                        "xyz": np.asarray(r["xyz"], dtype=np.float32),
                    }
                    if return_qc:
                        item["qc_node"] = qc_n[n0:n1].detach().cpu().numpy()
                        item["qc_edge"] = qc_e[e0:e1].detach().cpu().numpy()
                        item["qc_node_names"] = list(self.node_names)
                        item["qc_edge_names"] = list(self.edge_names)
                        item["starred_unreliable_channels"] = list(self.starred_unreliable_channels)
                        item["starred_element_npa_charge"] = list(self.starred_element_npa_charge)
                    if return_decoded_xyz:
                        stages = out["decoded"]["xyz"][:, n0:n1].detach().cpu().numpy()
                        item["decoded_xyz"] = stages[-1]
                        item["decoded_xyz_stages"] = stages
                    outputs.append(item)

        return outputs[0] if single_input else outputs

    def encode_smiles(
        self,
        smiles: str | Sequence[str],
        *,
        return_qc: bool = False,
        return_decoded_xyz: bool = False,
        batch_size: int = 16,
    ) -> dict | list[dict]:
        single_input = isinstance(smiles, str)
        smi_list = [smiles] if single_input else list(smiles)
        mols = []
        for s in smi_list:
            m = Chem.MolFromSmiles(s)
            if m is None:
                raise ValueError(f"Invalid SMILES string: {s!r}")
            mols.append(m)
        res = self.encode_mol(
            mols,
            generate_conformer=True,
            return_qc=return_qc,
            return_decoded_xyz=return_decoded_xyz,
            batch_size=batch_size,
        )
        return res[0] if single_input else res

    def encode_sdf(
        self,
        sdf_path: str | Path,
        *,
        return_qc: bool = False,
        return_decoded_xyz: bool = False,
        batch_size: int = 16,
    ) -> list[dict]:
        sup = Chem.SDMolSupplier(str(Path(sdf_path).expanduser()), removeHs=False)
        mols = []
        for idx, m in enumerate(sup):
            if m is None:
                raise ValueError(f"Failed to parse molecule index {idx} in SDF: {sdf_path}")
            mols.append(m)
        return self.encode_mol(
            mols,
            generate_conformer=False,
            return_qc=return_qc,
            return_decoded_xyz=return_decoded_xyz,
            batch_size=batch_size,
        )
