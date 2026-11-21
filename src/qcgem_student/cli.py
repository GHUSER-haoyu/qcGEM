"""Command-line interface (`qcgem-embed`) for computing representations from external checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .api import QCGEMStudent


def main():
    p = argparse.ArgumentParser(description="Compute qcGEM-Student molecular representations from an explicit checkpoint.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="Single SMILES string")
    src.add_argument("--smiles-file", help="Path to text file with one SMILES per line")
    src.add_argument("--sdf", help="Path to 3D SDF file")
    p.add_argument("--checkpoint", required=True, help="Local path to qcgem_student_v1.pt (downloaded from GitHub Releases)")
    p.add_argument("--output", required=True, help="Output .pt file path")
    p.add_argument("--device", default="cpu", help="PyTorch compute device (e.g. cpu, cuda:0)")
    p.add_argument("--mode", choices=["cpu", "gpu"], default="cpu", help="Feature mode (cpu=RDKit/MMFF/EHT, gpu=+AIMNet2)")
    p.add_argument("--aim-python", default=None, help="AIMNet2 Python executable when --mode gpu")
    p.add_argument("--cache-aim-script", default=None, help="Path to cache_aim.py when --mode gpu")
    p.add_argument("--return-qc", action="store_true", help="Include calibrated 37-channel QC predictions")
    p.add_argument("--return-decoded-xyz", action="store_true", help="Include decoder 3D coordinates (decoded_xyz & 6 stages)")
    p.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    a = p.parse_args()

    torch.set_num_threads(1)
    model = QCGEMStudent(
        a.checkpoint,
        device=a.device,
        mode=a.mode,
        aim_python=a.aim_python,
        cache_aim_script=a.cache_aim_script,
    )

    if a.smiles:
        results = [
            model.encode_smiles(
                a.smiles.strip(),
                return_qc=a.return_qc,
                return_decoded_xyz=a.return_decoded_xyz,
                batch_size=a.batch_size,
            )
        ]
    elif a.smiles_file:
        smis = [x.split()[0] for x in Path(a.smiles_file).expanduser().read_text().splitlines() if x.strip()]
        results = model.encode_smiles(
            smis,
            return_qc=a.return_qc,
            return_decoded_xyz=a.return_decoded_xyz,
            batch_size=a.batch_size,
        )
    else:
        results = model.encode_sdf(
            a.sdf,
            return_qc=a.return_qc,
            return_decoded_xyz=a.return_decoded_xyz,
            batch_size=a.batch_size,
        )

    dest = Path(a.output).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "results": results,
            "mode": a.mode,
            "device": a.device,
            "qc_node_names": model.node_names,
            "qc_edge_names": model.edge_names,
            "starred_unreliable_channels": model.starred_unreliable_channels,
            "starred_element_npa_charge": model.starred_element_npa_charge,
        },
        dest,
    )
    print(json.dumps({"output": str(dest), "count": len(results), "return_qc": a.return_qc, "return_decoded_xyz": a.return_decoded_xyz}))


if __name__ == "__main__":
    main()
