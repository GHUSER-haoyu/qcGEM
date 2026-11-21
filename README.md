# qcgem-student

Minimal standalone Python package for computing DFT-free **qcGEM** 3D molecular representations (`384-d` readout, `128-d` node/edge/global latents, decoded 3D coordinates) and calibrated 37-channel quantum-chemical properties.

## Installation

```bash
pip install "git+https://github.com/GHUSER-haoyu/qcGEM.git@pip"
# Or install from the default main branch:
# pip install "git+https://github.com/GHUSER-haoyu/qcGEM.git"
```

Uses pure Python wheel dependencies (`torch`, `numpy`, `scipy`, `rdkit`, `torch-geometric`) with a built-in deterministic PyTorch scatter fallback (no C++ scatter compilation required).

## Download Weights (Required)

Pretrained weights are **not bundled** in git or in the package and are never downloaded automatically at runtime. Download `qcgem_student_v1.pt` (`165,897,014` B, SHA256 `b8f4680e4c6d06d70112fd1d477607173a53833cbf167fda474be90b60c70a59`) from [Release `v0.1.0`](https://github.com/GHUSER-haoyu/qcGEM/releases/tag/v0.1.0):

```bash
mkdir -p ./checkpoints
curl -L -o ./checkpoints/qcgem_student_v1.pt \
  https://github.com/GHUSER-haoyu/qcGEM/releases/download/v0.1.0/qcgem_student_v1.pt
# Or via GitHub CLI:
# gh release download v0.1.0 -R GHUSER-haoyu/qcGEM -p qcgem_student_v1.pt -D ./checkpoints
```

## Quickstart (Python)

```python
from qcgem_student import QCGEMStudent, kabsch_align

model = QCGEMStudent(checkpoint_path="./checkpoints/qcgem_student_v1.pt", device="cpu")

# 1. Single SMILES (384-d representation + calibrated QC + decoded 3D coords)
out = model.encode_smiles(
    "CCCC[C@H](B(O)O)NC(=O)[C@H](N)CCN",
    return_qc=True,
    return_decoded_xyz=True,
)
print(out["readout"].shape)      # (384,)
print(out["node_emb"].shape)     # (N, 128)
print(out["qc_node"].shape)      # (N, 18)
print(out["decoded_xyz"].shape)  # (N, 3)

aligned, rmsd = kabsch_align(out["decoded_xyz"], out["xyz"])
print(f"Decoded 3D all-atom RMSD = {rmsd:.4f} A")

# 2. Batch of SMILES
outs = model.encode_smiles(["CCO", "c1ccccc1"], batch_size=16)

# 3. Pre-generated 3D conformers from SDF
sdf_outs = model.encode_sdf("ligands.sdf", return_qc=True)
```

## CLI

```bash
qcgem-embed \
  --checkpoint ./checkpoints/qcgem_student_v1.pt \
  --smiles "CCO" \
  --device cpu \
  --return-qc \
  --return-decoded-xyz \
  --output ./ethanol_rep.pt
```
