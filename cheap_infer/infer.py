"""qcGEM-Student deployment inference (unified checkpoint & legacy split checkpoints).

Uses cheap geometry/quantum descriptors (and optional AIMNet2 features) without
any DFT/QC labels or Teacher network. Weights must be downloaded separately
and passed explicitly via ``--checkpoint``.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

CHEAP_INFER_DIR = Path(__file__).resolve().parent
BACKBONE_DIR = Path(os.environ.get("QCMOL_BACKBONE_DIR", CHEAP_INFER_DIR / "training" / "backbone"))
HEAD_DIR = Path(os.environ.get("QCMOL_HEAD_DIR", CHEAP_INFER_DIR / "training" / "calibrated_head"))
for _p in (BACKBONE_DIR, HEAD_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from checkpoint_state import checkpoint_warm  # noqa: E402
from common import EDGE_NAMES, NODE_NAMES, cache_loads  # noqa: E402
from data import device_batch, groups  # noqa: E402
from features import BONDS, CHIRAL, extract, topology  # noqa: E402
from head_model import CalibratedQCHead  # noqa: E402
from model import Student  # noqa: E402

HYBRID = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER", "UNSPECIFIED"]
STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]
STARRED_UNRELIABLE_CHANNELS = ["LP_1", "LP_3", "NPA_rydberg", "BD_3", "BD_5", "BD_10", "BD_12"]
STARRED_ELEMENT_NPA_CHARGE = ["Br"]
RELEASE_URL = "https://github.com/GHUSER-haoyu/qcGEM/releases/tag/v0.1.0"


def onehot(x, values):
    a = np.zeros(len(values), np.float32)
    a[values.index(str(x))] = 1
    return a


def normal_features(m):
    n = m.GetNumAtoms()
    nodes = []
    edges = np.zeros((n, n, 34), np.float32)
    for a in m.GetAtoms():
        z = a.GetAtomicNum()
        el = np.zeros(36, np.float32)
        if 1 <= z <= 36:
            el[z - 1] = 1
        nodes.append(
            np.r_[
                [z, a.GetTotalDegree(), a.GetFormalCharge(), a.GetTotalNumHs(includeNeighbors=True), a.GetNumRadicalElectrons()],
                onehot(a.GetChiralTag(), CHIRAL),
                el,
                onehot(a.GetHybridization(), HYBRID),
                [not a.GetIsAromatic(), a.GetIsAromatic(), not a.IsInRing(), a.IsInRing()],
            ]
        )
    for b in m.GetBonds():
        x = np.r_[
            onehot(b.GetStereo(), STEREO),
            [not b.GetIsConjugated(), b.GetIsConjugated(), not b.IsInRing(), b.IsInRing(), not b.GetIsAromatic(), b.GetIsAromatic()],
            onehot(b.GetBondType(), BONDS),
        ]
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges[i, j] = edges[j, i] = x
    return np.asarray(nodes, np.float32), edges.reshape(n * n, 34)


def make_record(m, cid="query", generate=False):
    m = Chem.Mol(m)
    if generate:
        m = Chem.AddHs(m)
        p = AllChem.EmbedMolecule(m, randomSeed=101, maxAttempts=50)
        if p < 0:
            p = AllChem.EmbedMolecule(m, randomSeed=101, maxAttempts=50, useRandomCoords=True)
        if p < 0:
            raise ValueError("RDKit embedding failed")
        if AllChem.MMFFHasAllMoleculeParams(m):
            AllChem.MMFFOptimizeMolecule(m, mmffVariant="MMFF94", maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(m, maxIters=200)
        Chem.SanitizeMol(m)
    elif m.GetNumConformers() == 0:
        raise ValueError("Given-conformer mode requires coordinates")
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    nc, ec = normal_features(m)
    xyz = np.asarray(m.GetConformer().GetPositions(), np.float32)
    top = topology(m, nc, ec)
    v = extract(m, top)
    # Compatibility envelope for the AIM worker; expensive columns remain absent.
    return dict(cid=cid, xyz=xyz, top=top, node=np.c_[np.zeros((len(nc), 18), np.float32), nc], views=[v])


def missing_aim(r, dim, status="disabled"):
    n = len(r["xyz"])
    return dict(
        node=np.zeros((n, dim + 7), np.float32),
        edge=np.zeros((n * n, 3), np.float32),
        global_=np.zeros(8, np.float32),
        forces=np.zeros((n, 3), np.float32),
        status=status,
    )


def input_batch(r, a):
    v = r["views"][0]
    n = len(r["xyz"])
    b = {k: torch.tensor(x, dtype=torch.float32) for k, x in groups(r, v, a).items()}
    b.update(
        xyz=torch.tensor(v["xyz"]),
        forces=torch.tensor(a["forces"]),
        z=torch.tensor(r["top"]["z"], dtype=torch.long),
        edge_index=torch.tensor([np.repeat(np.arange(n), n), np.tile(np.arange(n), n)], dtype=torch.long),
        node_batch=torch.zeros(n, dtype=torch.long),
        edge_batch=torch.zeros(n * n, dtype=torch.long),
        ptr=torch.tensor([0, n]),
        cids=[r["cid"]],
        num_graphs=1,
    )
    return b


def input_batch_many(records, aa):
    single = [input_batch(r, a) for r, a in zip(records, aa)]
    out = {}
    offset = 0
    edges = []
    ptr = [0]
    for k in groups(records[0], records[0]["views"][0], aa[0]):
        out[k] = torch.cat([b[k] for b in single])
    for b in single:
        edges.append(b["edge_index"] + offset)
        offset += len(b["z"])
        ptr.append(offset)
    for k in ("xyz", "forces", "z"):
        out[k] = torch.cat([b[k] for b in single])
    out.update(
        edge_index=torch.cat(edges, 1),
        node_batch=torch.cat([torch.full((len(b["z"]),), i, dtype=torch.long) for i, b in enumerate(single)]),
        edge_batch=torch.cat([torch.full((len(b["z"]) ** 2,), i, dtype=torch.long) for i, b in enumerate(single)]),
        ptr=torch.tensor(ptr),
        cids=[r["cid"] for r in records],
        num_graphs=len(records),
    )
    return out


def aim_features(records, python):
    if not python or not Path(python).exists():
        raise FileNotFoundError(
            "GPU AIM mode requires a valid --aim-python interpreter with AIMNet2 installed. "
            "Use --mode cpu --device cuda:0 if you want CUDA execution with cheap-only descriptors."
        )
    with tempfile.TemporaryDirectory() as d:
        ip = Path(d) / "input.pkl"
        op = Path(d) / "output.pkl"
        ip.write_bytes(pickle.dumps(records, protocol=5))
        subprocess.run([python, str(BACKBONE_DIR / "cache_aim.py"), "--input", str(ip), "--output", str(op)], check=True)
        return cache_loads(op.read_bytes())


def load_models(
    checkpoint_path: str | Path,
    device: str = "cpu",
    stats_path: Optional[str | Path] = None,
    head_checkpoint_path: Optional[str | Path] = None,
    priors_path: Optional[str | Path] = None,
):
    ckpt = Path(checkpoint_path).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt}\n"
            f"Download 'qcgem_student_v1.pt' from {RELEASE_URL} and pass --checkpoint <path>."
        )
    st = torch.load(ckpt, weights_only=False, map_location="cpu")
    assert not st.get("diagnostic", False), "Refusing to run inference on a diagnostic checkpoint"

    if "stats" in st:
        stats = st["stats"]
    elif stats_path is not None:
        stats = json.loads(Path(stats_path).expanduser().read_text())
    else:
        raise FileNotFoundError("Raw training checkpoint requires --stats /path/to/stats.json (or use unified qcgem_student_v1.pt).")

    student = Student(stats, device=device, initialize_official=False)
    student.load_state_dict(st["model"])
    student.set_stage(checkpoint_warm(st))
    student.eval()

    head: Optional[CalibratedQCHead] = None
    channel_meta = st.get("channel_metadata", {})
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
            raise FileNotFoundError("Split head_best.pt requires --priors /path/to/priors.pt (or use calibrated_head_v1.pt).")
        head = CalibratedQCHead(stats, priors).to(device)
        head.load_state_dict(hst["head_state_dict"])
        head.eval()
        channel_meta = hst.get("channel_metadata", channel_meta)
    elif "head_state_dict" in st and "priors" in st:
        head = CalibratedQCHead(stats, st["priors"]).to(device)
        head.load_state_dict(st["head_state_dict"])
        head.eval()

    return student, head, stats, channel_meta


def main():
    p = argparse.ArgumentParser(description="Run qcGEM-Student representation & calibrated QC inference.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="Single SMILES string")
    src.add_argument("--smiles-file", help="Path to file containing one SMILES per line")
    src.add_argument("--sdf", help="Path to 3D SDF file")
    p.add_argument("--checkpoint", required=True, help="Path to qcgem_student_v1.pt (or raw best.pt)")
    p.add_argument("--stats", help="Path to stats.json (only needed for raw best.pt)")
    p.add_argument("--head-checkpoint", help="Optional path to calibrated_head_v1.pt or head_best.pt")
    p.add_argument("--priors", help="Path to priors.pt (only needed when passing raw head_best.pt)")
    p.add_argument("--mode", choices=["cpu", "gpu"], default="cpu", help="cpu=cheap descriptors only; gpu=cheap+AIMNet2")
    p.add_argument("--device", default="cpu", help="PyTorch compute device (e.g. cpu, cuda:0)")
    p.add_argument("--aim-python", default=None, help="Python executable for AIMNet2 worker when --mode gpu")
    p.add_argument("--output", required=True, help="Output .pt file path")
    a = p.parse_args()

    torch.set_num_threads(1)
    student, head, stats, channel_meta = load_models(
        a.checkpoint,
        device=a.device,
        stats_path=a.stats,
        head_checkpoint_path=a.head_checkpoint,
        priors_path=a.priors,
    )

    t = time.time()
    records = []
    failures = []
    if a.smiles:
        inputs = [("0", Chem.MolFromSmiles(a.smiles.strip()))]
        generate = True
    elif a.smiles_file:
        inputs = [(str(i), Chem.MolFromSmiles(x.split()[0])) for i, x in enumerate(Path(a.smiles_file).read_text().splitlines()) if x.strip()]
        generate = True
    else:
        inputs = [(str(i), m) for i, m in enumerate(Chem.SDMolSupplier(a.sdf, removeHs=False))]
        generate = False

    for cid, m in inputs:
        try:
            if m is None:
                raise ValueError("invalid structure")
            records.append(make_record(m, cid, generate=generate))
        except Exception as ex:
            failures.append(dict(id=cid, error=str(ex)))
    feature_seconds = time.time() - t

    dim = len(stats["features"]["aim_node"]["std"]) - 7
    t = time.time()
    aa = aim_features(records, a.aim_python) if a.mode == "gpu" and records else [missing_aim(r, dim) for r in records]
    aim_seconds = time.time() - t

    results = []
    t = time.time()
    with torch.no_grad():
        for r, ai in zip(records, aa):
            b = device_batch(input_batch(r, ai), a.device)
            out = student(b, a.mode, decode=True)
            item = dict(
                id=r["cid"],
                node=out["node"].cpu().numpy(),
                edge=out["edge"].cpu().numpy(),
                global_=out["global_"][0].cpu().numpy(),
                readout384=out["readout"][0].cpu().numpy(),
                decoded_xyz=out["decoded"]["xyz"][-1].cpu().numpy(),
                reference_qc_node=out["raw_node"].cpu().numpy(),
                reference_qc_edge=out["raw_edge"].cpu().numpy(),
                validity=dict(
                    aim=ai["status"],
                    eht=bool(r["views"][0]["eht_global"][0]),
                    mmff=bool(r["views"][0]["u1"][-2]),
                    feature_failures=r["views"][0]["failures"],
                ),
            )
            if head is not None:
                pred_n, pred_e = head(out, b)
                item["qc_node"] = pred_n.cpu().numpy()
                item["qc_edge"] = pred_e.cpu().numpy()
            results.append(item)

    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            results=results,
            failures=failures,
            mode=a.mode,
            device=a.device,
            calibrated_head_loaded=head is not None,
            node_qc_names=channel_meta.get("node_names", NODE_NAMES),
            edge_qc_names=channel_meta.get("edge_names", EDGE_NAMES),
            starred_unreliable_channels=channel_meta.get("starred_unreliable_channels", STARRED_UNRELIABLE_CHANNELS),
            starred_element_npa_charge=channel_meta.get("starred_element_npa_charge", STARRED_ELEMENT_NPA_CHARGE),
            feature_seconds=feature_seconds,
            aim_seconds=aim_seconds,
            encode_decode_seconds=time.time() - t,
        ),
        dest,
    )
    print(json.dumps(dict(output=str(dest), success=len(results), failed=len(failures), calibrated=head is not None)))


if __name__ == "__main__":
    main()
