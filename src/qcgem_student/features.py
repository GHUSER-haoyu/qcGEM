"""Cheap RDKit/MMFF94/EHT molecular descriptor extraction and batching."""
from __future__ import annotations

import io
import multiprocessing as mp
import os
import pickle
import subprocess
import time
from pathlib import Path
import tempfile

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdEHTTools, rdMolDescriptors
from rdkit.Chem.EState import EStateIndices

RDLogger.DisableLog("rdApp.warning")

BONDS = [
    "UNSPECIFIED", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE", "QUINTUPLE", "HEXTUPLE",
    "ONEANDAHALF", "TWOANDAHALF", "THREEANDAHALF", "FOURANDAHALF", "FIVEANDAHALF",
    "AROMATIC", "IONIC", "HYDROGEN", "THREECENTER", "DATIVEONE", "DATIVE",
    "DATIVEL", "DATIVER", "OTHER", "ZERO",
]
CHIRAL = [
    "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER",
    "CHI_TETRAHEDRAL", "CHI_ALLENE", "CHI_SQUAREPLANAR", "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_OCTAHEDRAL",
]
HYBRID = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER", "UNSPECIFIED"]
STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]
FG_SMARTS = [
    "[OX2H]", "[NX3;H2,H1;!$(N-C=O)]", "[NX3;H0;!$(N-C=O)]", "C(=O)[OX2H1]", "C(=O)[OX2H0]",
    "C(=O)N", "C(=O)[#6]", "[CX3H1](=O)", "[CX2]#N", "[N+](=O)[O-]",
    "S(=O)(=O)", "S(=O)", "[SX2H]", "[SX2H0]", "[nH]",
    "[nH0]", "[F,Cl,Br,I]", "P(=O)", "[CX3]=[CX3]", "[CX2]#[CX2]",
]
FG = [Chem.MolFromSmarts(x) for x in FG_SMARTS]
EN = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 33: 2.18, 34: 2.55, 35: 2.96, 53: 2.66}
IE = {1: 13.598, 5: 8.298, 6: 11.260, 7: 14.534, 8: 13.618, 9: 17.423, 14: 8.152, 15: 10.487, 16: 10.360, 17: 12.968, 33: 9.789, 34: 9.752, 35: 11.814, 53: 10.451}
PT = Chem.GetPeriodicTable()
GROUPS = ("g1", "g2", "eht_node", "aim_node", "e1", "eht_edge", "aim_edge", "u1", "eht_global", "aim_global")


class _CacheUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _eht_child(mol, send):
    try:
        fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        ok, r = rdEHTTools.RunMol(mol, confId=0)
        if not ok:
            send.send((False, "RunMol returned false"))
            os._exit(0)
        value = (
            np.array(r.GetOrbitalEnergies(), copy=True),
            np.array(r.GetReducedChargeMatrix(), copy=True),
            int(r.numElectrons),
            np.array(r.GetAtomicCharges(), copy=True),
            np.array(r.GetReducedOverlapPopulationMatrix(), copy=True),
        )
        send.send((True, value))
        send.close()
        os._exit(0)
    except BaseException as e:
        try:
            send.send((False, type(e).__name__ + ":" + str(e)))
        finally:
            os._exit(1)


def safe_eht(mol):
    """Run RDKit extended Hueckel theory with fork isolation on POSIX platforms."""
    if "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
        recv, send = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_eht_child, args=(mol, send))
        p.start()
        send.close()
        try:
            if not recv.poll(20):
                raise RuntimeError("EHT timeout 20s")
            try:
                ok, value = recv.recv()
            except EOFError:
                raise RuntimeError("EHT native process exited without result")
            if not ok:
                raise RuntimeError(value)
            return value
        finally:
            recv.close()
            p.join(0.2)
            if p.is_alive():
                p.terminate()
                p.join()

    ok, r = rdEHTTools.RunMol(mol, confId=0)
    if not ok:
        raise RuntimeError("RunMol returned false")
    return (
        np.array(r.GetOrbitalEnergies(), copy=True),
        np.array(r.GetReducedChargeMatrix(), copy=True),
        int(r.numElectrons),
        np.array(r.GetAtomicCharges(), copy=True),
        np.array(r.GetReducedOverlapPopulationMatrix(), copy=True),
    )


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


def topology(mol, node_cheap, edge_cheap):
    n = mol.GetNumAtoms()
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], np.int16)
    node_cheap = node_cheap.copy()
    node_cheap[:, 50:58] = 0
    for a in mol.GetAtoms():
        node_cheap[a.GetIdx(), 50 + HYBRID.index(str(a.GetHybridization()))] = 1
    edge_cheap = edge_cheap.copy()
    ec = edge_cheap.reshape(n, n, 34)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        conj = float(bond.GetIsConjugated())
        ec[i, j, 6:8] = ec[j, i, 6:8] = [1 - conj, conj]
    rings = np.array([[a.IsInRingSize(k) for k in range(3, 9)] for a in mol.GetAtoms()], np.float32)
    cip = np.array([[a.HasProp("_CIPCode") and a.GetProp("_CIPCode") == x for x in ("R", "S")] for a in mol.GetAtoms()], np.float32)
    fg = np.zeros((n, len(FG)), np.float32)
    for k, pat in enumerate(FG):
        for match in mol.GetSubstructMatches(pat):
            fg[list(match), k] = 1
    constants = np.array(
        [[EN.get(int(a), 0), PT.GetRcovalent(int(a)), PT.GetRvdw(int(a)), PT.GetNOuterElecs(int(a)), IE.get(int(a), 0), int(a in EN)] for a in z],
        np.float32,
    )
    g1 = np.concatenate([node_cheap, rings, cip, fg, constants], 1)
    hop = np.minimum(Chem.GetDistanceMatrix(mol), 9).astype(int)
    rot = np.zeros((n, n), np.float32)
    for i, j in mol.GetSubstructMatches(Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")):
        rot[i, j] = rot[j, i] = 1
    ec = edge_cheap.reshape(n, n, 34)
    e1 = np.concatenate([ec, (ec[:, :, 12:].sum(-1) == 0)[:, :, None], rot[:, :, None], np.eye(10, dtype=np.float32)[hop]], 2).reshape(n * n, -1)
    return dict(z=z, g1=g1.astype(np.float32), e1_top=e1.astype(np.float32), constants=constants)


def extract(mol, top):
    feature_start = time.time()
    n = mol.GetNumAtoms()
    xyz = np.asarray(mol.GetConformer().GetPositions(), np.float32)
    z = top["z"]
    delta = xyz[:, None, :] - xyz[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    off = ~np.eye(n, dtype=bool)
    failures = []

    mm = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    has_mm = mm is not None
    mq = np.array([mm.GetMMFFPartialCharge(i) for i in range(n)], np.float32) if has_mm else np.zeros(n, np.float32)
    energy = 0.0
    energy_ok = False
    try:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mm) if has_mm else AllChem.UFFGetMoleculeForceField(mol)
        if ff is not None:
            energy = float(ff.CalcEnergy())
            energy_ok = np.isfinite(energy)
    except Exception as e:
        failures.append("forcefield_energy:" + type(e).__name__)
    Chem.SanitizeMol(mol)

    gas = np.zeros(n, np.float32)
    gas_ok = False
    try:
        AllChem.ComputeGasteigerCharges(mol)
        gas = np.array([float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()], np.float32)
        gas_ok = bool(np.isfinite(gas).all())
        gas = np.nan_to_num(gas)
    except Exception as e:
        failures.append("gasteiger:" + type(e).__name__)

    estate = np.asarray(EStateIndices(mol), np.float32)
    cr = np.asarray(Crippen._GetAtomContribs(mol), np.float32)
    asa = np.asarray(rdMolDescriptors._CalcLabuteASAContribs(mol)[0], np.float32)
    phi = (mq[None, :] / (dist + 0.1) * off).sum(1)

    eh = np.zeros((n, 3), np.float32)
    overlap = np.zeros((n, n), np.float32)
    levels = np.zeros(3, np.float32)
    has_eht = False
    try:
        en, rc, ne, charges, packed = safe_eht(mol)
        h = (ne - 1) // 2
        l = h + 1
        assert rc.shape == (n, len(en)) and 0 <= h < l < len(en)
        eh = np.stack([charges, rc[:, h], rc[:, l]], 1).astype(np.float32)
        ii, jj = np.tril_indices(n)
        assert len(packed) == len(ii)
        overlap[ii, jj] = packed
        overlap[jj, ii] = packed
        levels = np.array([en[h], en[l], en[l] - en[h]], np.float32)
        has_eht = bool(np.isfinite(eh).all() and np.isfinite(levels).all() and np.isfinite(overlap).all())
        if not has_eht:
            eh[:] = 0
            levels[:] = 0
            overlap[:] = 0
    except Exception as e:
        failures.append("eht:" + type(e).__name__ + ":" + str(e)[:120])

    g2 = np.column_stack([np.full(n, has_mm), mq, np.full(n, gas_ok), gas, estate, cr, asa, phi]).astype(np.float32)
    eht_node = np.column_stack([np.full(n, has_eht), eh]).astype(np.float32)
    c = top["constants"]
    vdw = (c[:, 2, None] + c[None, :, 2]) / (dist + 0.1) * off
    empirical = np.stack([np.abs(c[:, 0, None] - c[None, :, 0]), vdw, mq[:, None] * mq[None, :] / (dist + 0.1) * off], -1).reshape(n * n, 3)
    e1 = np.concatenate([top["e1_top"], empirical], 1).astype(np.float32)
    eht_edge = np.stack([np.full((n, n), has_eht), overlap], -1).reshape(n * n, 2).astype(np.float32)
    center = xyz - np.average(xyz, axis=0, weights=[PT.GetAtomicWeight(int(a)) for a in z])
    u1 = np.array(
        [
            Descriptors.MolWt(mol),
            int((z > 1).sum()),
            sum(PT.GetNOuterElecs(int(a)) for a in z) - Chem.GetFormalCharge(mol),
            Chem.GetFormalCharge(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Crippen.MolMR(mol),
            energy if energy_ok else 0,
            np.linalg.norm((mq[:, None] * center).sum(0)),
            has_mm,
            energy_ok,
        ],
        np.float32,
    )
    result = dict(xyz=xyz, g2=g2, e1=e1, u1=u1, eht_node=eht_node, eht_edge=eht_edge, eht_global=np.r_[has_eht, levels].astype(np.float32), failures=failures)
    for k, v in result.items():
        if isinstance(v, np.ndarray) and not np.isfinite(v).all():
            raise ValueError("nonfinite cheap " + k)
    result["feature_seconds"] = time.time() - feature_start
    return result


def make_record(m: Chem.Mol, cid: str = "query", generate: bool = False) -> dict:
    m = Chem.Mol(m)
    if generate:
        m = Chem.AddHs(m)
        p = AllChem.EmbedMolecule(m, randomSeed=101, maxAttempts=50)
        if p < 0:
            p = AllChem.EmbedMolecule(m, randomSeed=101, maxAttempts=50, useRandomCoords=True)
        if p < 0:
            raise ValueError("RDKit 3D conformer embedding failed")
        if AllChem.MMFFHasAllMoleculeParams(m):
            AllChem.MMFFOptimizeMolecule(m, mmffVariant="MMFF94", maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(m, maxIters=200)
        Chem.SanitizeMol(m)
    elif m.GetNumConformers() == 0:
        raise ValueError("Given-conformer mode requires 3D coordinates (or pass generate_conformer=True).")
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    nc, ec = normal_features(m)
    xyz = np.asarray(m.GetConformer().GetPositions(), np.float32)
    top = topology(m, nc, ec)
    v = extract(m, top)
    return dict(cid=cid, xyz=xyz, top=top, node=np.c_[np.zeros((len(nc), 18), np.float32), nc], views=[v])


def missing_aim(r: dict, dim: int, status: str = "disabled") -> dict:
    n = len(r["xyz"])
    return dict(
        node=np.zeros((n, dim + 7), np.float32),
        edge=np.zeros((n * n, 3), np.float32),
        global_=np.zeros(8, np.float32),
        forces=np.zeros((n, 3), np.float32),
        status=status,
    )


def groups(r: dict, v: dict, a: dict) -> dict:
    g1 = np.asarray(r["top"]["g1"], np.float32).copy()
    # Canonicalize student stereo block to unspecified/R/S
    cip = g1[:, 68:70].copy()
    g1[:, 5:14] = 0
    g1[:, 5] = cip.sum(-1) == 0
    g1[:, 6:8] = cip
    return dict(
        g1=g1,
        g2=v["g2"],
        eht_node=v["eht_node"],
        aim_node=a["node"],
        e1=v["e1"],
        eht_edge=v["eht_edge"],
        aim_edge=a["edge"],
        u1=v["u1"][None],
        eht_global=v["eht_global"][None],
        aim_global=a["global_"][None],
    )


def input_batch(r: dict, a: dict) -> dict:
    v = r["views"][0]
    n = len(r["xyz"])
    b = {k: torch.from_numpy(np.asarray(x, np.float32).copy()) for k, x in groups(r, v, a).items()}
    edge_idx = np.stack([np.repeat(np.arange(n), n), np.tile(np.arange(n), n)], axis=0).astype(np.int64)
    b.update(
        xyz=torch.from_numpy(np.asarray(v["xyz"], np.float32).copy()),
        forces=torch.from_numpy(np.asarray(a["forces"], np.float32).copy()),
        z=torch.from_numpy(np.asarray(r["top"]["z"], np.int64).copy()),
        edge_index=torch.from_numpy(edge_idx),
        node_batch=torch.zeros(n, dtype=torch.long),
        edge_batch=torch.zeros(n * n, dtype=torch.long),
        ptr=torch.tensor([0, n], dtype=torch.long),
        eptr=torch.tensor([0, n * n], dtype=torch.long),
        cids=[r["cid"]],
        num_graphs=1,
    )
    return b


def input_batch_many(records: list[dict], aa: list[dict]) -> dict:
    single = [input_batch(r, a) for r, a in zip(records, aa)]
    out = {}
    offset = 0
    e_offset = 0
    edges = []
    ptr = [0]
    eptr = [0]
    for k in GROUPS:
        out[k] = torch.cat([b[k] for b in single], dim=0)
    for b in single:
        n = len(b["z"])
        edges.append(b["edge_index"] + offset)
        offset += n
        e_offset += n * n
        ptr.append(offset)
        eptr.append(e_offset)
    for k in ("xyz", "forces", "z"):
        out[k] = torch.cat([b[k] for b in single], dim=0)
    out.update(
        edge_index=torch.cat(edges, 1),
        node_batch=torch.cat([torch.full((len(b["z"]),), i, dtype=torch.long) for i, b in enumerate(single)]),
        edge_batch=torch.cat([torch.full((len(b["z"]) ** 2,), i, dtype=torch.long) for i, b in enumerate(single)]),
        ptr=torch.tensor(ptr, dtype=torch.long),
        eptr=torch.tensor(eptr, dtype=torch.long),
        cids=[r["cid"] for r in records],
        num_graphs=len(records),
    )
    return out


def device_batch(b: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in b.items()}


def aim_features(records: list[dict], aim_python: str | Path, cache_aim_script: str | Path) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        ip = Path(d) / "input.pkl"
        op = Path(d) / "output.pkl"
        ip.write_bytes(pickle.dumps(records, protocol=5))
        subprocess.run([str(aim_python), str(cache_aim_script), "--input", str(ip), "--output", str(op)], check=True)
        return _CacheUnpickler(io.BytesIO(op.read_bytes())).load()
