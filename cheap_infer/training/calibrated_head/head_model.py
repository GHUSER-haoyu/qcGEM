"""Isolated post-hoc calibrated QC output head for frozen qcGEM-Student."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(os.environ.get("QCMOL_STUDENT_ROOT", Path.cwd()))
os.environ["QCMOL_STUDENT_ROOT"] = str(ROOT)
BACKBONE_DIR = Path(os.environ.get("QCMOL_BACKBONE_DIR", Path(__file__).resolve().parents[1] / "backbone"))
if str(BACKBONE_DIR) not in sys.path:
    sys.path.insert(0, str(BACKBONE_DIR))

HEAD_DIR = Path(__file__).resolve().parent
STATS_PATH = Path(os.environ.get("QCMOL_STATS_PATH", ROOT / "stats.json"))
PRIORS_PATH = Path(os.environ.get("QCMOL_PRIORS_PATH", HEAD_DIR / "priors.pt"))
NUM_Z = 119  # 0..118 clamped


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def compute_pair_indices(edge_index: torch.Tensor, z: torch.Tensor, e1: torch.Tensor):
    """Map edges to clamped endpoint elements (za, zb), pair_type (0 nonbond, 1 bonded, 2 self), and flat prior idx."""
    i, j = edge_index[0], edge_index[1]
    za = z[i].clamp(0, NUM_Z - 1)
    zb = z[j].clamp(0, NUM_Z - 1)
    is_self = (i == j)
    is_bonded = (e1[:, 12:34].sum(-1) > 0.5) & (~is_self)
    pair_type = torch.where(
        is_self,
        torch.full_like(za, 2),
        torch.where(is_bonded, torch.ones_like(za), torch.zeros_like(za)),
    )
    z_lo = torch.minimum(za, zb)
    z_hi = torch.maximum(za, zb)
    pair_idx = (z_lo * NUM_Z + z_hi) * 3 + pair_type
    return za, zb, pair_type, pair_idx


def clean_node_targets(raw_node: torch.Tensor) -> torch.Tensor:
    """Return [N, 18] targets with H core placeholder 5.0 in col 0 masked to NaN (matching evaluate.py)."""
    yn = raw_node[:, :18].float().clone()
    col0 = yn[:, 0]
    yn[:, 0] = torch.where(col0 == 5.0, torch.full_like(col0, float("nan")), col0)
    return yn


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CalibratedQCHead(nn.Module):
    """Per-channel + element/pair-stratum conditioned calibration & prediction head on frozen Student features."""

    def __init__(self, stats: dict, priors: dict, hidden: int = 384):
        super().__init__()
        self.register_buffer("ns", torch.tensor(stats["node_qc"]["std"], dtype=torch.float32).clamp_min(1e-5))
        self.register_buffer("es", torch.tensor(stats["edge_qc"]["std"], dtype=torch.float32).clamp_min(1e-5))
        feats = stats["features"]
        for k in ("g1", "g2", "eht_node", "e1", "eht_edge"):
            self.register_buffer(f"scale_{k}", torch.tensor(feats[k]["std"], dtype=torch.float32).clamp_min(1e-5))

        self.register_buffer("node_prior_mean", priors["node_prior_mean"].float())
        self.register_buffer("node_within_std", torch.minimum(priors["node_within_std"].float(), self.ns).clamp_min(1e-4))
        self.register_buffer("pair_prior_mean", priors["pair_prior_mean"].float())
        self.register_buffer("edge_within_std", torch.minimum(priors["edge_within_std"].float(), self.es).clamp_min(1e-4))
        self.register_buffer("z_weight", priors["z_weight"].float())
        self.register_buffer("ptype_weight", priors["ptype_weight"].float())

        # Node embeddings & trunk (input = 128 + 32 + 18 + 18 + 96 + 9 + 4 = 305)
        self.z_emb = nn.Embedding(NUM_Z, 32)
        self.node_in = nn.Sequential(nn.LayerNorm(305), nn.Linear(305, hidden), nn.GELU())
        self.node_blocks = nn.Sequential(ResBlock(hidden), ResBlock(hidden))
        self.node_norm = nn.LayerNorm(hidden)
        self.node_out = nn.Linear(hidden, 18)
        self.node_skip = nn.Linear(36, 18)
        self.elem_bias_node = nn.Embedding(NUM_Z, 18)
        self.raw_gate_node = nn.Parameter(torch.zeros(18))

        # Edge embeddings & trunk (input = 128 + 16 + 16 + 8 + 19 + 19 + 49 + 2 = 257)
        self.ze_emb = nn.Embedding(NUM_Z, 16)
        self.ptype_emb = nn.Embedding(3, 8)
        self.edge_in = nn.Sequential(nn.LayerNorm(257), nn.Linear(257, hidden), nn.GELU())
        self.edge_blocks = nn.Sequential(ResBlock(hidden), ResBlock(hidden))
        self.edge_norm = nn.LayerNorm(hidden)
        self.edge_out = nn.Linear(hidden, 19)
        self.edge_skip = nn.Linear(38, 19)
        self.ze_bias_edge = nn.Embedding(NUM_Z, 19)
        self.ptype_bias_edge = nn.Embedding(3, 19)
        self.raw_gate_edge = nn.Parameter(torch.zeros(19))

        self._init_residuals()

    def _init_residuals(self):
        for layer in (self.node_out, self.node_skip, self.edge_out, self.edge_skip):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.elem_bias_node.weight)
        nn.init.zeros_(self.ze_bias_edge.weight)
        nn.init.zeros_(self.ptype_bias_edge.weight)

    def forward(self, out: dict, b: dict):
        nb = b["node_batch"]
        eb = b["edge_batch"]
        z = b["z"].clamp(0, NUM_Z - 1)
        eht_keep = out["eht_keep"].float()

        dec_n = out["decoded"]["node_features"][:, :18].float()
        raw_n = out["raw_node"].float()
        asinh_raw_n = torch.asinh(raw_n / self.ns)
        x_n = torch.cat(
            [
                out["node"].float(),
                self.z_emb(z),
                dec_n,
                asinh_raw_n,
                torch.asinh(b["g1"].float() / self.scale_g1),
                torch.asinh(b["g2"].float() / self.scale_g2),
                torch.asinh((b["eht_node"].float() * eht_keep[nb, None]) / self.scale_eht_node),
            ],
            dim=-1,
        )
        h_n = self.node_norm(self.node_blocks(self.node_in(x_n)))
        skip_n = torch.cat([dec_n, asinh_raw_n], dim=-1)
        prior_n = self.node_prior_mean[z]
        pred_node = (
            prior_n
            + torch.sigmoid(self.raw_gate_node) * (raw_n - prior_n)
            + (self.elem_bias_node(z) + self.node_out(h_n) + self.node_skip(skip_n)) * self.node_within_std
        )

        za, zb, pair_type, pair_idx = compute_pair_indices(b["edge_index"], z, b["e1"])
        dec_e = out["decoded"]["edge_features"][:, :19].float()
        raw_e = out["raw_edge"].float()
        asinh_raw_e = torch.asinh(raw_e / self.es)
        x_e = torch.cat(
            [
                out["edge"].float(),
                self.ze_emb(za),
                self.ze_emb(zb),
                self.ptype_emb(pair_type),
                dec_e,
                asinh_raw_e,
                torch.asinh(b["e1"].float() / self.scale_e1),
                torch.asinh((b["eht_edge"].float() * eht_keep[eb, None]) / self.scale_eht_edge),
            ],
            dim=-1,
        )
        h_e = self.edge_norm(self.edge_blocks(self.edge_in(x_e)))
        skip_e = torch.cat([dec_e, asinh_raw_e], dim=-1)
        prior_e = self.pair_prior_mean[pair_idx]
        pred_edge = (
            prior_e
            + torch.sigmoid(self.raw_gate_edge) * (raw_e - prior_e)
            + (self.ze_bias_edge(za) + self.ze_bias_edge(zb) + self.ptype_bias_edge(pair_type) + self.edge_out(h_e) + self.edge_skip(skip_e))
            * self.edge_within_std
        )
        return pred_node, pred_edge

    def compute_loss(self, pred_node: torch.Tensor, pred_edge: torch.Tensor, b: dict):
        yn = clean_node_targets(b["raw_node"])
        valid_n = torch.isfinite(yn)
        err_n = torch.where(valid_n, (pred_node - yn) / self.node_within_std, torch.zeros_like(pred_node))
        loss_mat_n = 0.5 * F.smooth_l1_loss(err_n, torch.zeros_like(err_n), beta=1.0, reduction="none") + 0.5 * err_n.square().clamp_max(25.0)
        z = b["z"].clamp(0, NUM_Z - 1)
        wn = self.z_weight[z][:, None] * valid_n.float()
        loss_node = (loss_mat_n * wn).sum() / wn.sum().clamp_min(1.0)

        ye = b["raw_edge"][:, :19].float()
        valid_e = torch.isfinite(ye)
        err_e = torch.where(valid_e, (pred_edge - ye) / self.edge_within_std, torch.zeros_like(pred_edge))
        loss_mat_e = 0.5 * F.smooth_l1_loss(err_e, torch.zeros_like(err_e), beta=1.0, reduction="none") + 0.5 * err_e.square().clamp_max(25.0)
        za, zb, pair_type, _ = compute_pair_indices(b["edge_index"], z, b["e1"])
        we_row = (self.ptype_weight[pair_type] * 0.5 * (self.z_weight[za] + self.z_weight[zb])).clamp(0.5, 4.0)
        we = we_row[:, None] * valid_e.float()
        loss_edge = (loss_mat_e * we).sum() / we.sum().clamp_min(1.0)

        total = loss_node + loss_edge
        return total, {"loss_node": float(loss_node.detach()), "loss_edge": float(loss_edge.detach())}


class CheapAimRecords(torch.utils.data.Dataset):
    """Lightweight Dataset loading only cheap + aim LMDBs (zero teacher I/O)."""

    def __init__(self, split: str, epoch: int = 0):
        self.rows = [r for r in json.loads((ROOT / "manifest.json").read_text()) if r["split"] == split]
        self.epoch = epoch
        self.split = split
        self.readers = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k: int):
        from common import Reader
        if self.readers is None:
            self.readers = [Reader("cheap"), Reader("aim")]
        row = self.rows[k]
        r = self.readers[0].get(row["i"])
        a = self.readers[1].get(row["i"])
        if self.epoch == 0:
            j = 0
        elif self.split == "train":
            gen = np.random.default_rng(row["i"] + 1000003 * self.epoch)
            j = 0 if gen.random() < 0.1 else int(gen.integers(1, len(r["views"])))
        else:
            j = 1
        return {"record": r, "view": r["views"][j], "aim": a[j]}

    def close(self):
        if self.readers:
            for rdr in self.readers:
                for db in rdr.dbs.values():
                    db.close()
                rdr.dbs.clear()
            self.readers = None


def collate_cheap_aim(items):
    from data import GROUPS, groups
    b = {k: [] for k in GROUPS}
    nb, eb, edges, xyz, forces, nodes, rawedges, z = [], [], [], [], [], [], [], []
    ptr, eptr = [0], [0]
    for g, it in enumerate(items):
        r, v, a = it["record"], it["view"], it["aim"]
        n = len(r["xyz"])
        offset = ptr[-1]
        for k, x in groups(r, v, a).items():
            b[k].append(np.asarray(x, np.float32))
        edges.append(np.array([np.repeat(np.arange(n), n), np.tile(np.arange(n), n)]) + offset)
        nb.append(np.full(n, g))
        eb.append(np.full(n * n, g))
        ptr.append(offset + n)
        eptr.append(eptr[-1] + n * n)
        xyz.append(v["xyz"])
        forces.append(a["forces"])
        nodes.append(r["node"])
        rawedges.append(r["edge"])
        z.append(r["top"]["z"])
    out = {k: torch.from_numpy(np.concatenate(v)) for k, v in b.items()}
    out.update(
        xyz=torch.tensor(np.concatenate(xyz)),
        forces=torch.tensor(np.concatenate(forces)),
        raw_node=torch.tensor(np.concatenate(nodes)),
        raw_edge=torch.tensor(np.concatenate(rawedges)),
        z=torch.tensor(np.concatenate(z), dtype=torch.long),
        edge_index=torch.tensor(np.concatenate(edges, 1), dtype=torch.long),
        node_batch=torch.tensor(np.concatenate(nb), dtype=torch.long),
        edge_batch=torch.tensor(np.concatenate(eb), dtype=torch.long),
        ptr=torch.tensor(ptr),
        eptr=torch.tensor(eptr),
        ids=[it["record"]["i"] for it in items],
        cids=[it["record"]["cid"] for it in items],
        num_graphs=len(items),
    )
    return out
