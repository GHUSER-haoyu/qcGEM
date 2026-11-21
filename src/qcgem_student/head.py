"""Post-hoc calibrated 37-channel QC output head for frozen qcGEM-Student."""
from __future__ import annotations

import torch
from torch import nn

NUM_Z = 119
NODE_NAMES = (
    [f"NAO_{i}" for i in range(6)]
    + [f"LP_{i}" for i in range(5)]
    + ["LP_occupancy", "ADCH", "LI", "NPA_charge", "NPA_core", "NPA_valence", "NPA_rydberg"]
)
EDGE_NAMES = [f"BD_{i}" for i in range(14)] + ["BD_occupancy", "Mayer", "LBO", "ELF", "DI"]
STARRED_UNRELIABLE_CHANNELS = ["LP_1", "LP_3", "NPA_rydberg", "BD_3", "BD_5", "BD_10", "BD_12"]
STARRED_ELEMENT_NPA_CHARGE = ["Br"]


def compute_pair_indices(edge_index: torch.Tensor, z: torch.Tensor, e1: torch.Tensor):
    """Map edges to clamped endpoint elements (za, zb), pair_type (0 nonbond, 1 bonded, 2 self), and flat prior idx."""
    i, j = edge_index[0], edge_index[1]
    za = z[i].clamp(0, NUM_Z - 1)
    zb = z[j].clamp(0, NUM_Z - 1)
    is_self = i == j
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

        self.z_emb = nn.Embedding(NUM_Z, 32)
        self.node_in = nn.Sequential(nn.LayerNorm(305), nn.Linear(305, hidden), nn.GELU())
        self.node_blocks = nn.Sequential(ResBlock(hidden), ResBlock(hidden))
        self.node_norm = nn.LayerNorm(hidden)
        self.node_out = nn.Linear(hidden, 18)
        self.node_skip = nn.Linear(36, 18)
        self.elem_bias_node = nn.Embedding(NUM_Z, 18)
        self.raw_gate_node = nn.Parameter(torch.zeros(18))

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
