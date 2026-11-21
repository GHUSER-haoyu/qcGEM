"""qcGEM-Student distilled encoder/decoder wrapper."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .official import create
from .stable_ops import deterministic_scatter, reset_group_plans


def checkpoint_warm(state: dict) -> bool:
    """Restore the inference stage used by a saved Student checkpoint."""
    epoch = int(state["epoch_next"]) - int(bool(state.get("resume_at_epoch_boundary", False)))
    if epoch < 0:
        raise ValueError("Invalid checkpoint epoch metadata")
    return epoch == 0


def mean(x, index, n):
    return deterministic_scatter(x, index, dim_size=n, reduce="mean")


class Projection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 256))

    def forward(self, x):
        return self.layers(x)


class Fusion(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(256 * n, 256), nn.SiLU(), nn.Linear(256, 256 * n))
        self.norm = nn.LayerNorm(256)
        self.n = n

    def forward(self, z):
        gate = self.gate(torch.cat(z, -1)).sigmoid().reshape(-1, self.n, 256)
        return self.norm((torch.stack(z, 1) * gate).sum(1))


class GeoInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo = nn.Linear(12, 256)

    def forward(self, x):
        return x[:, :256] + self.geo(x[:, 256:])


class First(nn.Module):
    def forward(self, x):
        return x[:, :256]


class Student(nn.Module):
    def __init__(self, stats, device="cpu", initialize_official=False):
        super().__init__()
        self.stats = stats
        self.core, self.geo = create(device, stable=True, load_weights=initialize_official)
        self.core.pre_process.requires_grad_(False)
        self.core.decoder.requires_grad_(False)
        enc = self.core.encoder
        enc.embedding_node_input = GeoInput()
        enc.embedding_edge_input = nn.Identity()
        enc.embedding_md_input = nn.Identity()
        enc.embedding_fp_input = nn.Identity()
        enc.embedding_gf_input = First()
        self.proj = nn.ModuleDict({k: Projection(len(v["std"])) for k, v in stats["features"].items()})
        self.fuse_node = Fusion(4)
        self.fuse_edge = Fusion(3)
        self.fuse_global = Fusion(3)
        self.step = nn.Sequential(
            nn.Linear(len(stats["features"]["aim_node"]["std"]) - 7, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.node_scale = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))
        self.edge_scale = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))
        self.register_buffer("energy_coeff", torch.tensor(stats["energy_coeff"], dtype=torch.float32))
        for k, v in stats["features"].items():
            self.register_buffer("scale_" + k, torch.tensor(v["std"], dtype=torch.float32).clamp_min(1e-5))
        for name, p in enc.named_parameters():
            if name.startswith(("Global_Norm_", "Global_Update_layer_NecOut_15.", "Edge_Update_layer_NecOut_15.")):
                p.requires_grad_(False)
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.core.decoder.eval()
        self.core.pre_process.eval()
        return self

    def set_stage(self, warm):
        for name, p in self.core.encoder.named_parameters():
            p.requires_grad_(
                not warm and not name.startswith(("Global_Norm_", "Global_Update_layer_NecOut_15.", "Edge_Update_layer_NecOut_15."))
            )
        for p in self.core.encoder.embedding_node_input.parameters():
            p.requires_grad_(True)
        for p in self.step.parameters():
            p.requires_grad_(not warm)
        self.warm = warm

    def forward(self, b, mode="cpu", decode=True):
        reset_group_plans()
        self.geo._feature_cache.clear()
        nb = b["node_batch"]
        eb = b["edge_batch"]
        ng = b["num_graphs"]
        aim_keep = b["aim_global"][:, 0].clone()
        eht_keep = b["eht_global"][:, 0].clone()
        if mode == "cpu":
            aim_keep.zero_()
        elif self.training:
            aim_keep *= torch.rand(ng, device=nb.device) >= 0.2
        if self.training:
            eht_keep *= torch.rand(ng, device=nb.device) >= 0.1
        projected = {}
        for k, layer in self.proj.items():
            x = b[k].float()
            mask = None
            if k == "aim_global":
                x = x.clone()
                x[:, 1] -= mean(self.energy_coeff[b["z"]].unsqueeze(1), nb, ng).squeeze(1) * x[:, 0]
            if k.startswith("aim"):
                mask = aim_keep[nb] if k.endswith("node") else aim_keep[eb] if k.endswith("edge") else aim_keep
            if k.startswith("eht"):
                mask = eht_keep[nb] if k.endswith("node") else eht_keep[eb] if k.endswith("edge") else eht_keep
            x = torch.asinh(x / getattr(self, "scale_" + k))
            if mask is not None:
                x = x * mask[:, None]
            projected[k] = layer(x)
            if mask is not None:
                projected[k] = projected[k] * mask[:, None]
        node = self.fuse_node([projected[k] for k in ("g1", "g2", "eht_node", "aim_node")])
        edge = self.fuse_edge([projected[k] for k in ("e1", "eht_edge", "aim_edge")])
        glob = self.fuse_global([projected[k] for k in ("u1", "eht_global", "aim_global")])
        pos = b["xyz"].double()
        f = b["forces"].double()
        fn = f.norm(dim=-1, keepdim=True)
        if not getattr(self, "warm", False):
            latent = b["aim_node"][:, 1:-6].float()
            step = 0.15 * self.step(latent).sigmoid().double() * aim_keep[nb, None]
            correction = step * torch.tanh(fn / 2) * f / fn.clamp_min(1e-8)
            pos = pos + correction
        pos = pos - mean(pos, nb, ng)[nb]
        i, j = b["edge_index"]
        d2 = (pos[i] - pos[j]).square().sum(-1)
        keep = (i != j) & (d2 < 64.0) & (d2 > 1e-12)
        geoedges = b["edge_index"][:, keep]
        self.geo.radius_graph = lambda *a, **kw: geoedges
        inp = dict(
            global_MD=glob,
            global_FP=torch.zeros_like(glob),
            xyz=pos.float(),
            pos=pos,
            node_features=node,
            edge_features=edge,
            edge_index=b["edge_index"],
            atom_type=b["z"],
            batch=nb,
            ptr=b["ptr"],
            CID_list=b["cids"],
        )
        enc, states = self.core.encoder(inp)
        rep = self.core.encoder.pool_graph_embedding(enc, states)
        out = dict(
            node=enc["node_features"],
            edge=enc["edge_features"],
            global_=enc["global_features"],
            readout=rep,
            mid_node=torch.stack([states["Node"][l] for l in (1, 2, 3)]),
            mid_global=torch.stack([states["Global"][l] for l in (1, 2, 3)], 1),
            corrected_xyz=pos,
            aim_keep=aim_keep,
            eht_keep=eht_keep,
        )
        if decode:
            out = self.decode(out, b)
        return out

    def decode(self, out, b):
        enc = dict(
            node_features=out["node"].float(),
            edge_features=out["edge"].float(),
            global_features=out["global_"].float(),
            edge_index=b["edge_index"],
            atom_type=b["z"],
            batch=b["node_batch"],
        )
        with torch.autocast(device_type=out["node"].device.type, enabled=False):
            dec = self.core.decoder(enc)
        sn = self.node_scale(out["node"].float())
        se = self.edge_scale(out["edge"].float())
        out.update(
            decoded=dec,
            node_affine=sn,
            edge_affine=se,
            raw_node=dec["node_features"][:, :18] * (F.softplus(sn[:, 1:]) + 1e-5) + sn[:, :1],
            raw_edge=dec["edge_features"][:, :19] * (F.softplus(se[:, 1:]) + 1e-5) + se[:, :1],
        )
        return out
