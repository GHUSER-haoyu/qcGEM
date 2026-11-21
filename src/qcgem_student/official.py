"""Package-local loader for the geometry-corrected qcGEM encoder/decoder core."""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

from . import stable_ops as s
from . import symmetric_geometry

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def load_module(stable: bool = True):
    name = "qcgem_student._core_stable" if stable else "qcgem_student._core_original"
    if name in sys.modules:
        return sys.modules[name], sys.modules[name + "_geo"]

    spec_geo = importlib.util.spec_from_file_location(name + "_geo", VENDOR_DIR / "GeometryModule.py")
    geo = importlib.util.module_from_spec(spec_geo)
    sys.modules[spec_geo.name] = geo
    spec_geo.loader.exec_module(geo)

    spec_utils = importlib.util.spec_from_file_location(name + "_utils", VENDOR_DIR / "utils.py")
    utils_mod = importlib.util.module_from_spec(spec_utils)
    sys.modules[spec_utils.name] = utils_mod
    spec_utils.loader.exec_module(utils_mod)

    tree = ast.parse((VENDOR_DIR / "models.py").read_text())
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and n.module in ("GeometryModule", "utils"))
    ]

    m = types.ModuleType(name)
    m.__file__ = str(VENDOR_DIR / "models.py")
    for k, v in utils_mod.__dict__.items():
        if not k.startswith("__"):
            m.__dict__[k] = v
    m.GeometryModule_ = geo.GeometryModule_
    sys.modules[name] = m
    exec(compile(tree, m.__file__, "exec"), m.__dict__)

    original = m.MaskProcess

    def clean(*a, **kw):
        kw["noise_size"] = 0.0
        return original(*a, **kw)

    m.MaskProcess = clean
    if stable:
        geo.scatter = s.deterministic_scatter
        geo.scatter_min = s.stable_min
        geo.EdgeGraphConv.aggregate = s.stable_aggregate
        symmetric_geometry.install(geo, s.deterministic_scatter)
        m.softmax = s.stable_softmax
        m.qcGEM_Transformer.aggregate = s.stable_aggregate
    return m, geo


def create(device: str = "cpu", stable: bool = True, load_weights: bool = False, checkpoint_path=None):
    m, geo = load_module(stable)
    model = m.qcGEM(
        input_global_dim=[200, 512, 512],
        global_head_dim=32,
        botnec_global_dim=128,
        input_node_dim=80,
        node_head_dim=32,
        BotNec_node_dim=128,
        input_edge_dim=53,
        edge_head_dim=32,
        BotNec_edge_dim=128,
        heads=8,
        device=device,
        act_fn=nn.GELU(),
        norm="layer",
        remove_self_loop=False,
        global_mask_ratio=1.0,
        mask_ratio=0.0,
        replace_ratio=0.0,
        remask_ratio=0.0,
        encoder_method="qcGEM_Encoder",
        decoder_method="qcGEM_Decoder",
        encoder_layers=16,
        decoder_layers=0,
        gm_cutoff=8.0,
        gm_output_dim=12,
        gm_interact_time=4,
        gm_layer_num=3,
    )
    if load_weights:
        if not checkpoint_path:
            raise FileNotFoundError("load_weights=True requires an explicit checkpoint_path.")
        state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")["model_state_dict"]
        model.load_state_dict(state, strict=True)
    if stable:
        def stable_mean(self, x, index=None, ptr=None, dim_size=None, dim=-2, **kwargs):
            assert index is not None and dim in (-2, 0)
            return s.deterministic_scatter(x, index, dim_size=dim_size, reduce="mean")

        for block in model.encoder.modules():
            if isinstance(block, m.aggr.MeanAggregation):
                block.forward = types.MethodType(stable_mean, block)
        for g in (model.encoder.GM_1, model.encoder.GM_2):
            for block in g.interaction_blocks:
                block.norm_layer.forward = types.MethodType(s.stable_graph_norm, block.norm_layer)
    return model, geo
