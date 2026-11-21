"""Deterministic, FP32/FP64-safe graph reductions with pure-PyTorch fallback."""
from __future__ import annotations

import torch

_GROUP_PLANS = {}


def reset_group_plans():
    _GROUP_PLANS.clear()


def _pure_segment_csr(
    values: torch.Tensor,
    ptr: torch.Tensor,
    count: torch.Tensor,
    size: int,
    sorted_index: torch.Tensor,
    reduce: str,
) -> torch.Tensor:
    if values.shape[0] == 0:
        return values.new_zeros((size, *values.shape[1:]))

    if reduce in ("sum", "add", "mean"):
        try:
            res = torch.segment_reduce(values, reduce="sum", offsets=ptr, axis=0, initial=0.0)
        except (RuntimeError, TypeError, ValueError):
            idx = sorted_index.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
            res = values.new_zeros((size, *values.shape[1:])).scatter_add_(0, idx, values)
        if reduce == "mean":
            denom = count.clamp_min(1).to(res.dtype).view(-1, *([1] * (res.ndim - 1)))
            res = res / denom
        return res

    # reduce == 'max'
    try:
        res = torch.segment_reduce(values, reduce="max", offsets=ptr, axis=0, initial=float("-inf"))
    except (RuntimeError, TypeError, ValueError):
        idx = sorted_index.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
        res = values.new_full((size, *values.shape[1:]), float("-inf"))
        res.scatter_reduce_(0, idx, values, reduce="amax", include_self=True)
    nonempty = (count > 0).view(-1, *([1] * (res.ndim - 1)))
    return torch.where(nonempty, res, torch.zeros_like(res))


def _run_segment_csr(
    values: torch.Tensor,
    ptr: torch.Tensor,
    count: torch.Tensor,
    size: int,
    sorted_index: torch.Tensor,
    reduce: str,
) -> torch.Tensor:
    op = "sum" if reduce == "add" else reduce
    try:
        from torch_scatter import segment_csr
    except ImportError:
        segment_csr = None
    if segment_csr is not None:
        return segment_csr(values, ptr, reduce=op)
    return _pure_segment_csr(values, ptr, count, size, sorted_index, op)


def deterministic_scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
    if dim != 0 or out is not None or reduce not in ("sum", "add", "mean", "max"):
        raise ValueError("unsupported stable reduction")
    size = dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
    key = (index.data_ptr(), index.numel(), index.stride(), size)
    if key not in _GROUP_PLANS:
        order = torch.argsort(index, stable=True)
        count = torch.bincount(index, minlength=size)
        ptr = torch.cat([count.new_zeros(1), count.cumsum(0)])
        # Hold a live reference to `index` so its storage address cannot be recycled before reset_group_plans().
        _GROUP_PLANS[key] = (index, index[order], order, count, ptr)
    _, sorted_index, order, count, ptr = _GROUP_PLANS[key]
    values = src[order]
    if values.dtype in (torch.float16, torch.bfloat16) and reduce in ("sum", "add", "mean"):
        return _run_segment_csr(values.float(), ptr, count, size, sorted_index, reduce)
    if values.dtype == torch.float32 and reduce in ("sum", "add", "mean"):
        return _run_segment_csr(values.double(), ptr, count, size, sorted_index, reduce).float()
    return _run_segment_csr(values, ptr, count, size, sorted_index, reduce)


def stable_min(src, index, dim_size=None):
    with torch.no_grad():
        value = torch.full((dim_size,), float("inf"), device=src.device, dtype=src.dtype)
        value.scatter_reduce_(0, index, src.detach(), reduce="amin", include_self=True)
        pos = torch.arange(len(src), device=src.device)
        candidate = torch.where(src.detach() == value[index], pos, len(src))
        arg = torch.full((dim_size,), len(src), device=src.device, dtype=torch.long)
        arg.scatter_reduce_(0, index, candidate, reduce="amin", include_self=True)
    return value, arg


def stable_cross(a, b, dim=-1):
    value = torch.cross(a, b, dim=dim)
    scale = a.norm(dim=dim, keepdim=True) * b.norm(dim=dim, keepdim=True)
    return torch.where(value.norm(dim=dim, keepdim=True) <= scale * 1e-10, torch.zeros_like(value), value)


def stable_atan2(b, a):
    scale = torch.sqrt(a.square() + b.square())
    b = torch.where(b.abs() <= scale * 1e-10, torch.zeros_like(b), b)
    return torch.atan2(b, a)


def stable_softmax(src, index, ptr=None, num_nodes=None, dim=0):
    maximum = deterministic_scatter(src.detach(), index, dim_size=num_nodes, reduce="max")
    value = (src - maximum[index]).exp()
    denom = deterministic_scatter(value, index, dim_size=num_nodes, reduce="sum")
    return value / (denom[index] + 1e-16)


def stable_aggregate(self, inputs, index, ptr=None, dim_size=None):
    return deterministic_scatter(inputs, index, dim_size=dim_size, reduce="sum")


stable_aggregate.__name__ = "aggregate"


def stable_graph_norm(self, x, batch=None, batch_size=None):
    size = int(batch.max()) + 1 if batch_size is None else batch_size
    mean = deterministic_scatter(x, batch, dim_size=size, reduce="mean")
    out = x - mean[batch] * self.mean_scale
    var = deterministic_scatter(out.square(), batch, dim_size=size, reduce="mean")
    return self.weight * out / torch.sqrt(var[batch] + self.eps) + self.bias


def stable_squared_distances(pos, edge_index):
    delta = pos.double()[edge_index[0]] - pos.double()[edge_index[1]]
    return delta.square().sum(-1, keepdim=True).float()
