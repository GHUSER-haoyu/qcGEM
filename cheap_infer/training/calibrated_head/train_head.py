"""DDP training of CalibratedQCHead on frozen refined Student representations."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from head_model import (
    BACKBONE_DIR,
    HEAD_DIR,
    PRIORS_PATH,
    ROOT,
    STATS_PATH,
    CalibratedQCHead,
    CheapAimRecords,
    atomic_json,
    atomic_torch_save,
    clean_node_targets,
    collate_cheap_aim,
    sha256_file,
)

if str(BACKBONE_DIR) not in sys.path:
    sys.path.insert(0, str(BACKBONE_DIR))
from checkpoint_state import checkpoint_warm  # noqa: E402
from common import EDGE_NAMES, NODE_NAMES  # noqa: E402
from data import BucketBatches, device_batch  # noqa: E402
from model import Student  # noqa: E402


def eval_validation(student, head_mod, device, rank, world, stats, micro=16):
    head_mod.eval()
    ds = CheapAimRecords("val", 1)
    dl = DataLoader(Subset(ds, list(range(rank, len(ds), world))), batch_size=micro, collate_fn=collate_cheap_aim, num_workers=2)

    ns = torch.tensor(stats["node_qc"]["std"], dtype=torch.float64, device=device)
    es = torch.tensor(stats["edge_qc"]["std"], dtype=torch.float64, device=device)

    # Accumulate sufficient statistics: [2 modes, 5 stats (n, sy, syy, ae, se), C]
    n_suff = torch.zeros((2, 5, 18), dtype=torch.float64, device=device)
    e_suff = torch.zeros((2, 5, 19), dtype=torch.float64, device=device)
    # Rare element NPA charge (idx 14) for F(9), Cl(17), Br(35)
    rare_z = [9, 17, 35]
    rare_suff = torch.zeros((2, len(rare_z), 5), dtype=torch.float64, device=device)

    def upd(tgt, y, p):
        v = torch.isfinite(y) & torch.isfinite(p)
        yy = torch.where(v, y.double(), torch.zeros_like(y, dtype=torch.float64))
        er = torch.where(v, (p - y).double(), torch.zeros_like(y, dtype=torch.float64))
        tgt[0] += v.sum(0)
        tgt[1] += yy.sum(0)
        tgt[2] += (yy * yy).sum(0)
        tgt[3] += er.abs().sum(0)
        tgt[4] += (er * er).sum(0)

    with torch.no_grad():
        for b in dl:
            b = device_batch(b, device)
            yn = clean_node_targets(b["raw_node"])
            ye = b["raw_edge"][:, :19].float()
            for mi, mode in enumerate(("gpu", "cpu")):
                out = student(b, mode=mode, decode=True)
                pn, pe = head_mod(out, b)
                upd(n_suff[mi], yn, pn)
                upd(e_suff[mi], ye, pe)
                for zi, zv in enumerate(rare_z):
                    m = b["z"] == zv
                    if m.any():
                        y1 = yn[m, 14]
                        p1 = pn[m, 14]
                        v1 = torch.isfinite(y1) & torch.isfinite(p1)
                        yy1 = torch.where(v1, y1.double(), torch.zeros_like(y1, dtype=torch.float64))
                        er1 = torch.where(v1, (p1 - y1).double(), torch.zeros_like(y1, dtype=torch.float64))
                        rare_suff[mi, zi, 0] += v1.sum()
                        rare_suff[mi, zi, 1] += yy1.sum()
                        rare_suff[mi, zi, 2] += (yy1 * yy1).sum()
                        rare_suff[mi, zi, 3] += er1.abs().sum()
                        rare_suff[mi, zi, 4] += (er1 * er1).sum()

    dist.all_reduce(n_suff)
    dist.all_reduce(e_suff)
    dist.all_reduce(rare_suff)

    def summarize(s, std_vec):
        n, sy, syy, ae, se = s
        sst = (syy - sy * sy / n.clamp_min(1.0)).clamp_min(1e-12)
        r2 = torch.where(n >= 2, 1.0 - se / sst, torch.full_like(n, float("nan")))
        mse_norm = se / (n.clamp_min(1.0) * std_vec.square())
        mae = ae / n.clamp_min(1.0)
        return r2.cpu().numpy(), mae.cpu().numpy(), mse_norm.cpu().numpy()

    report = {}
    scores = []
    for mi, mode in enumerate(("gpu", "cpu")):
        nr2, nmae, nmse = summarize(n_suff[mi], ns)
        er2, emae, emse = summarize(e_suff[mi], es)
        mean_mse = 0.5 * (float(np.nanmean(nmse)) + float(np.nanmean(emse)))
        scores.append(mean_mse)
        rare_rep = {}
        for zi, zv in enumerate(rare_z):
            n, sy, syy, ae, se = rare_suff[mi, zi].tolist()
            sst = max(syy - sy * sy / max(n, 1.0), 1e-12)
            rare_rep[f"Z{zv}"] = {"n": int(n), "r2": 1.0 - se / sst if n >= 2 else None, "mae": ae / max(n, 1.0)}
        report[mode] = {
            "norm_mse": mean_mse,
            "node_r2": {NODE_NAMES[k]: float(nr2[k]) for k in range(18)},
            "edge_r2": {EDGE_NAMES[k]: float(er2[k]) for k in range(19)},
            "rare_npa_charge": rare_rep,
        }
    report["score"] = float(np.mean(scores))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--microbatch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/best.pt"))
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    device = f"cuda:{rank}"

    stats = json.loads(STATS_PATH.read_text())
    priors = torch.load(PRIORS_PATH, map_location="cpu", weights_only=False)
    ckpt_path = os.path.abspath(args.checkpoint)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert not st.get("diagnostic", False)

    student = Student(stats, device=device, initialize_official=False)
    student.load_state_dict(st["model"])
    student.set_stage(checkpoint_warm(st))
    student.eval()
    student.requires_grad_(False)

    head = CalibratedQCHead(stats, priors).to(device)
    ddp_head = DDP(head, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)

    decay_params = [p for n, p in head.named_parameters() if p.ndim >= 2 and "bias" not in n and "emb" not in n]
    no_decay_params = [p for n, p in head.named_parameters() if not (p.ndim >= 2 and "bias" not in n and "emb" not in n)]
    opt = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 1e-4}, {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.lr,
    )

    sample_ds = CheapAimRecords("train", 1)
    steps_per_epoch = len(BucketBatches(sample_ds, rank, world, args.microbatch, epoch=1))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(200, max(10, total_steps // 20))

    best_score = float("inf")
    history = []
    t_start = time.time()
    global_step = 0
    best_ckpt = HEAD_DIR / "checkpoints/head_best.pt"

    if rank == 0:
        atomic_json(
            HEAD_DIR / "TRAINING_START.json",
            {
                "started": time.time(),
                "world": world,
                "epochs": args.epochs,
                "microbatch": args.microbatch,
                "steps_per_epoch": steps_per_epoch,
                "backbone_checkpoint": ckpt_path,
                "backbone_sha256": sha256_file(ckpt_path),
                "backbone_best_val": float(st["best"]),
                "backbone_epoch_next": int(st["epoch_next"]),
            },
        )

    for epoch in range(1, args.epochs + 1):
        ds = CheapAimRecords("train", epoch)
        sampler = BucketBatches(ds, rank, world, args.microbatch, epoch=epoch)
        dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_cheap_aim, num_workers=2, pin_memory=True)
        ddp_head.train()

        for b in dl:
            if global_step < warmup_steps:
                lr = args.lr * float(global_step + 1) / float(warmup_steps)
            else:
                prog = float(global_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * prog))
            for pg in opt.param_groups:
                pg["lr"] = lr

            b = device_batch(b, device)
            ng = b["num_graphs"]
            mode = "gpu" if global_step % 2 == 0 else "cpu"

            # Explicit EHT 10% dropout (and 15% AIM dropout when in gpu mode) matching Student robustness
            b["eht_global"] = b["eht_global"].clone()
            b["eht_global"][:, 0] *= (torch.rand(ng, device=device) >= 0.1).float()
            if mode == "gpu":
                b["aim_global"] = b["aim_global"].clone()
                b["aim_global"][:, 0] *= (torch.rand(ng, device=device) >= 0.15).float()

            with torch.no_grad():
                out = student(b, mode=mode, decode=True)

            opt.zero_grad(set_to_none=True)
            pred_n, pred_e = ddp_head(out, b)
            loss, parts = head.compute_loss(pred_n, pred_e, b)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(ddp_head.parameters(), 5.0))
            opt.step()

            global_step += 1
            if rank == 0 and (global_step % 100 == 0 or global_step == total_steps):
                rec = {
                    "time": time.time(),
                    "epoch": epoch,
                    "step": global_step,
                    "lr": lr,
                    "mode": mode,
                    "loss": float(loss.detach()),
                    "parts": parts,
                    "grad_norm": gn,
                    "elapsed": time.time() - t_start,
                }
                with (HEAD_DIR / "train.jsonl").open("a") as f:
                    f.write(json.dumps(rec) + "\n")

        val_report = eval_validation(student, ddp_head.module, device, rank, world, stats, micro=args.microbatch)
        if rank == 0:
            history.append({"epoch": epoch, "step": global_step, "val": val_report})
            improved = val_report["score"] < best_score
            if improved:
                best_score = val_report["score"]
                atomic_torch_save(
                    {
                        "head_state_dict": ddp_head.module.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "best_val_score": best_score,
                        "val_report": val_report,
                        "backbone_checkpoint": ckpt_path,
                        "backbone_sha256": sha256_file(ckpt_path),
                    },
                    best_ckpt,
                )
            print(f"[Head Epoch {epoch}] val_score={val_report['score']:.4f} best={best_score:.4f} gpu_bd_r2={val_report['gpu']['edge_r2']['BD_occupancy']:.4f} gpu_F_r2={val_report['gpu']['rare_npa_charge']['Z9']['r2']}")
        dist.barrier()

    if rank == 0:
        atomic_json(
            HEAD_DIR / "TRAINING_COMPLETE.json",
            {
                "completed": time.time(),
                "elapsed_seconds": time.time() - t_start,
                "best_val_score": best_score,
                "best_head_checkpoint": str(best_ckpt),
                "best_head_sha256": sha256_file(best_ckpt),
                "backbone_sha256": sha256_file(ckpt_path),
                "history": history,
            },
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
