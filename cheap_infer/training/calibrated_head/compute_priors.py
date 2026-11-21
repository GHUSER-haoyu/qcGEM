"""Stream all training records to compute exact element and pair-stratum empirical-Bayes priors."""
import concurrent.futures
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from head_model import HEAD_DIR, NUM_Z, PRIORS_PATH, ROOT, atomic_json, atomic_torch_save

NUM_PAIRS = NUM_Z * NUM_Z * 3
ELEMENTS = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def _process_chunk(args):
    chunk_ids, check_bond = args
    from common import Reader

    reader = Reader("cheap")
    ncnt = np.zeros((NUM_Z, 18), dtype=np.float64)
    nsum = np.zeros((NUM_Z, 18), dtype=np.float64)
    nsumsq = np.zeros((NUM_Z, 18), dtype=np.float64)

    ecnt = np.zeros((NUM_PAIRS, 19), dtype=np.float64)
    esum = np.zeros((NUM_PAIRS, 19), dtype=np.float64)
    esumsq = np.zeros((NUM_PAIRS, 19), dtype=np.float64)

    checked = 0
    mismatches = 0

    for idx in chunk_ids:
        r = reader.get(idx)
        z = np.clip(np.asarray(r["top"]["z"], np.int64), 0, NUM_Z - 1)
        n = len(z)
        yn = np.asarray(r["node"][:, :18], np.float64).copy()
        yn[yn[:, 0] == 5.0, 0] = np.nan
        valid_n = np.isfinite(yn)
        yn_clean = np.where(valid_n, yn, 0.0)

        for uz in np.unique(z):
            m = z == uz
            ncnt[uz] += valid_n[m].sum(0)
            nsum[uz] += yn_clean[m].sum(0)
            nsumsq[uz] += (yn_clean[m] ** 2).sum(0)

        ye = np.asarray(r["edge"][:, :19], np.float64)
        valid_e = np.isfinite(ye)
        ye_clean = np.where(valid_e, ye, 0.0)
        cov = np.asarray(r["edge"][:, 31:53]).sum(-1) > 0
        diag = np.eye(n, dtype=bool).reshape(-1)
        if check_bond and checked < 8:
            e1_bond = np.asarray(r["views"][1]["e1"][:, 12:34]).sum(-1) > 0.5
            if not np.array_equal(cov, e1_bond):
                mismatches += 1
            checked += 1

        za = np.repeat(z, n)
        zb = np.tile(z, n)
        ptype = np.where(diag, 2, np.where(cov, 1, 0))
        pidx = (np.minimum(za, zb) * NUM_Z + np.maximum(za, zb)) * 3 + ptype

        # Aggregate within molecule first so np.add.at touches only unique pair strata
        u, inv = np.unique(pidx, return_inverse=True)
        for k, up in enumerate(u):
            mk = inv == k
            ecnt[up] += valid_e[mk].sum(0)
            esum[up] += ye_clean[mk].sum(0)
            esumsq[up] += (ye_clean[mk] ** 2).sum(0)

    return ncnt, nsum, nsumsq, ecnt, esum, esumsq, checked, mismatches


def main():
    t0 = time.time()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    train_ids = sorted(r["i"] for r in manifest if r["split"] == "train")
    n_workers = 8
    chunks = [train_ids[k::n_workers] for k in range(n_workers)]

    ncnt = np.zeros((NUM_Z, 18), dtype=np.float64)
    nsum = np.zeros((NUM_Z, 18), dtype=np.float64)
    nsumsq = np.zeros((NUM_Z, 18), dtype=np.float64)
    ecnt = np.zeros((NUM_PAIRS, 19), dtype=np.float64)
    esum = np.zeros((NUM_PAIRS, 19), dtype=np.float64)
    esumsq = np.zeros((NUM_PAIRS, 19), dtype=np.float64)
    checked_total = 0
    mismatches_total = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork")) as pool:
        for res in pool.map(_process_chunk, [(c, idx == 0) for idx, c in enumerate(chunks)]):
            a, b, c, d, e, f, chk, mm = res
            ncnt += a
            nsum += b
            nsumsq += c
            ecnt += d
            esum += e
            esumsq += f
            checked_total += chk
            mismatches_total += mm

    assert checked_total == 8 and mismatches_total == 0, f"Bond mask check failed: checked={checked_total}, mismatches={mismatches_total}"

    # Node global + empirical Bayes element prior (m=16 smoothing)
    g_ncnt = ncnt.sum(0)
    g_nmean = nsum.sum(0) / np.maximum(g_ncnt, 1.0)
    node_prior_mean = (nsum + 16.0 * g_nmean[None, :]) / (ncnt + 16.0)

    # Exact within-element residual variance
    z_ok = ncnt > 0
    z_mean_exact = np.divide(nsum, ncnt, out=np.zeros_like(nsum), where=z_ok)
    sse_n = np.where(z_ok, np.maximum(nsumsq - ncnt * (z_mean_exact ** 2), 0.0), 0.0).sum(0)
    node_within_std = np.sqrt(np.maximum(sse_n / np.maximum(g_ncnt, 1.0), 0.0))

    # Pair-stratum fallback to pair_type mean (m=32 smoothing)
    ecnt_3d = ecnt.reshape(NUM_Z * NUM_Z, 3, 19)
    esum_3d = esum.reshape(NUM_Z * NUM_Z, 3, 19)
    esumsq_3d = esumsq.reshape(NUM_Z * NUM_Z, 3, 19)
    pt_cnt = ecnt_3d.sum(0)  # [3, 19]
    pt_mean = np.divide(esum_3d.sum(0), np.maximum(pt_cnt, 1.0))  # [3, 19]
    pair_prior_3d = (esum_3d + 32.0 * pt_mean[None, :, :]) / (ecnt_3d + 32.0)
    pair_prior_mean = pair_prior_3d.reshape(NUM_PAIRS, 19)

    p_ok = ecnt > 0
    p_mean_exact = np.divide(esum, ecnt, out=np.zeros_like(esum), where=p_ok)
    sse_e = np.where(p_ok, np.maximum(esumsq - ecnt * (p_mean_exact ** 2), 0.0), 0.0).sum(0)
    edge_within_std = np.sqrt(np.maximum(sse_e / np.maximum(ecnt.sum(0), 1.0), 0.0))

    # Inverse-sqrt frequency element weights (normalized so mean atom weight ~ 1.0, capped [0.5, 6.0])
    atom_cnt_z = ncnt[:, 14]  # NPA_charge valid across all elements
    tot_atoms = np.maximum(atom_cnt_z.sum(), 1.0)
    freq_z = np.where(atom_cnt_z > 0, atom_cnt_z / tot_atoms, 1.0)
    raw_zw = np.where(atom_cnt_z > 0, 1.0 / np.sqrt(np.maximum(freq_z, 1e-6)), 1.0)
    avg_zw = np.sum(raw_zw * atom_cnt_z) / tot_atoms
    z_weight = np.clip(raw_zw / np.maximum(avg_zw, 1e-6), 0.5, 6.0)

    # Pair-type balance weights (bonded edges are sparse relative to N^2 nonbonded; emphasize bonded/self slightly)
    pt_totals = pt_cnt[:, 15]  # Mayer valid across all edges
    tot_edges = np.maximum(pt_totals.sum(), 1.0)
    raw_ptw = 1.0 / np.sqrt(np.maximum(pt_totals / tot_edges, 1e-4))
    avg_ptw = np.sum(raw_ptw * pt_totals) / tot_edges
    ptype_weight = np.clip(raw_ptw / np.maximum(avg_ptw, 1e-6), 0.5, 3.5)

    priors = {
        "node_prior_mean": torch.tensor(node_prior_mean, dtype=torch.float32),
        "node_within_std": torch.tensor(node_within_std, dtype=torch.float32),
        "pair_prior_mean": torch.tensor(pair_prior_mean, dtype=torch.float32),
        "edge_within_std": torch.tensor(edge_within_std, dtype=torch.float32),
        "z_weight": torch.tensor(z_weight, dtype=torch.float32),
        "ptype_weight": torch.tensor(ptype_weight, dtype=torch.float32),
        "node_counts": torch.tensor(ncnt, dtype=torch.float64),
        "ptype_counts": torch.tensor(pt_cnt, dtype=torch.float64),
    }
    atomic_torch_save(priors, PRIORS_PATH)

    from common import EDGE_NAMES, NODE_NAMES
    summary = {
        "train_molecules": len(train_ids),
        "elapsed_seconds": time.time() - t0,
        "bond_mask_verification": {"checked_records": checked_total, "mismatches": mismatches_total},
        "element_counts_and_weights": {
            f"Z{z}_{ELEMENTS.get(z, str(z))}": {"atoms": int(atom_cnt_z[z]), "loss_weight": float(z_weight[z]), "npa_charge_mean": float(node_prior_mean[z, 14])}
            for z in sorted(ELEMENTS)
        },
        "pair_type_counts_and_weights": {
            name: {"edges": int(pt_totals[k]), "loss_weight": float(ptype_weight[k])}
            for k, name in enumerate(["nonbonded", "bonded", "self"])
        },
        "node_within_std": {NODE_NAMES[k]: float(node_within_std[k]) for k in range(18)},
        "edge_within_std": {EDGE_NAMES[k]: float(edge_within_std[k]) for k in range(19)},
    }
    atomic_json(HEAD_DIR / "PRIOR_STATS_SUMMARY.json", summary)
    print(f"Computed priors on {len(train_ids)} train molecules in {summary['elapsed_seconds']:.2f}s (bond mismatches={mismatches_total})")


if __name__ == "__main__":
    main()
