# qcGEM-Student Weights (`v0.1.0`)

Binary checkpoints are hosted in **GitHub Releases** (`v0.1.0`) rather than raw git or Git LFS:
- `qcgem_student_v1.pt` is `165.9 MB`, exceeding GitHub's `100 MB` git blob limit.
- GitHub Free enforces a `1 GB/month` LFS bandwidth quota that breaks public clones and `pip install` once exhausted.
- Keeping weights in Releases keeps `git clone` and `pip install` lightweight and requires explicit local checkpoint selection at runtime.

## Assets

| File | Size | SHA256 | Contents |
| :--- | :--- | :--- | :--- |
| `qcgem_student_v1.pt` | `165,897,014` B (~158.2 MiB) | `b8f4680e4c6d06d70112fd1d477607173a53833cbf167fda474be90b60c70a59` | **Recommended single-file bundle**: FP32 refined Student backbone (no optimizer state), `CalibratedQCHead`, empirical-Bayes priors, feature/QC stats, and channel metadata. |
| `calibrated_head_v1.pt` | `12,306,811` B (~11.7 MiB) | `dd2584d9de8230357a6804bc07936d763c7c8befa392f475e77bde98ca4d580a` | Standalone `CalibratedQCHead` + priors + stats for pairing with raw `best.pt`. |

> Neither runtime package nor inference requires the upstream `qcGEM_ckpt_Index649.pt` Teacher checkpoint; `Student(..., initialize_official=False)` loads all layers directly from `qcgem_student_v1.pt`.

## Download & Verify

```bash
mkdir -p ./checkpoints
curl -L -o ./checkpoints/qcgem_student_v1.pt \
  https://github.com/GHUSER-haoyu/qcGEM/releases/download/v0.1.0/qcgem_student_v1.pt

# Or via GitHub CLI:
gh release download v0.1.0 -R GHUSER-haoyu/qcGEM -p qcgem_student_v1.pt -D ./checkpoints

# Verify SHA256:
(cd ./checkpoints && shasum -a 256 -c ../cheap_infer/weights/SHA256SUMS.txt --ignore-missing)
```

See [WEIGHTS_MANIFEST.json](WEIGHTS_MANIFEST.json) for machine-readable provenance hashes and channel reliability caveats (`starred_unreliable_channels`).
