#!/usr/bin/env python3
import os
import re
import csv
import numpy as np
from scipy import linalg

# =========================
# Defaults (edit here)
# =========================
OUT_ROOT = "./clip_cls_embeddings"   # contains subfolders: FFHQ, DigiFace, ...
REF_NAME = "FFHQ"

# KID settings
KID_SUBSETS = 100
KID_SUBSET_SIZE = 2000
SEED = 0

# If you want to limit RAM/time, cap how many embeddings to load per dataset for KID.
# None = load all available.
KID_MAX_SAMPLES = None  # e.g., 50000

# Output CSV
SAVE_CSV = "./fid_kid_results_ffhq.csv"

# =========================
# Helpers
# =========================
BUCKET_RE = re.compile(r"bucket_(\d+)_emb\.npy$")


def list_bucket_files(folder):
    files = []
    for fn in os.listdir(folder):
        m = BUCKET_RE.match(fn)
        if m:
            files.append((int(m.group(1)), os.path.join(folder, fn)))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def iter_embeddings(folder, max_items=None):
    """Yield float64 chunks (N,D) from bucket files."""
    files = list_bucket_files(folder)
    seen = 0
    for fp in files:
        x = np.load(fp, mmap_mode="r")  # (N,D)
        if x.ndim != 2:
            raise ValueError("Bad embedding array shape in %s: %s" % (fp, str(x.shape)))

        if max_items is None:
            yield np.asarray(x, dtype=np.float64)
            seen += x.shape[0]
        else:
            remaining = max_items - seen
            if remaining <= 0:
                break
            if x.shape[0] <= remaining:
                yield np.asarray(x, dtype=np.float64)
                seen += x.shape[0]
            else:
                yield np.asarray(x[:remaining], dtype=np.float64)
                seen += remaining
                break


def compute_mean_and_cov(folder):
    """
    Streaming mean/cov:
      mu = sum(x)/N
      cov = (X^T X)/N - mu mu^T
    """
    sum_x = None
    sum_xx = None
    n = 0
    d = None

    for chunk in iter_embeddings(folder, max_items=None):
        if d is None:
            d = chunk.shape[1]
            sum_x = np.zeros((d,), dtype=np.float64)
            sum_xx = np.zeros((d, d), dtype=np.float64)
        elif chunk.shape[1] != d:
            raise ValueError("Dim mismatch in %s: expected %d got %d" % (folder, d, chunk.shape[1]))

        sum_x += chunk.sum(axis=0)
        sum_xx += chunk.T @ chunk
        n += chunk.shape[0]

    if n == 0:
        raise ValueError("No embeddings found in %s" % folder)

    mu = sum_x / n
    cov = (sum_xx / n) - np.outer(mu, mu)
    cov = (cov + cov.T) / 2.0
    return mu, cov, n, d


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    cov1 = np.asarray(cov1, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)

    diff = mu1 - mu2
    diff_sq = diff @ diff

    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)

    if not np.isfinite(covmean).all():
        d = cov1.shape[0]
        covmean, _ = linalg.sqrtm((cov1 + np.eye(d) * eps) @ (cov2 + np.eye(d) * eps), disp=False)

    if np.iscomplexobj(covmean):
        if np.max(np.abs(np.imag(covmean))) > 1e-3:
            raise ValueError("sqrtm returned large imaginary component; covariance may be ill-conditioned.")
        covmean = np.real(covmean)

    return float(diff_sq + np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(covmean))


def load_all_embeddings(folder, max_samples=None):
    """Load embeddings into RAM as float64 (N,D)."""
    chunks = []
    total = 0
    d = None
    for chunk in iter_embeddings(folder, max_items=max_samples):
        if d is None:
            d = chunk.shape[1]
        elif chunk.shape[1] != d:
            raise ValueError("Dim mismatch in %s: expected %d got %d" % (folder, d, chunk.shape[1]))
        chunks.append(chunk)
        total += chunk.shape[0]
    if total == 0:
        raise ValueError("No embeddings loaded from %s" % folder)
    return np.concatenate(chunks, axis=0)


def kid_mmd2_unbiased_poly(X, Y):
    """
    Unbiased MMD^2 with polynomial kernel:
      k(x,y) = (x·y / d + 1)^3
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    m, d = X.shape
    if Y.shape != (m, d):
        raise ValueError("KID subset shape mismatch: %s vs %s" % (X.shape, Y.shape))

    Kxx = (X @ X.T) / d
    Kyy = (Y @ Y.T) / d
    Kxy = (X @ Y.T) / d

    Kxx = (Kxx + 1.0) ** 3
    Kyy = (Kyy + 1.0) ** 3
    Kxy = (Kxy + 1.0) ** 3

    # unbiased: drop diagonals for xx and yy
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = Kxy.mean()

    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def compute_kid(X_ref, X_gen, subsets=50, subset_size=1000, seed=0):
    rng = np.random.default_rng(seed)

    n_ref = X_ref.shape[0]
    n_gen = X_gen.shape[0]
    d_ref = X_ref.shape[1]
    d_gen = X_gen.shape[1]

    if d_ref != d_gen:
        raise ValueError("Dim mismatch: ref %d vs gen %d" % (d_ref, d_gen))
    if n_ref < subset_size or n_gen < subset_size:
        raise ValueError("Not enough samples for KID subset_size=%d: ref=%d gen=%d" %
                         (subset_size, n_ref, n_gen))

    vals = []
    for _ in range(subsets):
        idx_r = rng.choice(n_ref, size=subset_size, replace=False)
        idx_g = rng.choice(n_gen, size=subset_size, replace=False)
        vals.append(kid_mmd2_unbiased_poly(X_ref[idx_r], X_gen[idx_g]))

    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1))


def find_datasets(out_root):
    names = []
    for name in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, name)
        if os.path.isdir(p) and len(list_bucket_files(p)) > 0:
            names.append(name)
    return names


# =========================
# Main
# =========================
def main():
    if not os.path.isdir(OUT_ROOT):
        raise FileNotFoundError("OUT_ROOT not found: %s" % OUT_ROOT)

    ref_dir = os.path.join(OUT_ROOT, REF_NAME)
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError("Reference dataset folder not found: %s" % ref_dir)

    datasets = find_datasets(OUT_ROOT)
    if REF_NAME not in datasets:
        raise RuntimeError("Reference folder exists but has no bucket_XXXXX_emb.npy files: %s" % ref_dir)

    print("[INFO] Found datasets:", datasets)
    print("[INFO] Reference:", REF_NAME)

    # --- FID reference stats (streaming) ---
    print("\n[INFO] Computing reference mean/cov for FID...")
    mu_r, cov_r, n_r, d_r = compute_mean_and_cov(ref_dir)
    print("[INFO] Ref stats: N=%d, D=%d" % (n_r, d_r))

    # --- KID reference embeddings (RAM) ---
    print("\n[INFO] Loading reference embeddings for KID (may take time)...")
    X_ref = load_all_embeddings(ref_dir, max_samples=KID_MAX_SAMPLES)
    print("[INFO] Ref loaded for KID:", X_ref.shape)

    results = []

    for name in datasets:
        if name == REF_NAME:
            continue

        gen_dir = os.path.join(OUT_ROOT, name)
        print("\n===================================")
        print("Comparing:", name, "vs", REF_NAME)

        # FID
        mu_g, cov_g, n_g, d_g = compute_mean_and_cov(gen_dir)
        if d_g != d_r:
            raise ValueError("Dim mismatch: ref D=%d vs %s D=%d" % (d_r, name, d_g))

        fid = frechet_distance(mu_r, cov_r, mu_g, cov_g)
        print("FID: %.6f (ref N=%d, gen N=%d)" % (fid, n_r, n_g))

        # KID
        X_gen = load_all_embeddings(gen_dir, max_samples=KID_MAX_SAMPLES)
        print("[INFO] Gen loaded for KID:", X_gen.shape)

        kid_mean, kid_std = compute_kid(
            X_ref, X_gen, subsets=KID_SUBSETS, subset_size=KID_SUBSET_SIZE, seed=SEED
        )
        print("KID: %.8f ± %.8f (subsets=%d, subset_size=%d)" %
              (kid_mean, kid_std, KID_SUBSETS, KID_SUBSET_SIZE))

        results.append({
            "ref": REF_NAME,
            "gen": name,
            "fid": fid,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
            "ref_n": n_r,
            "gen_n": n_g,
            "dim": d_r,
            "kid_loaded_ref_n": int(X_ref.shape[0]),
            "kid_loaded_gen_n": int(X_gen.shape[0]),
        })

    # sort by FID
    results.sort(key=lambda r: r["fid"])

    print("\n\n===== SUMMARY (sorted by FID) =====")
    for r in results:
        print("%15s  FID=%10.6f  KID=% .8f±%.8f" %
              (r["gen"], r["fid"], r["kid_mean"], r["kid_std"]))

    # save csv
    if results:
        with open(SAVE_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print("\n[OK] wrote:", SAVE_CSV)
    else:
        print("\n[WARN] No datasets to compare (only reference found).")


if __name__ == "__main__":
    main()
