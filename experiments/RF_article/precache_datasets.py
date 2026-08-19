"""Pre-cache every dataset the RF-article sweeps need, ONCE, on a networked login node.

Concurrent SLURM array tasks would otherwise race the same download and corrupt the
cache. Run this from PROJECT_ROOT before submitting any array; it is idempotent, so
re-running it only fills in whatever is missing.

    python experiments/RF_article/precache_datasets.py
    python experiments/RF_article/precache_datasets.py --force-notmnist   # re-copy notMNIST

Torchvision sets land in data/torchvision (the same root ImageDataStreamer uses).
notMNIST comes from deeplake and is deep-copied to data/datasets/notmnist_dl so the
compute nodes never need network; point NOTMNIST_LOCAL elsewhere to override.
"""
import argparse
import os
import sys

TORCH_ROOT = os.path.join("data", "torchvision")
NOTMNIST_LOCAL = os.environ.get(
    "NOTMNIST_LOCAL", os.path.join("data", "datasets", "notmnist_dl")
)
NOTMNIST_HUB = "hub://activeloop/not-mnist-small"


def precache_torchvision():
    from torchvision import datasets

    os.makedirs(TORCH_ROOT, exist_ok=True)
    ok = True
    # (label, callable) — SVHN takes split=, the rest take train=
    jobs = [
        ("MNIST", lambda tr: datasets.MNIST(TORCH_ROOT, train=tr, download=True)),
        ("KMNIST", lambda tr: datasets.KMNIST(TORCH_ROOT, train=tr, download=True)),
        ("FashionMNIST", lambda tr: datasets.FashionMNIST(TORCH_ROOT, train=tr, download=True)),
        ("SVHN", lambda tr: datasets.SVHN(
            TORCH_ROOT, split=("train" if tr else "test"), download=True)),
    ]
    for name, ctor in jobs:
        for tr in (True, False):
            split = "train" if tr else "test"
            try:
                ds = ctor(tr)
                print(f"  [ok]   {name:<13s} {split:<5s} n={len(ds)}", flush=True)
            except Exception as exc:
                ok = False
                print(f"  [FAIL] {name:<13s} {split:<5s} {type(exc).__name__}: {exc}",
                      flush=True)
    return ok


def precache_notmnist(force=False):
    try:
        import deeplake
    except ImportError:
        print("  [FAIL] notMNIST needs deeplake -> pip install deeplake", flush=True)
        return False

    if os.path.isdir(NOTMNIST_LOCAL) and os.listdir(NOTMNIST_LOCAL) and not force:
        try:
            n = len(deeplake.load(NOTMNIST_LOCAL, read_only=True))
            print(f"  [skip] notMNIST already at {NOTMNIST_LOCAL} (n={n})", flush=True)
            return True
        except Exception as exc:
            print(f"  [warn] local copy unreadable ({exc}); re-copying", flush=True)

    os.makedirs(os.path.dirname(NOTMNIST_LOCAL) or ".", exist_ok=True)
    try:
        deeplake.deepcopy(NOTMNIST_HUB, NOTMNIST_LOCAL, overwrite=True)
        n = len(deeplake.load(NOTMNIST_LOCAL, read_only=True))
        print(f"  [ok]   notMNIST -> {NOTMNIST_LOCAL} (n={n})", flush=True)
        return True
    except Exception as exc:
        print(f"  [FAIL] notMNIST {type(exc).__name__}: {exc}", flush=True)
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-notmnist", action="store_true",
                    help="re-copy notMNIST even if a local copy exists")
    ap.add_argument("--skip-notmnist", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir("neurosnn"):
        sys.exit("Run this from PROJECT_ROOT (the dir containing neurosnn/).")

    print(f"Torchvision root : {os.path.abspath(TORCH_ROOT)}")
    print(f"notMNIST local   : {os.path.abspath(NOTMNIST_LOCAL)}\n")
    good = precache_torchvision()
    if not a.skip_notmnist:
        good &= precache_notmnist(force=a.force_notmnist)

    print("\nAll datasets cached." if good else "\nSOME DATASETS FAILED — see [FAIL] above.")
    sys.exit(0 if good else 1)
