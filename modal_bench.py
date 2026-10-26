"""
modal_bench.py - run bench_matmul.py on a rented Modal GPU and bring the results home.

Modal has no RTX 4090. The default is the L40S: the same Ada generation (sm_89),
with 91.6 FP32 TFLOPS peak, 864 GB/s and a 96 MB L2. Pick another GPU with the
GPU environment variable (e.g. GPU=H100); bench_matmul.py reads the GPU it lands on.

One-time setup on your machine:
    pip3 install modal
    python3 -m modal setup

Usage (from this folder):
    modal run modal_bench.py --bench-args="-k naive --quick"
    modal run modal_bench.py --bench-args="-k naive"
    modal run modal_bench.py --bench-args="-k naive --verbose-build"   # see nvcc errors
    GPU=H100 modal run modal_bench.py --bench-args="-k naive"

Each run:
  * uploads this folder (your .cu files, headers, scripts and existing results/)
  * builds and benchmarks on the GPU; compiled extensions are cached in a Modal
    Volume so an unchanged kernel isn't recompiled next time
  * runs plot_bench.py, then writes results/ and plots/ back into this folder
"""
import os
import shlex
from pathlib import Path

import modal

LOCAL_DIR = Path(__file__).resolve().parent
SRC = "/root/src"      # read-only upload of this folder
WORK = "/root/work"    # writable copy the benchmark runs in
CACHE = "/cache"       # Modal Volume: build cache that survives between runs
GPU = os.environ.get("GPU", "L40S")

image = (
    # devel image: the benchmark JIT-compiles your kernels, so it needs nvcc
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("ninja", "numpy", "pandas", "matplotlib")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL_DIR, remote_path=SRC,
                   ignore=["build", "plots", "**/__pycache__", "**/.DS_Store"])
)

app = modal.App("cuda-gemm-bench", image=image)
build_cache = modal.Volume.from_name("cuda-gemm-build-cache", create_if_missing=True)


@app.function(gpu=GPU, timeout=60 * 60, volumes={CACHE: build_cache})
def bench(bench_args: list[str], plot: bool) -> dict:
    import shutil
    import subprocess
    import sys

    work = Path(WORK)
    shutil.copytree(SRC, work, dirs_exist_ok=True)

    # Point build/ at the volume so compiled kernels persist across runs.
    cache_build = Path(CACHE) / "build"
    cache_build.mkdir(parents=True, exist_ok=True)
    for lock in cache_build.rglob("lock"):  # left behind if a previous build was killed
        lock.unlink()
    if (work / "build").exists():
        shutil.rmtree(work / "build")
    (work / "build").symlink_to(cache_build)

    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,memory.total",
                    "--format=csv"], check=False)
    code = subprocess.run([sys.executable, "bench_matmul.py", *bench_args], cwd=work).returncode
    build_cache.commit()

    if plot and any((work / "results").glob("*.csv")):
        subprocess.run([sys.executable, "plot_bench.py"], cwd=work, check=False)

    files = {}
    for sub in ("results", "plots"):
        for path in (work / sub).rglob("*"):
            if path.is_file():
                files[str(path.relative_to(work))] = path.read_bytes()
    return {"returncode": code, "files": files}


@app.local_entrypoint()
def main(bench_args: str = "-k naive --quick", plot: bool = True):
    print(f"[modal] GPU={GPU} | bench_matmul.py {bench_args}")
    out = bench.remote(shlex.split(bench_args), plot)

    for rel, data in out["files"].items():
        dest = LOCAL_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    if out["files"]:
        print(f"[modal] wrote {len(out['files'])} files into results/ and plots/")
    if out["returncode"] != 0:
        raise SystemExit(f"[modal] bench_matmul.py exited with {out['returncode']}")
