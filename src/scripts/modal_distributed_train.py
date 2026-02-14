import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME = "jaide-v40-training"
GPU_SPEC = "B200:8"
DATA_VOLUME_NAME = "jaide-training-data"
CHECKPOINT_VOLUME_NAME = "jaide-checkpoints"

DATA_MOUNT_PATH = Path("/data")
CHECKPOINT_MOUNT_PATH = Path("/checkpoints")
PROJECT_MOUNT_PATH = Path("/jaide")

DATASET_DIR = DATA_MOUNT_PATH / "tower9b"
DATASET_FILE = DATASET_DIR / "hun_Latn_full.jsonl"

MODELS_DIR = PROJECT_MOUNT_PATH / "models"
BINARY_PATH = PROJECT_MOUNT_PATH / "main"

CPU_REQUEST = 64.0
CPU_LIMIT = 80.0
MEMORY_REQUEST_MB = 262144
MEMORY_LIMIT_MB = 262144
EPHEMERAL_DISK_MB = 3145728
TIMEOUT_SECONDS = 86400

LOCAL_PROJECT_DIR = (Path(__file__).resolve().parent / "../..").resolve()

IGNORE_PATTERNS = [
    "node_modules",
    ".git",
    "zig-cache",
    ".pythonlibs",
    ".cache",
    ".upm",
    "__pycache__",
    ".local",
    ".replit",
    "*.bin",
]

app = modal.App(APP_NAME)

jaide_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-change-held-packages git curl xz-utils build-essential wget ca-certificates",
        "rm -rf /var/lib/apt/lists/*",
    )
    .pip_install("pyarrow", "requests", "datasets", "huggingface_hub", "zstandard")
    .run_commands(
        "mkdir -p /opt",
        "curl -sL https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz | tar -xJ -C /opt",
        "ln -sf /opt/zig-linux-x86_64-0.13.0/zig /usr/local/bin/zig",
        "zig version",
    )
    .env(
        {
            "PATH": "/opt/zig-linux-x86_64-0.13.0:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HOME": "/data/hf_home",
            "HF_DATASETS_CACHE": "/data/hf_datasets_cache",
            "TRANSFORMERS_CACHE": "/data/hf_transformers_cache",
        }
    )
    .add_local_dir(
        str(LOCAL_PROJECT_DIR),
        remote_path=str(PROJECT_MOUNT_PATH),
        ignore=IGNORE_PATTERNS,
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)

def _run_checked(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _dataset_exists() -> bool:
    try:
        return DATASET_FILE.is_file() and DATASET_FILE.stat().st_size > 0
    except OSError:
        return False

def download_hunsum1_to_jsonl(volume: modal.Volume) -> Tuple[str, int, int]:
    from datasets import load_dataset

    _ensure_dir(DATASET_DIR)

    if _dataset_exists():
        size = int(DATASET_FILE.stat().st_size)
        line_count = 0
        with open(DATASET_FILE, "r", encoding="utf-8", errors="replace") as f_in:
            for _ in f_in:
                line_count += 1
        return str(DATASET_FILE), size, line_count

    ds = load_dataset("SZTAKI-HLT/HunSum-1", split="train")

    line_count = 0
    with open(DATASET_FILE, "w", encoding="utf-8") as f_out:
        for i, row in enumerate(ds):
            text = ""
            for key in ("text", "article", "lead", "content", "sentence"):
                val = row.get(key) if isinstance(row, dict) else None
                if isinstance(val, str) and val.strip():
                    text = val.strip()
                    break
            if not text and isinstance(row, dict):
                for _, val in row.items():
                    if isinstance(val, str) and len(val) > 50:
                        text = val.strip()
                        break
            if text and len(text) > 20:
                f_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                line_count += 1

    size = int(DATASET_FILE.stat().st_size)
    volume.commit()
    return str(DATASET_FILE), size, line_count

def _patch_zig_sources_for_0_13(project_dir: str) -> None:
    cmd = [
        "bash",
        "-lc",
        "find src -name '*.zig' -print0 | xargs -0 -I {} sed -i "
        "'s/@fabs/@abs/g; s/\\.Little/.little/g; s/\\.AcqRel/.acq_rel/g; s/\\.SeqCst/.seq_cst/g' {}",
    ]
    rc, out, err = _run_checked(cmd, cwd=project_dir)
    if rc != 0:
        raise RuntimeError(f"Zig patch failed: {err[:4000]}")

def _build_zig(project_dir: str) -> None:
    rc, out, err = _run_checked(["zig", "build-exe", "src/main.zig", "-O", "ReleaseFast"], cwd=project_dir)
    if rc != 0:
        raise RuntimeError(f"Build failed: {err[:8000]}")
    if not BINARY_PATH.is_file():
        raise FileNotFoundError(f"Built binary not found at {BINARY_PATH}")

def _detect_gpus() -> Tuple[int, str]:
    p = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
    lines = [l for l in (p.stdout or "").splitlines() if l.strip()]
    return len(lines), p.stdout or ""

def _extract_loss(stdout: str) -> Optional[float]:
    if not stdout:
        return None
    for line in stdout.splitlines():
        if "loss" in line.lower():
            m = re.search(r"[Ll]oss[:\s]+([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
    return None

@app.function(
    image=jaide_image,
    gpu=GPU_SPEC,
    cpu=(CPU_REQUEST, CPU_LIMIT),
    memory=(MEMORY_REQUEST_MB, MEMORY_LIMIT_MB),
    ephemeral_disk=EPHEMERAL_DISK_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={
        str(DATA_MOUNT_PATH): data_volume,
        str(CHECKPOINT_MOUNT_PATH): checkpoint_volume,
    },
)
def train_jaide(
    epochs: int = 20,
    batch_size: int = 256,
    per_epoch_timeout_seconds: int = 3600,
) -> Dict[str, Any]:
    data_volume.reload()
    checkpoint_volume.reload()

    gpu_count, gpu_list = _detect_gpus()

    dataset_path, dataset_size, sample_count = download_hunsum1_to_jsonl(data_volume)

    os.chdir(str(PROJECT_MOUNT_PATH))
    _ensure_dir(MODELS_DIR)
    _ensure_dir(CHECKPOINT_MOUNT_PATH)

    _patch_zig_sources_for_0_13(str(PROJECT_MOUNT_PATH))
    _build_zig(str(PROJECT_MOUNT_PATH))

    num_epochs = int(epochs)
    bs = int(batch_size)

    total_start = time.time()
    training_results: List[Dict[str, Any]] = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(8))

        cmd = [
            str(BINARY_PATH),
            "--mode",
            "train",
            "--epochs",
            "1",
            "--batch-size",
            str(bs),
            "--dataset-path",
            str(dataset_path),
            "--samples",
            str(sample_count),
            "--output-dir",
            str(MODELS_DIR),
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=int(per_epoch_timeout_seconds),
        )

        epoch_time = float(time.time() - epoch_start)
        loss = _extract_loss(proc.stdout)
        if loss is None:
            loss = 0.0

        checkpoint_dir = CHECKPOINT_MOUNT_PATH / f"epoch_{epoch:03d}"
        _ensure_dir(checkpoint_dir)

        with open(checkpoint_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "duration_seconds": epoch_time,
                    "batch_size": bs,
                    "dataset_samples": sample_count,
                    "dataset_size_mb": float(dataset_size) / 1e6,
                    "gpu_config": f"{gpu_count}x B200",
                    "return_code": int(proc.returncode),
                    "stdout_tail": (proc.stdout or "")[-8000:],
                    "stderr_tail": (proc.stderr or "")[-8000:],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        for model_file in MODELS_DIR.glob("*.bin"):
            shutil.copy2(model_file, checkpoint_dir / model_file.name)

        checkpoint_volume.commit()

        ckpt_files = list(checkpoint_dir.glob("*.bin"))
        ckpt_size = int(sum(f.stat().st_size for f in ckpt_files)) if ckpt_files else 0

        training_results.append(
            {
                "epoch": epoch,
                "loss": loss,
                "time_seconds": epoch_time,
                "checkpoint_mb": float(ckpt_size) / 1e6,
                "files": len(ckpt_files),
                "return_code": int(proc.returncode),
            }
        )

        if proc.returncode != 0:
            break

    total_time = float(time.time() - total_start)

    latest_dir = CHECKPOINT_MOUNT_PATH / "latest"
    if latest_dir.is_symlink():
        latest_dir.unlink()
    elif latest_dir.exists():
        shutil.rmtree(latest_dir)

    final_epoch = training_results[-1]["epoch"] if training_results else 0
    if final_epoch > 0:
        shutil.copytree(CHECKPOINT_MOUNT_PATH / f"epoch_{final_epoch:03d}", latest_dir)

    final_loss = float(training_results[-1]["loss"]) if training_results else 0.0
    status = "completed" if training_results and int(training_results[-1]["return_code"]) == 0 and len(training_results) == num_epochs else "failed"

    with open(CHECKPOINT_MOUNT_PATH / "training_complete.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": status,
                "total_epochs": num_epochs,
                "completed_epochs": len(training_results),
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60.0,
                "final_loss": final_loss,
                "dataset_path": str(dataset_path),
                "dataset_size_mb": float(dataset_size) / 1e6,
                "sample_count": sample_count,
                "gpu_config": f"{gpu_count}x NVIDIA B200",
                "epochs": training_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    checkpoint_volume.commit()

    return {
        "status": status,
        "epochs": num_epochs,
        "completed_epochs": len(training_results),
        "final_loss": final_loss,
        "total_time_minutes": total_time / 60.0,
        "dataset_size_mb": float(dataset_size) / 1e6,
        "sample_count": sample_count,
        "gpu_config": f"{gpu_count}x NVIDIA B200",
        "checkpoints": len(training_results),
    }

@app.local_entrypoint()
def main(
    epochs: int = 20,
    batch_size: int = 256,
    per_epoch_timeout_seconds: int = 3600,
) -> None:
    result = train_jaide.remote(
        epochs=epochs,
        batch_size=batch_size,
        per_epoch_timeout_seconds=per_epoch_timeout_seconds,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))