import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import modal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("jaide-v40-training")

volume = modal.Volume.from_name("jaide-training-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential",
        "wget",
        "curl",
        "git",
        "ca-certificates",
        "xz-utils",
    )
    .run_commands(
        "curl -sSf https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz -o /tmp/zig.tar.xz",
        "tar -xf /tmp/zig.tar.xz -C /tmp",
        "mv /tmp/zig-linux-x86_64-0.11.0 /usr/local/zig",
        "ln -s /usr/local/zig/zig /usr/local/bin/zig",
        "zig version",
    )
    .add_local_dir("src", remote_path="/jaide_src", copy=True)
    .add_local_file(
        "arxiv_hungarian_dataset_2.jsonl",
        remote_path="/dataset/arxiv_hungarian_dataset_2.jsonl",
        copy=True,
    )
    .run_commands(
        "zig build-exe /jaide_src/main.zig -O ReleaseFast -fstrip -femit-bin=/root/main",
        "chmod +x /root/main",
    )
)

WORK_DIR = Path("/workspace")
SRC_MOUNT = Path("/jaide_src")
DATASET_PATH = Path("/dataset/arxiv_hungarian_dataset_2.jsonl")
MODELS_DIR = Path("/models")
BINARY_PATH = Path("/root/main")

GPU_TYPES = ["B200"]
GPU_COUNT = 8
DEFAULT_GPU_CONFIG = f"{GPU_COUNT}x {GPU_TYPES[0]}"

class TrainingParameters(TypedDict):
    epochs: int
    batch_size: int
    learning_rate: float
    dim: int
    layers: int
    sample_limit: int
    noise_level: float
    gradient_clip: float

class TrainingResult(TypedDict):
    status: str
    exit_code: int
    duration_seconds: float
    gpu_config: str
    model_path: Optional[str]
    stdout: str
    stderr: str
    timestamp: float
    parameters: Optional[TrainingParameters]

class ModelInfo(TypedDict):
    filename: str
    size_bytes: int
    size_mb: float
    modified: float
    path: str

class ListModelsResult(TypedDict):
    models: List[ModelInfo]
    training_logs: List[TrainingResult]

def _ensure_positive(name: str, value: Union[int, float]) -> float:
    val = float(value)
    if val <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return val

def _ensure_non_negative(name: str, value: Union[int, float]) -> float:
    val = float(value)
    if val < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return val

def _validate_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    if not isinstance(value, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value}")
    return value

@app.function(
    image=image,
    gpu=[f"{t}:{GPU_COUNT}" for t in GPU_TYPES],
    timeout=86400,
    volumes={str(MODELS_DIR): volume},
    cpu=64.0,
    memory=128 * 1024,
)
def train_jaide_rsf(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dim: int,
    layers: int,
    sample_limit: int,
    noise_level: float,
    gradient_clip: float,
) -> TrainingResult:
    start_wall = time.time()

    try:
        epochs = _validate_int("epochs", epochs)
        batch_size = _validate_int("batch_size", batch_size)
        dim = _validate_int("dim", dim)
        layers = _validate_int("layers", layers)
        sample_limit = _validate_int("sample_limit", sample_limit)

        learning_rate = _ensure_positive("learning_rate", learning_rate)
        noise_level = _ensure_non_negative("noise_level", noise_level)
        gradient_clip = _ensure_non_negative("gradient_clip", gradient_clip)

        WORK_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not BINARY_PATH.is_file():
            raise FileNotFoundError(f"Compiled binary not found at: {BINARY_PATH}")

        if not DATASET_PATH.is_file():
            raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

        unique_id = uuid.uuid4().hex
        model_output = MODELS_DIR / f"rsf_trained_{GPU_COUNT}x_{unique_id}.bin"

        train_args = [
            str(BINARY_PATH),
            "--mode",
            "train",
            "--dataset",
            str(DATASET_PATH),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--dim",
            str(dim),
            "--layers",
            str(layers),
            "--sample-limit",
            str(sample_limit),
            "--noise-level",
            str(noise_level),
            "--gradient-clip",
            str(gradient_clip),
            "--model-output",
            str(model_output),
        ]

        env = os.environ.copy()
        env.update({
            "JAIDE_GPU_COUNT": str(GPU_COUNT),
            "JAIDE_GPU_TYPE": GPU_TYPES[0],
        })

        train_stdout_path = WORK_DIR / "train_stdout.log"
        train_stderr_path = WORK_DIR / "train_stderr.log"

        train_start = time.time()
        with open(train_stdout_path, "w", encoding="utf-8") as tout, open(train_stderr_path, "w", encoding="utf-8") as terr:
            train_result = subprocess.run(
                train_args,
                stdout=tout,
                stderr=terr,
                env=env,
                cwd=WORK_DIR,
            )
        train_end = time.time()

        train_stdout_txt = train_stdout_path.read_text(encoding="utf-8")
        train_stderr_txt = train_stderr_path.read_text(encoding="utf-8")

        params = TrainingParameters(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dim=dim,
            layers=layers,
            sample_limit=sample_limit,
            noise_level=noise_level,
            gradient_clip=gradient_clip,
        )

        status = "completed" if train_result.returncode == 0 else "failed"
        model_path_str = str(model_output) if train_result.returncode == 0 and model_output.is_file() else None

        training_log = TrainingResult(
            status=status,
            exit_code=train_result.returncode,
            duration_seconds=train_end - train_start,
            gpu_config=DEFAULT_GPU_CONFIG,
            model_path=model_path_str,
            stdout=train_stdout_txt,
            stderr=train_stderr_txt,
            timestamp=train_end,
            parameters=params,
        )

        log_path = MODELS_DIR / f"training_log_{int(train_end)}_{unique_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

        volume.commit()

        return training_log

    except Exception as e:
        end_wall = time.time()
        logger.exception("Training failed with exception")
        return TrainingResult(
            status="exception",
            exit_code=1,
            duration_seconds=end_wall - start_wall,
            gpu_config=DEFAULT_GPU_CONFIG,
            model_path=None,
            stdout="",
            stderr=f"{type(e).__name__}: {e}",
            timestamp=end_wall,
            parameters=None,
        )

@app.function(image=image, volumes={str(MODELS_DIR): volume})
def list_models() -> ListModelsResult:
    volume.reload()
    models: List[ModelInfo] = []
    try:
        for f in MODELS_DIR.glob("rsf_trained_*.bin"):
            stat = f.stat()
            models.append(
                ModelInfo(
                    filename=f.name,
                    size_bytes=stat.st_size,
                    size_mb=stat.st_size / (1024 * 1024),
                    modified=stat.st_mtime,
                    path=str(f),
                )
            )
    except OSError as e:
        logger.error(f"Failed to list models directory: {e}")

    logs: List[TrainingResult] = []
    try:
        for f in MODELS_DIR.glob("training_log_*.json"):
            try:
                with open(f, "r", encoding="utf-8") as log_file:
                    log_data = json.load(log_file)
                if isinstance(log_data, dict):
                    logs.append(TrainingResult(**log_data))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"Skipping malformed log file {f.name}: {e}")
    except OSError as e:
        logger.error(f"Failed to scan for log files: {e}")

    def _ts(x: TrainingResult) -> float:
        t = x.get("timestamp")
        if t is None:
            return 0.0
        try:
            return float(t)
        except (TypeError, ValueError):
            return 0.0

    return ListModelsResult(
        models=sorted(models, key=lambda x: float(x.get("modified", 0.0)), reverse=True),
        training_logs=sorted(logs, key=_ts, reverse=True),
    )

@app.function(image=image, volumes={str(MODELS_DIR): volume})
def get_model_bytes(model_filename: str) -> bytes:
    volume.reload()
    safe_name = Path(model_filename).name
    if not safe_name.startswith("rsf_trained_") or not safe_name.endswith(".bin"):
        raise ValueError(f"Invalid model filename format: {model_filename}")

    model_path = MODELS_DIR / safe_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {safe_name}")

    return model_path.read_bytes()

@app.local_entrypoint()
def main(
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    dim: int = 128,
    layers: int = 4,
    sample_limit: int = 100,
    noise_level: float = 0.05,
    gradient_clip: float = 5.0,
):
    separator = "=" * 70
    print(separator)
    print("JAIDE v40 - Root-Level AGI Training on 8x GPUs")
    print(separator)
    print("Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Embedding Dimension: {dim}")
    print(f"  RSF Layers: {layers}")
    print(f"  Sample Limit: {sample_limit}")
    print(f"  Noise Level: {noise_level}")
    print(f"  Gradient Clip: {gradient_clip}")
    print(separator)

    result = train_jaide_rsf.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dim=dim,
        layers=layers,
        sample_limit=sample_limit,
        noise_level=noise_level,
        gradient_clip=gradient_clip,
    )

    print("\n" + separator)
    print("TRAINING RESULTS")
    print(separator)
    print(f"Status: {result.get('status')}")
    duration_seconds = result.get("duration_seconds")
    if duration_seconds is not None:
        ds = float(duration_seconds)
        print(f"Duration: {ds:.2f} seconds ({ds/60.0:.2f} minutes)")

    print(f"GPU Configuration: {result.get('gpu_config')}")

    model_path = result.get("model_path")
    if model_path:
        print(f"Model saved to: {model_path}")

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")

    if stdout:
        print("\n--- Training Output ---")
        print(stdout)

    if stderr:
        print("\n--- Errors/Warnings ---")
        print(stderr)

    print(separator)

    if result.get("status") == "completed":
        print("\nListing all trained models...")
        models_info = list_models.remote()
        models_list = models_info.get("models", [])
        print(f"\nAvailable models: {len(models_list)}")
        for model in models_list[:5]:
            fn = model.get("filename")
            mb = model.get("size_mb")
            try:
                mb_f = float(mb)
                print(f"  - {fn} ({mb_f:.2f} MB)")
            except (TypeError, ValueError):
                print(f"  - {fn}")