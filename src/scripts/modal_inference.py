import modal
import os
import time
from pathlib import Path
from typing import Dict, Any, List

app = modal.App("jaide-v40-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential",
        "wget",
        "curl",
        "ca-certificates",
        "xz-utils"
    )
    .run_commands(
        "wget -q https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz -O /tmp/zig.tar.xz",
        "tar -xJf /tmp/zig.tar.xz -C /usr/local",
        "ln -s /usr/local/zig-linux-x86_64-0.11.0/zig /usr/local/bin/zig",
        "rm /tmp/zig.tar.xz"
    )
    .add_local_dir("../../src", "/jaide_src")
)

volume = modal.Volume.from_name("jaide-training-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="B200:1",
    volumes={"/data": volume},
    cpu=16,
    memory=65536,
    timeout=3600,

)
def inference(
    prompt: str,
    model_filename: str,
    max_tokens: int = 256,
    temperature: float = 1.0
) -> Dict[str, Any]:
    import subprocess
    import shutil

    work_dir = Path("/workspace")
    work_dir.mkdir(exist_ok=True)

    src_target = work_dir / "jaide40" / "jaide" / "src"
    src_target.mkdir(parents=True, exist_ok=True)

    src_path = Path("/jaide_src")
    for item in src_path.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src_path)
            target_path = src_target / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_path)

    os.chdir(work_dir)

    build_result = subprocess.run(
        [
            "zig", "build-exe",
            str(src_target / "main.zig"),
            "-O", "ReleaseFast",
            "-fstrip"
        ],
        capture_output=True,
        text=True
    )

    if build_result.returncode != 0:
        return {
            "status": "build_failed",
            "error": build_result.stderr
        }

    binary_path = work_dir / "main"
    model_path = Path("/data/models") / model_filename

    if not model_path.exists():
        return {
            "status": "model_not_found",
            "error": f"Model file not found: {model_filename}"
        }

    infer_args = [
        str(binary_path),
        "--mode", "infer",
        "--model", str(model_path),
        "--prompt", prompt,
        "--max-tokens", str(max_tokens)
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    start_time = time.time()

    infer_result = subprocess.run(
        infer_args,
        capture_output=True,
        text=True,
        env=env
    )

    end_time = time.time()

    return {
        "status": "completed" if infer_result.returncode == 0 else "failed",
        "prompt": prompt,
        "output": infer_result.stdout,
        "error": infer_result.stderr if infer_result.returncode != 0 else None,
        "inference_time_seconds": end_time - start_time,
        "max_tokens": max_tokens,
        "model": model_filename
    }

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=1),
    volumes={"/data": volume},
    mounts=[jaide_source_mount],
    cpu=16,
    memory=65536,
    timeout=7200,

)
def batch_inference(
    prompts: List[str],
    model_filename: str,
    max_tokens: int = 256
) -> Dict[str, Any]:
    results = []
    total_start = time.time()

    for idx, prompt in enumerate(prompts):
        result = inference.local(
            prompt=prompt,
            model_filename=model_filename,
            max_tokens=max_tokens
        )
        results.append({
            "index": idx,
            "prompt": prompt,
            "result": result
        })

    total_end = time.time()

    return {
        "total_prompts": len(prompts),
        "total_time_seconds": total_end - total_start,
        "average_time_per_prompt": (total_end - total_start) / len(prompts) if prompts else 0,
        "results": results
    }

@app.local_entrypoint()
def main(
    prompt: str = "Mi az általános relativitáselmélet lényege?",
    model: str = None,
    max_tokens: int = 256
):
    from modal_train import list_models

    if model is None:
        print("Fetching latest model...")
        models_info = list_models.remote()
        if not models_info["models"]:
            print("No trained models found. Run training first.")
            return
        model = models_info["models"][0]["filename"]
        print(f"Using latest model: {model}")

    print("="*70)
    print("JAIDE v40 Inference on B200 GPU")
    print("="*70)
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Max Tokens: {max_tokens}")
    print("="*70)

    result = inference.remote(
        prompt=prompt,
        model_filename=model,
        max_tokens=max_tokens
    )

    print("\n" + "="*70)
    print("INFERENCE RESULT")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Inference Time: {result['inference_time_seconds']:.3f} seconds")

    if result['status'] == 'completed':
        print(f"\nPrompt: {result['prompt']}")
        print(f"\nOutput:\n{result['output']}")
    else:
        print(f"\nError: {result['error']}")

    print("="*70)