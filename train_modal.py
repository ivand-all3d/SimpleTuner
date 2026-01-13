# train_modal.py
"""
Run training on Modal with the following command:

modal run train_modal.py --config-path simpletuner/examples/flux.peft-lora/config.json

You can specify other arguments like:
--env-name default
--num-processes 1
--mixed-precision bf16
--debug
"""

import modal
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# GPU Configuration
GPU_CONFIG = "H100"
TIMEOUT_HOURS = 24

# Define the image with dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["git", "libgl1-mesa-glx", "libglib2.0-0", "jq"])
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("setup.py", "/root/setup.py", copy=True)
    .add_local_dir("simpletuner", remote_path="/root/simpletuner", copy=True)
    .add_local_dir("scripts", remote_path="/root/scripts", copy=True)
    # .add_local_file("st_cli.py", remote_path="/root/st_cli.py", copy=True)
    .workdir("/root")
    .run_commands(["uv venv --system-site-packages", 'uv pip install -e ".[cuda]"'])
    # Ensure root is in PYTHONPATH so we can import simpletuner
    .env(
        {
            "PYTHONPATH": "/root:$PYTHONPATH",
            "VIRTUAL_ENV": "/root/.venv",
            "PATH": "/root/.venv/bin:$PATH",
        }
    )
    # Create config directory
    .run_commands("mkdir -p /root/config")
)

# If a local config directory exists, mount it
if Path("config").exists():
    image = image.add_local_dir("config", remote_path="/root/config")

# Define volumes for persistence
model_volume = modal.Volume.from_name("simpletuner-models", create_if_missing=True)
data_volume = modal.Volume.from_name("qwen-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("simpletuner-cache", create_if_missing=True)

app = modal.App(
    name="simpletuner-training",
    image=image,
    volumes={
        "/root/output": model_volume,  # Standardize output path
        "/root/data": data_volume,
        "/root/cache": cache_volume,
    },
)


@app.function(
    gpu=GPU_CONFIG,
    timeout=3600 * TIMEOUT_HOURS,
    cpu=8,
    memory=64 * 1024,  # 64GB RAM
    # Mount volumes to persist data
    volumes={
        "/root/output": model_volume,
        "/root/data": data_volume,
        "/root/cache": cache_volume,
    },
)
def train(
    config_path: Optional[str] = None,
    env_name: str = "default",
    num_processes: int = 1,
    mixed_precision: str = "bf16",
    debug: bool = False,
):
    """
    Run SimpleTuner training on Modal.

    Args:
        config_path: Path to a config file (JSON/TOML/ENV) relative to project root.
                     e.g. "simpletuner/examples/flux.peft-lora/config.json"
        env_name: Environment name if using config.env (default: "default")
        num_processes: Number of processes for accelerate (default: 1)
        mixed_precision: Mixed precision mode (no, fp16, bf16)
        debug: Enable debug logging
    """
    print(f"Starting training on {GPU_CONFIG}...")

    # Set Environment Variables
    env_vars = os.environ.copy()
    env_vars["HF_HOME"] = "/root/cache/huggingface"
    env_vars["HF_HUB_CACHE"] = "/root/cache/huggingface/hub"
    env_vars["WANDB_CACHE_DIR"] = "/root/cache/wandb"
    env_vars["DISABLE_UPDATES"] = (
        "true"  # Prevent train.sh from trying to run poetry install
    )
    env_vars["MIXED_PRECISION"] = mixed_precision
    env_vars["TRAINING_NUM_PROCESSES"] = str(num_processes)
    env_vars["TRAINING_NUM_MACHINES"] = "1"
    env_vars["TRAINING_DYNAMO_BACKEND"] = "no"

    if debug:
        env_vars["ACCELERATE_LOG_LEVEL"] = "INFO"
        env_vars["CURL_CA_BUNDLE"] = ""  # Sometimes helps with SSL issues in containers

    # Determine command to run
    # We can invoke simpletuner/train.py directly via accelerate

    cmd = ["accelerate", "launch"]

    # Add accelerate args
    cmd.extend(
        [
            "--mixed_precision",
            mixed_precision,
            "--num_processes",
            str(num_processes),
            "--num_machines",
            "1",
            "--dynamo_backend",
            "no",
        ]
    )

    # Script to launch
    cmd.append("simpletuner/train.py")

    # Handle Config
    if config_path:
        print(f"Using config file: {config_path}")
        if config_path.endswith(".json"):
            env_vars["CONFIG_BACKEND"] = "json"
            # Remove extension for CONFIG_PATH
            env_vars["CONFIG_PATH"] = config_path.rsplit(".", 1)[0]
        elif config_path.endswith(".toml"):
            env_vars["CONFIG_BACKEND"] = "toml"
            env_vars["CONFIG_PATH"] = config_path.rsplit(".", 1)[0]
        elif config_path.endswith(".env"):
            env_vars["CONFIG_BACKEND"] = "env"
            print(
                "Warning: .env config support in this script assumes SimpleTuner loads it internaly."
            )

    else:
        # Use ENV variable defaults
        env_vars["ENV"] = env_name
        print(f"Using ENV={env_name}")

    # Ensure output directory exists
    if not os.path.exists("/root/output"):
        os.makedirs("/root/output", exist_ok=True)

    print("Running command:", " ".join(cmd))

    # Execute
    try:
        subprocess.run(cmd, check=True, env=env_vars, cwd="/root")
        print("Training completed successfully.")

        # Commit volumes
        model_volume.commit()
        cache_volume.commit()
        data_volume.commit()

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    pass
