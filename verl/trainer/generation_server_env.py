"""Helpers for launching generation servers from long-lived trainer processes.

Ray actors inherit the driver's environment by default. When evaluation is
triggered from a `torchrun` process, stale torch distributed / torchelastic
variables can leak into Ray workers and cause them to rendezvous against the
training job's TCP store instead of the fresh Ray worker-group store created
for evaluation.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


GENERATION_SERVER_ENV_VARS = {
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "VLLM_USE_V1": "1",
    # Disable OTLP exporters in Ray/vLLM actors. The grpc/opentelemetry-cpp
    # metric exporter can segfault in worker background threads.
    "OTEL_SDK_DISABLED": "true",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",
    # Avoid vLLM torch.compile cache corruption across repeated standalone
    # rollout launches in long-lived Slurm allocations.
    "VLLM_DISABLE_COMPILE_CACHE": "1",
}


GENERATION_SERVER_PASSTHROUGH_ENV_KEYS = (
    "VLLM_CACHE_ROOT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "XDG_CACHE_HOME",
)


# These keys are commonly injected by torchrun / torchelastic. They are valid
# for the trainer job itself, but they must not leak into standalone Ray-based
# generation workers.
TORCH_LAUNCH_ENV_KEYS = (
    "DIST_INIT_METHOD",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_ERROR_FILE",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "WORLD_SIZE",
)


def build_generation_server_runtime_env() -> dict[str, dict[str, str]]:
    """Return the runtime env used by standalone generation servers."""
    env_vars = GENERATION_SERVER_ENV_VARS.copy()
    for key in GENERATION_SERVER_PASSTHROUGH_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return {"env_vars": env_vars}


@contextmanager
def temporarily_clear_torch_launch_env() -> Iterator[None]:
    """Temporarily remove torchrun/torchelastic env vars from the current process."""
    saved_env = {key: os.environ[key] for key in TORCH_LAUNCH_ENV_KEYS if key in os.environ}
    for key in TORCH_LAUNCH_ENV_KEYS:
        os.environ.pop(key, None)

    try:
        yield
    finally:
        for key in TORCH_LAUNCH_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(saved_env)
