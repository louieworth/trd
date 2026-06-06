#!/usr/bin/env python3
#
# Configuration for KL Divergence Training

from dataclasses import dataclass, field
from typing import Literal, Optional


def get_prompt_mode_tag(use_initial_response: bool) -> str:
    """Return the prompt-mode suffix used in paths and evaluation names.

    The historical "correction" suffix is kept for compatibility with existing
    output paths even though the prompt now means rewrite-with-initial-response.
    """
    return "correction" if use_initial_response else "rewrite"


@dataclass
class KLTrainingConfig:
    """Configuration for KL Divergence Training."""

    # Distillation Mode
    # opsd: on-policy self-distillation — teacher = student, teacher prompt embeds y* (expert solution).
    # opd : on-policy distillation       — teacher ≠ student, teacher prompt has NO expert reference.
    distill_mode: Literal["opsd", "opd"] = "opsd"
    task: Literal["math", "code"] = "math"

    # KL Settings
    kl_type: Literal["reverse", "forward", "jsd"] = "reverse"
    kl_method: Literal["monte_carlo", "full_vocab"] = "monte_carlo"
    temperature: float = 0.7
    # Per-token KL clip (OPSD `jsd_token_clip`). Caps each token's KL
    # contribution to prevent outlier tokens (where teacher >> student)
    # from dominating the gradient. 0 disables.
    kl_token_clip: float = 0.1
    # Mixture coefficient for generalized JSD (only used when kl_type="jsd").
    # OPSD convention: beta=0 → forward KL, beta=1 → reverse KL, beta∈(0,1) → JSD mixture.
    beta: float = 0.0

    # Model Settings
    student_model_path: str = "Qwen/Qwen3-1.7B"
    teacher_model_path: str = ""  # Empty means same as student (memory efficient with LoRA)
    base_model_name: str = ""  # Stable name used for result keys/paths across multi-epoch runs
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # Training Settings
    learning_rate: float = 2e-5
    train_batch_size: int = 1  # per-GPU batch size
    gradient_accumulation_steps: int = 4
    total_epochs: int = 1
    max_length: int = 20480
    warmup_steps_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    min_lr_ratio: float = 0.1

    # Data Settings
    data_path: str = ""  # Reverse: y_o responses. Forward: y_r rewritten responses.
    corrected_responses_path: str = ""  # Optional legacy second file for forward KL rewrite targets
    max_samples: Optional[int] = None
    use_initial_response: bool = False  # Teacher prompt mode: False=rewrite from expert only, True=rewrite using initial response + expert guidance
    prompt_truncation: bool = False  # If True, when prompt+response > max_length, trim the **Your Initial Solution:** block in the teacher prompt instead of right-truncating the response.
    num_workers: int = 4

    # verl FSDP Settings
    fsdp_strategy: Literal["fsdp", "fsdp2"] = "fsdp2"
    fsdp_size: int = -1
    ulysses_sequence_parallel_size: int = 1
    max_token_len_per_gpu: Optional[int] = None
    use_remove_padding: bool = True
    use_torch_compile: bool = True
    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False

    # Distributed Training
    local_rank: int = -1
    world_size: int = 1
    nnodes: int = 1
    n_gpus_per_node: int = 1
    distributed_backend: str = "nccl"

    # Output Settings
    epoch_index: int = 1
    output_dir: str = "outputs/kl_training"
    model_save_dir: str = "./checkpoints"
    gen_results_dir: str = ""  # Intermediate generation files (auto-generated if empty)
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 500
    save_merged_model: bool = True  # Merge LoRA adapters after training
    max_ckpt_to_keep: int = 1  # Rolling window for intra-epoch FSDP ckpts; per-epoch hf_merged is always preserved separately
    resume_checkpoint_path: str = ""  # Optional explicit FSDP checkpoint to load before training
    resume_checkpoint_mode: Literal["continue", "initialize"] = "continue"
    async_hf_export: bool = False  # Launch rank-0 HF export in the background when eval does not need it

    # Evaluation Settings
    eval_datasets: list = field(default_factory=lambda: ["aime24", "aime25", "math500", "hmmt25"])
    run_eval_after_training: bool = False  # Set to True to run evaluation automatically
    eval_datasets_dir: str = "./data/eval"
    eval_dataset_paths: dict = field(default_factory=lambda: {
        "aime24": "data/aime24.parquet",
        "aime25": "data/aime25.parquet",
        "math500": "data/math500.parquet",
        "hmmt24": "data/hmmt24.parquet",
        "hmmt25": "data/hmmt25.parquet",
        "amc23": "data/amc23.parquet",
        "beyondaime": "data/beyondaime.parquet",
        "amobench": "data/amobench.parquet",
        "gsm8k": "data/gsm8k.parquet",
    })

    # Optimization
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # Wandb
    wandb_project: str = "trd"
    wandb_run_name: str = ""

    top_k: int = 0

    def __post_init__(self):
        """Validate configuration."""
        if not self.base_model_name:
            self.base_model_name = self.student_model_path.rstrip("/").split("/")[-1]

        if self.max_token_len_per_gpu is None:
            self.max_token_len_per_gpu = self.max_length

        if self.bf16 and self.fp16:
            raise ValueError("Cannot use both bf16 and fp16")

        if not self.bf16 and not self.fp16:
            print("Warning: Neither bf16 nor fp16 is enabled. Training will use fp32.")

        if self.distill_mode == "opd" and not self.teacher_model_path:
            raise ValueError(
                "distill_mode='opd' requires a distinct teacher: set teacher_model_path. "
                "Leave it empty only for distill_mode='opsd' (teacher = student)."
            )

        if self.resume_checkpoint_mode not in {"continue", "initialize"}:
            raise ValueError(
                "resume_checkpoint_mode must be 'continue' or 'initialize', "
                f"got {self.resume_checkpoint_mode!r}"
            )

        if self.task == "code":
            self.eval_dataset_paths = {
                "humaneval_plus": "evalplus:humaneval",
                "mbpp_plus": "evalplus:mbpp",
                "livecodebench_v6": "livecodebench:release_v6",
            }
            return

        self.eval_dataset_paths = {
            "aime24": f"{self.eval_datasets_dir}/aime24/aime24_test.parquet",
            "aime25": f"{self.eval_datasets_dir}/aime25/aime25_test.parquet",
            "math500": f"{self.eval_datasets_dir}/math500/math500_test.parquet",
            "hmmt24": f"{self.eval_datasets_dir}/hmmt24/hmmt24_test.parquet",
            "hmmt25": f"{self.eval_datasets_dir}/hmmt25/hmmt25_test.parquet",
            "amc23": f"{self.eval_datasets_dir}/amc23/amc23_test.parquet",
            "beyondaime": f"{self.eval_datasets_dir}/beyondaime/beyondaime_test.parquet",
            "amobench": f"{self.eval_datasets_dir}/amobench/amobench_test.parquet",
            "gsm8k": f"{self.eval_datasets_dir}/gsm8k/gsm8k_test.parquet",
        }

    @property
    def prompt_mode_tag(self) -> str:
        return get_prompt_mode_tag(self.use_initial_response)


# ============================================================================
# Preset Configurations
# ============================================================================

def get_reverse_kl_monte_carlo_config(**kwargs) -> KLTrainingConfig:
    """Preset for Reverse KL + Monte Carlo (most efficient)."""
    config = KLTrainingConfig(
        kl_type="reverse",
        kl_method="monte_carlo",
        temperature=1.0,
        learning_rate=2e-6,
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def get_reverse_kl_full_vocab_config(**kwargs) -> KLTrainingConfig:
    """Preset for Reverse KL + Full Vocabulary (more accurate)."""
    config = KLTrainingConfig(
        kl_type="reverse",
        kl_method="full_vocab",
        temperature=1.0,
        learning_rate=1e-6,
        gradient_checkpointing=True,
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def get_forward_kl_monte_carlo_config(**kwargs) -> KLTrainingConfig:
    """Preset for Forward KL + Monte Carlo (distribution covering)."""
    config = KLTrainingConfig(
        kl_type="forward",
        kl_method="monte_carlo",
        temperature=1.0,
        learning_rate=1e-6,
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def get_forward_kl_full_vocab_config(**kwargs) -> KLTrainingConfig:
    """Preset for Forward KL + Full Vocabulary (precise distribution matching)."""
    config = KLTrainingConfig(
        kl_type="forward",
        kl_method="full_vocab",
        temperature=1.0,
        learning_rate=1e-6,
        gradient_checkpointing=True,
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config
