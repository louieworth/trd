#!/usr/bin/env python3
#
# Entry point for KL Divergence Training

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recipe.opd.config import KLTrainingConfig
from recipe.opd.kl_trainer import KLTrainer
from recipe.opd._tokenizer_compat import apply_qwen2_tokenizer_vllm_compat


def parse_args():
    parser = argparse.ArgumentParser(description="KL Divergence Training for Math Reasoning")

    # Distillation Mode
    parser.add_argument(
        "--task", type=str, default="math", choices=["math", "code"],
        help="Training/evaluation task adapter: math uses DeepScaleR-style prompts; code uses TACO/code prompts.",
    )

    parser.add_argument(
        "--distill_mode", type=str, default="opsd", choices=["opsd", "opd"],
        help="opsd: teacher = student, teacher prompt embeds expert solution. "
             "opd: teacher != student, teacher prompt has no expert reference.",
    )

    # KL Settings
    parser.add_argument("--kl_type", type=str, default="reverse", choices=["reverse", "forward", "jsd"])
    parser.add_argument("--kl_method", type=str, default="monte_carlo", choices=["monte_carlo", "full_vocab"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl_token_clip", type=float, default=0,
                        help="Per-token KL clip (OPSD jsd_token_clip). 0 disables.")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Mixture coefficient for generalized JSD (only used when kl_type=jsd). "
                             "beta=0 → forward KL, beta=1 → reverse KL, beta∈(0,1) → JSD mixture.")

    # Model Settings
    parser.add_argument("--student_model_path", type=str, required=True)
    parser.add_argument("--teacher_model_path", type=str, default="")
    parser.add_argument("--base_model_name", type=str, default="")
    parser.add_argument("--use_lora", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)

    # Training Settings
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=20480)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Data Settings
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--corrected_responses_path", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--use_initial_response",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Teacher prompt mode: false=rewrite from expert only, true=rewrite using the initial response plus expert guidance.",
    )
    parser.add_argument(
        "--prompt_truncation",
        type=lambda x: x.lower() == "true",
        default=False,
        help="When prompt+response > max_length, truncate the **Your Initial Solution:** block in the teacher prompt instead of the response tail. Preserves response loss signal at the cost of dropping initial-solution context.",
    )
    parser.add_argument("--num_workers", type=int, default=4)

    # verl FSDP Settings
    parser.add_argument("--fsdp_strategy", type=str, default="fsdp2", choices=["fsdp", "fsdp2"])
    parser.add_argument("--fsdp_size", type=int, default=-1)
    parser.add_argument("--ulysses_sequence_parallel_size", type=int, default=1)
    parser.add_argument("--max_token_len_per_gpu", type=int, default=None)
    parser.add_argument("--use_remove_padding", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--use_torch_compile", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--param_offload", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--optimizer_offload", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--offload_policy", type=lambda x: x.lower() == "true", default=False)

    # Distributed Training
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
    )
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n_gpus_per_node", type=int, default=1)

    # Output Settings
    parser.add_argument("--epoch_index", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_save_dir", type=str, default="./checkpoints")
    parser.add_argument("--gen_results_dir", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="verl-kl-training")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--save_merged_model", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save FSDP checkpoint every N optimizer steps. Lower = better crash safety, more disk.")
    parser.add_argument("--max_ckpt_to_keep", type=int, default=1,
                        help="Rolling window for intra-epoch FSDP ckpts. Per-epoch hf_merged is preserved separately.")
    parser.add_argument("--resume_checkpoint_path", type=str, default="",
                        help="Explicit FSDP checkpoint to load before training.")
    parser.add_argument("--resume_checkpoint_mode", type=str, default="continue", choices=["continue", "initialize"],
                        help="continue=resume same run and skip completed optimizer steps; initialize=load weights/optimizer but run this batch from scratch.")
    parser.add_argument("--async_hf_export", type=lambda x: x.lower() == "true", default=False,
                        help="Run final HF export in a background rank-0 subprocess when possible.")

    # Evaluation Settings
    parser.add_argument("--run_eval_after_training", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--eval_datasets", type=str, default="aime24,aime25,math500")
    parser.add_argument("--eval_datasets_dir", type=str, default="./data/eval")

    # Top-K teacher local support matching (Fu et al. 2026, arXiv:2603.25562)
    parser.add_argument("--top_k", type=int, default=0,
                        help="When > 0 and kl_method=full_vocab, replace per-position KL with truncated KL "
                             "over the top-K teacher-selected tokens (renormalized in support). "
                             "Paper default 32. JSD is not supported with top-K.")

    return parser.parse_args()


def main():
    pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if SCRIPT_DIR not in pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([SCRIPT_DIR, *pythonpath_entries])

    apply_qwen2_tokenizer_vllm_compat()

    args = parse_args()

    if not args.data_path:
        raise ValueError("--data_path is required")

    # Create config
    config = KLTrainingConfig(
        # Distillation Mode
        distill_mode=args.distill_mode,
        task=args.task,
        # KL Settings
        kl_type=args.kl_type,
        kl_method=args.kl_method,
        temperature=args.temperature,
        kl_token_clip=args.kl_token_clip,
        beta=args.beta,
        # Model Settings
        student_model_path=args.student_model_path,
        teacher_model_path=args.teacher_model_path,
        base_model_name=args.base_model_name,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        # Training Settings
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        total_epochs=args.total_epochs,
        max_length=args.max_length,
        warmup_steps_ratio=args.warmup_steps_ratio,
        weight_decay=args.weight_decay,
        min_lr_ratio=args.min_lr_ratio,
        # Data Settings
        data_path=args.data_path,
        corrected_responses_path=args.corrected_responses_path,
        max_samples=args.max_samples,
        use_initial_response=args.use_initial_response,
        prompt_truncation=args.prompt_truncation,
        num_workers=args.num_workers,
        # verl FSDP Settings
        fsdp_strategy=args.fsdp_strategy,
        fsdp_size=args.fsdp_size,
        ulysses_sequence_parallel_size=args.ulysses_sequence_parallel_size,
        max_token_len_per_gpu=args.max_token_len_per_gpu,
        use_remove_padding=args.use_remove_padding,
        use_torch_compile=args.use_torch_compile,
        param_offload=args.param_offload,
        optimizer_offload=args.optimizer_offload,
        offload_policy=args.offload_policy,
        # Distributed Training
        local_rank=args.local_rank,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        # Output Settings
        epoch_index=args.epoch_index,
        output_dir=args.output_dir,
        model_save_dir=args.model_save_dir,
        gen_results_dir=args.gen_results_dir if args.gen_results_dir else args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_merged_model=args.save_merged_model,
        save_steps=args.save_steps,
        max_ckpt_to_keep=args.max_ckpt_to_keep,
        resume_checkpoint_path=args.resume_checkpoint_path,
        resume_checkpoint_mode=args.resume_checkpoint_mode,
        async_hf_export=args.async_hf_export,
        # Evaluation Settings
        run_eval_after_training=args.run_eval_after_training,
        eval_datasets=args.eval_datasets.split(",") if args.eval_datasets else [],
        eval_datasets_dir=args.eval_datasets_dir,
        # Top-K local support matching
        top_k=args.top_k,
    )

    trainer = KLTrainer(config)
    try:
        trainer.train()
    except BaseException:
        # A single rank failing inside train() must not call destroy_process_group
        # while peers are still inside an FSDP collective — that hangs the whole
        # job (the other ranks busy-wait on NCCL forever). Print the traceback,
        # then hard-exit so torchrun reaps every worker.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    trainer.close()


if __name__ == "__main__":
    main()
