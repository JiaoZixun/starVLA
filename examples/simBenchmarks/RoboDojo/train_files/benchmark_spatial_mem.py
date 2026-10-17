#!/usr/bin/env python3
"""Measure real loader + model + backward + optimizer time on the active GPU."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("STARVLA_DISABLE_DEEPSPEED", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.tools import get_training_dtype
from starVLA.training.train_starvla import (
    VLATrainer, build_framework, create_accelerator, prepare_data,
    setup_directories, setup_optimizer_and_scheduler,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("starvla_robodojo_v30_qwen35_0p8b_spatial_mem.yaml")))
    parser.add_argument("--stage", choices=("a", "b"), default="b")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--target-seconds", type=float, default=1.0)
    parser.add_argument("--output", default="playground/benchmarks/robodojo_spatial_mem.json")
    parser.add_argument("--require-fast-kernels", action="store_true", help="Reject Qwen3.5's serial PyTorch delta-rule fallback")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.warmup < 1 or args.steps < 2:
        parser.error("A CUDA GPU, warmup>=1 and steps>=2 are required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist([
        f"run_id=benchmark_spatial_mem_{args.stage}",
        f"trainer.freeze_modules={'qwen_vl_interface' if args.stage == 'a' else ''}",
        *args.overrides]))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config
    from transformers.models.qwen3_5 import modeling_qwen3_5
    fast_kernels = bool(modeling_qwen3_5.is_fast_path_available)
    if args.require_fast_kernels and not fast_kernels:
        parser.error("Install compatible flash-linear-attention and causal-conv1d in the training environment first")
    accelerator = create_accelerator(cfg)
    if accelerator.num_processes != 1 or accelerator.gradient_accumulation_steps != 1:
        raise ValueError("This benchmark measures one GPU, one optimizer update per batch")
    out_dir = setup_directories(cfg)
    model = build_framework(cfg)
    loader = prepare_data(cfg, accelerator, out_dir)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, cfg)
    trainer = VLATrainer(cfg, model, loader, optimizer, scheduler, accelerator)
    trainer.prepare_training()
    trainer.model.train()
    trainer._create_data_iterators()
    timings, data_timings, peaks, losses = [], [], [], []
    skipped_updates = 0
    for index in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = trainer._get_next_batch()
        data_end = time.perf_counter()
        metrics = trainer._train_step(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if index >= args.warmup:
            timings.append(elapsed)
            data_timings.append(data_end - start)
            peaks.append(metrics["memory/cuda_peak_allocated_gb"])
            losses.extend(value for key, value in metrics.items() if "loss" in key)
            skipped_updates += int(trainer.optimizer.step_was_skipped)
        trainer.completed_steps += int(not trainer.optimizer.step_was_skipped)
        print(f"{index + 1}/{args.warmup + args.steps}: {elapsed:.3f}s, data={data_end-start:.3f}s, skipped={trainer.optimizer.step_was_skipped}", flush=True)
    mean, median, p95 = float(np.mean(timings)), float(np.median(timings)), float(np.percentile(timings, 95))
    report = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "dtype": str(get_training_dtype()), "stage": args.stage,
        "qwen35_fast_kernels": fast_kernels,
        "batch_size": cfg.datasets.vla_data.per_device_batch_size,
        "views": 3, "image_size": list(cfg.datasets.vla_data.obs_image_size),
        "gradient_checkpointing": cfg.trainer.gradient_checkpointing,
        "num_workers": cfg.datasets.vla_data.num_workers,
        "warmup": args.warmup, "steps": args.steps,
        "seconds_per_batch_mean": mean, "seconds_per_batch_median": median,
        "seconds_per_batch_p95": p95, "data_seconds_mean": float(np.mean(data_timings)),
        "samples_per_second": cfg.datasets.vla_data.per_device_batch_size / mean,
        "peak_allocated_gib": max(peaks), "all_losses_finite": bool(np.isfinite(losses).all()),
        "target_seconds": args.target_seconds,
        "optimizer_updates": args.steps - skipped_updates,
        "skipped_optimizer_updates": skipped_updates,
        "target_met": mean <= args.target_seconds and bool(np.isfinite(losses).all()) and skipped_updates == 0,
        "scope": "Real dataset decode/preprocess + VLM/spatial/memory + backward/clip/optimizer; excludes startup/checkpoint save",
        "batch_seconds": timings, "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"config", "batch_seconds"}}, indent=2))
    print(f"Report: {output}")
    if not report["all_losses_finite"] or skipped_updates:
        raise SystemExit("Benchmark encountered nonfinite losses or skipped optimizer updates; increase warmup/check precision")


if __name__ == "__main__":
    main()
