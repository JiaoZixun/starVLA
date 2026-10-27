#!/usr/bin/env python3
"""DexYCB Spatial Pretraining Script.

Trains WaypointModel + Qwen on DexYCB wrist trajectories with geometry supervision only.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from examples.human2robots.DexYCBSpatial.dexycb_dataset import DexYCBWristDataset
from starVLA.model.factory import create_model_by_name


def setup_ddp():
    """Initialize DDP."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank, world_size
    else:
        # Single GPU
        return 0, 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/preprocessed_dexycb")
    parser.add_argument("--output_dir", type=str, default="playground/Checkpoints/dexycb_pretrain_20e")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_qwen", type=float, default=1e-6)
    parser.add_argument("--lr_waypoint", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_every_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    local_rank, world_size = setup_ddp()
    is_main = local_rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Training DexYCB Spatial Pretraining")
        print(f"World size: {world_size}")

    # Create model
    model_config = {
        "name": "QwenSpatialOFT",
        "qwenvl": {
            "base_vlm": "Qwen/Qwen3.5-0.8B",
            "local_files_only": True,
            "attn_implementation": "sdpa",
            "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        },
        "action_model": {
            "action_dim": 7,
            "action_hidden_dim": 1024,
            "action_horizon": 8,
        },
        "spatial": {
            "width": 256,
            "heads": 4,
            "ffn_width": 1024,
            "layers": 2,
            "tau": [1, 2, 4, 8],
            "waypoint_weight": 1.0,
            "waypoint_rotation_weight": 0.25,
            "gripper_weight": 0.10,
        },
    }

    model = create_model_by_name(model_config)
    model = model.cuda(local_rank)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Datasets
    train_dataset = DexYCBWristDataset(args.data_root, split="train")
    val_dataset = DexYCBWristDataset(args.data_root, split="val")

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer
    qwen_params = []
    waypoint_params = []

    for name, param in model.named_parameters():
        if "qwen_vl_interface" in name:
            qwen_params.append(param)
        elif "waypoint_model" in name or "spatial" in name:
            waypoint_params.append(param)
        else:
            # action_model: freeze during pretraining
            param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {"params": qwen_params, "lr": args.lr_qwen},
        {"params": waypoint_params, "lr": args.lr_waypoint},
    ], betas=(0.9, 0.95), weight_decay=0.01)

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )

    # Training loop
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_trans_loss = 0.0
        epoch_rot_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)

        for batch in pbar:
            # Warmup: freeze qwen for first 500 steps
            if global_step < args.warmup_steps:
                for param in qwen_params:
                    param.requires_grad = False
            elif global_step == args.warmup_steps:
                for param in qwen_params:
                    param.requires_grad = True
                if is_main:
                    print(f"Warmup complete at step {global_step}, unfreezing Qwen")

            # Forward
            images = batch["image"].cuda(local_rank)
            instructions = batch["instruction"]

            # Prepare examples
            examples = []
            for i in range(len(instructions)):
                examples.append({
                    "instruction": instructions[i],
                    "geometry_labels": {
                        k: v[i].cuda(local_rank) for k, v in batch["geometry_labels"].items()
                    },
                    "action": batch["action"][i].cuda(local_rank),
                })

            # Model forward
            model_module = model.module if isinstance(model, DDP) else model
            _, _, _, metrics = model_module.forward_train(
                images,
                instructions,
                examples,
            )

            # Only use geometry loss (no BC loss for pretraining)
            loss = metrics.get("waypoint_translation_loss", 0) + \
                   model_config["spatial"]["waypoint_rotation_weight"] * metrics.get("waypoint_rotation_loss", 0)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at step {global_step}, skipping")
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += loss.item()
            epoch_trans_loss += metrics.get("waypoint_translation_loss", 0).item()
            epoch_rot_loss += metrics.get("waypoint_rotation_loss", 0).item()
            global_step += 1

            if is_main:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "trans": f"{metrics.get('waypoint_translation_loss', 0):.4f}",
                    "rot": f"{metrics.get('waypoint_rotation_loss', 0):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Epoch summary
        if is_main:
            avg_loss = epoch_loss / len(train_loader)
            avg_trans = epoch_trans_loss / len(train_loader)
            avg_rot = epoch_rot_loss / len(train_loader)
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, trans={avg_trans:.4f}, rot={avg_rot:.4f}")

        # Save checkpoint
        if is_main and (epoch + 1) % args.save_every_epochs == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch{epoch+1}.pt"
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    if is_main:
        print("Training complete!")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
