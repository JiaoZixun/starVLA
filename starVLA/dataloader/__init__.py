import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def _build_human_wrist_dataloader(data_cfg, waypoint_tau):
    import torch
    from starVLA.dataloader.human_wrist_dataset import HumanWristGeometryDataset, collate_human_wrist_batch

    dataset = HumanWristGeometryDataset(
        sidecar_root=data_cfg.geometry_sidecar_root,
        split=data_cfg.get("split", "train"),
        waypoint_tau=waypoint_tau,
        image_size=data_cfg.get("image_size", 224),
    )
    workers = int(data_cfg.get("num_workers", 8))
    kwargs = {
        "batch_size": int(data_cfg.per_device_batch_size),
        "collate_fn": collate_human_wrist_batch,
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "shuffle": True,
        "drop_last": bool(data_cfg.get("drop_last", False)),
        "generator": torch.Generator(),
    }
    if workers:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "human_wrist_dataset":
        data_cfg = cfg.datasets.vla_data
        return _build_human_wrist_dataloader(data_cfg, data_cfg.get("waypoint_tau", [1, 2, 4, 8]))
    elif dataset_py == "cotrain_human_robot":
        # Robot batches come from the regular LeRobot mixture configured in vla_data;
        # human batches from vla_data.human_data, interleaved one domain per batch.
        from starVLA.dataloader.cotrain_dataloader import HumanRobotCotrainLoader

        data_cfg = cfg.datasets.vla_data
        human_cfg = data_cfg.get("human_data")
        if human_cfg is None:
            raise ValueError("dataset_py=cotrain_human_robot requires datasets.vla_data.human_data")
        robot_loader = build_dataloader(cfg, dataset_py="lerobot_datasets")
        # Both domains must use the same waypoint slots as the model.
        human_loader = _build_human_wrist_dataloader(human_cfg, data_cfg.get("waypoint_tau", [1, 2, 4, 8]))
        return HumanRobotCotrainLoader(robot_loader, human_loader, human_every=int(human_cfg.get("human_every", 4)))
    elif dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )
        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        dataloader_kwargs = {
            "batch_size": cfg.datasets.vla_data.per_device_batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
            # shuffle=True
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            dataloader_kwargs["prefetch_factor"] = int(vla_dataset_cfg.get("prefetch_factor", 2))

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
    elif dataset_py == "umi_datasets":
        from starVLA.dataloader.umi_datasets import make_umi_dataloader

        umi_train_dataloader = make_umi_dataloader(cfg)
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            umi_train_dataloader.dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return umi_train_dataloader
    raise ValueError(f"Unsupported dataset_py: {dataset_py!r}")
