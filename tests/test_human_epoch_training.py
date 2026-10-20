from collections import Counter
from types import SimpleNamespace

import torch
import pytest
from torch.utils.data import DataLoader, Dataset

from starVLA.training.train_starvla import VLATrainer


class _IndexDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return index


def test_human_epoch_loop_trains_every_sample_each_epoch_without_extra_batches():
    loader = DataLoader(_IndexDataset(), batch_size=2, shuffle=True, drop_last=False)
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            max_train_epochs=3,
            max_train_steps=len(loader) * 3,
            task_mode="human_geometry_pretrain",
            eval_interval=0,
            save_interval=100,
            gradient_accumulation_steps=1,
        )
    )
    trainer.vla_train_dataloader = loader
    trainer.accelerator = SimpleNamespace(sync_gradients=True, is_local_main_process=False)
    trainer.completed_steps = 0
    seen = []
    trainer._log_training_config = lambda: None
    trainer._train_step = lambda batch: (seen.extend(batch.tolist()) or {})
    trainer._log_metrics = lambda metrics: None
    trainer._finalize_training = lambda: None
    saved_steps = []
    trainer._save_checkpoint = lambda: saved_steps.append(trainer.completed_steps)

    trainer.train()

    assert trainer.completed_steps == len(loader) * 3
    assert Counter(seen) == Counter({index: 3 for index in range(len(loader.dataset))})
    assert saved_steps == [3, 6, 9]


@pytest.mark.parametrize("completed_steps", [3, 9])
def test_human_epoch_resume_at_boundary_runs_only_remaining_epochs(completed_steps):
    loader = DataLoader(_IndexDataset(), batch_size=2, shuffle=True, drop_last=False, generator=torch.Generator())
    seed = 17
    expected_orders = []
    for epoch in range(3):
        loader.generator.manual_seed(seed + epoch)
        expected_orders.append([sample for batch in loader for sample in batch.tolist()])

    # Three of nine updates are already complete. Resume at the next full
    # epoch boundary and train exactly the remaining two epochs.
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        seed=seed,
        trainer=SimpleNamespace(
            max_train_epochs=3,
            max_train_steps=len(loader) * 3,
            task_mode="human_geometry_pretrain",
            eval_interval=0,
            save_interval=100,
            gradient_accumulation_steps=1,
        ),
    )
    trainer.vla_train_dataloader = loader
    trainer.accelerator = SimpleNamespace(sync_gradients=True, is_local_main_process=False)
    trainer.completed_steps = completed_steps
    seen = []
    trainer._log_training_config = lambda: None
    trainer._train_step = lambda batch: (seen.extend(batch.tolist()) or {})
    trainer._log_metrics = lambda metrics: None
    trainer._finalize_training = lambda: None
    saved_steps = []
    trainer._save_checkpoint = lambda: saved_steps.append(trainer.completed_steps)

    trainer.train()

    expected_remaining = [sample for epoch in expected_orders[completed_steps // len(loader):] for sample in epoch]
    assert seen == expected_remaining
    assert trainer.completed_steps == 9
    assert saved_steps == [step for step in [3, 6, 9] if step > completed_steps]


def test_resume_epoch_is_derived_from_completed_epoch_boundary(tmp_path):
    loader = DataLoader(_IndexDataset(), batch_size=2, drop_last=False)
    checkpoint = tmp_path / "steps_3_pytorch_model.pt"

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(max_train_epochs=3, gradient_accumulation_steps=1)
    )
    trainer.vla_train_dataloader = loader
    trainer.completed_steps = 3
    trainer.resume_epoch = 0
    trainer.resume_batch = 0

    trainer._restore_epoch_cursor(str(checkpoint))

    assert trainer.resume_epoch == 1
    assert trainer.resume_batch == 0


def test_epoch_resume_rejects_mid_epoch_checkpoint(tmp_path):
    loader = DataLoader(_IndexDataset(), batch_size=2, drop_last=False)
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(max_train_epochs=3, gradient_accumulation_steps=1)
    )
    trainer.vla_train_dataloader = loader
    trainer.completed_steps = 2
    trainer.resume_epoch = 0
    trainer.resume_batch = 0

    try:
        trainer._restore_epoch_cursor(str(tmp_path / "steps_2_pytorch_model.pt"))
    except ValueError as error:
        assert "only supports complete epoch boundaries" in str(error)
    else:
        raise AssertionError("mid-epoch resume must fail with an explicit boundary-only message")
