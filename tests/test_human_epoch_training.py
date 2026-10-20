from collections import Counter
from types import SimpleNamespace

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
        )
    )
    trainer.vla_train_dataloader = loader
    trainer.accelerator = SimpleNamespace(sync_gradients=True, is_local_main_process=False)
    trainer.completed_steps = 0
    seen = []
    trainer._log_training_config = lambda: None
    trainer._train_step = lambda batch: (seen.extend(batch) or {})
    trainer._log_metrics = lambda metrics: None
    trainer._finalize_training = lambda: None

    trainer.train()

    assert trainer.completed_steps == len(loader) * 3
    assert Counter(seen) == Counter({index: 3 for index in range(len(loader.dataset))})
