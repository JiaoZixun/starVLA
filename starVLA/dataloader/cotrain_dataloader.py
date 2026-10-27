"""Interleave robot and human batches for co-training (one domain per batch)."""

from __future__ import annotations


class HumanRobotCotrainLoader:
    """Yield one human batch after every ``human_every - 1`` robot batches, deterministically.

    ``human_every=4`` gives robot, robot, robot, human, ... Each source restarts on its own
    when exhausted, so the robot loader keeps its epoch semantics and ``len()`` (used for
    epoch logging) is the robot loader's length.
    """

    def __init__(self, robot_loader, human_loader, human_every: int):
        if int(human_every) < 2:
            raise ValueError(f"human_every must be >= 2 (at least one robot batch per cycle), got {human_every}")
        self.robot_loader = robot_loader
        self.human_loader = human_loader
        self.human_every = int(human_every)
        self.epochs = {"robot": 0, "human": 0}

    def __len__(self):
        return len(self.robot_loader)

    def _next(self, iterators: dict, domain: str):
        source = self.robot_loader if domain == "robot" else self.human_loader
        try:
            return next(iterators[domain])
        except StopIteration:
            self.epochs[domain] += 1
            if hasattr(source, "sampler") and callable(getattr(source.sampler, "set_epoch", None)):
                source.sampler.set_epoch(self.epochs[domain])
            iterators[domain] = iter(source)
            return next(iterators[domain])

    def __iter__(self):
        # Never ends: the trainer stops on max_train_steps.
        iterators = {"robot": iter(self.robot_loader), "human": iter(self.human_loader)}
        position = 0
        while True:
            position += 1
            domain = "human" if position % self.human_every == 0 else "robot"
            yield self._next(iterators, domain)
