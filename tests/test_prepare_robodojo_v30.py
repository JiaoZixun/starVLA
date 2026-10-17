from pathlib import Path

from examples.simBenchmarks.RoboDojo.train_files.prepare_robodojo_v30 import _link_or_validate


def test_adapter_links_are_relative_and_repair_absolute_links(tmp_path):
    raw = tmp_path / "raw/data"
    raw.mkdir(parents=True)
    adapter = tmp_path / "bundle/adapter"
    adapter.mkdir(parents=True)
    link = adapter / "data"

    _link_or_validate(raw, link)
    assert link.is_symlink()
    assert not link.readlink().is_absolute()
    assert link.resolve() == raw

    link.unlink()
    link.symlink_to(raw)
    _link_or_validate(raw, link)
    assert not link.readlink().is_absolute()
    assert link.resolve() == raw
