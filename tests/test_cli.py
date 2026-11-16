import os
import signal

import pytest

import sbayes.cli as cli


def fake_runner(task: cli.RunTask) -> None:
    """Stands in for a run: run 1 is killed, the others leave a marker file."""
    if task.i_run == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    open(task.config / f"done_{task.i_run}", "w").close()


def test_killed_run_does_not_block_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "runner", fake_runner)
    tasks = [
        cli.RunTask(i_run=i, n_clusters=3, config=tmp_path, experiment_name="test",
                    custom_settings=None, resume=False, use_gpu=False, num_cpus=1)
        for i in range(4)
    ]
    with pytest.raises(RuntimeError, match="1 of 4 runs failed") as error:
        cli.run_in_parallel(tasks, processes=2)
    assert "K=3, run 1" in str(error.value)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["done_0", "done_2", "done_3"]
