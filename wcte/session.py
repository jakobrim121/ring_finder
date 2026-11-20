"""Fit one event in an isolated process using the current Python installation."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


def fit_event(
    pmt_ids, charges_pe, times_ns, *, incoming_particle, event_id=0,
    pmt_id_space="wcte", survey=None, prompt_window_ns=None, clock_decimals=None,
):
    """Return fit arrays and ring count without changing the calling kernel's survey."""
    with TemporaryDirectory(prefix="wcte-event-") as directory:
        source, output = Path(directory) / "digits.npz", Path(directory) / "fit"
        ids = np.asarray(pmt_ids)
        if ids.shape == (0,):
            ids = ids.astype(np.int64)
        np.savez(source, raw_pmt_ids=ids, charges=charges_pe, times=times_ns)
        command = [
            sys.executable, "-m", "wcte", "--input", str(source), "--output", str(output),
            "--incoming-particle", incoming_particle, "--event-id", str(event_id),
            "--pmt-id-space", pmt_id_space,
        ]
        if survey is not None:
            command.extend(("--survey", str(Path(survey).resolve(strict=True))))
        if prompt_window_ns is not None:
            command.extend(("--prompt-window-ns", *map(str, prompt_window_ns)))
        if clock_decimals is not None:
            command.extend(("--clock-decimals", str(clock_decimals)))
        environment = os.environ.copy()
        environment.pop("WCTE_SURVEY", None)
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        with np.load(output / "prediction.npz", allow_pickle=False) as prediction:
            fit = dict(prediction)
    return dict(fit, ring_count=fit["fractions"].shape[1] - 1)
