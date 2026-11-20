"""Run the qualified WCTE fitter on individual events or a manifest."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from time import process_time

from . import engine
from .api import PARTICLES, PMT_ID_SPACES, fit_hits


def write_json(path, value):
    """Serialize NumPy values from the fit diagnostics as ordinary JSON."""

    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, Path):
            return str(item)
        if hasattr(item, "tolist"):
            return clean(item.tolist())
        return item

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def fit_one(raw_path, event_id, particle, folder, options):
    """Read detector digits, fit the event and write the standard result files."""
    import numpy as np

    total_start = process_time()
    with np.load(raw_path, allow_pickle=False) as archive:
        digits = {key: archive[key] for key in ("raw_pmt_ids", "charges", "times")}
    start = process_time()
    prediction, detail, comparison = fit_hits(
        digits["raw_pmt_ids"], digits["charges"], digits["times"],
        event_id=event_id, incoming_particle=particle, **options,
    )
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / "prediction.npz", **prediction)
    prediction_cpu = process_time() - start
    write_json(folder / "original_fit.json", detail)
    write_json(folder / "comparison.json", comparison)
    return dict(
        count=prediction["fractions"].shape[1] - 1,
        event=event_id,
        incoming_particle=particle,
        output=str(folder),
        reconstruction_and_prediction_cpu_seconds=prediction_cpu,
        including_input_and_diagnostics_cpu_seconds=process_time() - total_start,
    )


def run_job(spec):
    return fit_one(*spec)


def read_jobs(args, parser):
    """Resolve manifest paths before creating worker processes."""
    out = args.output.resolve()
    options = dict(pmt_id_space=args.pmt_id_space, prompt_window_ns=args.prompt_window_ns,
                   clock_decimals=args.clock_decimals)
    if out.exists() or args.workers < 1:
        parser.error("Use a new output directory and one or more workers")
    if args.input:
        if not args.incoming_particle:
            parser.error("--incoming-particle is required with --input")
        return [(args.input.resolve(), args.event_id, args.incoming_particle, out, options)]
    manifest = args.events.resolve()
    try:
        rows = json.loads(manifest.read_text())["events"]
        jobs = [
            (
                (manifest.parent / row["file"]).resolve(),
                int(row["event"]),
                row["incoming_particle"],
                out / f"{index:06d}",
                options,
            )
            for index, row in enumerate(rows)
        ]
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(f"Invalid event manifest: {error}")
    if not jobs or any(row[2] not in PARTICLES for row in jobs):
        parser.error("Provide file, event and incoming_particle for each event")
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="NPZ with the detector digit arrays")
    source.add_argument("--events", type=Path, help="JSON manifest with an events list")
    parser.add_argument("--event-id", type=int, default=0)
    parser.add_argument("--incoming-particle", choices=PARTICLES)
    parser.add_argument("--pmt-id-space", choices=PMT_ID_SPACES, default="wcsim")
    parser.add_argument("--survey", type=Path, help="Run-specific active PMTs and detector positions")
    parser.add_argument("--prompt-window-ns", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument("--clock-decimals", type=int, help="Round median-relative times after prompt selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    jobs = read_jobs(args, parser)
    if args.survey is not None:
        os.environ["WCTE_SURVEY"] = str(args.survey.resolve(strict=True))
    engine.limit_threads()
    if args.workers == 1:
        engine.initialize()
        reports = [run_job(spec) for spec in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=engine.initialize) as pool:
            reports = list(pool.map(run_job, jobs))
    write_json(args.output.resolve() / "runs.json", {"events": reports})
    print(f"Fitted {len(reports)} events.")
