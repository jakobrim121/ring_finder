# WCTE ring finder

Reconstruct Cherenkov rings from one event's PMT IDs, charges, and hit times.
The fitter returns a ring count and a shared-charge partition. The incoming
particle hypothesis can be a pion, proton, or muon. The package uses the
Third Retention fitter with support for run-specific active PMTs and geometry.

## Run the example notebook

Use Python 3.11 or 3.12 on Linux or macOS.

```sh
git clone https://github.com/jakobrim121/ring_finder.git
cd ring_finder
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook]'
jupyter lab examples/WCTE_Ring_Finder_Tutorial.ipynb
```

Run the notebook from top to bottom. The checkout includes the detector inputs,
calibrations, event-display code, and both example events. After installing the
Python dependencies, the notebook runs offline without a CERN login or a
LicketyFit checkout. The first fit in each detector configuration can take
longer while numerical kernels compile.
[Example notebook](examples/WCTE_Ring_Finder_Tutorial.ipynb)

| Example | Included input |
| --- | --- |
| Simulation | Digitized hits from event 0 of the 300 MeV WCSim pion sample |
| Real data | A detector-only ROOT excerpt of WCTE Run 2079, original entry 15, and its active-PMT survey |

The excerpt holds one entry at index 0 and preserves the original event number
15. The notebook shows how to apply the PMT readout mask and an upstream hit
time window, then convert the example run's calibrated charge from ADC to PE.
Use your analysis to select the event and apply any trigger, T5, and upstream
timing cuts.

## Fit your own arrays

```python
from wcte.api import fit_hits

prediction, optical, reviews = fit_hits(
    pmt_ids, charges_pe, times_ns, incoming_particle="pion"
)
ring_count = prediction["fractions"].shape[1] - 1
```

Pass aligned one-dimensional arrays for a single event. Use WCTE IDs encoded
as `100 * slot + position` with zero-based PMT positions, or set
`pmt_id_space="wcsim"` for zero-based WCSim tube IDs. Charges are in PE and
times are in ns. The standard fit window is `[-8, +7)` ns around the detected
prompt peak. To use your own final gate, pass `prompt_window_ns=(low, high)`
in the input clock. The timing likelihood uses that same interval.

The result's `fractions` and `shared_charge_pe` have one row per retained
pulse and one column per component. Column zero is background. The following
columns are fitted rings. `input_indices` indexes the three input arrays.
The display colors each PMT by the component with the most shared PE there.

Set `WCTE_SURVEY` before a standalone fit process starts to use a different
detector survey. A survey NPZ contains `pmt_ids`, `positions_mm`,
`inward_normals`, `module_slots`, and `module_centers_mm`. Use one survey per
process. The notebook uses `wcte.session.fit_event` to run simulation and a
surveyed detector in separate processes within the same kernel environment.
Set `WCTE_NUMBA_CACHE` to choose where compiled kernels are cached.

## From the command line

```sh
python -m pip install -e .
python -m wcte --input examples/data/wcsim_pion_300mev_event0.npz \
  --incoming-particle pion --output fit-output
```

The input NPZ has `raw_pmt_ids`, `charges`, and `times`. The command defaults
to WCSim IDs. Use `--pmt-id-space wcte` for WCTE IDs and `--survey FILE` for a
run-specific survey. Use `--events MANIFEST.json` for a JSON object with an
`events` list. Each row has `file`, `event`, and `incoming_particle`.
Paths are relative to the manifest. `--workers N` runs independent events
in separate processes.

| Output | Contents |
| --- | --- |
| `prediction.npz` | Retained IDs and charges, input indices, shared fractions and PE, and display labels |
| `original_fit.json` | Optical and finite-track hypotheses entering the final reviews |
| `comparison.json` | Review decisions, supporting evidence, and timings |
| `runs.json` | Input IDs, ring counts, output paths, and process CPU timings |

## Source layout

| Path | What it does |
| --- | --- |
| `wcte/api.py` and `wcte/cli.py` | Array and command-line entry points |
| `wcte/pipeline.py` and `wcte/engine.py` | Reconstruction stages and numerical initialization |
| `wcte/model/search/` | Optical seeds and finite-track hypothesis comparisons |
| `wcte/reviews/` | Particle-specific reviews and charge reassignment |
| `wcte/model/detector/` and `wcte/data/` | Detector geometry, response tables, and calibrations |
| `wcte/display.py` and `wcte/display_data/` | WCTE 2D event display and its projection |
| `examples/` | Notebook and its stand-alone input files |

The display uses the WCTE module projection and channel placement from
LicketyFit. Third-party notices for the numerical sources are in `licenses/`.
[LicketyFit event display](https://github.com/jakobrim121/LicketyFit_official/tree/main/event_display)
