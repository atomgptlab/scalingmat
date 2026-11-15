# scalingmat

Code, results and job logs for

> J. Lee and K. Choudhary, *Scaling a Universal Graph Neural Network Force Field to Billions of Atoms on Desk-Side GPUs* (2026, submitted).

A universal graph neural network force field, ALIGNN-FF, evaluates the energies and forces of more than 10^9 atoms on NVIDIA GB10 desk-side computers (128 GB of CPU–GPU unified memory). The periodic cell is cut into cubes, and each cube is evaluated together with a halo of every atom within the model's 20 Å receptive field. Wherever the model's 52-neighbor cap does not bind, the result is exact: chunked and full-cell forces agree to about 1e-12 eV/Å in double precision. Because the GPU holds only one piece at a time, GPU memory no longer limits the size of the cell.

All results use **AFF-r2SCAN-smooth**, ALIGNN-FF retrained on MatPES r2SCAN with smooth switches on its bond (4–5 Å) and angle (3.0–3.5 Å) messages. The published AFF-r2SCAN's forces jump at its 3.5 Å three-body cutoff. That jump stalls relaxations and breaks energy conservation in molecular dynamics.

- Weights of AFF-r2SCAN-smooth: [doi:10.6084/m9.figshare.35008936](https://doi.org/10.6084/m9.figshare.35008936)
- Patch that adds the switches to ALIGNN: [`gb10/alignn_ff_max_atoms/smooth_messages.patch`](gb10/alignn_ff_max_atoms/smooth_messages.patch)
- Detailed notes on every run, check and job: [`gb10/alignn_ff_max_atoms/README.md`](gb10/alignn_ff_max_atoms/README.md)

## Main results (AFF-r2SCAN-smooth, single precision, GB10)

| System | Atoms | Forces | GB10s (tasks) | Largest piece | GPU (GB) | GB10-hours |
|---|---:|:---:|---:|---:|---:|---:|
| Si | 8,000,000 | yes | 1 (1) | 106,067 | 47.5 | 0.29 |
| Si | 64,000,000 | yes | 6 (6) | 106,054 | 47.5 | 2.34 |
| Si | 1,000,000,000 | no | 8 (10) | 261,222 | 49.3 | 9.29 |
| Si | 1,000,000,000 | yes | 8 (12) | 71,578 | 31.9 | 43.2 |
| Si | 2,000,376,000 | no | 9 (10) | 328,336 | 62.1 | 17.7 |
| Si | 2,000,376,000 | yes | 9 (20) | 104,194 | 46.6 | 72.3 |
| Si polycrystal, 4,800 grains | 1,021,299,267 | yes | 9 (10) | 69,056 | 30.8 | 45.1 |
| Si/SiO2/Al gate stack | 102,361,540 | yes | 3 (6) | 39,059 | 52.8 | 10.7 |
| STMV virus in water | 1,066,628 | yes | 1 (1) | 22,263 | 58.3 | 0.92 |

GB10-hours are the summed wall time of all tasks, which is the time the run would take on one GB10. None of the runs reached a memory limit. As a single graph, one GB10 holds 216,000 Si atoms with forces (layer checkpointing, 118.8 GB) and 551,368 atoms for the energy alone. Si cells are rattled by 0.05 Å per component. Results with the original AFF-r2SCAN and three other ALIGNN-FF models are in the detailed notes.

## Running it

Requirements: PyTorch with CUDA, the pure-PyTorch [ALIGNN](https://github.com/atomgptlab/alignn), `ase`, `matscipy` and `jarvis-tools`. DGL is not needed. The paper used ALIGNN commit `6a20fb5` with PyTorch 2.13 and CUDA 13.0. ALIGNN 2026.11.2 gives the same results.

```bash
# ALIGNN with the switches (models without them evaluate unchanged)
git clone https://github.com/atomgptlab/alignn.git alignn_smooth
(cd alignn_smooth && git apply ../scalingmat/gb10/alignn_ff_max_atoms/smooth_messages.patch)
export PYTHONPATH=$PWD/alignn_smooth

# weights: unzip the Figshare record; the model folder holds config.json and best_model.pt
M=/path/to/aff_r2scan_smooth
cd scalingmat/gb10/alignn_ff_max_atoms

# exactness check: 8,000 atoms of rattled Si, chunked against the full cell, float64
python smooth_eval.py ff_stream.py chunk --model_dir $M --kind Si --rep 10 --rattle 0.05 \
    --k 2 --forces --ckpt --validate --float64 --out check

# 8,000,000 atoms with forces on one GPU (6^3 cubes)
python smooth_eval.py ff_stream.py chunk --model_dir $M --kind Si --rep 100 --rattle 0.05 \
    --k 6 --forces --ckpt --out si_8M
python ff_stream.py reduce --out si_8M

# 10^9 atoms in W tasks: run ranks 0 to W-1 (one GPU each, or one after another), then reduce
python smooth_eval.py ff_stream.py chunk --model_dir $M --kind Si --rep 500 --rattle 0.05 \
    --k 36 --forces --ckpt --rank 0 --world 12 --out si_1B
```

- `ff_stream.py` generates crystals piece by piece from the lattice, so no task holds the whole cell. `ff_any.py` evaluates any orthorhombic cell written by `cells.py`, such as the polycrystal, the gate stack or water. `ff_max_atoms.py` holds the model, graph and energy code and the single-graph sweeps.
- `smooth_eval.py` runs either script with the per-atom energies of a model with switches. Without it, such a model stops with an error instead of being evaluated wrongly.
- The `*.sbatch` and `stage*_submit.sh` files are the Slurm jobs that produced the results. Their paths point to the cluster where they ran.

## Reproducing the figures

The figures are drawn from the committed result JSONs and need only `numpy` and `matplotlib`. Run the scripts from inside `analysis/`. The versions in the paper are in [`analysis/paper_figures/`](analysis/paper_figures/).

```bash
cd analysis
# Fig. 2 (decomposition), Fig. 3 (cost and memory) and Fig. S2 (halo width)
GB10_MODEL=smooth python gb10_billion_figures.py
# Fig. 4 (per-atom energy maps)
MAPS_DIR=../gb10/alignn_ff_max_atoms/results/aff_smooth/maps \
    FIG_OUT=gb10_energy_maps_smooth.png python gb10_energy_maps_figure.py
# Fig. S1 (other force-field families on one GB10)
python gb10_other_ff_figure.py
```

Structures, force arrays and per-atom energies of the largest runs are too large for the repository. Their summaries, and the energy maps drawn from them, are committed.

## Repository layout

```
gb10/alignn_ff_max_atoms/   the paper: chunked evaluation, retraining, checks, results/, logs/, README
gb10/other_ff_max_atoms/    SNAP, NEP, Allegro and Deep Potential size sweeps on one GB10 (Fig. S1)
gb10/alignn_ff_tensorrt/    TensorRT probe (1.1 to 1.3 times faster; fails at large sizes)
analysis/                   figure scripts, paper_figures/
b200/, h200/, gb10/alignn_ff/, gb10/alignn_ff_b200settings/,
analysis/scaling_*.py, cross_gpu_figure.py, results_report.py
                            earlier single-graph benchmarks (below)
```

## Earlier single-graph benchmarks

Before the decomposition, ALIGNN-FF was timed as a single graph on one NVIDIA B200 (`b200/alignn_ff/`, Cu FCC, energy only, DGL), one H200 NVL (`h200/`) and one GB10 (`gb10/alignn_ff/`, `gb10/alignn_ff_b200settings/`). These runs found the following.

- Replacing DGL's line-graph construction with a sparse one (`scale5b_v5.py`) made 10^5 to 10^6 atoms tractable on one GPU.
- In single precision, the energy of a single graph drifts and then jumps past about 470,000 atoms (`b200/alignn_ff/analysis/v6/v6_summary.md`). The chunked evaluation does not have this problem: it gives 0.604014 eV/atom up to 780,448 atoms on a GB10.
- With the same settings, model inference on the H200 is within 13% of the B200 and about 8 times faster than on the GB10. Through the ASE calculator, energy, forces and stress need 1.95 MB/atom on both the GB10 and the H200.

Each folder has its own README or summary. `analysis/results_report.py` collects these runs into `results_report.pdf`.

## License

Apache License 2.0, see `LICENSE`.
