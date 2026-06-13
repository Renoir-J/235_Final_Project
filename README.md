# Phys 235 Schwinger VQS Final Project

This project reproduces the Schwinger-model quench workflow from Nagano, Bapat, and Bauer, "Quench dynamics of the Schwinger model via variational quantum algorithms" (arXiv:2302.10933). The main entry point is `code/main_skeleton.ipynb`; the physics and algorithm implementations live in the numbered Python modules under `code/`.

## Layout

```text
235_Final_Project/
  code/
    main_skeleton.ipynb        # main runnable workflow
    schwinger_core.py          # Hamiltonian, ED, observables, fidelity
    module1_vqe.py             # VQE q=0 ground-state preparation
    module2_quench.py          # q=0 -> q=2 quench setup
    module3_trotter.py         # second-order Suzuki-Trotter baseline
    module4_mclachlan_vqs.py   # McLachlan VQS evolution
    test_fixture_integrity.py  # fixture schema check
  test_data/                   # reusable fixture data
  docs/
  project_rule.md
```

## Setup on a New Machine

Keep the full project folder structure unchanged, especially `code/` and `test_data/`.

Windows PowerShell:

```powershell
cd path\to\235_Final_Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pennylane jupyter
```

macOS/Linux:

```bash
cd path/to/235_Final_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pennylane jupyter
```

For Google Colab, upload or mount `235_Final_Project`, open `code/main_skeleton.ipynb`, and uncomment the install cell:

```python
# %pip install -q pennylane scipy matplotlib
```

Before running the notebook, check the fixture files:

```bash
python code/test_fixture_integrity.py
```

Expected output:

```text
PASS fixture metadata and arrays
```

## Run the Simulation

Interactive run:

```bash
jupyter lab code/main_skeleton.ipynb
```

Then run all cells from top to bottom.

Headless run:

```bash
jupyter nbconvert --to notebook --execute code/main_skeleton.ipynb --output main_skeleton_executed.ipynb --output-dir code --ExecutePreprocessor.timeout=-1
```

Use the unlimited timeout because full VQE/VQS regeneration can be slow.

## Workflow

`main_skeleton.ipynb` runs the project in this order:

1. Load dependencies and locate `code/`.
2. Define physics and numerical options.
3. Load or regenerate fixture data.
4. Run Module 1 VQE and Module 2 quench setup.
5. Validate the VQE/quench state before dynamics.
6. Run Module 3 Suzuki-Trotter dynamics.
7. Run Module 4 McLachlan VQS ensemble.
8. Plot fidelity, observables, and VQS/ED or Suzuki/ED ratios.

The main callable interfaces are:

```python
run_module1_from_config(Module1Config(...))
run_module2_from_config(Module2Config(...), theta_opt)
run_module3_from_config(Module3Config(...), psi_0)
module4_lib.run_vqs_evolution(...)
run_vqe_restart_ensemble(...)
```

## Core Options

All main options are in Section 2 of `code/main_skeleton.ipynb`.

### Physics

```python
PHYSICS_CONFIG = {
    "N": 4,
    "ag": 1.0,
    "m_over_g": 1.0,
    "q_initial": 0.0,
    "q_final": 2.0,
    "g": 1.0,
    "layer_count": 5,
}
```

These are the report-level defaults: 4 lattice sites/qubits, initial field `q=0`, post-quench field `q=2`, and HVA depth `L=5`.

### VQE

```python
VQE_REPRO_CONFIG = {
    "n_restarts": 10,
    "seed": 1234,
    "learning_rate": 0.05,
    "max_steps": 200,
    "grad_tol": 1e-4,
    "stall_window": 100,
    "stall_tol": 1e-9,
    "use_lbfgs_polish": False,
}
VQE_FIXTURE_CONFIG = {**VQE_REPRO_CONFIG, "n_restarts": 5, "max_steps": 50}
ACTIVE_VQE_CONFIG = VQE_REPRO_CONFIG
```

Use `VQE_REPRO_CONFIG` for final results. Use `VQE_FIXTURE_CONFIG` only for faster development runs.

### Module 1-3 Fixture Mode

```python
USE_TEST_DATA_INPUT = True
REGENERATE_TEST_DATA = False
```

- Fast reproduction: keep the defaults above to load existing Module 1-3 fixtures.
- Clean recomputation: set `USE_TEST_DATA_INPUT = False` and `REGENERATE_TEST_DATA = True`.

### Optional Layer Sweep

```python
RUN_LAYER_SWEEP = False
LAYER_SWEEP_L_VALUES = [1, 2, 3, 4, 5]
```

Set `RUN_LAYER_SWEEP = True` only when generating the optional VQE layer-depth distribution.

### Trotter Baseline

```python
TROTTER_CONFIG = {
    "total_time": 5.0,
    "n_steps": 100,
    "n_steps_scan": [10, 20, 40, 80, 160],
}
```

`n_steps` controls the main second-order Suzuki-Trotter trajectory. `n_steps_scan` is used for convergence checks.

### VQS and Module 4 Ensemble

```python
VQS_CONFIG = {
    "total_time": 5.0,
    "n_steps": 400,
    "n_steps_scan": [100, 200, 400, 800],
    "regularization": 1e-8,
    "use_projector": True,
}

RUN_MODULE4_ENSEMBLE = True
USE_MODULE4_FIXTURE = True
REGENERATE_MODULE4_DATA = False
MODULE4_LAYER_VALUES = [3, 4, 5]
MODULE4_DT_VALUES = [0.01, 0.02, 0.04]
MODULE4_FIXED_LAYER = 5
MODULE4_FIXED_DT = 0.01
MODULE4_TOTAL_TIME = 5.0
MODULE4_SAMPLE_COUNT = 20
```

- Fast reproduction: keep `USE_MODULE4_FIXTURE = True` and `REGENERATE_MODULE4_DATA = False`.
- Full regeneration: set `USE_MODULE4_FIXTURE = False` and `REGENERATE_MODULE4_DATA = True`.
- Development smoke run: reduce `MODULE4_SAMPLE_COUNT`, `MODULE4_LAYER_VALUES`, and `MODULE4_DT_VALUES`; do not use reduced settings for final figures.

## Outputs

The notebook displays plots inline. Regenerated fixture outputs are saved under:

```text
test_data/module1_vqe/
test_data/module2_quench/
test_data/module3_trotter/
test_data/module4_mclachlan/
```

Each fixture folder contains `metadata.json` plus compressed NumPy arrays. Metadata records the parameters and validation metrics used to create the fixture.

## Troubleshooting

- `Could not find code/module1_vqe.py`: run the notebook from `235_Final_Project/` or `235_Final_Project/code/`, and keep the folder layout unchanged.
- `Module 4 ensemble plots skipped`: provide the Module 4 fixture files or set `RUN_MODULE4_ENSEMBLE = True`.
- Validation assertion fails: restore the default configs first, then regenerate from Module 1 onward.
- Full regeneration is slow: use fixture mode for quick checks; reserve full regeneration for final verification.
