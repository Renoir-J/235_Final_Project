"""Fixture integrity checks for the relabeled Phys 235 modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as onp

ROOT = Path(__file__).parent.parent / "test_data"
EXPECTED = {
    "module1_vqe": {"theta_opt", "statevector", "exact_ground_state", "H_initial_matrix"},
    "module2_quench": {"theta_opt", "psi_0", "H_initial_matrix", "H_final_matrix"},
    "module3_trotter": {
        "times", "trotter_electric_field", "trotter_chiral_condensate", "trotter_charge",
        "exact_electric_field", "exact_chiral_condensate", "exact_charge", "fidelity",
        "conv_n_steps", "conv_final_fidelity", "conv_final_state_error",
    },
}
MODULE4_ENSEMBLE_KEYS = {
    "layer_3_times", "layer_3_fidelity", "layer_3_sample_energy", "layer_3_sample_r_E",
    "layer_4_times", "layer_4_fidelity", "layer_4_sample_energy", "layer_4_sample_r_E",
    "layer_5_times", "layer_5_fidelity", "layer_5_sample_energy", "layer_5_sample_r_E",
    "layer_5_suzuki_fidelity",
    "dt_0p01_times", "dt_0p01_fidelity", "dt_0p01_suzuki_fidelity",
    "dt_0p02_times", "dt_0p02_fidelity", "dt_0p02_suzuki_fidelity",
    "dt_0p04_times", "dt_0p04_fidelity", "dt_0p04_suzuki_fidelity",
    "obs_times",
    "obs_vqs_electric_field", "obs_ed_electric_field", "obs_suzuki_electric_field",
    "obs_vqs_chiral_condensate", "obs_ed_chiral_condensate", "obs_suzuki_chiral_condensate",
    "obs_ratio_vqs_electric_field", "obs_ratio_suzuki_electric_field",
    "obs_ratio_vqs_chiral_condensate", "obs_ratio_suzuki_chiral_condensate",
}
SCHEMA_ONLY_IF_PRESENT = {"module4_mclachlan": MODULE4_ENSEMBLE_KEYS}


def test_fixture_metadata_and_array_keys_match():
    for module_name, expected_keys in EXPECTED.items():
        module_dir = ROOT / module_name
        metadata = json.loads((module_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["schema_version"] == 2
        assert metadata["module"] == module_name
        assert set(metadata["array_keys"]) == expected_keys
        arrays = onp.load(module_dir / metadata["arrays_file"], allow_pickle=False)
        assert set(arrays.files) == expected_keys


def test_optional_fixture_metadata_and_array_keys_match():
    for module_name, expected_keys in SCHEMA_ONLY_IF_PRESENT.items():
        module_dir = ROOT / module_name
        metadata_path = module_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["schema_version"] == 2
        assert metadata["module"] == module_name
        arrays = onp.load(module_dir / metadata["arrays_file"], allow_pickle=False)
        assert set(metadata["array_keys"]) == set(arrays.files)
        assert set(arrays.files) == expected_keys


if __name__ == "__main__":
    test_fixture_metadata_and_array_keys_match()
    test_optional_fixture_metadata_and_array_keys_match()
    print("PASS fixture metadata and arrays")
