"""Tests for Module 5 (Suzuki-Trotter baseline).

Run with `python test_module_5_trotter.py` (no extra deps) or `pytest`.
"""

from __future__ import annotations

import os

import numpy as onp
import scipy.linalg

import module_4_vqe_quench as m4
import module_5_trotter as m5

N, AG, MG, QF, G = 4, 1.0, 1.0, 2.0, 1.0
H = m4.build_schwinger_hamiltonian(N=N, ag=AG, m_over_g=MG, external_field=QF, g=G)
HF = m4.hamiltonian_matrix(H)

_HERE = os.path.dirname(os.path.abspath(__file__))
PSI0 = onp.load(os.path.join(_HERE, "..", "test_data", "module_4", "arrays.npz"))["psi_0"]


def test_split_partitions_all_terms():
    diag, hop = m5.split_hamiltonian_terms(H)
    assert len(diag) == 8 and len(hop) == 6
    assert len(diag) + len(hop) == len(H.terms)
    assert all(set(t.word) <= {"I", "Z"} for t in diag)
    assert all(not (set(t.word) <= {"I", "Z"}) for t in hop)


def test_group_matrices_reconstruct_full_hamiltonian():
    diag, hop = m5.split_hamiltonian_terms(H)
    h_diag = m5.group_matrix(H, diag)
    h_hop = m5.group_matrix(H, hop)
    assert onp.linalg.norm(h_diag + h_hop - HF) < 1e-10
    assert onp.linalg.norm(h_diag - onp.diag(onp.diag(h_diag))) < 1e-12


def _groups():
    diag, hop = m5.split_hamiltonian_terms(H)
    return m5.group_matrix(H, diag), m5.group_matrix(H, hop)


def test_trotter_step_is_unitary():
    h_diag, h_hop = _groups()
    step = m5.second_order_trotter_step(h_diag, h_hop, 0.1)
    identity = onp.eye(step.shape[0])
    assert onp.linalg.norm(step.conj().T @ step - identity) < 1e-10


def test_trotter_evolve_shapes_and_norm():
    h_diag, h_hop = _groups()
    times, states = m5.trotter_evolve(PSI0, h_diag, h_hop, 2.0, 50)
    assert times.shape == (51,)
    assert states.shape == (51, 16)
    assert onp.allclose(onp.linalg.norm(states, axis=1), 1.0, atol=1e-8)
    assert onp.allclose(states[0], PSI0.astype(complex))


def test_trotter_matches_exact_for_fine_steps():
    h_diag, h_hop = _groups()
    _, states = m5.trotter_evolve(PSI0, h_diag, h_hop, 3.0, 200)
    psi_exact = scipy.linalg.expm(-1.0j * HF * 3.0) @ PSI0
    assert m4.state_fidelity(psi_exact, states[-1]) > 0.9999


def test_observable_trajectory_matches_compute_observables_at_t0():
    h_diag, h_hop = _groups()
    times, states = m5.trotter_evolve(PSI0, h_diag, h_hop, 1.0, 10)
    traj = m5.observable_trajectory(times, states, N, AG, QF, G)
    obs0 = m4.compute_observables(PSI0, N=N, ag=AG, external_field=QF, g=G)
    assert traj.electric_field.shape == (11,)
    assert traj.chiral_condensate.shape == (11,)
    assert traj.charge.shape == (11,)
    assert abs(traj.electric_field[0] - obs0["electric_field"]) < 1e-12
    assert abs(traj.chiral_condensate[0] - obs0["chiral_condensate"]) < 1e-12


def test_fidelity_series_starts_at_one():
    h_diag, h_hop = _groups()
    times, trotter_states = m5.trotter_evolve(PSI0, h_diag, h_hop, 1.0, 10)
    exact_states = m4.exact_time_evolution(HF, PSI0, times)
    fidelity = m5.fidelity_series(exact_states, trotter_states)
    assert fidelity.shape == (11,)
    assert abs(fidelity[0] - 1.0) < 1e-10
    assert onp.all(fidelity <= 1.0 + 1e-9)


def test_convergence_monotonic_and_second_order():
    h_diag, h_hop = _groups()
    conv = m5.run_trotter_convergence(PSI0, h_diag, h_hop, HF, 3.0, (10, 20, 40, 80))
    assert conv.n_steps_values == (10, 20, 40, 80)
    assert conv.final_fidelity.shape == (4,)
    assert onp.all(onp.diff(conv.final_fidelity) >= -1e-12)       # fidelity improves
    assert onp.all(onp.diff(conv.final_state_error) <= 1e-12)     # error decreases
    assert 1.8 < conv.order_estimate < 2.2                         # second order


def test_config_to_dict_roundtrips_scan_as_list():
    cfg = m5.Module5Config(N=4, ag=1.0, m_over_g=1.0, q_final=2.0, g=1.0,
                           total_time=3.0, n_steps=100, n_steps_scan=(10, 20, 40))
    data = cfg.to_dict()
    assert data["n_steps_scan"] == [10, 20, 40]
    assert data["total_time"] == 3.0


def test_config_validate_rejects_bad_input():
    base = dict(N=4, ag=1.0, m_over_g=1.0, q_final=2.0, g=1.0,
                total_time=3.0, n_steps=100, n_steps_scan=(10, 20))
    for override in ({"total_time": -1.0}, {"n_steps": 0}, {"n_steps_scan": (10,)}, {"N": 1}):
        bad = m5.Module5Config(**{**base, **override})
        raised = False
        try:
            bad.validate()
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {override}"


def test_validate_module5_setup_keys():
    h_diag, h_hop = _groups()
    times, trotter_states = m5.trotter_evolve(PSI0, h_diag, h_hop, 3.0, 80)
    exact_states = m4.exact_time_evolution(HF, PSI0, times)
    fidelity = m5.fidelity_series(exact_states, trotter_states)
    conv = m5.run_trotter_convergence(PSI0, h_diag, h_hop, HF, 3.0, (10, 20, 40, 80))
    validation = m5.validate_module5_setup(0.0, fidelity, conv)
    assert set(validation) == {
        "split_sum_error", "final_fidelity", "min_fidelity",
        "fidelity_monotonic", "trotter_order_estimate",
    }
    assert validation["fidelity_monotonic"] is True
    assert validation["final_fidelity"] > 0.99


def test_acceptance_gate_logic():
    good = {"split_sum_error": 0.0, "final_fidelity": 0.9999, "min_fidelity": 0.999,
            "fidelity_monotonic": True, "trotter_order_estimate": 2.0}
    assert m5.module5_acceptance_passed(good)
    assert not m5.module5_acceptance_passed({**good, "final_fidelity": 0.5})
    assert not m5.module5_acceptance_passed({**good, "fidelity_monotonic": False})
    assert not m5.module5_acceptance_passed({**good, "split_sum_error": 1.0})
    assert not m5.module5_acceptance_passed({**good, "trotter_order_estimate": 1.0})


def test_run_module5_from_config_end_to_end():
    cfg = m5.Module5Config(N=4, ag=1.0, m_over_g=1.0, q_final=2.0, g=1.0,
                           total_time=3.0, n_steps=100, n_steps_scan=(10, 20, 40, 80, 160))
    result = m5.run_module5_from_config(cfg, PSI0)
    assert result.fidelity.shape == (101,)
    assert result.trotter.electric_field.shape == (101,)
    assert result.exact.electric_field.shape == (101,)
    assert result.split_sum_error < 1e-10
    assert m5.module5_acceptance_passed(result.validation)
    assert result.validation["final_fidelity"] > 0.99


def test_run_module5_workflow_tuple_interface():
    out = m5.run_module5_workflow(
        N=4, ag=1.0, m_over_g=1.0, q_final=2.0, g=1.0,
        total_time=3.0, n_steps=80, n_steps_scan=(10, 20, 40, 80), psi_0=PSI0,
    )
    trotter, exact, fidelity, convergence, validation = out
    assert fidelity.shape == (81,)
    assert convergence.order_estimate > 1.8
    assert m5.module5_acceptance_passed(validation)


if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL", fn.__name__, exc)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR", fn.__name__, repr(exc))
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
