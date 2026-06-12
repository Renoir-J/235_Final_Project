"""Module 3: second-order Suzuki-Trotter baseline for the post-quench dynamics.

Reproduces target 2 of the project proposal: evolve the Module 2 quench-ready
state |psi_0> under the post-quench Schwinger Hamiltonian H(q=2) with a
second-order Suzuki-Trotter (Strang) product formula, and compare observables
and fidelity against the exact scipy.linalg.expm reference from schwinger_core.

The Hamiltonian is split into two non-commuting groups H = H_diag + H_hop, where
H_diag collects the Pauli terms diagonal in the computational basis (words over
{I, Z}: the electric J-term and the staggered mass term) and H_hop collects the
hopping terms (words containing X or Y). Each group is exponentiated exactly; the
only error is the Trotter splitting error between the two groups, which vanishes
as O(dt^2) globally for the symmetric Strang step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as onp
from numpy.typing import NDArray
import scipy.linalg

from schwinger_core import (
    ObservableTrajectory,
    PauliTerm,
    SchwingerHamiltonian,
    build_schwinger_hamiltonian,
    exact_time_evolution,
    fidelity_series,
    hamiltonian_matrix,
    observable_trajectory,
    state_fidelity,
)

__all__ = [
    "Module3Config",
    "Module3WorkflowResult",
    "TrotterConvergence",
    "split_hamiltonian_terms",
    "group_matrix",
    "second_order_trotter_step",
    "trotter_evolve",
    "run_trotter_convergence",
    "validate_module3_setup",
    "module3_acceptance_passed",
    "run_module3_from_config",
    "run_module3_workflow",
]


def split_hamiltonian_terms(
    hamiltonian: SchwingerHamiltonian,
) -> tuple[list[PauliTerm], list[PauliTerm]]:
    """Partition terms into (diagonal over {I, Z}, hopping with X/Y)."""

    diagonal: list[PauliTerm] = []
    hopping: list[PauliTerm] = []
    for term in hamiltonian.terms:
        if set(term.word) <= {"I", "Z"}:
            diagonal.append(term)
        else:
            hopping.append(term)
    return diagonal, hopping


def group_matrix(hamiltonian: SchwingerHamiltonian, terms: list[PauliTerm]) -> NDArray:
    """Dense matrix for a subset of terms, reusing schwinger_core.hamiltonian_matrix."""

    return hamiltonian_matrix(replace(hamiltonian, terms=terms))


def second_order_trotter_step(H_diag: NDArray, H_hop: NDArray, dt: float) -> NDArray:
    """Symmetric Strang step exp(-i H_diag dt/2) exp(-i H_hop dt) exp(-i H_diag dt/2)."""

    half = scipy.linalg.expm(-0.5j * dt * H_diag)
    full = scipy.linalg.expm(-1.0j * dt * H_hop)
    return half @ full @ half


def trotter_evolve(
    psi_0: NDArray,
    H_diag: NDArray,
    H_hop: NDArray,
    total_time: float,
    n_steps: int,
) -> tuple[NDArray, NDArray]:
    """Evolve psi_0 with n_steps Strang steps; return (times, states) over [0, total_time]."""

    dt = total_time / n_steps
    step = second_order_trotter_step(H_diag, H_hop, dt)
    states = onp.empty((n_steps + 1, psi_0.shape[0]), dtype=complex)
    states[0] = onp.asarray(psi_0, dtype=complex)
    for k in range(n_steps):
        states[k + 1] = step @ states[k]
    times = onp.linspace(0.0, total_time, n_steps + 1)
    return times, states


@dataclass
class TrotterConvergence:
    n_steps_values: tuple[int, ...]
    final_fidelity: NDArray
    final_state_error: NDArray
    order_estimate: float


def run_trotter_convergence(
    psi_0: NDArray,
    H_diag: NDArray,
    H_hop: NDArray,
    H_final_matrix: NDArray,
    total_time: float,
    n_steps_scan: Iterable[int],
) -> TrotterConvergence:
    """Compare Trotter vs exact at t=total_time across increasing n_steps.

    Returns final-time fidelity F(T) and state error ||psi_trot(T) - psi_exact(T)||
    per n_steps, plus the fitted convergence order (expected ~2 for the symmetric
    Strang step). The exact final state is computed once and reused.
    """

    n_values = tuple(sorted(int(n) for n in n_steps_scan))
    psi_exact_T = exact_time_evolution(H_final_matrix, psi_0, onp.array([total_time]))[0]
    final_fidelity = onp.empty(len(n_values))
    final_state_error = onp.empty(len(n_values))
    for idx, n in enumerate(n_values):
        _, states = trotter_evolve(psi_0, H_diag, H_hop, total_time, n)
        psi_trot_T = states[-1]
        final_fidelity[idx] = state_fidelity(psi_exact_T, psi_trot_T)
        final_state_error[idx] = float(onp.linalg.norm(psi_trot_T - psi_exact_T))
    slope = onp.polyfit(onp.log(onp.asarray(n_values, dtype=float)), onp.log(final_state_error), 1)[0]
    return TrotterConvergence(
        n_steps_values=n_values,
        final_fidelity=final_fidelity,
        final_state_error=final_state_error,
        order_estimate=float(-slope),
    )


@dataclass(frozen=True)
class Module3Config:
    """Stable public configuration for the Module 3 Trotter baseline.

    Concrete parameter choices belong in main_skeleton.ipynb, not here.
    """

    N: int
    ag: float
    m_over_g: float
    q_final: float
    g: float
    total_time: float
    n_steps: int
    n_steps_scan: tuple[int, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["n_steps_scan"] = list(self.n_steps_scan)
        return data

    def validate(self) -> None:
        if self.N < 2:
            raise ValueError("N must be at least 2.")
        if self.total_time <= 0:
            raise ValueError("total_time must be positive.")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1.")
        if len(self.n_steps_scan) < 2:
            raise ValueError("n_steps_scan needs at least two values for an order fit.")


@dataclass
class Module3WorkflowResult:
    config: Module3Config
    trotter: ObservableTrajectory
    exact: ObservableTrajectory
    fidelity: NDArray
    convergence: TrotterConvergence
    split_sum_error: float
    validation: dict


def validate_module3_setup(
    split_sum_error: float,
    fidelity: NDArray,
    convergence: TrotterConvergence,
) -> dict:
    """Collect the physics-meaningful Module 3 checks."""

    final_values = convergence.final_fidelity
    monotonic = bool(onp.all(onp.diff(final_values) >= -1e-12))
    return {
        "split_sum_error": float(split_sum_error),
        "final_fidelity": float(fidelity[-1]),
        "min_fidelity": float(onp.min(fidelity)),
        "fidelity_monotonic": monotonic,
        "trotter_order_estimate": float(convergence.order_estimate),
    }


def module3_acceptance_passed(validation: dict, required_fidelity: float = 0.99) -> bool:
    """Return True when the Trotter baseline is ready for downstream comparison."""

    split_ok = float(validation.get("split_sum_error", onp.inf)) < 1e-9
    fidelity_ok = float(validation.get("final_fidelity", -onp.inf)) > required_fidelity
    monotonic_ok = bool(validation.get("fidelity_monotonic", False))
    order_ok = float(validation.get("trotter_order_estimate", 0.0)) > 1.8
    return split_ok and fidelity_ok and monotonic_ok and order_ok


def run_module3_from_config(config: Module3Config, psi_0: NDArray) -> Module3WorkflowResult:
    """Run the full Module 3 Trotter baseline from a stable config and initial state."""

    config.validate()
    hamiltonian = build_schwinger_hamiltonian(
        N=config.N,
        ag=config.ag,
        m_over_g=config.m_over_g,
        external_field=config.q_final,
        g=config.g,
    )
    H_final_matrix = hamiltonian_matrix(hamiltonian)
    diagonal_terms, hopping_terms = split_hamiltonian_terms(hamiltonian)
    H_diag = group_matrix(hamiltonian, diagonal_terms)
    H_hop = group_matrix(hamiltonian, hopping_terms)
    split_sum_error = float(onp.linalg.norm(H_diag + H_hop - H_final_matrix))

    psi_0 = onp.asarray(psi_0, dtype=complex)
    times, trotter_states = trotter_evolve(psi_0, H_diag, H_hop, config.total_time, config.n_steps)
    exact_states = exact_time_evolution(H_final_matrix, psi_0, times)

    trotter = observable_trajectory(times, trotter_states, config.N, config.ag, config.q_final, config.g)
    exact = observable_trajectory(times, exact_states, config.N, config.ag, config.q_final, config.g)
    fidelity = fidelity_series(exact_states, trotter_states)
    convergence = run_trotter_convergence(
        psi_0, H_diag, H_hop, H_final_matrix, config.total_time, config.n_steps_scan
    )
    validation = validate_module3_setup(split_sum_error, fidelity, convergence)

    return Module3WorkflowResult(
        config=config,
        trotter=trotter,
        exact=exact,
        fidelity=fidelity,
        convergence=convergence,
        split_sum_error=split_sum_error,
        validation=validation,
    )


def run_module3_workflow(
    N: int,
    ag: float,
    m_over_g: float,
    q_final: float,
    g: float,
    total_time: float,
    n_steps: int,
    n_steps_scan: Iterable[int],
    psi_0: NDArray,
) -> tuple[ObservableTrajectory, ObservableTrajectory, NDArray, TrotterConvergence, dict]:
    """Tuple workflow interface for notebook cells that do not need config objects."""

    result = run_module3_from_config(
        Module3Config(
            N=N,
            ag=ag,
            m_over_g=m_over_g,
            q_final=q_final,
            g=g,
            total_time=total_time,
            n_steps=n_steps,
            n_steps_scan=tuple(n_steps_scan),
        ),
        psi_0=psi_0,
    )
    return result.trotter, result.exact, result.fidelity, result.convergence, result.validation
