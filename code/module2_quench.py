"""Module 2: quench setup for post-VQE Schwinger dynamics.

This module takes optimized HVA parameters from Module 1, constructs the q=0
and q=2 Hamiltonians, and packages the quench-ready state and reference objects
used by the Trotter and McLachlan dynamics modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as onp
from numpy.typing import NDArray

from module1_vqe import hva_state
from schwinger_core import (
    SchwingerHamiltonian,
    build_schwinger_hamiltonian,
    commutator_norm,
    compute_observables,
    expectation,
    hamiltonian_matrix,
    total_charge_matrix,
)

__all__ = [
    "Module2Config",
    "Module2WorkflowResult",
    "QuenchSetup",
    "module2_acceptance_passed",
    "prepare_quench_state",
    "run_module2_from_config",
    "run_module2_workflow",
    "validate_module2_setup",
]


@dataclass
class QuenchSetup:
    psi_0: NDArray
    H_initial: SchwingerHamiltonian
    H_final: SchwingerHamiltonian
    H_initial_matrix: NDArray
    H_final_matrix: NDArray
    initial_energy_q0: float
    initial_energy_q2: float
    initial_variance_q2: float
    initial_observables_q2: dict[str, float]


@dataclass(frozen=True)
class Module2Config:
    """Stable public configuration for Module 2 quench setup."""

    N: int
    ag: float
    m_over_g: float
    q_initial: float
    q_final: float
    g: float
    layer_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def validate(self) -> None:
        if self.N < 2:
            raise ValueError("N must be at least 2.")
        if self.ag <= 0:
            raise ValueError("ag must be positive.")
        if self.g <= 0:
            raise ValueError("g must be positive.")
        if self.layer_count < 1:
            raise ValueError("layer_count must be at least 1.")


@dataclass
class Module2WorkflowResult:
    config: Module2Config
    quench: QuenchSetup
    validation: dict[str, float | bool]


def prepare_quench_state(
    lambda_opt: NDArray,
    layer_count: int,
    N: int,
    ag: float,
    m_over_g: float,
    q_initial: float,
    q_final: float,
    g: float,
) -> QuenchSetup:
    """Build the q=0 initial state and q=2 post-quench reference objects."""

    H_initial = build_schwinger_hamiltonian(
        N=N,
        ag=ag,
        m_over_g=m_over_g,
        external_field=q_initial,
        g=g,
    )
    H_final = build_schwinger_hamiltonian(
        N=N,
        ag=ag,
        m_over_g=m_over_g,
        external_field=q_final,
        g=g,
    )
    H_initial_matrix = hamiltonian_matrix(H_initial)
    H_final_matrix = hamiltonian_matrix(H_final)
    psi_0 = hva_state(lambda_opt, layer_count=layer_count, N=N)

    initial_energy_q0 = float(expectation(psi_0, H_initial_matrix).real)
    initial_energy_q2 = float(expectation(psi_0, H_final_matrix).real)
    H2_expectation = expectation(psi_0, H_final_matrix @ H_final_matrix).real
    initial_variance_q2 = float(H2_expectation - initial_energy_q2**2)

    return QuenchSetup(
        psi_0=psi_0,
        H_initial=H_initial,
        H_final=H_final,
        H_initial_matrix=H_initial_matrix,
        H_final_matrix=H_final_matrix,
        initial_energy_q0=initial_energy_q0,
        initial_energy_q2=initial_energy_q2,
        initial_variance_q2=initial_variance_q2,
        initial_observables_q2=compute_observables(
            psi_0,
            N=N,
            ag=ag,
            external_field=q_final,
            g=g,
        ),
    )


def validate_module2_setup(quench: QuenchSetup) -> dict[str, float | bool]:
    """Collect the visible checks required before moving to dynamics."""

    N = quench.H_initial.N
    charge = total_charge_matrix(N)
    return {
        "initial_energy_q0": quench.initial_energy_q0,
        "initial_energy_q2": quench.initial_energy_q2,
        "q2_energy_variance": quench.initial_variance_q2,
        "commutator_norm_q0": commutator_norm(quench.H_initial_matrix, charge),
        "commutator_norm_q2": commutator_norm(quench.H_final_matrix, charge),
    }


def module2_acceptance_passed(validation: dict[str, float | bool]) -> bool:
    """Return True when Module 2 is ready for dynamics modules."""

    charge_ok = float(validation.get("commutator_norm_q0", onp.inf)) < 1e-8
    charge_ok = charge_ok and float(validation.get("commutator_norm_q2", onp.inf)) < 1e-8
    quench_ok = float(validation.get("q2_energy_variance", 0.0)) > 1e-10
    return charge_ok and quench_ok


def run_module2_from_config(config: Module2Config, theta_opt: NDArray) -> Module2WorkflowResult:
    """Run the full Module 2 quench setup from a stable config and VQE parameters."""

    config.validate()
    quench = prepare_quench_state(
        lambda_opt=theta_opt,
        layer_count=config.layer_count,
        N=config.N,
        ag=config.ag,
        m_over_g=config.m_over_g,
        q_initial=config.q_initial,
        q_final=config.q_final,
        g=config.g,
    )
    validation = validate_module2_setup(quench)
    return Module2WorkflowResult(config=config, quench=quench, validation=validation)


def run_module2_workflow(
    N: int,
    ag: float,
    m_over_g: float,
    q_initial: float,
    q_final: float,
    g: float,
    layer_count: int,
    theta_opt: NDArray,
) -> tuple[QuenchSetup, dict[str, float | bool]]:
    """Tuple workflow interface for notebook cells that do not need config objects."""

    result = run_module2_from_config(
        Module2Config(
            N=N,
            ag=ag,
            m_over_g=m_over_g,
            q_initial=q_initial,
            q_final=q_final,
            g=g,
            layer_count=layer_count,
        ),
        theta_opt=theta_opt,
    )
    return result.quench, result.validation
