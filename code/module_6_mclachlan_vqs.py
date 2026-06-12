"""Module 6: McLachlan variational quantum simulation (VQS) of the post-quench dynamics.

Reproduces targets 3 and 4 of the project proposal: evolve the Module 4 HVA
parameters theta(t) (starting from the VQE optimum theta_opt) under the
post-quench Schwinger Hamiltonian H(q=2) by solving the projected McLachlan
equation of motion M(theta) theta_dot = V(theta) with a regularized Euler step,
then compare observables and fidelity against the exact scipy.linalg.expm
reference (and, in the notebook, the Module 5 Trotter baseline).

The variational manifold is the same paper HVA used for VQE, so the evolution
stays at constant circuit depth. All linear algebra runs classically on dense
16-dim statevectors. The state |psi(theta)> and the tangent vectors
|d_i psi> = d|psi>/d theta_i are built in closed form from the HVA gate sequence
(each gate is a Pauli rotation exp(-i phi P / 2) = cos(phi/2) I - i sin(phi/2) P,
with exact derivative (-i P / 2) times the gate). The metric and force use the
phase-projected (Q_psi) form derived in vqs_notes.tex, which tracks the exact
evolution where the bare form fails for this real-parameterized ansatz.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as onp
from numpy.typing import NDArray

from module_4_vqe_quench import (
    build_schwinger_hamiltonian,
    exact_time_evolution,
    hamiltonian_matrix,
    state_fidelity,
    total_charge_matrix,
)
from module_5_trotter import (
    ObservableTrajectory,
    fidelity_series,
    observable_trajectory,
)

__all__ = [
    "Module6Config",
    "Module6WorkflowResult",
    "VQSTrajectory",
    "VQSConvergence",
    "hva_tangent_vectors",
    "mclachlan_matrices",
    "regularized_solve",
    "mclachlan_residual",
    "mclachlan_step",
    "run_vqs_evolution",
    "run_vqs_convergence",
    "charge_drift",
    "validate_module6_setup",
    "module6_acceptance_passed",
    "run_module6_from_config",
    "run_module6_workflow",
]


# These Pauli/link helpers intentionally duplicate small pieces of module_4 (its
# PAULI_MATS, pauli_word_matrix, _even_then_odd_links) so this module depends only
# on module_4's public, documented interface and never on its private symbols. The
# duplication is ~25 trivial, tested lines; the decoupling is worth it.
_PAULI = {
    "I": onp.eye(2, dtype=complex),
    "X": onp.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": onp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": onp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _pauli_word_matrix(letters: tuple[str, ...]) -> NDArray:
    matrix = _PAULI[letters[0]]
    for letter in letters[1:]:
        matrix = onp.kron(matrix, _PAULI[letter])
    return matrix


def _word(N: int, entries: list[tuple[int, str]]) -> NDArray:
    letters = ["I"] * N
    for wire, pauli in entries:
        letters[wire] = pauli
    return _pauli_word_matrix(tuple(letters))


def _even_then_odd_links(N: int) -> list[int]:
    return [n for n in range(N - 1) if n % 2 == 0] + [n for n in range(N - 1) if n % 2 == 1]


def _hva_gate_specs(N: int, layer_count: int) -> list[tuple[NDArray, int, float]]:
    """Ordered (generator P, flat-parameter index, dphi/dtheta) for every HVA gate.

    Mirrors module_4_vqe_quench.apply_hva_ansatz exactly. phi for each gate equals
    (dphi/dtheta) * theta[index], so the gate is reconstructed from theta alone. The
    list is in circuit-application order (spec 0 acts first on |init>), which the
    forward/suffix products in hva_tangent_vectors rely on. Note the XX and YY gates
    of a layer's link share one alpha index, so two specs carry the same index.
    """

    links = _even_then_odd_links(N)
    n_links = N - 1
    specs: list[tuple[NDArray, int, float]] = []
    for layer in range(layer_count):
        for n in links:
            alpha_idx = layer * n_links + n
            specs.append((_word(N, [(n, "X"), (n + 1, "X")]), alpha_idx, -0.5))
            specs.append((_word(N, [(n, "Y"), (n + 1, "Y")]), alpha_idx, -0.5))
        for n in links:
            beta_idx = layer_count * n_links + layer * n_links + n
            specs.append((_word(N, [(n, "Z"), (n + 1, "Z")]), beta_idx, -1.0))
        for n in range(N):
            gamma_idx = 2 * layer_count * n_links + layer * N + n
            specs.append((_word(N, [(n, "Z")]), gamma_idx, -1.0))
    return specs


def _init_state(N: int) -> NDArray:
    state = onp.zeros(2**N, dtype=complex)
    state[0] = 1.0
    for wire in range(0, N, 2):
        state = _word(N, [(wire, "X")]) @ state
    return state


def hva_tangent_vectors(theta: NDArray, layer_count: int, N: int) -> tuple[NDArray, NDArray]:
    """Return (|psi(theta)>, tangents) where tangents[i] = d|psi>/d theta_i.

    Closed-form, exact, and consistent with module_4's apply_hva_ansatz. Each gate
    G = cos(phi/2) I - i sin(phi/2) P has derivative dG/dphi = (-i P / 2) G.
    """

    theta = onp.asarray(theta, dtype=float)
    dim = 2**N
    specs = _hva_gate_specs(N, layer_count)
    identity = onp.eye(dim, dtype=complex)

    mats: list[NDArray] = []
    for generator, index, slope in specs:
        phi = slope * theta[index]
        mats.append(onp.cos(phi / 2.0) * identity - 1j * onp.sin(phi / 2.0) * generator)

    forward = [_init_state(N)]
    for gate in mats:
        forward.append(gate @ forward[-1])
    psi = forward[-1]

    n_gates = len(mats)
    suffix = [identity] * (n_gates + 1)
    running = identity
    for j in range(n_gates - 1, -1, -1):
        running = running @ mats[j]
        suffix[j] = running

    tangents = onp.zeros((theta.shape[0], dim), dtype=complex)
    for j, (generator, index, slope) in enumerate(specs):
        contribution = suffix[j + 1] @ ((-0.5j) * (generator @ forward[j + 1]))
        tangents[index] += slope * contribution
    return psi, tangents


def mclachlan_matrices(
    psi: NDArray,
    dpsi: NDArray,
    H_matrix: NDArray,
    projected: bool = True,
) -> tuple[NDArray, NDArray, float]:
    """Build the McLachlan metric M, force V, and energy <H>.

    projected=True  -> M_ij = Re<d_i psi|Q_psi|d_j psi>, V_i = Im<d_i psi|(H-<H>)|psi>
    projected=False -> bare A_ij = Re<d_i psi|d_j psi>,  C_i = Im<d_i psi|H|psi>
    """

    n_params = dpsi.shape[0]
    overlap = onp.array([onp.vdot(dpsi[i], psi) for i in range(n_params)])  # <d_i|psi>
    gram = dpsi.conj() @ dpsi.T                                             # <d_i|d_j>
    h_psi = H_matrix @ psi
    energy = float(onp.vdot(psi, h_psi).real)
    grad_h = onp.array([onp.vdot(dpsi[i], h_psi) for i in range(n_params)])  # <d_i|H|psi>

    if projected:
        metric = (gram - onp.outer(overlap, overlap.conj())).real
        force = (grad_h - energy * overlap).imag
    else:
        metric = gram.real
        force = grad_h.imag
    return onp.asarray(metric, dtype=float), onp.asarray(force, dtype=float), energy


def regularized_solve(
    M: NDArray,
    V: NDArray,
    regularization: float = 1e-8,
    gated: bool = True,
) -> NDArray:
    """Solve M theta_dot = V with a (determinant-gated) ridge for stability.

    gated=True adds regularization*I only when |det M| < regularization (the
    notes' determinant-gated ridge); gated=False always adds it.
    """

    matrix = onp.asarray(M, dtype=float)
    if gated:
        if abs(onp.linalg.det(matrix)) < regularization:
            matrix = matrix + regularization * onp.eye(matrix.shape[0])
    else:
        matrix = matrix + regularization * onp.eye(matrix.shape[0])
    try:
        return onp.linalg.solve(matrix, V)
    except onp.linalg.LinAlgError:
        return onp.linalg.lstsq(matrix, V, rcond=None)[0]


def mclachlan_residual(
    psi: NDArray,
    V: NDArray,
    theta_dot: NDArray,
    H_matrix: NDArray,
    energy: float,
) -> float:
    """McLachlan distance L_min = Var(H) - V^T theta_dot (projected form)."""

    h2 = float(onp.vdot(psi, H_matrix @ (H_matrix @ psi)).real)
    variance = h2 - energy**2
    return float(variance - V @ theta_dot)


def mclachlan_step(
    theta: NDArray,
    H_matrix: NDArray,
    layer_count: int,
    N: int,
    dt: float,
    regularization: float = 1e-8,
    projected: bool = True,
) -> tuple[NDArray, float]:
    """One Euler step of the projected McLachlan EOM; returns (theta_next, residual)."""

    psi, dpsi = hva_tangent_vectors(theta, layer_count, N)
    M, V, energy = mclachlan_matrices(psi, dpsi, H_matrix, projected=projected)
    theta_dot = regularized_solve(M, V, regularization)
    residual = mclachlan_residual(psi, V, theta_dot, H_matrix, energy)
    return onp.asarray(theta, dtype=float) + dt * theta_dot, residual


@dataclass
class VQSTrajectory:
    times: NDArray
    states: NDArray
    params: NDArray
    residual: NDArray


def run_vqs_evolution(
    theta_0: NDArray,
    H_matrix: NDArray,
    layer_count: int,
    N: int,
    total_time: float,
    n_steps: int,
    regularization: float = 1e-8,
    projected: bool = True,
) -> VQSTrajectory:
    """Integrate the projected McLachlan EOM with n_steps Euler steps."""

    dt = total_time / n_steps
    theta = onp.asarray(theta_0, dtype=float).copy()
    dim = 2**N
    states = onp.empty((n_steps + 1, dim), dtype=complex)
    params = onp.empty((n_steps + 1, theta.shape[0]), dtype=float)
    residual = onp.empty(n_steps + 1)

    states[0] = hva_tangent_vectors(theta, layer_count, N)[0]
    params[0] = theta
    for k in range(n_steps):
        theta, residual[k] = mclachlan_step(
            theta, H_matrix, layer_count, N, dt, regularization, projected
        )
        params[k + 1] = theta
        states[k + 1] = hva_tangent_vectors(theta, layer_count, N)[0]
    residual[-1] = residual[-2] if n_steps > 0 else 0.0
    times = onp.linspace(0.0, total_time, n_steps + 1)
    return VQSTrajectory(times=times, states=states, params=params, residual=residual)


def charge_drift(states: NDArray, N: int) -> float:
    """Max deviation of <Q(t)> from <Q(0)> along a trajectory, Q = (1/N) sum_n Z_n."""

    charge_op = total_charge_matrix(N)
    values = onp.array([onp.vdot(state, charge_op @ state).real for state in states])
    return float(onp.max(onp.abs(values - values[0])))


@dataclass
class VQSConvergence:
    n_steps_values: tuple[int, ...]
    final_fidelity: NDArray
    final_infidelity: NDArray
    order_estimate: float


def run_vqs_convergence(
    theta_0: NDArray,
    H_matrix: NDArray,
    psi_exact_T: NDArray,
    layer_count: int,
    N: int,
    total_time: float,
    n_steps_scan: Iterable[int],
    regularization: float = 1e-8,
    projected: bool = True,
) -> VQSConvergence:
    """Final-time fidelity and infidelity vs n_steps, plus the fitted infidelity order.

    Euler is first order in the state, so the phase-invariant infidelity 1 - F(T)
    scales as ~dt^2 -> order ~2. The exact final state is supplied once and reused.
    """

    n_values = tuple(sorted(int(n) for n in n_steps_scan))
    final_fidelity = onp.empty(len(n_values))
    final_infidelity = onp.empty(len(n_values))
    for idx, n in enumerate(n_values):
        traj = run_vqs_evolution(
            theta_0, H_matrix, layer_count, N, total_time, n, regularization, projected
        )
        fidelity = state_fidelity(psi_exact_T, traj.states[-1])
        final_fidelity[idx] = fidelity
        final_infidelity[idx] = max(1.0 - fidelity, 1e-15)
    slope = onp.polyfit(
        onp.log(onp.asarray(n_values, dtype=float)), onp.log(final_infidelity), 1
    )[0]
    return VQSConvergence(
        n_steps_values=n_values,
        final_fidelity=final_fidelity,
        final_infidelity=final_infidelity,
        order_estimate=float(-slope),
    )


@dataclass(frozen=True)
class Module6Config:
    """Stable public configuration for the Module 6 McLachlan VQS baseline.

    Concrete parameter choices belong in main_skeleton.ipynb, not here.
    """

    N: int
    ag: float
    m_over_g: float
    q_final: float
    g: float
    layer_count: int
    total_time: float
    n_steps: int
    n_steps_scan: tuple[int, ...]
    regularization: float
    use_projector: bool

    def to_dict(self) -> dict:
        data = asdict(self)
        data["n_steps_scan"] = list(self.n_steps_scan)
        return data

    def validate(self) -> None:
        if self.N < 2:
            raise ValueError("N must be at least 2.")
        if self.layer_count < 1:
            raise ValueError("layer_count must be at least 1.")
        if self.total_time <= 0:
            raise ValueError("total_time must be positive.")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1.")
        if len(self.n_steps_scan) < 2:
            raise ValueError("n_steps_scan needs at least two values for an order fit.")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive.")


@dataclass
class Module6WorkflowResult:
    config: Module6Config
    trajectory: VQSTrajectory
    vqs: ObservableTrajectory
    exact: ObservableTrajectory
    fidelity: NDArray
    convergence: VQSConvergence
    validation: dict


def validate_module6_setup(
    fidelity: NDArray,
    convergence: VQSConvergence,
    charge_drift_value: float,
    residual: NDArray,
) -> dict:
    """Collect the physics-meaningful Module 6 checks."""

    return {
        "final_fidelity": float(fidelity[-1]),
        "min_fidelity": float(onp.min(fidelity)),
        "charge_drift": float(charge_drift_value),
        "max_residual": float(onp.max(residual)),
        "infidelity_order_estimate": float(convergence.order_estimate),
    }


def module6_acceptance_passed(validation: dict, required_fidelity: float = 0.99) -> bool:
    """Return True when the VQS evolution is a faithful reproduction baseline."""

    fidelity_ok = float(validation.get("final_fidelity", -onp.inf)) > required_fidelity
    charge_ok = float(validation.get("charge_drift", onp.inf)) < 1e-8
    order_ok = float(validation.get("infidelity_order_estimate", 0.0)) > 1.5
    return fidelity_ok and charge_ok and order_ok


def run_module6_from_config(config: Module6Config, theta_0: NDArray) -> Module6WorkflowResult:
    """Run the full Module 6 McLachlan VQS baseline from a stable config and initial params."""

    config.validate()
    hamiltonian = build_schwinger_hamiltonian(
        N=config.N,
        ag=config.ag,
        m_over_g=config.m_over_g,
        external_field=config.q_final,
        g=config.g,
    )
    H_matrix = hamiltonian_matrix(hamiltonian)

    theta_0 = onp.asarray(theta_0, dtype=float)
    psi_0 = hva_tangent_vectors(theta_0, config.layer_count, config.N)[0]

    trajectory = run_vqs_evolution(
        theta_0, H_matrix, config.layer_count, config.N,
        config.total_time, config.n_steps, config.regularization, config.use_projector,
    )
    exact_states = exact_time_evolution(H_matrix, psi_0, trajectory.times)

    vqs = observable_trajectory(
        trajectory.times, trajectory.states, config.N, config.ag, config.q_final, config.g
    )
    exact = observable_trajectory(
        trajectory.times, exact_states, config.N, config.ag, config.q_final, config.g
    )
    fidelity = fidelity_series(exact_states, trajectory.states)
    convergence = run_vqs_convergence(
        theta_0, H_matrix, exact_states[-1], config.layer_count, config.N,
        config.total_time, config.n_steps_scan, config.regularization, config.use_projector,
    )
    drift = charge_drift(trajectory.states, config.N)
    validation = validate_module6_setup(fidelity, convergence, drift, trajectory.residual)

    return Module6WorkflowResult(
        config=config,
        trajectory=trajectory,
        vqs=vqs,
        exact=exact,
        fidelity=fidelity,
        convergence=convergence,
        validation=validation,
    )


def run_module6_workflow(
    N: int,
    ag: float,
    m_over_g: float,
    q_final: float,
    g: float,
    layer_count: int,
    total_time: float,
    n_steps: int,
    n_steps_scan: Iterable[int],
    regularization: float,
    use_projector: bool,
    theta_0: NDArray,
) -> tuple[VQSTrajectory, ObservableTrajectory, ObservableTrajectory, NDArray, VQSConvergence, dict]:
    """Tuple workflow interface for notebook cells that do not need config objects."""

    result = run_module6_from_config(
        Module6Config(
            N=N,
            ag=ag,
            m_over_g=m_over_g,
            q_final=q_final,
            g=g,
            layer_count=layer_count,
            total_time=total_time,
            n_steps=n_steps,
            n_steps_scan=tuple(n_steps_scan),
            regularization=regularization,
            use_projector=use_projector,
        ),
        theta_0=theta_0,
    )
    return (
        result.trajectory,
        result.vqs,
        result.exact,
        result.fidelity,
        result.convergence,
        result.validation,
    )
