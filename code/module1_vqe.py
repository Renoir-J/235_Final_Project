"""Module 1: VQE ground-state preparation for the Schwinger model.

This module prepares the q=0 ground state with the paper Hamiltonian
Variational Ansatz (HVA). Quench construction is handled separately by
module2_quench.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as onp
from numpy.typing import NDArray
import pennylane as qml
from pennylane import numpy as np
import scipy.optimize

from schwinger_core import (
    SchwingerHamiltonian,
    build_schwinger_hamiltonian,
    commutator_norm,
    exact_ground_state,
    hamiltonian_matrix,
    state_fidelity,
    total_charge_matrix,
)

__all__ = [
    "Module1Config",
    "Module1WorkflowResult",
    "VQEEnsembleResult",
    "VQERestartSample",
    "VQEResult",
    "apply_hva_ansatz",
    "hva_state",
    "make_vqe_initial_guesses",
    "module1_acceptance_passed",
    "n_hva_params",
    "run_module1_from_config",
    "run_module1_workflow",
    "run_vqe",
    "run_vqe_restart_ensemble",
    "validate_module1_setup",
]


@dataclass
class VQEResult:
    theta_opt: NDArray
    best_energy: float
    adam_energy: float
    exact_ground_energy: float
    exact_max_energy: float
    r_E: float
    statevector: NDArray
    exact_ground_state: NDArray
    restart_history: list[dict]
    polished: bool


@dataclass
class VQERestartSample:
    restart_index: int
    theta_opt: NDArray
    energy: float
    r_E: float
    steps: int
    reason: str


@dataclass
class VQEEnsembleResult:
    layer_count: int
    samples: list[VQERestartSample]
    exact_ground_energy: float
    exact_max_energy: float


@dataclass(frozen=True)
class Module1Config:
    """Stable public configuration for Module 1 VQE."""

    N: int
    ag: float
    m_over_g: float
    q_initial: float
    g: float
    layer_count: int
    n_restarts: int
    seed: int
    learning_rate: float
    max_steps: int
    grad_tol: float
    stall_window: int
    stall_tol: float
    use_lbfgs_polish: bool

    def to_dict(self) -> dict[str, float | int | bool]:
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
        if self.n_restarts < 1 or self.n_restarts > 20:
            raise ValueError("n_restarts must be between 1 and 20.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.grad_tol <= 0:
            raise ValueError("grad_tol must be positive.")
        if self.stall_window < 1:
            raise ValueError("stall_window must be at least 1.")
        if self.stall_tol < 0:
            raise ValueError("stall_tol must be non-negative.")


@dataclass
class Module1WorkflowResult:
    config: Module1Config
    vqe: VQEResult
    H_initial_matrix: NDArray
    validation: dict[str, float | bool]


def n_hva_params(layer_count: int, N: int) -> int:
    """Number of flattened HVA parameters."""

    return layer_count * ((N - 1) + (N - 1) + N)


def unpack_hva_params(
    theta: Any,
    layer_count: int,
    N: int,
) -> tuple[Any, Any, Any]:
    """Unpack flat HVA parameters into alpha, beta, gamma arrays."""

    n_links = N - 1
    alpha_size = layer_count * n_links
    beta_size = layer_count * n_links

    alpha = np.reshape(theta[:alpha_size], (layer_count, n_links))
    beta = np.reshape(theta[alpha_size : alpha_size + beta_size], (layer_count, n_links))
    gamma = np.reshape(theta[alpha_size + beta_size :], (layer_count, N))
    return alpha, beta, gamma


def _even_then_odd_links(N: int) -> list[int]:
    even_links = [n for n in range(N - 1) if n % 2 == 0]
    odd_links = [n for n in range(N - 1) if n % 2 == 1]
    return even_links + odd_links


def apply_hva_ansatz(theta: Any, layer_count: int, N: int) -> None:
    """Apply the paper HVA circuit.

    The algebraic layer is U_l = Z * ZZ * XY. Since operators act on a
    state from right to left, the circuit applies XY first, then ZZ, then Z.
    PauliRot(phi, P) implements exp(-i phi P / 2), so the signs below
    implement exp(+i angle P / 2) conventions from the paper.
    """

    alpha, beta, gamma = unpack_hva_params(theta, layer_count, N)

    for wire in range(0, N, 2):
        qml.PauliX(wire)

    link_order = _even_then_odd_links(N)
    for layer in range(layer_count):
        for n in link_order:
            qml.PauliRot(-alpha[layer, n] / 2.0, "XX", wires=[n, n + 1])
            qml.PauliRot(-alpha[layer, n] / 2.0, "YY", wires=[n, n + 1])

        for n in link_order:
            qml.PauliRot(-beta[layer, n], "ZZ", wires=[n, n + 1])

        for n in range(N):
            qml.PauliRot(-gamma[layer, n], "Z", wires=[n])


def make_energy_qnode(
    hamiltonian: SchwingerHamiltonian,
    layer_count: int,
) -> Callable[[Any], Any]:
    """Create a differentiable PennyLane energy function."""

    dev = qml.device("default.qubit", wires=hamiltonian.N)
    qml_hamiltonian = hamiltonian.to_qml()

    @qml.qnode(dev, interface="autograd")
    def energy(theta: Any) -> Any:
        apply_hva_ansatz(theta, layer_count=layer_count, N=hamiltonian.N)
        return qml.expval(qml_hamiltonian)

    return energy


def make_state_qnode(N: int, layer_count: int) -> Callable[[Any], Any]:
    """Create a PennyLane statevector function for the HVA."""

    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="autograd")
    def state(theta: Any) -> Any:
        apply_hva_ansatz(theta, layer_count=layer_count, N=N)
        return qml.state()

    return state


def hva_state(theta: NDArray, layer_count: int, N: int) -> NDArray:
    """Return a dense statevector for HVA parameters."""

    state_fn = make_state_qnode(N=N, layer_count=layer_count)
    theta_pl = np.array(theta, requires_grad=False)
    return onp.asarray(state_fn(theta_pl), dtype=complex)


def make_vqe_initial_guesses(
    layer_count: int,
    N: int,
    seed: int,
) -> list[NDArray]:
    """Generate 20 deterministic VQE initial guesses."""

    rng = onp.random.default_rng(seed)
    size = n_hva_params(layer_count, N)
    guesses: list[NDArray] = [onp.zeros(size)]
    guesses.extend(rng.normal(loc=0.0, scale=0.05, size=size) for _ in range(9))
    guesses.extend(rng.uniform(low=-onp.pi, high=onp.pi, size=size) for _ in range(10))
    return guesses


def _adam_minimize(
    objective: Callable[[Any], Any],
    theta0: NDArray,
    learning_rate: float,
    max_steps: int,
    grad_tol: float,
    stall_window: int,
    stall_tol: float,
) -> dict:
    """Small Adam implementation that reuses one autograd gradient per step."""

    grad_fn = qml.grad(objective)
    theta = np.array(theta0, requires_grad=True)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    energy_history: list[float] = []
    grad_norm_history: list[float] = []
    best_theta = onp.asarray(theta, dtype=float).copy()
    best_energy = float(objective(theta))
    reason = "max_steps"

    for step in range(1, max_steps + 1):
        grad = grad_fn(theta)
        grad_array = onp.asarray(grad, dtype=float)
        grad_norm = float(onp.linalg.norm(grad_array))

        if grad_norm < grad_tol:
            reason = "grad_tol"
            break

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        theta = np.array(theta, requires_grad=True)

        energy = float(objective(theta))
        energy_history.append(energy)
        grad_norm_history.append(grad_norm)

        if energy < best_energy:
            best_energy = energy
            best_theta = onp.asarray(theta, dtype=float).copy()

        if len(energy_history) > stall_window:
            previous_best = min(energy_history[:-stall_window])
            recent_best = min(energy_history[-stall_window:])
            if previous_best - recent_best < stall_tol:
                reason = "stall_tol"
                break
    else:
        step = max_steps

    return {
        "theta": best_theta,
        "energy": best_energy,
        "steps": step,
        "reason": reason,
        "energy_history": energy_history,
        "grad_norm_history": grad_norm_history,
    }


def _lbfgs_polish(
    objective: Callable[[Any], Any],
    theta0: NDArray,
    maxiter: int = 300,
) -> scipy.optimize.OptimizeResult:
    """Optional L-BFGS-B polish using PennyLane autograd gradients."""

    grad_fn = qml.grad(objective)

    def fun(x: NDArray) -> float:
        theta = np.array(x, requires_grad=True)
        return float(objective(theta))

    def jac(x: NDArray) -> NDArray:
        theta = np.array(x, requires_grad=True)
        return onp.asarray(grad_fn(theta), dtype=float)

    return scipy.optimize.minimize(
        fun,
        onp.asarray(theta0, dtype=float),
        jac=jac,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )


def run_vqe(
    H_initial: SchwingerHamiltonian,
    layer_count: int,
    n_restarts: int,
    seed: int,
    learning_rate: float,
    max_steps: int,
    grad_tol: float,
    stall_window: int,
    stall_tol: float,
    use_lbfgs_polish: bool,
) -> VQEResult:
    """Run VQE for H(q=0) with the paper HVA."""

    H_matrix = hamiltonian_matrix(H_initial)
    spectrum = exact_ground_state(H_matrix)
    energy_fn = make_energy_qnode(H_initial, layer_count=layer_count)
    guesses = make_vqe_initial_guesses(layer_count=layer_count, N=H_initial.N, seed=seed)

    restart_history: list[dict] = []
    best_theta: NDArray = guesses[0]
    best_energy = onp.inf

    for restart_idx, theta0 in enumerate(guesses[:n_restarts]):
        result = _adam_minimize(
            objective=energy_fn,
            theta0=theta0,
            learning_rate=learning_rate,
            max_steps=max_steps,
            grad_tol=grad_tol,
            stall_window=stall_window,
            stall_tol=stall_tol,
        )
        restart_record = {
            "restart": restart_idx,
            "energy": result["energy"],
            "steps": result["steps"],
            "reason": result["reason"],
            "energy_history": result["energy_history"],
            "grad_norm_history": result["grad_norm_history"],
        }
        restart_history.append(restart_record)

        if result["energy"] < best_energy:
            best_energy = result["energy"]
            best_theta = result["theta"]

    adam_energy = float(best_energy)
    polished = False
    if use_lbfgs_polish:
        polish = _lbfgs_polish(energy_fn, best_theta)
        if polish.success and float(polish.fun) <= best_energy:
            best_theta = onp.asarray(polish.x, dtype=float)
            best_energy = float(polish.fun)
            polished = True
        restart_history.append(
            {
                "restart": "lbfgs_polish",
                "energy": float(polish.fun),
                "steps": int(polish.nit),
                "reason": polish.message,
                "success": bool(polish.success),
            }
        )

    r_E = (spectrum.max_energy - best_energy) / (spectrum.max_energy - spectrum.ground_energy)
    statevector = hva_state(best_theta, layer_count=layer_count, N=H_initial.N)

    return VQEResult(
        theta_opt=best_theta,
        best_energy=float(best_energy),
        adam_energy=adam_energy,
        exact_ground_energy=spectrum.ground_energy,
        exact_max_energy=spectrum.max_energy,
        r_E=float(r_E),
        statevector=statevector,
        exact_ground_state=spectrum.ground_state,
        restart_history=restart_history,
        polished=polished,
    )


def run_vqe_restart_ensemble(
    H_initial: SchwingerHamiltonian,
    layer_count: int,
    n_samples: int = 20,
    seed: int = 1234,
    learning_rate: float = 0.05,
    max_steps: int = 200,
    grad_tol: float = 1e-4,
    stall_window: int = 100,
    stall_tol: float = 1e-9,
) -> VQEEnsembleResult:
    """Run deterministic VQE restarts and retain every optimized sample.

    The sample order is exactly the one from make_vqe_initial_guesses: restart 0
    is the zero vector, followed by seeded local and global random guesses. The
    n_samples parameter is kept for tests and ablations; production paper plots
    pass n_samples=20 to use the full restart set.
    """

    if n_samples < 1 or n_samples > 20:
        raise ValueError("n_samples must be between 1 and 20.")
    if layer_count < 1:
        raise ValueError("layer_count must be at least 1.")

    H_matrix = hamiltonian_matrix(H_initial)
    spectrum = exact_ground_state(H_matrix)
    energy_fn = make_energy_qnode(H_initial, layer_count=layer_count)
    guesses = make_vqe_initial_guesses(layer_count=layer_count, N=H_initial.N, seed=seed)
    denom = spectrum.max_energy - spectrum.ground_energy

    samples: list[VQERestartSample] = []
    for restart_idx, theta0 in enumerate(guesses[:n_samples]):
        result = _adam_minimize(
            objective=energy_fn,
            theta0=theta0,
            learning_rate=learning_rate,
            max_steps=max_steps,
            grad_tol=grad_tol,
            stall_window=stall_window,
            stall_tol=stall_tol,
        )
        energy = float(result["energy"])
        r_E = (spectrum.max_energy - energy) / denom
        samples.append(
            VQERestartSample(
                restart_index=restart_idx,
                theta_opt=onp.asarray(result["theta"], dtype=float).copy(),
                energy=energy,
                r_E=float(r_E),
                steps=int(result["steps"]),
                reason=str(result["reason"]),
            )
        )

    return VQEEnsembleResult(
        layer_count=layer_count,
        samples=samples,
        exact_ground_energy=spectrum.ground_energy,
        exact_max_energy=spectrum.max_energy,
    )


def validate_module1_setup(
    vqe_result: VQEResult,
    H_initial_matrix: NDArray,
    N: int,
) -> dict[str, float | bool]:
    """Collect the checks required before constructing the quench state."""

    charge = total_charge_matrix(N)
    return {
        "r_E": vqe_result.r_E,
        "vqe_energy": vqe_result.best_energy,
        "exact_ground_energy": vqe_result.exact_ground_energy,
        "ground_state_fidelity": state_fidelity(vqe_result.statevector, vqe_result.exact_ground_state),
        "commutator_norm_q0": commutator_norm(H_initial_matrix, charge),
    }


def module1_acceptance_passed(validation: dict[str, float | bool], required_r_E: float = 0.99) -> bool:
    """Return True when Module 1 VQE is ready for quench construction."""

    charge_ok = float(validation.get("commutator_norm_q0", onp.inf)) < 1e-8
    vqe_ok = float(validation.get("r_E", -onp.inf)) > required_r_E
    return charge_ok and vqe_ok


def run_module1_from_config(config: Module1Config) -> Module1WorkflowResult:
    """Run the complete Module 1 VQE workflow from a stable config object."""

    config.validate()
    H_initial = build_schwinger_hamiltonian(
        N=config.N, ag=config.ag, m_over_g=config.m_over_g,
        external_field=config.q_initial, g=config.g,
    )
    H_initial_matrix = hamiltonian_matrix(H_initial)
    vqe_result = run_vqe(
        H_initial=H_initial,
        layer_count=config.layer_count,
        n_restarts=config.n_restarts,
        seed=config.seed,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        grad_tol=config.grad_tol,
        stall_window=config.stall_window,
        stall_tol=config.stall_tol,
        use_lbfgs_polish=config.use_lbfgs_polish,
    )
    validation = validate_module1_setup(vqe_result, H_initial_matrix, config.N)
    return Module1WorkflowResult(config=config, vqe=vqe_result, H_initial_matrix=H_initial_matrix, validation=validation)


def run_module1_workflow(
    N: int,
    ag: float,
    m_over_g: float,
    q_initial: float,
    g: float,
    layer_count: int,
    n_restarts: int,
    seed: int,
    learning_rate: float,
    max_steps: int,
    grad_tol: float,
    stall_window: int,
    stall_tol: float,
    use_lbfgs_polish: bool,
) -> tuple[VQEResult, NDArray, dict[str, float | bool]]:
    """Tuple workflow interface for notebook cells that do not need config objects."""

    result = run_module1_from_config(
        Module1Config(
            N=N, ag=ag, m_over_g=m_over_g, q_initial=q_initial, g=g,
            layer_count=layer_count, n_restarts=n_restarts, seed=seed,
            learning_rate=learning_rate, max_steps=max_steps, grad_tol=grad_tol,
            stall_window=stall_window, stall_tol=stall_tol,
            use_lbfgs_polish=use_lbfgs_polish,
        )
    )
    return result.vqe, result.H_initial_matrix, result.validation
