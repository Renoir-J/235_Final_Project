"""Module 4: VQE ground-state preparation and quench initialization.

This script implements the paper-faithful Module 4 workflow for

    Nagano, Bapat, Bauer, "Quench dynamics of the Schwinger model via
    variational quantum algorithms" (arXiv:2302.10933).

It prepares the q=0 ground state with the Hamiltonian Variational Ansatz
(HVA), then constructs the q=2 post-quench Hamiltonian and saves the
quench-ready state/reference objects for later Trotter and McLachlan VQS
modules.

The source paper uses q=2 for the external electric field and L=5 for the
ansatz layer count. Keep those two symbols separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import numpy as onp
from numpy.typing import NDArray
import pennylane as qml
from pennylane import numpy as np
import scipy.linalg
import scipy.optimize


PAULI_MATS = {
    "I": onp.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "X": onp.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": onp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": onp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

__all__ = [
    "ExactSpectrum",
    "Module4Config",
    "Module4WorkflowResult",
    "PauliTerm",
    "QuenchSetup",
    "SchwingerHamiltonian",
    "VQEResult",
    "build_schwinger_hamiltonian",
    "commutator_norm",
    "compute_observables",
    "exact_ground_state",
    "exact_time_evolution",
    "hamiltonian_matrix",
    "hva_state",
    "make_vqe_initial_guesses",
    "module4_acceptance_passed",
    "prepare_quench_state",
    "run_module4_from_config",
    "run_module4_workflow",
    "run_vqe",
    "state_fidelity",
    "total_charge_matrix",
    "validate_module4_setup",
]


@dataclass(frozen=True)
class PauliTerm:
    coeff: float
    word: tuple[str, ...]


@dataclass
class SchwingerHamiltonian:
    terms: list[PauliTerm]
    N: int
    ag: float
    m_over_g: float
    external_field: float
    g: float
    a: float
    m: float
    J: float
    w: float

    def to_qml(self) -> qml.Hamiltonian:
        coeffs: list[float] = []
        ops: list[qml.operation.Operator] = []
        for term in self.terms:
            coeffs.append(term.coeff)
            ops.append(pauli_word_to_qml(term.word))
        return qml.Hamiltonian(coeffs, ops)


@dataclass
class ExactSpectrum:
    eigenvalues: NDArray
    eigenvectors: NDArray
    ground_energy: float
    max_energy: float
    ground_state: NDArray


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
class Module4Config:
    """Stable public configuration passed in by main_skeleton.ipynb.

    This class intentionally has no project-specific defaults. Concrete
    parameter choices belong in the main notebook, not in this algorithm module.
    """

    N: int
    ag: float
    m_over_g: float
    q_initial: float
    q_final: float
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
        if self.layer_count < 1:
            raise ValueError("layer_count must be at least 1.")
        if self.n_restarts < 1 or self.n_restarts > 20:
            raise ValueError("n_restarts must be between 1 and 20.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


@dataclass
class Module4WorkflowResult:
    """Top-level result passed from Module 4 to later workflow stages."""

    config: Module4Config
    vqe: VQEResult
    quench: QuenchSetup
    validation: dict[str, float | bool]


def n_hva_params(layer_count: int, N: int) -> int:
    """Number of flattened HVA parameters."""

    return layer_count * ((N - 1) + (N - 1) + N)


def _empty_word(N: int) -> tuple[str, ...]:
    return tuple("I" for _ in range(N))


def _with_paulis(N: int, entries: Iterable[tuple[int, str]]) -> tuple[str, ...]:
    word = ["I"] * N
    for wire, pauli in entries:
        word[wire] = pauli
    return tuple(word)


def _add_term(
    accumulator: dict[tuple[str, ...], float],
    coeff: float,
    word: tuple[str, ...],
    atol: float = 1e-14,
) -> None:
    if abs(coeff) < atol:
        return
    accumulator[word] = accumulator.get(word, 0.0) + float(coeff)


def _finalize_terms(
    accumulator: dict[tuple[str, ...], float],
    atol: float = 1e-12,
) -> list[PauliTerm]:
    terms = [
        PauliTerm(coeff=coeff, word=word)
        for word, coeff in accumulator.items()
        if abs(coeff) > atol
    ]
    terms.sort(key=lambda term: ("".join(term.word), term.coeff))
    return terms


def build_schwinger_hamiltonian(
    N: int,
    ag: float,
    m_over_g: float,
    external_field: float,
    g: float,
) -> SchwingerHamiltonian:
    """Build the spin Hamiltonian from Eq. (8) of arXiv:2302.10933.

    H(q) =
        J sum_n [sum_{i<=n} (Z_i + (-1)^i)/2 + q]^2
      + (w/2) sum_n (X_n X_{n+1} + Y_n Y_{n+1})
      + (m/2) sum_n (-1)^n Z_n.

    The constant terms are preserved.
    """

    if N < 2:
        raise ValueError("N must be at least 2.")
    if ag <= 0:
        raise ValueError("ag must be positive.")
    if g <= 0:
        raise ValueError("g must be positive.")

    a = ag / g
    m = m_over_g * g
    J = g**2 * a / 2.0
    w = 1.0 / (2.0 * a)
    terms: dict[tuple[str, ...], float] = {}
    identity = _empty_word(N)

    # Electric term: J * [q + sum_{i<=n} (Z_i + (-1)^i)/2]^2.
    for n in range(N - 1):
        active_wires = list(range(n + 1))
        constant = external_field + 0.5 * sum((-1) ** i for i in active_wires)

        # A^2 = c^2 I + sum_i z_i^2 I + 2c sum_i z_i Z_i
        #       + 2 sum_{i<j} z_i z_j Z_i Z_j, with z_i = 1/2.
        identity_coeff = J * (constant**2 + 0.25 * len(active_wires))
        _add_term(terms, identity_coeff, identity)

        for i in active_wires:
            _add_term(terms, J * constant, _with_paulis(N, [(i, "Z")]))

        for left_idx, i in enumerate(active_wires):
            for j in active_wires[left_idx + 1 :]:
                _add_term(terms, 0.5 * J, _with_paulis(N, [(i, "Z"), (j, "Z")]))

    # Hopping term: (w/2) sum_n (X_n X_{n+1} + Y_n Y_{n+1}).
    for n in range(N - 1):
        _add_term(terms, 0.5 * w, _with_paulis(N, [(n, "X"), (n + 1, "X")]))
        _add_term(terms, 0.5 * w, _with_paulis(N, [(n, "Y"), (n + 1, "Y")]))

    # Staggered mass term: (m/2) sum_n (-1)^n Z_n.
    for n in range(N):
        _add_term(terms, 0.5 * m * ((-1) ** n), _with_paulis(N, [(n, "Z")]))

    return SchwingerHamiltonian(
        terms=_finalize_terms(terms),
        N=N,
        ag=ag,
        m_over_g=m_over_g,
        external_field=external_field,
        g=g,
        a=a,
        m=m,
        J=J,
        w=w,
    )


def pauli_word_to_qml(word: tuple[str, ...]) -> qml.operation.Operator:
    """Convert a Pauli word such as ('Z', 'I', 'X') to a PennyLane op."""

    active_ops: list[qml.operation.Operator] = []
    for wire, pauli in enumerate(word):
        if pauli == "I":
            continue
        if pauli == "X":
            active_ops.append(qml.PauliX(wire))
        elif pauli == "Y":
            active_ops.append(qml.PauliY(wire))
        elif pauli == "Z":
            active_ops.append(qml.PauliZ(wire))
        else:
            raise ValueError(f"Unsupported Pauli character: {pauli}")

    if not active_ops:
        return qml.Identity(0)

    op = active_ops[0]
    for next_op in active_ops[1:]:
        op = op @ next_op
    return op


def pauli_word_matrix(word: tuple[str, ...]) -> NDArray:
    """Dense matrix for a Pauli word, using wire 0 as the leftmost Kronecker factor."""

    matrix = PAULI_MATS[word[0]]
    for pauli in word[1:]:
        matrix = onp.kron(matrix, PAULI_MATS[pauli])
    return matrix


def hamiltonian_matrix(hamiltonian: SchwingerHamiltonian) -> NDArray:
    """Convert the Pauli-term Hamiltonian to a dense matrix."""

    dim = 2**hamiltonian.N
    matrix = onp.zeros((dim, dim), dtype=complex)
    for term in hamiltonian.terms:
        matrix = matrix + term.coeff * pauli_word_matrix(term.word)
    return matrix


def exact_ground_state(H_matrix: NDArray) -> ExactSpectrum:
    """Exact diagonalization reference."""

    eigenvalues, eigenvectors = scipy.linalg.eigh(H_matrix)
    return ExactSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        ground_energy=float(eigenvalues[0].real),
        max_energy=float(eigenvalues[-1].real),
        ground_state=eigenvectors[:, 0],
    )


def is_hermitian(matrix: NDArray, atol: float = 1e-10) -> bool:
    return bool(onp.allclose(matrix, matrix.conj().T, atol=atol))


def unpack_hva_params(
    theta: Any,
    layer_count: int,
    N: int,
) -> tuple[Any, Any, Any]:
    """Unpack flat HVA parameters into alpha, beta, gamma arrays."""

    expected = n_hva_params(layer_count, N)
    if len(theta) != expected:
        raise ValueError(f"Expected {expected} HVA parameters, got {len(theta)}.")

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

    if n_restarts < 1 or n_restarts > 20:
        raise ValueError("n_restarts must be between 1 and 20 for the configured guess policy.")

    H_matrix = hamiltonian_matrix(H_initial)
    spectrum = exact_ground_state(H_matrix)
    energy_fn = make_energy_qnode(H_initial, layer_count=layer_count)
    guesses = make_vqe_initial_guesses(layer_count=layer_count, N=H_initial.N, seed=seed)

    restart_history: list[dict] = []
    best_theta: NDArray | None = None
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

    if best_theta is None:
        raise RuntimeError("VQE did not run any restart.")

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


def expectation(state: NDArray, operator: NDArray) -> complex:
    """Return <state|operator|state>."""

    return onp.vdot(state, operator @ state)


def state_fidelity(left: NDArray, right: NDArray) -> float:
    """Return |<left|right>|^2."""

    return float(abs(onp.vdot(left, right)) ** 2)


def total_charge_matrix(N: int) -> NDArray:
    """Q = (1/N) sum_n Z_n, the U(1) charge check used by the paper."""

    matrix = onp.zeros((2**N, 2**N), dtype=complex)
    for n in range(N):
        matrix += pauli_word_matrix(_with_paulis(N, [(n, "Z")])) / N
    return matrix


def commutator_norm(left: NDArray, right: NDArray) -> float:
    return float(onp.linalg.norm(left @ right - right @ left))


def compute_observables(
    state: NDArray,
    N: int,
    ag: float,
    external_field: float,
    g: float,
) -> dict[str, float]:
    """Compute paper observables E(t), Sigma(t), and Q(t)."""

    z_expectations = []
    for n in range(N):
        z_matrix = pauli_word_matrix(_with_paulis(N, [(n, "Z")]))
        z_expectations.append(float(expectation(state, z_matrix).real))

    electric_sum = 0.0
    electric_background = 0.0
    for n in range(N):
        for k in range(n + 1):
            electric_sum += z_expectations[k]
            electric_background += (-1) ** k

    electric_field = g * electric_sum / (2.0 * N)
    electric_field += g * electric_background / (2.0 * N)
    electric_field += g * external_field

    chiral_condensate = ag * sum(((-1) ** n) * z_expectations[n] for n in range(N)) / N
    charge = sum(z_expectations) / N

    return {
        "electric_field": float(electric_field),
        "chiral_condensate": float(chiral_condensate),
        "charge": float(charge),
    }


def exact_time_evolution(
    H_matrix: NDArray,
    psi_0: NDArray,
    times: NDArray,
) -> NDArray:
    """Exact statevector evolution exp(-i H t) psi_0 for a list of times."""

    evolved_states = []
    for t in times:
        evolved_states.append(scipy.linalg.expm(-1.0j * H_matrix * float(t)) @ psi_0)
    return onp.asarray(evolved_states, dtype=complex)


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


def validate_module4_setup(vqe_result: VQEResult, quench: QuenchSetup) -> dict[str, float | bool]:
    """Collect the visible checks required before moving to dynamics."""

    N = quench.H_initial.N
    H0 = quench.H_initial_matrix
    Hf = quench.H_final_matrix
    charge = total_charge_matrix(N)

    return {
        "H0_shape_ok": H0.shape == (2**N, 2**N),
        "Hf_shape_ok": Hf.shape == (2**N, 2**N),
        "H0_hermitian": is_hermitian(H0),
        "Hf_hermitian": is_hermitian(Hf),
        "state_norm": float(onp.linalg.norm(quench.psi_0)),
        "r_E": vqe_result.r_E,
        "vqe_energy": vqe_result.best_energy,
        "exact_ground_energy": vqe_result.exact_ground_energy,
        "ground_state_fidelity": state_fidelity(
            quench.psi_0,
            vqe_result.exact_ground_state,
        ),
        "q2_energy_variance": quench.initial_variance_q2,
        "commutator_norm_q0": commutator_norm(H0, charge),
        "commutator_norm_q2": commutator_norm(Hf, charge),
    }


def module4_acceptance_passed(
    validation: dict[str, float | bool],
    required_r_E: float = 0.99,
) -> bool:
    """Return True when Module 4 is ready for later dynamics modules."""

    required_checks = [
        "H0_shape_ok",
        "Hf_shape_ok",
        "H0_hermitian",
        "Hf_hermitian",
    ]
    checks_ok = all(bool(validation.get(key, False)) for key in required_checks)
    norm_ok = abs(float(validation.get("state_norm", onp.inf)) - 1.0) < 1e-8
    charge_ok = float(validation.get("commutator_norm_q0", onp.inf)) < 1e-9
    charge_ok = charge_ok and float(validation.get("commutator_norm_q2", onp.inf)) < 1e-9
    vqe_ok = float(validation.get("r_E", -onp.inf)) > required_r_E
    quench_ok = float(validation.get("q2_energy_variance", 0.0)) > 1e-10
    return checks_ok and norm_ok and charge_ok and vqe_ok and quench_ok


def run_module4_from_config(config: Module4Config) -> Module4WorkflowResult:
    """Run the complete Module 4 workflow from a stable config object."""

    config.validate()
    H_initial = build_schwinger_hamiltonian(
        N=config.N,
        ag=config.ag,
        m_over_g=config.m_over_g,
        external_field=config.q_initial,
        g=config.g,
    )
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
    quench = prepare_quench_state(
        lambda_opt=vqe_result.theta_opt,
        layer_count=config.layer_count,
        N=config.N,
        ag=config.ag,
        m_over_g=config.m_over_g,
        q_initial=config.q_initial,
        q_final=config.q_final,
        g=config.g,
    )
    validation = validate_module4_setup(vqe_result, quench)
    return Module4WorkflowResult(
        config=config,
        vqe=vqe_result,
        quench=quench,
        validation=validation,
    )


def run_module4_workflow(
    N: int,
    ag: float,
    m_over_g: float,
    q_initial: float,
    q_final: float,
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
) -> tuple[VQEResult, QuenchSetup, dict[str, float | bool]]:
    """Tuple workflow interface for notebook cells that do not need config objects."""

    result = run_module4_from_config(
        Module4Config(
            N=N,
            ag=ag,
            m_over_g=m_over_g,
            q_initial=q_initial,
            q_final=q_final,
            g=g,
            layer_count=layer_count,
            n_restarts=n_restarts,
            seed=seed,
            learning_rate=learning_rate,
            max_steps=max_steps,
            grad_tol=grad_tol,
            stall_window=stall_window,
            stall_tol=stall_tol,
            use_lbfgs_polish=use_lbfgs_polish,
        )
    )
    return result.vqe, result.quench, result.validation
