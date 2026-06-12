"""Shared Schwinger-model physics helpers for the Phys 235 workflow.

This module owns Hamiltonian construction, dense matrix conversion, exact
references, observables, and simple trajectory comparison utilities. The numbered
workflow modules import these helpers instead of depending on each other for
common physics code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as onp
from numpy.typing import NDArray
import pennylane as qml
import scipy.linalg

__all__ = [
    "PauliTerm",
    "SchwingerHamiltonian",
    "ExactSpectrum",
    "ObservableTrajectory",
    "build_schwinger_hamiltonian",
    "pauli_word_to_qml",
    "pauli_word_matrix",
    "hamiltonian_matrix",
    "exact_ground_state",
    "expectation",
    "state_fidelity",
    "total_charge_matrix",
    "commutator_norm",
    "compute_observables",
    "exact_time_evolution",
    "observable_trajectory",
    "fidelity_series",
]


PAULI_MATS = {
    "I": onp.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "X": onp.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": onp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": onp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


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


@dataclass
class ObservableTrajectory:
    times: NDArray
    states: NDArray
    electric_field: NDArray
    chiral_condensate: NDArray
    charge: NDArray


def observable_trajectory(
    times: NDArray,
    states: NDArray,
    N: int,
    ag: float,
    q_final: float,
    g: float,
) -> ObservableTrajectory:
    """Compute paper observables along a trajectory of statevectors."""

    count = states.shape[0]
    electric_field = onp.empty(count)
    chiral_condensate = onp.empty(count)
    charge = onp.empty(count)
    for idx in range(count):
        obs = compute_observables(states[idx], N=N, ag=ag, external_field=q_final, g=g)
        electric_field[idx] = obs["electric_field"]
        chiral_condensate[idx] = obs["chiral_condensate"]
        charge[idx] = obs["charge"]
    return ObservableTrajectory(
        times=onp.asarray(times, dtype=float),
        states=onp.asarray(states, dtype=complex),
        electric_field=electric_field,
        chiral_condensate=chiral_condensate,
        charge=charge,
    )


def fidelity_series(states_ref: NDArray, states: NDArray) -> NDArray:
    """Per-time fidelity |<ref|state>|^2 between two trajectories."""

    return onp.asarray(
        [state_fidelity(states_ref[idx], states[idx]) for idx in range(states.shape[0])],
        dtype=float,
    )
