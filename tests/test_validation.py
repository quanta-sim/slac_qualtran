"""Regression checks for the extracted final construction."""

import numpy as np
import pytest

from slac_qualtran.analysis import validate_complete_inverse_square_action
from slac_qualtran.circuits import (
    CompleteMultiscaleInverseSquare,
    inverse_square_column,
    shannon_dense_circuit_reference,
    shannon_multiscale_reference,
    shannon_rectangular_matrix_eq18,
)
from slac_qualtran.resources import complete_inverse_square_counts, theoretical_slac_counts


@pytest.mark.parametrize("n", [3, 4])
def test_classical_oracles_and_encoded_action(n):
    check = validate_complete_inverse_square_action(n, 3)
    assert check["finite_M_block_error"] < 2e-14
    assert check["random_vector_action_error"] < 2e-14
    assert check["raw_operator_rounding_error"] <= check["rounding_error_bound"] + 1e-14


@pytest.mark.parametrize("n,r", [(3, 1), (4, 2), (5, 7), (16, 36)])
def test_resource_graph_matches_structural_formula(n, r):
    actual = complete_inverse_square_counts(CompleteMultiscaleInverseSquare(n, r, True))
    assert actual["toffoli"] == theoretical_slac_counts(n, r)["full"]


@pytest.mark.parametrize("n", [3, 4])
def test_only_midpoint_is_dropped(n):
    column = inverse_square_column(n)
    size = 1 << n
    assert np.flatnonzero(column == 0).tolist() == [size // 2]
    assert column[0] == -np.pi**2 / 3
    for j in range(1, size // 2):
        assert column[j] == column[size-j] == 2*(-1)**(1+j)/j**2


@pytest.mark.parametrize("n", [3, 4])
def test_shannon_reference_actions(n):
    filters, _, _ = shannon_rectangular_matrix_eq18(n)
    np.testing.assert_allclose(shannon_dense_circuit_reference(n), filters, atol=2e-12)
    w = shannon_multiscale_reference(n)
    np.testing.assert_allclose(w.conj().T @ w, np.eye(1 << n), atol=2e-12)
