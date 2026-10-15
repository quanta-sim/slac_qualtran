"""SLAC analysis: extracted final implementation, with no import-time benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .circuits import (
    CompleteInverseSquareBlockEncoding,
    CompleteMultiscaleInverseSquare,
    SLACInverseSquareTest,
    build_slac_laplacian_select_halfrange_cbloq,
    circulant_from_column,
    inverse_square_column,
    shannon_multiscale_reference,
    slac_inverse_square_alpha,
    wavelet_scale_weights,
)
from .resources import (
    complete_inverse_square_counts,
    theoretical_slac_counts,
)


def validate_complete_inverse_square_action(n, n_ref):
    """Compositional verification via reversible classical actions, not simulation.

    Preparation probabilities are combined with the actual reversible
    inequality and SELECT actions, then checked against Fourier convolution.
    COPY keeps every failed branch orthogonal to the all-zero signal block.
    """
    q, size, samples = (n - 1, 1 << n, 1 << n_ref)
    alpha = slac_inverse_square_alpha(n)
    angle = 2 * np.arccos(np.sqrt(np.pi ** 2 / 3 / alpha))
    p_zero, p_nonzero = (np.cos(angle / 2) ** 2, np.sin(angle / 2) ** 2)
    geometric_normalization = sum((2.0 ** (-mu) for mu in range(q)))
    test = SLACInverseSquareTest(n, n_ref).decompose_bloq()
    select = build_slac_laplacian_select_halfrange_cbloq(n)
    action = -p_zero * np.eye(size, dtype=complex)
    success_probability = p_zero
    checked = 0
    for j in range(1, size // 2):
        mu = j.bit_length() - 1
        mu_wire = 1 << q - 1 - mu
        j_wire = int(f'{j:0{q}b}'[::-1], 2)
        accepted = 0
        for m in range(samples):
            for f_in in (0, 1):
                actual = test.call_classically(a=1, mu=mu_wire, j=j_wire, ref=m, f=f_in)
                failure = int(m * j * j >= 1 << 2 * mu + n_ref)
                assert actual == (1, mu_wire, j_wire, m, f_in ^ failure)
                checked += 1
            accepted += 1 - failure
        p_j = p_nonzero * (2.0 ** (-mu) / geometric_normalization) / (1 << mu)
        branch_weight = p_j * accepted / samples
        success_probability += branch_weight
        phase = (-1) ** (1 + j)
        for d in (0, 1):
            for x in range(size):
                d_out, j_out, y = select.call_classically(d=d, j=j, sys=x)
                assert (d_out, j_out) == (d, j)
                assert y == (x + (2 * d - 1) * j) % size
                action[y, x] += phase * branch_weight / 2
    for f_in in (0, 1):
        assert test.call_classically(a=0, mu=0, j=0, ref=0, f=f_in) == (0, 0, 0, 0, f_in)
    finite_target = circulant_from_column(inverse_square_column(n, n_ref)) / alpha
    exact_target = circulant_from_column(inverse_square_column(n)) / alpha
    np.testing.assert_allclose(action, finite_target, atol=2e-14)
    rng = np.random.default_rng(731 + n)
    action_error = 0.0
    for _ in range(4):
        vector = rng.normal(size=size) + 1j * rng.normal(size=size)
        vector /= np.linalg.norm(vector)
        ref = np.fft.ifft(np.fft.fft(inverse_square_column(n, n_ref)) * np.fft.fft(vector)) / alpha
        action_error = max(action_error, float(np.max(np.abs(action @ vector - ref))))
    assert action_error < 2e-14
    weight_error = float(np.max(np.abs(np.fft.fft(inverse_square_column(n, n_ref) - inverse_square_column(n)))))
    bound = 8 * (1 - 2.0 ** (-q)) / samples
    assert weight_error < bound + 1e-14
    return {'n': n, 'n_ref': n_ref, 'alpha_n': alpha, 'classical_oracle_cases': checked + 2, 'finite_M_block_error': float(np.max(np.abs(action - finite_target))), 'random_vector_action_error': action_error, 'raw_operator_rounding_error': weight_error, 'rounding_error_bound': bound, 'success_probability': success_probability, 'encoded_action': action, 'exact_target': exact_target}


def benchmark_complete_inverse_square_condition(n_values=range(3, 8)):
    rows = []
    for n in n_values:
        size, r = (1 << n, 2 * n + 4)
        w = shannon_multiscale_reference(n)
        p = np.diag(wavelet_scale_weights(n))
        fourier = np.exp(2j * np.pi * np.outer(np.arange(size), np.arange(size)) / size) / np.sqrt(size)
        q = w @ fourier[:, 1:]
        for kind, precision in [('Exact truncated 1/j^2 target', None), ('Finite-M complete circuit', r)]:
            column = inverse_square_column(n, precision)
            a = circulant_from_column(column)
            multi = w @ a @ w.conj().T
            before = q.conj().T @ multi @ q
            after = q.conj().T @ p @ multi @ p @ q
            rows.append({'n': n, 'N': size, 'n_ref': r, 'operator': kind, 'kappa_constant_excluded': np.linalg.cond(before), 'kappa_with_P_constant_excluded': np.linalg.cond(after), 'constant_mode_eigenvalue': float(column.sum())})
    return pd.DataFrame(rows)


def compare_three_slac_circuits(n_values=range(3, 17)):
    rows = []
    for n in n_values:
        r = 2 * n + 4
        theory = theoretical_slac_counts(n, r)
        base = complete_inverse_square_counts(CompleteInverseSquareBlockEncoding(n, r))['toffoli']
        multi = complete_inverse_square_counts(CompleteMultiscaleInverseSquare(n, r))['toffoli']
        pre = complete_inverse_square_counts(CompleteMultiscaleInverseSquare(n, r, True))['toffoli']
        expressions = {'LCU': 100 * n * n + 142 * n - 32, 'multiscale': (4 * n ** 3 + 327 * n * n + 371 * n - 126) // 3, 'preconditioned': (4 * n ** 3 + 327 * n * n + 383 * n - 150) // 3}
        assert expressions['LCU'] == theory['base'] == base
        assert expressions['multiscale'] == theory['base'] + theory['W_pair'] == multi
        assert expressions['preconditioned'] == theory['full'] == pre
        rows.append({'n': n, 'n_ref': r, 'LCU_analytical': expressions['LCU'], 'LCU_circuit': base, 'multiscale_analytical': expressions['multiscale'], 'multiscale_circuit': multi, 'preconditioned_analytical': expressions['preconditioned'], 'preconditioned_circuit': pre})
    table = pd.DataFrame(rows)
    x = np.linspace(table['n'].min(), table['n'].max(), 500)
    curves = [('LCU', 'Complete LCU', '#1976b9', 'o', 100 * x * x + 142 * x - 32), ('multiscale', 'Multiscale', '#d55e00', 's', (4 * x ** 3 + 327 * x * x + 371 * x - 126) / 3), ('preconditioned', 'Preconditioned', '#00875a', '+', (4 * x ** 3 + 327 * x * x + 383 * x - 150) / 3)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, color, marker, y in curves:
        ax.plot(x, y, '--', color=color, linewidth=1.7, label=label + ': analytical')
        ax.plot(table['n'], table[key + '_circuit'], linestyle='none', marker=marker, markersize=7 if marker != '+' else 10, markerfacecolor='none', markeredgewidth=1.5, color=color, label=label + ': Qualtran')
    ax.set(xlabel='n = log2(N)', ylabel='Coherent Toffoli count', title='Complete SLAC circuits: analytical counts vs Qualtran')
    ax.set_xticks(table['n'])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    fig.text(0.5, 0.02, 'n_ref = 2n + 4. No fitted constants. The two P circuits add only 4(n - 2) Toffolis.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return (table, fig)
