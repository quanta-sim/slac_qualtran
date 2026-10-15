"""SLAC resources: extracted final implementation, with no import-time benchmarks."""

from collections import Counter
from qualtran import Controlled
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.basic_gates import CHadamard
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.basic_gates import CZ
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.basic_gates import Ry
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.basic_gates import SGate
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.basic_gates import TwoBitCSwap
from qualtran.bloqs.basic_gates import TwoBitSwap
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.basic_gates import YGate
from qualtran.bloqs.basic_gates import ZGate
from qualtran.bloqs.basic_gates import ZPowGate
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting.generalizers import ignore_split_join
import numpy as np
import pandas as pd
from .circuits import (
    CompleteInverseSquareBlockEncoding,
    CompleteMultiscaleInverseSquare,
    PrefixMultiscaleShannonQSWT,
    SLACInverseSquareTest,
    SLACProductAccumulate,
    WaveletScalePreconditioner,
    _slac_controlled_and,
    build_slac_laplacian_select_halfrange_cbloq,
)


def _slac_control_bits(spec):
    return tuple((int(bit) for dtype, values in zip(spec.qdtypes, spec.cvs) for value in np.asarray(values).flat for bit in dtype.to_bits(int(value))))


def _slac_phase_cost(exponent):
    exponent = float(exponent) % 2
    if exponent == 0:
        return Counter()
    if (2 * exponent).is_integer():
        return Counter(clifford=1)
    if (4 * exponent).is_integer():
        return Counter(T=1)
    return Counter(rotation=1)


def _slac_leaf_cost(bloq):
    """Lower every leaf to one common gate basis; reject unknown leaves."""
    if isinstance(bloq, And):
        return Counter(toffoli=1, clifford=2 * (int(bloq.cv1 == 0) + int(bloq.cv2 == 0)))
    if _slac_controlled_and(bloq):
        cvs = _slac_control_bits(bloq.ctrl_spec) + (int(bloq.subbloq.cv1), int(bloq.subbloq.cv2))
        return Counter(toffoli=2 * len(cvs) - 3, clifford=2 * cvs.count(0))
    if isinstance(bloq, Toffoli):
        return Counter(toffoli=1)
    if isinstance(bloq, (Hadamard, XGate, YGate, ZGate, SGate, CNOT, CZ)):
        return Counter(clifford=1)
    if isinstance(bloq, TGate):
        return Counter(T=1)
    if isinstance(bloq, Ry):
        return Counter(rotation=1)
    if isinstance(bloq, ZPowGate):
        return _slac_phase_cost(bloq.exponent)
    if isinstance(bloq, CHadamard):
        return Counter(rotation=2, clifford=1)
    if isinstance(bloq, TwoBitSwap):
        return Counter(clifford=3)
    if isinstance(bloq, TwoBitCSwap):
        return Counter(toffoli=1, clifford=2)
    if isinstance(bloq, Controlled) and isinstance(bloq.subbloq, SGate):
        bits = _slac_control_bits(bloq.ctrl_spec)
        if len(bits) == 1:
            return Counter(T=3, clifford=2 + 2 * bits.count(0))
    raise NotImplementedError(f'Uncounted leaf: {bloq!r}')


def complete_inverse_square_counts(bloq):
    _, leaves = bloq.call_graph(generalizer=ignore_split_join, keep=_slac_controlled_and)
    total = Counter()
    for gate, multiplicity in leaves.items():
        if isinstance(gate, Rz):
            cost = Counter(rotation=1)
        elif type(gate).__name__ in ('Allocate', 'Free', 'Cast'):
            cost = Counter()
        else:
            cost = _slac_leaf_cost(gate)
        for name, amount in cost.items():
            total[name] += int(multiplicity) * amount
    return dict(total)


def theoretical_slac_counts(n, n_ref):
    if n < 3 or n_ref < 1:
        raise ValueError('Require n >= 3 and n_ref >= 1.')
    q, r = (n - 1, n_ref)
    width = 2 * q + r
    square = q * (5 * q - 1)
    product = r * (8 * q + r - 1)
    comparator = 4 * width - 2
    clean_test = 2 * square + 2 * product + 2 * comparator + 1
    select = 2 * q
    base = 2 * clean_test + select
    qsw = lambda k: 2 * k * k + 2 * k - 10
    w_pair = 2 * qsw(n) + 2 * sum((qsw(k) + 5 * k for k in range(2, n))) + 4 * (n - 3)
    p_pair = 4 * (n - 2)
    assert base == 20 * q * q + 32 * q * r + 4 * r * r + 30 * q + 12 * r - 6
    assert 3 * w_pair == 4 * n ** 3 + 27 * n * n - 55 * n - 30
    full = base + w_pair + p_pair
    if r == 2 * n + 4:
        assert 3 * full == 4 * n ** 3 + 327 * n * n + 383 * n - 150
    return {'square': square, 'product': product, 'comparator': comparator, 'clean_test': clean_test, 'select': select, 'base': base, 'W_pair': w_pair, 'P_pair': p_pair, 'full': full}


def slac_rotation_audit(bloq):
    """Audit generic angles, synthesized phase angles, and fixed pi/4-grid rotations."""
    _, leaves = bloq.call_graph(generalizer=ignore_split_join, keep=_slac_controlled_and)
    count = Counter()
    for gate, multiplicity in leaves.items():
        multiplicity = int(multiplicity)
        if isinstance(gate, CHadamard):
            count['fixed_pi_over_4_grid'] += 2 * multiplicity
        elif isinstance(gate, (Ry, Rz)):
            units = float(gate.angle) * 4 / np.pi
            kind = 'fixed_pi_over_4_grid' if np.isclose(units, np.rint(units), atol=1e-12, rtol=0) else 'generic_Ry_Rz'
            count[kind] += multiplicity
        elif isinstance(gate, ZPowGate) and _slac_phase_cost(gate.exponent)['rotation']:
            count['non_Clifford_T_phase'] += multiplicity
    count['unsynthesized_rotations'] = count['generic_Ry_Rz'] + count['non_Clifford_T_phase']
    count['old_rotation_bucket'] = count['unsynthesized_rotations'] + count['fixed_pi_over_4_grid']
    assert count['old_rotation_bucket'] == complete_inverse_square_counts(bloq).get('rotation', 0)
    return dict(count)


def theoretical_component_table(n=16, n_ref=36):
    q, r = (n - 1, n_ref)
    theory = theoretical_slac_counts(n, n_ref)
    rows = []
    specs = [('Square via schoolbook product', 'q(5q-1)', 'square', SLACProductAccumulate(q, q)), ('Product m*j^2', 'r(8q+r-1)', 'product', SLACProductAccumulate(r, 2 * q)), ('Comparator (one call)', '4(2q+r)-2', 'comparator', LessThanEqual(2 * q + r, 2 * q + r)), ('Clean inequality oracle', '2 Square + 2 Product + 2 Comparator + 1', 'clean_test', SLACInverseSquareTest(n, r)), ('SELECT', '2q', 'select', build_slac_laplacian_select_halfrange_cbloq(n)), ('Complete base LCU', '20q^2+32qr+4r^2+30q+12r-6', 'base', CompleteInverseSquareBlockEncoding(n, r)), ('W and W-adjoint', '(4n^3+27n^2-55n-30)/3', 'W_pair', None), ('Both P circuits', '4(n-2)', 'P_pair', None), ('Full preconditioned circuit', 'base + W pair + P pair', 'full', CompleteMultiscaleInverseSquare(n, r, True))]
    for label, formula, key, bloq in specs:
        if key == 'W_pair':
            actual = Counter(complete_inverse_square_counts(PrefixMultiscaleShannonQSWT(n)))
            actual.update(complete_inverse_square_counts(PrefixMultiscaleShannonQSWT(n).adjoint()))
        elif key == 'P_pair':
            actual = Counter({k: 2 * v for k, v in complete_inverse_square_counts(WaveletScalePreconditioner(n)).items()})
        else:
            actual = Counter(complete_inverse_square_counts(bloq))
        assert actual['toffoli'] == theory[key], (label, actual, theory[key])
        rows.append({'component': label, 'Toffoli_formula': formula, 'theory_Toffoli': theory[key], 'Qualtran_Toffoli': actual['toffoli'], 'Clifford': actual['clifford'], 'explicit_T': actual['T'], 'logical_rotations': actual['rotation']})
    return pd.DataFrame(rows)
