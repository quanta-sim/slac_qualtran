"""SLAC circuits: extracted final implementation, with no import-time benchmarks."""

from functools import cached_property
from qualtran import Bloq
from qualtran import BloqBuilder
from qualtran import Controlled
from qualtran import CtrlSpec
from qualtran import QAny
from qualtran import QBit
from qualtran import QUInt
from qualtran import Register
from qualtran import Signature
from qualtran.bloqs.arithmetic import CAdd
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.arithmetic.controlled_add_or_subtract import ControlledAddOrSubtract
from qualtran.bloqs.basic_gates import CHadamard
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.basic_gates import Ry
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.basic_gates import SGate
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.basic_gates import TwoBitCSwap
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.basic_gates import ZGate
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.qft import QFTTextBook
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
import attrs
import importlib
import inspect
import math
import numpy as np
import pandas as pd


def add_register_robust(bb, name: str, bitsize: int):
    try:
        return bb.add_register(name, bitsize)
    except TypeError:
        try:
            from qualtran import QAny
            return bb.add_register(name, QAny(bitsize))
        except Exception:
            from qualtran import QUInt
            return bb.add_register(name, QUInt(bitsize))


def soq_array(bits):
    return np.asarray(bits, dtype=object).reshape(-1)


def split_bits(bb, reg):
    return soq_array(bb.split(reg))


def join_bits(bb, bits):
    return bb.join(soq_array(bits))


def find_ry_gate_class():
    candidate_modules = ['qualtran.bloqs.basic_gates', 'qualtran.bloqs.basic_gates.rotation', 'qualtran.bloqs.basic_gates.rotations', 'qualtran.bloqs.basic_gates.y_basis']
    candidate_names = ['Ry', 'RY', 'RyGate', 'RotationY']
    for mod_name in candidate_modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name in candidate_names:
            if hasattr(mod, name):
                return getattr(mod, name)
    raise ImportError('Could not find Ry gate class.')


def make_ry(theta: float):
    RyClass = find_ry_gate_class()
    attempts = [lambda: RyClass(theta), lambda: RyClass(angle=theta), lambda: RyClass(theta=theta), lambda: RyClass(rads=theta), lambda: RyClass(angle_rads=theta)]
    last_exc = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last_exc = exc
    print('Ry class:', RyClass)
    print('Signature:', inspect.signature(RyClass))
    raise last_exc


def add_ry(bb, q, theta: float):
    ry_gate = make_ry(theta)
    for reg_name in ['q', 'target', 'qubit']:
        try:
            out = bb.add_d(ry_gate, **{reg_name: q})
            return out[reg_name]
        except Exception:
            pass
    try:
        out = bb.add(ry_gate, q=q)
        if isinstance(out, tuple):
            return out[0]
        return out
    except Exception as exc:
        print('Failed to add Ry gate.')
        print('Ry gate:', ry_gate)
        raise exc


def find_cnot_gate_class():
    candidate_modules = ['qualtran.bloqs.basic_gates', 'qualtran.bloqs.basic_gates.cnot', 'qualtran.bloqs.basic_gates.clifford']
    candidate_names = ['CNOT', 'CNOTGate', 'CX', 'CXGate']
    for mod_name in candidate_modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name in candidate_names:
            if hasattr(mod, name):
                return getattr(mod, name)
    raise ImportError('Could not find CNOT/CX gate class.')


def make_cnot():
    CNOTClass = find_cnot_gate_class()
    attempts = [lambda: CNOTClass(), lambda: CNOTClass]
    last_exc = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last_exc = exc
    raise last_exc


def add_cnot(bb, ctrl, target):
    cnot_gate = make_cnot()
    for c_name, t_name in [('ctrl', 'target'), ('control', 'target'), ('ctrl', 'q'), ('control', 'q')]:
        try:
            out = bb.add_d(cnot_gate, **{c_name: ctrl, t_name: target})
            return (out[c_name], out[t_name])
        except Exception:
            pass
    try:
        ctrl, target = bb.add(cnot_gate, ctrl=ctrl, target=target)
        return (ctrl, target)
    except Exception as exc:
        print('Failed to add CNOT.')
        print('CNOT gate:', cnot_gate)
        raise exc


def add_controlled_ry_decomposed(bb, ctrl, target, theta: float):
    """
    Implements CRy(theta) using:
        Ry(theta/2) -- CNOT -- Ry(-theta/2) -- CNOT
    """
    target = add_ry(bb, target, theta / 2)
    ctrl, target = add_cnot(bb, ctrl, target)
    target = add_ry(bb, target, -theta / 2)
    ctrl, target = add_cnot(bb, ctrl, target)
    return (ctrl, target)


def add_controlled_h(bb, ctrl, target):
    target = add_ry(bb, q=target, theta=np.pi / 4)
    ctrl, target = add_cnot(bb, ctrl=ctrl, target=target)
    target = add_ry(bb, q=target, theta=-np.pi / 4)
    return (ctrl, target)


def theta_initial_branch():
    """
    Step 1:
        |0>_a ->
        sqrt(pi^2/(pi^2+24)) |0>_a
        + sqrt(24/(pi^2+24)) |1>_a

    Standard Ry convention:
        Ry(phi)|0> = cos(phi/2)|0> + sin(phi/2)|1>

    Therefore:
        phi = 2 arctan(sqrt(24)/pi).
    """
    theta_ini = math.atan(math.sqrt(24) / math.pi)
    phi = 2 * theta_ini
    return (theta_ini, phi)


def theta_mu_geometric_active(k: int, n: int):
    """
    Eq. (28), active-register version.

    mu = 0,...,n-2.

    There are n-2 active mu qubits and n-2 rotations:
        k = 0,...,n-3.

    theta_k =
        2 arccos[
            1 / sqrt(2(1 - 2^{-(n-1-k)}))
        ].
    """
    if not 0 <= k <= n - 3:
        raise ValueError(f'k must satisfy 0 <= k <= n-3. Got k={k}, n={n}.')
    denom = math.sqrt(2 * (1 - 2 ** (-(n - 1 - k))))
    return 2 * math.acos(1 / denom)


def apply_step2_geometric_mu_cascade_active(bb, a_bit, mu_bits, n: int):
    """
    Active unary register has L=n-2 qubits.

    Unary convention:
        mu=0     -> 000...0
        mu=1     -> 100...0
        mu=2     -> 110...0
        ...
        mu=n-2   -> 111...1
    """
    L = n - 2
    if len(mu_bits) != L:
        raise ValueError(f'Expected len(mu_bits)=n-2={L}. Got {len(mu_bits)}.')
    if L <= 0:
        return (a_bit, mu_bits)
    theta0 = theta_mu_geometric_active(k=0, n=n)
    a_bit, mu_bits[0] = add_controlled_ry_decomposed(bb, ctrl=a_bit, target=mu_bits[0], theta=theta0)
    for k in range(1, L):
        theta_k = theta_mu_geometric_active(k=k, n=n)
        mu_bits[k - 1], mu_bits[k] = add_controlled_ry_decomposed(bb, ctrl=mu_bits[k - 1], target=mu_bits[k], theta=theta_k)
    return (a_bit, mu_bits)


def apply_step3_controlled_hadamards_active(bb, mu_bits, j_bits, n: int):
    """
    Active j register has L=n-2 qubits.

    If mu=r, the first r unary bits are 1.
    Therefore applying CH from mu_k to j_k creates
        j = 0,...,2^r-1.
    """
    L = n - 2
    if len(mu_bits) != L:
        raise ValueError(f'Expected len(mu_bits)=n-2={L}. Got {len(mu_bits)}.')
    if len(j_bits) != L:
        raise ValueError(f'Expected len(j_bits)=n-2={L}. Got {len(j_bits)}.')
    for k in range(L):
        mu_bits[k], j_bits[k] = add_controlled_h(bb, ctrl=mu_bits[k], target=j_bits[k])
    return (mu_bits, j_bits)


def apply_step4_unary_prefix_to_onehot_v2(bb, a_bit, mu_bits, n: int):
    """
    Convert prefix-unary mu to one-hot mu.

    Before Step 4, on the a=1 branch:
        mu=0     -> 000...0
        mu=1     -> 100...0
        mu=2     -> 110...0
        ...
        mu=n-2   -> 111...10

    After Step 4:
        mu=0     -> 100...0
        mu=1     -> 010...0
        mu=2     -> 001...0
        ...
        mu=n-2   -> 000...1

    The a=0 branch remains all-zero in mu.
    """
    if len(mu_bits) != n - 1:
        raise ValueError(f'Expected len(mu_bits)=n-1={n - 1}. Got {len(mu_bits)}.')
    for k in reversed(range(n - 2)):
        mu_bits[k], mu_bits[k + 1] = add_cnot(bb, ctrl=mu_bits[k], target=mu_bits[k + 1])
    a_bit, mu_bits[0] = add_cnot(bb, ctrl=a_bit, target=mu_bits[0])
    return (a_bit, mu_bits)


def apply_step5_shift_j_into_dyadic_boxes(bb, mu_bits, j_bits, n: int):
    """
    Apply CX(mu_i, j_i) for i=0,...,n-2.

    Since mu is one-hot after Step 4, exactly one j_i is flipped.
    This adds 2^mu to j and maps the interval

        0 <= j < 2^mu

    to

        2^mu <= j < 2^{mu+1}.
    """
    if len(mu_bits) != n - 1:
        raise ValueError(f'Expected len(mu_bits)=n-1={n - 1}. Got {len(mu_bits)}.')
    if len(j_bits) != n - 1:
        raise ValueError(f'Expected len(j_bits)=n-1={n - 1}. Got {len(j_bits)}.')
    for i in range(n - 1):
        mu_bits[i], j_bits[i] = add_cnot(bb, ctrl=mu_bits[i], target=j_bits[i])
    return (mu_bits, j_bits)


def build_slac_laplacian_prep_steps_1_2_3_4_5_v2(n: int):
    """
    Build Steps 1--5.

    Registers:
        a  : 1 qubit
        mu : n-1 qubits
        j  : n-1 qubits

    Step 1:
        branch rotation on a.

    Step 2:
        geometric unary-prefix preparation on mu[0:n-2].

    Step 3:
        controlled-H on j[0:n-2].

    Step 4:
        convert unary-prefix mu into one-hot mu.

    Step 5:
        apply CX(mu_i, j_i) for i=0,...,n-2 to shift j into B_mu.
    """
    if n < 3:
        raise ValueError('Need n >= 3.')
    bb = BloqBuilder()
    a = add_register_robust(bb, 'a', 1)
    mu = add_register_robust(bb, 'mu', n - 1)
    j = add_register_robust(bb, 'j', n - 1)
    a_bits = split_bits(bb, a)
    mu_bits = split_bits(bb, mu)
    j_bits = split_bits(bb, j)
    _, phi_a = theta_initial_branch()
    a_bits[0] = add_ry(bb, q=a_bits[0], theta=phi_a)
    active_mu_bits = mu_bits[:n - 2]
    a_bits[0], active_mu_bits = apply_step2_geometric_mu_cascade_active(bb, a_bit=a_bits[0], mu_bits=active_mu_bits, n=n)
    mu_bits[:n - 2] = active_mu_bits
    active_j_bits = j_bits[:n - 2]
    active_mu_bits, active_j_bits = apply_step3_controlled_hadamards_active(bb, mu_bits=mu_bits[:n - 2], j_bits=active_j_bits, n=n)
    mu_bits[:n - 2] = active_mu_bits
    j_bits[:n - 2] = active_j_bits
    a_bits[0], mu_bits = apply_step4_unary_prefix_to_onehot_v2(bb, a_bit=a_bits[0], mu_bits=mu_bits, n=n)
    mu_bits, j_bits = apply_step5_shift_j_into_dyadic_boxes(bb, mu_bits=mu_bits, j_bits=j_bits, n=n)
    a = join_bits(bb, a_bits)
    mu = join_bits(bb, mu_bits)
    j = join_bits(bb, j_bits)
    return bb.finalize(a=a, mu=mu, j=j)


def _qswt_v2_cnot(bb, ctrl, target):
    gate = CNOT()
    names = [r.name for r in gate.signature]
    ctrl_name = 'ctrl' if 'ctrl' in names else 'control' if 'control' in names else names[0]
    target_name = 'target' if 'target' in names else names[1]
    out = bb.add_d(gate, **{ctrl_name: ctrl, target_name: target})
    return (out[ctrl_name], out[target_name])


def _qswt_v2_join(bb, bits):
    bits = np.asarray(bits, dtype=object)
    return bb.join(bits, dtype=QUInt(len(bits)))


def _qswt_v2_pattern_controlled_1q(bb, gate, control, control_bitsize, control_value, target):
    """
    Apply one-qubit `gate` iff the QUInt control register is

        control == control_value.
    """
    cgate = gate.controlled(ctrl_spec=CtrlSpec(qdtypes=QUInt(control_bitsize), cvs=int(control_value)))
    names = [r.name for r in cgate.signature]
    control_names = [name for name in names if 'ctrl' in name.lower() or 'control' in name.lower()]
    if len(control_names) != 1:
        raise RuntimeError(f'Could not determine control register for {cgate.signature}')
    ctrl_name = control_names[0]
    target_names = [name for name in names if name != ctrl_name]
    if len(target_names) != 1:
        raise RuntimeError(f'Could not determine target register for {cgate.signature}')
    target_name = target_names[0]
    out = bb.add_d(cgate, **{ctrl_name: control, target_name: target})
    return (out[ctrl_name], out[target_name])


@attrs.frozen
class QSWTShiftV2(Bloq):
    bitsize: int

    @cached_property
    def signature(self):
        return Signature([Register('system', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, system):
        bits = bb.split(system)
        bits[0] = bb.add(XGate(), q=bits[0])
        system = bb.join(bits, dtype=QUInt(self.bitsize))
        return {'system': system}


@attrs.frozen
class QSWTMixV2(Bloq):
    bitsize: int

    @cached_property
    def signature(self):
        return Signature([Register('system', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, system):
        n = self.bitsize
        if n < 2:
            raise ValueError('QSWTMixV2 requires n >= 2.')
        bits = bb.split(system)
        msb = bits[0]
        lower = _qswt_v2_join(bb, bits[1:])
        pattern = 1 << n - 2
        lower, msb = _qswt_v2_pattern_controlled_1q(bb, Hadamard(), control=lower, control_bitsize=n - 1, control_value=pattern, target=msb)
        lower, msb = _qswt_v2_pattern_controlled_1q(bb, SGate(), control=lower, control_bitsize=n - 1, control_value=pattern, target=msb)
        lower_bits = bb.split(lower)
        bits = np.asarray([msb, *list(lower_bits)], dtype=object)
        system = bb.join(bits, dtype=QUInt(n))
        return {'system': system}


@attrs.frozen
class QSWTSplitV2(Bloq):
    bitsize: int

    @cached_property
    def signature(self):
        return Signature([Register('system', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, system):
        n = self.bitsize
        if n < 2:
            raise ValueError('QSWTSplitV2 requires n >= 2.')
        bits = bb.split(system)
        bits[1] = bb.add(XGate(), q=bits[1])
        bits[1], bits[0] = _qswt_v2_cnot(bb, ctrl=bits[1], target=bits[0])
        system = bb.join(bits, dtype=QUInt(n))
        return {'system': system}


@attrs.frozen
class QSWTBlockV2(Bloq):
    bitsize: int

    @cached_property
    def signature(self):
        return Signature([Register('system', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, system):
        n = self.bitsize
        if n < 2:
            raise ValueError('QSWTBlockV2 requires n >= 2.')
        bits = bb.split(system)
        msb = bits[0]
        lower = _qswt_v2_join(bb, bits[1:])
        lower, msb = _qswt_v2_pattern_controlled_1q(bb, ZGate(), control=lower, control_bitsize=n - 1, control_value=0, target=msb)
        lower_bits = bb.split(lower)
        bits = np.asarray([msb, *list(lower_bits)], dtype=object)
        bits[1] = bb.add(XGate(), q=bits[1])
        system = bb.join(bits, dtype=QUInt(n))
        return {'system': system}


@attrs.frozen
class ShannonQSWTStepV2(Bloq):
    bitsize: int

    @cached_property
    def signature(self):
        if self.bitsize < 2:
            raise ValueError('ShannonQSWTStepV2 requires n >= 2.')
        return Signature([Register('system', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, system):
        n = self.bitsize
        system = bb.add(QFTTextBook(n, with_reverse=True), q=system)
        system = bb.add(QSWTShiftV2(n), system=system)
        system = bb.add(QSWTMixV2(n), system=system)
        system = bb.add(QSWTSplitV2(n), system=system)
        system = bb.add(QSWTBlockV2(n), system=system)
        bits = bb.split(system)
        branch = bits[0]
        lower = bb.join(bits[1:], dtype=QUInt(n - 1))
        lower = bb.add(QFTTextBook(n - 1, with_reverse=True).adjoint(), q=lower)
        lower_bits = bb.split(lower)
        system = bb.join(np.asarray([branch, *list(lower_bits)], dtype=object), dtype=QUInt(n))
        return {'system': system}


def shannon_filters_eq18(N):
    """
    Compute S_IR[j] and S_UV[j] DIRECTLY from Eq. (18).

    This avoids any possible ambiguity in simplifying
    the closed-form Eq. (19).

    IR:

      S_IR[j]
       =
      2/N cos(pi j/2)
       +
      sqrt(2)/N
      sum_{q=-N/4+1}^{N/4-1}
          exp(-2 pi i j q/N)

    UV:

      S_UV[j]
       =
      2/N sin(pi j/2)
       +
      sqrt(2)/N [
          sum_{q=-N/2}^{-N/4-1}
              exp(-2 pi i j q/N)

        + sum_{q=N/4+1}^{N/2-1}
              exp(-2 pi i j q/N)
      ].
    """
    if N < 4 or N & N - 1 != 0:
        raise ValueError('N must be a power of two >= 4.')
    S_IR = np.zeros(N, dtype=np.complex128)
    S_UV = np.zeros(N, dtype=np.complex128)
    q_ir = np.arange(-N // 4 + 1, N // 4)
    q_uv_negative = np.arange(-N // 2, -N // 4)
    q_uv_positive = np.arange(N // 4 + 1, N // 2)
    for j in range(N):
        phase_ir = np.exp(-2j * np.pi * j * q_ir / N)
        phase_uv_negative = np.exp(-2j * np.pi * j * q_uv_negative / N)
        phase_uv_positive = np.exp(-2j * np.pi * j * q_uv_positive / N)
        S_IR[j] = 2.0 / N * np.cos(np.pi * j / 2.0) + np.sqrt(2.0) / N * np.sum(phase_ir)
        S_UV[j] = 2.0 / N * np.sin(np.pi * j / 2.0) + np.sqrt(2.0) / N * (np.sum(phase_uv_negative) + np.sum(phase_uv_positive))
    S_IR = np.real_if_close(S_IR, tol=1000)
    S_UV = np.real_if_close(S_UV, tol=1000)
    return (np.asarray(S_IR, dtype=np.complex128), np.asarray(S_UV, dtype=np.complex128))


def shannon_rectangular_matrix_eq18(n):
    N = 2 ** n
    S_IR, S_UV = shannon_filters_eq18(N)
    W = np.zeros((N, N), dtype=np.complex128)
    for m in range(N // 2):
        for j in range(N):
            idx = (2 * m - j) % N
            W[m, j] = S_IR[idx]
            W[N // 2 + m, j] = S_UV[idx]
    return (W, S_IR, S_UV)


def _qswt_dense_qft(N):
    a = np.arange(N)
    return np.exp(2j * np.pi * np.outer(a, a) / N) / np.sqrt(N)


def _qswt_dense_perm(N, mapping):
    U = np.zeros((N, N), dtype=np.complex128)
    for j in range(N):
        out, phase = mapping(j)
        U[out, j] = phase
    return U


def shannon_dense_circuit_reference(n):
    N = 2 ** n
    F_N = _qswt_dense_qft(N)
    U_shift = _qswt_dense_perm(N, lambda j: ((j - N // 2) % N, 1.0))
    U_mix = np.eye(N, dtype=np.complex128)
    i = N // 4
    j = 3 * N // 4
    U_mix[np.ix_([i, j], [i, j])] = np.array([[1.0, 1.0], [1j, -1j]], dtype=np.complex128) / np.sqrt(2.0)
    U_split = _qswt_dense_perm(N, lambda j: ((j - N // 4) % N, 1.0))
    U_block = _qswt_dense_perm(N, lambda j: (j ^ N // 4, -1.0 if j == N // 2 else 1.0))
    F_half = _qswt_dense_qft(N // 2)
    final_iqft = np.kron(np.eye(2, dtype=np.complex128), F_half.conj().T)
    return final_iqft @ U_block @ U_split @ U_mix @ U_shift @ F_N


def shannon_multiscale_reference(n):
    """Independent recursive Shannon multiscale matrix."""
    N = 1 << n
    if n == 1:
        return np.eye(2, dtype=complex)
    one_step, _, _ = shannon_rectangular_matrix_eq18(n)
    if n == 2:
        return one_step
    half = shannon_multiscale_reference(n - 1)
    recurse_on_ir = np.eye(N, dtype=complex)
    recurse_on_ir[:N // 2, :N // 2] = half
    return recurse_on_ir @ one_step


def _be_add_dtype_register(bb, name, dtype):
    """Add a typed register across the Qualtran versions used here."""
    try:
        return bb.add_register_from_dtype(name, dtype)
    except Exception:
        return bb.add_register(name, dtype.bitsize)


def _reverse_quint_register(bb, reg, bitsize):
    """Reverse wires between the notebook's little-endian j and QUInt."""
    bits = list(bb.split(reg))
    return bb.join(np.asarray(list(reversed(bits)), dtype=object), dtype=QUInt(bitsize))


def build_slac_laplacian_select_halfrange_cbloq(n: int):
    """SELECT for half-range shifts j=1,...,2^(n-1)-1."""
    if n < 2:
        raise ValueError('Need n >= 2.')
    bb = BloqBuilder()
    d = _be_add_dtype_register(bb, 'd', QBit())
    j = _be_add_dtype_register(bb, 'j', QUInt(n - 1))
    sys = _be_add_dtype_register(bb, 'sys', QUInt(n))
    select_adder = ControlledAddOrSubtract(QUInt(n - 1), QUInt(n))
    out = bb.add_d(select_adder, ctrl=d, a=j, b=sys)
    return bb.finalize(d=out['ctrl'], j=out['a'], sys=out['b'])


@attrs.frozen
class SharedControlQFT(Bloq):
    """Controlled textbook QFT, sharing a predicate across each phase gradient."""
    bitsize: int

    @cached_property
    def signature(self):
        return Signature([Register('ctrl', QBit()), Register('q', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, ctrl, q):
        bits = list(bb.split(q))
        for i in range(self.bitsize):
            if i:
                out = bb.add_d(And(), ctrl=np.asarray([ctrl, bits[i]], dtype=object))
                ctrl, bits[i] = out['ctrl']
                flag = out['target']
                phase = bb.join(np.asarray(bits[:i][::-1], dtype=object), dtype=QUInt(i))
                out = bb.add_d(PhaseGradientUnitary(i, exponent=0.5, is_controlled=True), ctrl=flag, phase_grad=phase)
                flag, phase = (out['ctrl'], out['phase_grad'])
                bits[:i] = list(bb.split(phase))[::-1]
                out = bb.add_d(And().adjoint(), ctrl=np.asarray([ctrl, bits[i]], dtype=object), target=flag)
                ctrl, bits[i] = out['ctrl']
            ctrl, bits[i] = bb.add(CHadamard(), ctrl=ctrl, target=bits[i])
        for i in range(self.bitsize // 2):
            ctrl, bits[i], bits[-i - 1] = bb.add(TwoBitCSwap(), ctrl=ctrl, x=bits[i], y=bits[-i - 1])
        return {'ctrl': ctrl, 'q': bb.join(np.asarray(bits, dtype=object), dtype=QUInt(self.bitsize))}


@attrs.frozen
class SharedControlQSWTStep(Bloq):
    bitsize: int

    @cached_property
    def signature(self):
        return Signature([Register('ctrl', QBit()), Register('system', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, ctrl, system):
        ctrl, system = bb.add(SharedControlQFT(self.bitsize), ctrl=ctrl, q=system)
        for cls in (QSWTShiftV2, QSWTMixV2, QSWTSplitV2, QSWTBlockV2):
            ctrl, system = bb.add(cls(self.bitsize).controlled(), ctrl=ctrl, system=system)
        bits = list(bb.split(system))
        lower = bb.join(np.asarray(bits[1:], dtype=object), dtype=QUInt(self.bitsize - 1))
        ctrl, lower = bb.add(SharedControlQFT(self.bitsize - 1).adjoint(), ctrl=ctrl, q=lower)
        bits[1:] = list(bb.split(lower))
        return {'ctrl': ctrl, 'system': bb.join(np.asarray(bits, dtype=object), dtype=QUInt(self.bitsize))}


@attrs.frozen
class PrefixMultiscaleShannonQSWT(Bloq):
    """Share the nested IR-prefix controls with a compute/uncompute And ladder.

    This is the prefix-sharing idea of unary iteration, specialized to a single
    nested IR path, not a SELECT over a fixed, external selection register.
    Each branch bit is produced by its QSWT before its prefix flag is computed.
    """
    bitsize: int
    share_qft_controls: bool = True

    @cached_property
    def signature(self):
        if self.bitsize < 2:
            raise ValueError('Use n >= 2.')
        return Signature([Register('system', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, system):

        def descend(flag, bits):
            width = len(bits)
            lower = bb.join(np.asarray(bits, dtype=object), dtype=QUInt(width))
            step = SharedControlQSWTStep(width) if self.share_qft_controls else ShannonQSWTStepV2(width).controlled()
            out = bb.add_d(step, ctrl=flag, system=lower)
            flag, bits = (out['ctrl'], list(bb.split(out['system'])))
            if width > 2:
                out = bb.add_d(And(1, 0), ctrl=np.asarray([flag, bits[0]], dtype=object))
                flag, bits[0] = out['ctrl']
                child, bits[1:] = descend(out['target'], bits[1:])
                out = bb.add_d(And(1, 0).adjoint(), ctrl=np.asarray([flag, bits[0]], dtype=object), target=child)
                flag, bits[0] = out['ctrl']
            return (flag, bits)
        system = bb.add(ShannonQSWTStepV2(self.bitsize), system=system)
        if self.bitsize > 2:
            bits = list(bb.split(system))
            flag = bb.add(XGate(), q=bits[0])
            flag, bits[1:] = descend(flag, bits[1:])
            bits[0] = bb.add(XGate(), q=flag)
            system = bb.join(np.asarray(bits, dtype=object), dtype=QUInt(self.bitsize))
        return {'system': system}


@attrs.frozen
class CoherentControlledAnd(Bloq):
    """Concrete repair for the single-controlled, non-thru And primitive."""
    original: Controlled

    @cached_property
    def signature(self):
        if self.original.ctrl_spec.num_qubits != 1 or self.original.ctrl_spec.shapes != ((),):
            raise ValueError('This circuit repair expects one scalar control bit.')
        return self.original.signature

    def build_composite_bloq(self, bb, **soqs):
        name = self.original.ctrl_reg_names[0]
        and_gate = self.original.subbloq
        bits = [soqs[name], *soqs['ctrl']]
        cvs = [self.original.ctrl_spec.get_single_ctrl_val(), int(and_gate.cv1), int(and_gate.cv2)]
        for i, cv in enumerate(cvs):
            if cv == 0:
                bits[i] = bb.add(XGate(), q=bits[i])
        target = soqs['target'] if and_gate.uncompute else bb.allocate(dtype=QUInt(1))
        scratch = bb.allocate(dtype=QUInt(1))
        out = bb.add_d(Toffoli(), ctrl=np.asarray(bits[:2], dtype=object), target=scratch)
        bits[:2], scratch = (list(out['ctrl']), out['target'])
        out = bb.add_d(Toffoli(), ctrl=np.asarray([scratch, bits[2]], dtype=object), target=target)
        scratch, bits[2] = out['ctrl']
        target = out['target']
        out = bb.add_d(Toffoli(), ctrl=np.asarray(bits[:2], dtype=object), target=scratch)
        bits[:2], scratch = (list(out['ctrl']), out['target'])
        bb.free(scratch)
        for i, cv in enumerate(cvs):
            if cv == 0:
                bits[i] = bb.add(XGate(), q=bits[i])
        result = {name: bits[0], 'ctrl': np.asarray(bits[1:], dtype=object)}
        if and_gate.uncompute:
            bb.free(target)
        else:
            result['target'] = target
        return result


def slac_coherent_cbloq(bloq):
    """Replace unsupported C[And] nodes with explicit coherent Toffoli circuits."""
    cb = bloq.decompose_bloq().flatten(pred=lambda inst: not _slac_controlled_and(inst.bloq))
    bb, _ = BloqBuilder.from_signature(cb.signature)
    soq_map = []
    for inst, inputs, outputs in cb.iter_bloqsoqs():
        replacement = CoherentControlledAnd(inst.bloq) if _slac_controlled_and(inst.bloq) else inst.bloq
        new_outputs = bb.add_t(replacement, **bb.map_soqs(inputs, soq_map))
        soq_map.extend(zip(outputs, new_outputs))
    return bb.finalize(**bb.map_soqs(cb.final_soqs(), soq_map))


def _slac_controlled_and(bloq):
    return isinstance(bloq, Controlled) and isinstance(bloq.subbloq, And)


def wavelet_scale_weights(n, order=2):
    """P_p = diag(I_2, 2^(-p/2) I_2, 2^(-p) I_4, ...)."""
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError('n must be an integer >= 2.')
    if not np.isfinite(order) or order <= 0:
        raise ValueError('The derivative order must be finite and positive.')
    scales = np.zeros(1 << n, dtype=int)
    for scale in range(1, n):
        scales[1 << scale:1 << scale + 1] = scale
    return np.exp2(-float(order) * scales / 2)


def wavelet_preconditioner_angles(n, order=2):
    wavelet_scale_weights(n, order)
    theta = np.arccos(np.exp2(-float(order) * np.arange(n) / 2))
    rows = [{'zero_prefix_length': 0, 'condition': 'unconditional', 'phase_increment': theta[-1], 'Rz_angle_radians': -2 * theta[-1]}]
    for length in range(1, n):
        delta = theta[n - 1 - length] - theta[n - length]
        rows.append({'zero_prefix_length': length, 'condition': '0' * length, 'phase_increment': delta, 'Rz_angle_radians': -2 * delta})
    return pd.DataFrame(rows)


@attrs.frozen
class WaveletScalePreconditioner(Bloq):
    """Ancilla-zero block is P_p; p=2 gives the pictured matrix P.

    H exp(i theta_s Z) H has diagonal amplitude cos(theta_s) = 2^(-p*s/2).
    Clean prefix flags implement nested controls with linear gate overhead.
    """
    bitsize: int
    order: float = 2

    @cached_property
    def signature(self):
        wavelet_scale_weights(self.bitsize, self.order)
        return Signature([Register('p_anc', QBit()), Register('sys', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, p_anc, sys):
        angles = wavelet_preconditioner_angles(self.bitsize, self.order)['Rz_angle_radians'].to_numpy()
        bits = list(bb.split(sys))
        p_anc = bb.add(Hadamard(), q=p_anc)
        p_anc = bb.add(Rz(float(angles[0])), q=p_anc)
        for i in range(self.bitsize - 1):
            bits[i] = bb.add(XGate(), q=bits[i])

        def controlled_rz(flag, target, angle):
            target = bb.add(Rz(float(angle / 2)), q=target)
            flag, target = bb.add(CNOT(), ctrl=flag, target=target)
            target = bb.add(Rz(float(-angle / 2)), q=target)
            flag, target = bb.add(CNOT(), ctrl=flag, target=target)
            return (flag, target)
        bits[0], p_anc = controlled_rz(bits[0], p_anc, angles[1])
        flags = []
        for i in range(1, self.bitsize - 1):
            parent = bits[0] if i == 1 else flags[-1]
            flag = bb.allocate(dtype=QBit())
            out = bb.add_d(Toffoli(), ctrl=np.asarray([parent, bits[i]], dtype=object), target=flag)
            parent, bits[i] = out['ctrl']
            if i == 1:
                bits[0] = parent
            else:
                flags[-1] = parent
            flag, p_anc = controlled_rz(out['target'], p_anc, angles[i + 1])
            flags.append(flag)
        for i in range(self.bitsize - 2, 0, -1):
            parent = bits[0] if i == 1 else flags[i - 2]
            out = bb.add_d(Toffoli(), ctrl=np.asarray([parent, bits[i]], dtype=object), target=flags[i - 1])
            parent, bits[i] = out['ctrl']
            if i == 1:
                bits[0] = parent
            else:
                flags[i - 2] = parent
            bb.free(out['target'])
        for i in range(self.bitsize - 1):
            bits[i] = bb.add(XGate(), q=bits[i])
        p_anc = bb.add(Hadamard(), q=p_anc)
        return {'p_anc': p_anc, 'sys': bb.join(np.asarray(bits, dtype=object), dtype=QUInt(self.bitsize))}


def slac_inverse_square_alpha(n):
    if not isinstance(n, (int, np.integer)) or n < 3:
        raise ValueError('Use integer n >= 3.')
    return np.pi ** 2 / 3 + 8 * (1 - 2.0 ** (-(n - 1)))


@attrs.frozen
class SLACProductAccumulate(Bloq):
    """Executable schoolbook out <- out + a*b, using native controlled adders."""
    a_bits: int
    b_bits: int

    @cached_property
    def signature(self):
        return Signature.build_from_dtypes(a=QUInt(self.a_bits), b=QUInt(self.b_bits), result=QUInt(self.a_bits + self.b_bits))

    def build_composite_bloq(self, bb, *, a, b, result):
        bits = list(bb.split(a))
        result_bits = list(bb.split(result))
        for shift in range(self.a_bits):
            width = self.a_bits + self.b_bits - shift
            upper = bb.join(np.asarray(result_bits[:width], dtype=object), dtype=QUInt(width))
            out = bb.add_d(CAdd(QUInt(self.b_bits), QUInt(width)), ctrl=bits[-1 - shift], a=b, b=upper)
            bits[-1 - shift], b = (out['ctrl'], out['a'])
            result_bits[:width] = list(bb.split(out['b']))
        return {'a': bb.join(np.asarray(bits, dtype=object), dtype=QUInt(self.a_bits)), 'b': b, 'result': bb.join(np.asarray(result_bits, dtype=object), dtype=QUInt(self.a_bits + self.b_bits))}


@attrs.frozen
class SLACInverseSquareTest(Bloq):
    """f ^= a AND [m*j^2 >= M*2^(2*mu)], with all arithmetic scratch cleaned."""
    bitsize: int
    n_ref: int

    @cached_property
    def signature(self):
        slac_inverse_square_alpha(self.bitsize)
        if not isinstance(self.n_ref, (int, np.integer)) or self.n_ref < 1:
            raise ValueError('n_ref must be a positive integer.')
        q = self.bitsize - 1
        return Signature.build_from_dtypes(a=QBit(), mu=QAny(q), j=QUInt(q), ref=QUInt(self.n_ref), f=QBit())

    def build_composite_bloq(self, bb, *, a, mu, j, ref, f):
        q, r = (self.bitsize - 1, self.n_ref)
        j = _reverse_quint_register(bb, j, q)
        j_bits = list(bb.split(j))
        copy_bits = []
        for i in range(q):
            bit = bb.allocate(dtype=QBit())
            j_bits[i], bit = bb.add(CNOT(), ctrl=j_bits[i], target=bit)
            copy_bits.append(bit)
        j = bb.join(np.asarray(j_bits, dtype=object), dtype=QUInt(q))
        copy = bb.join(np.asarray(copy_bits, dtype=object), dtype=QUInt(q))
        square = bb.allocate(dtype=QUInt(2 * q))
        square_op = SLACProductAccumulate(q, q)
        out = bb.add_d(square_op, a=j, b=copy, result=square)
        j, copy, square = (out['a'], out['b'], out['result'])
        product_op = SLACProductAccumulate(r, 2 * q)
        rhs = bb.allocate(dtype=QUInt(r + 2 * q))
        out = bb.add_d(product_op, a=ref, b=square, result=rhs)
        ref, square, rhs = (out['a'], out['b'], out['result'])
        mu_bits = list(bb.split(mu))
        lhs_bits = []
        active_positions = {r + 2 * i: i for i in range(q)}
        for power in range(r + 2 * q):
            lhs_bits.append(mu_bits[active_positions[power]] if power in active_positions else bb.allocate(dtype=QBit()))
        lhs = bb.join(np.asarray(lhs_bits[::-1], dtype=object), dtype=QUInt(r + 2 * q))
        compare = LessThanEqual(r + 2 * q, r + 2 * q)
        failed = bb.allocate(dtype=QBit())
        out = bb.add_d(compare, x=lhs, y=rhs, target=failed)
        lhs, rhs, failed = (out['x'], out['y'], out['target'])
        out = bb.add_d(Toffoli(), ctrl=np.asarray([a, failed], dtype=object), target=f)
        a, failed = out['ctrl']
        f = out['target']
        out = bb.add_d(compare.adjoint(), x=lhs, y=rhs, target=failed)
        lhs, rhs = (out['x'], out['y'])
        bb.free(out['target'])
        lhs_bits = list(bb.split(lhs))[::-1]
        for power, bit in enumerate(lhs_bits):
            if power in active_positions:
                mu_bits[active_positions[power]] = bit
            else:
                bb.free(bit)
        mu = bb.join(np.asarray(mu_bits, dtype=object), dtype=QAny(q))
        out = bb.add_d(product_op.adjoint(), a=ref, b=square, result=rhs)
        ref, square = (out['a'], out['b'])
        bb.free(out['result'])
        out = bb.add_d(square_op.adjoint(), a=j, b=copy, result=square)
        j, copy = (out['a'], out['b'])
        bb.free(out['result'])
        j_bits, copy_bits = (list(bb.split(j)), list(bb.split(copy)))
        for i in range(q):
            j_bits[i], copy_bits[i] = bb.add(CNOT(), ctrl=j_bits[i], target=copy_bits[i])
            bb.free(copy_bits[i])
        j = bb.join(np.asarray(j_bits, dtype=object), dtype=QUInt(q))
        j = _reverse_quint_register(bb, j, q)
        return {'a': a, 'mu': mu, 'j': j, 'ref': ref, 'f': f}


@attrs.frozen
class CompleteInverseSquarePrep(Bloq):
    bitsize: int
    n_ref: int

    @cached_property
    def signature(self):
        return Signature([Register('d', QBit()), *SLACInverseSquareTest(self.bitsize, self.n_ref).signature])

    def build_composite_bloq(self, bb, *, d, a, mu, j, ref, f):
        alpha = slac_inverse_square_alpha(self.bitsize)
        old_angle = 2 * np.arccos(np.sqrt(np.pi ** 2 / (np.pi ** 2 + 24)))
        new_angle = 2 * np.arccos(np.sqrt(np.pi ** 2 / 3 / alpha))
        a = bb.add(Ry(float(new_angle - old_angle)), q=a)
        out = bb.add_d(build_slac_laplacian_prep_steps_1_2_3_4_5_v2(self.bitsize), a=a, mu=mu, j=j)
        a, mu, j = (out['a'], out['mu'], out['j'])
        ref_bits = list(bb.split(ref))
        for i in range(self.n_ref):
            a, ref_bits[i] = bb.add(CHadamard(), ctrl=a, target=ref_bits[i])
        ref = bb.join(np.asarray(ref_bits, dtype=object), dtype=QUInt(self.n_ref))
        out = bb.add_d(SLACInverseSquareTest(self.bitsize, self.n_ref), a=a, mu=mu, j=j, ref=ref, f=f)
        out['a'], d = bb.add(CHadamard(), ctrl=out['a'], target=d)
        return {'d': d, **out}


@attrs.frozen
class CompleteInverseSquareBlockEncoding(Bloq):
    """PREP, exact -Z parity sign, failure COPY, SELECT, PREP-adjoint."""
    bitsize: int
    n_ref: int

    @cached_property
    def signature(self):
        return Signature([*CompleteInverseSquarePrep(self.bitsize, self.n_ref).signature, Register('marker', QBit()), Register('sys', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb, *, marker, sys, **soqs):
        prep = CompleteInverseSquarePrep(self.bitsize, self.n_ref)
        soqs = bb.add_d(prep, **soqs)
        j_bits = list(bb.split(soqs['j']))
        for gate in (XGate(), ZGate(), XGate()):
            j_bits[0] = bb.add(gate, q=j_bits[0])
        soqs['j'] = bb.join(np.asarray(j_bits, dtype=object), dtype=QUInt(self.bitsize - 1))
        soqs['f'], marker = bb.add(CNOT(), ctrl=soqs['f'], target=marker)
        j_select = _reverse_quint_register(bb, soqs['j'], self.bitsize - 1)
        out = bb.add_d(build_slac_laplacian_select_halfrange_cbloq(self.bitsize), d=soqs['d'], j=j_select, sys=sys)
        soqs['d'], sys = (out['d'], out['sys'])
        soqs['j'] = _reverse_quint_register(bb, out['j'], self.bitsize - 1)
        soqs = bb.add_d(prep.adjoint(), **soqs)
        return {**soqs, 'marker': marker, 'sys': sys}


@attrs.frozen
class CompleteMultiscaleInverseSquare(Bloq):
    bitsize: int
    n_ref: int
    precondition: bool = False

    @cached_property
    def signature(self):
        registers = list(CompleteInverseSquareBlockEncoding(self.bitsize, self.n_ref).signature)
        if self.precondition:
            registers = [Register('p_left', QBit()), Register('p_right', QBit()), *registers]
        return Signature(registers)

    def build_composite_bloq(self, bb, **soqs):
        left = soqs.pop('p_left') if self.precondition else None
        right = soqs.pop('p_right') if self.precondition else None
        p = WaveletScalePreconditioner(self.bitsize)
        if self.precondition:
            right, soqs['sys'] = bb.add(p, p_anc=right, sys=soqs['sys'])
        w = PrefixMultiscaleShannonQSWT(self.bitsize)
        soqs['sys'] = bb.add(w.adjoint(), system=soqs['sys'])
        soqs = bb.add_d(CompleteInverseSquareBlockEncoding(self.bitsize, self.n_ref), **soqs)
        soqs['sys'] = bb.add(w, system=soqs['sys'])
        if self.precondition:
            left, soqs['sys'] = bb.add(p, p_anc=left, sys=soqs['sys'])
            soqs.update(p_left=left, p_right=right)
        return soqs


def inverse_square_column(n, n_ref=None):
    """Paper's truncated D2 coefficients, or their exact finite-M circuit values."""
    size = 1 << n
    column = np.zeros(size)
    column[0] = -np.pi ** 2 / 3
    for j in range(1, size // 2):
        if n_ref is None:
            magnitude = 2 / j ** 2
        else:
            mu = j.bit_length() - 1
            numerator = 1 << 2 * mu + n_ref
            accepted = (numerator + j * j - 1) // (j * j)
            magnitude = 2 * accepted / (1 << 2 * mu + n_ref)
        column[j] = column[size - j] = (-1) ** (1 + j) * magnitude
    return column


def circulant_from_column(column):
    indices = np.arange(len(column))
    return np.asarray(column)[(indices[:, None] - indices[None, :]) % len(column)]
