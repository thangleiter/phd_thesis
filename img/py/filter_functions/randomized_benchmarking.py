# %% Imports
import datetime
import pathlib
import sys
import time
from collections import defaultdict
from itertools import compress
from pathlib import Path
from typing import Sequence

import filter_functions as ff
import lindblad_mc_tools as lmt
import matplotlib.pyplot as plt
import numpy as np
import quaternion as quat
import qutil.linalg as qla
import qutip as qt
from filter_functions import PulseSequence, util
from numpy import ndarray
from numpy.random import permutation, randn
from qutil.plotting.colors import RWTH_COLORS
from scipy import odr
from scipy.integrate import trapezoid
from scipy.io import loadmat
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, MARGINSTYLE, TOTALWIDTH, MARGINWIDTH, PATH, init

LINE_COLORS = list(RWTH_COLORS.values())
FILE_PATH = pathlib.Path(__file__).relative_to(pathlib.Path(__file__).parents[3])
DATA_PATH = PATH.parent / 'data/filter_functions'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)

init(MAINSTYLE, backend := 'pgf')
# %% Function definitions


def MC_ent_fidelity(gates, target):
    """Calculate the entanglement fidelity"""
    d = gates.shape[-1]
    return util.abs2(np.einsum('...ll', gates @ target.conj().T)/d)


def MC_avg_gate_fidelity(gates, target):
    """Calculate the average gate fidelity"""
    d = gates.shape[-1]
    return (d*MC_ent_fidelity(gates, target) + 1)/(d + 1)


def MC_state_fidelity(gates, psi: ndarray = None):
    """Calculate state fidelity for input state psi"""
    if psi is None:
        psi = np.c_[1:-1:-1]

    fidelity = np.abs(psi.T @ gates @ psi).squeeze()
    return fidelity


def FF_state_infidelity(pulse: PulseSequence, S: ndarray, omega: ndarray,
                        ind: int = 3) -> float:
    R = pulse.get_control_matrix(omega)
    F = np.einsum('jko->jo', ff.util.abs2(R[:, np.delete([0, 1, 2, 3], ind)]))
    return trapezoid(F*S, omega)/(2*np.pi*pulse.d)


def monte_carlo_gate(N_MC: int, S0: float, alpha: float,
                     f_min: float, f_max: float, dt: ndarray,
                     coeffs: Sequence[ndarray], seed: int | None = 42, threads: int = 1):
    """
    Return N_MC gates with ceil(dt.min()*f_max) noise steps per time step of
    the gate.
    """
    # Indices of nonzero control vector components
    coeff_idx = np.array([coeff is not None for coeff in coeffs])

    samplers = [
        lmt.noise.FFTSpectralSampler(N_MC, lmt.noise.PowerLawSpectralFunction(S0[i], -alpha),
                                     dt, f_max=f_max, f_min=f_min, seed=seed, threads=threads)
        if coeff_idx[i] else None for i in range(len(coeffs))
    ]
    N_n = samplers[0].shape[-1] // dt.size
    dt = np.repeat(dt, N_n)/N_n
    a = np.zeros((coeff_idx.sum(), N_MC, dt.size))

    i = 0
    for idx, val in enumerate(coeff_idx):
        if val:
            # Control on this pauli vector component
            a[i] = coeffs[idx] + samplers[idx]().squeeze()
            i += 1

    a *= dt/2
    U = qla.pauli_expm(a, coeff_idx)
    U_tot = qla.mdot(np.flip(U.transpose([3, 2, 0, 1]), axis=0))
    return U_tot


def run_randomized_benchmarking(N_G: int, N_m: int, N_MC: int,
                                m_min: int, m_max: int,
                                S0: ndarray, alpha: ndarray, omega: ndarray,
                                gate_type: str, run_MC: bool = True,
                                seed: int | None = 42, threads: int = 1):
    """
    Run monte carlo randomized benchmarking with gate independent errors
    """
    if run_MC:
        MC_gates = {a: np.empty((N_m, N_G, N_MC, 2, 2), dtype=complex) for a in alpha}

    FF_infid_tot = {a: np.empty((N_m, N_G, 3), dtype=float) for a in alpha}
    lengths = np.round(np.linspace(m_min, m_max, N_m)).astype(int)
    delta_t = []
    S = np.einsum('an,oa->ano', S0, np.power.outer(1/omega, alpha))

    t_now = [time.perf_counter()]
    print(f'Start simulation with {N_m} sequence lengths')
    print('------------------------------------------------')
    for m, length in enumerate(lengths):
        t_now.append(time.perf_counter())
        delta_t.append(t_now[-1] - t_now[-2])
        for j in tqdm(range(N_G), desc=f'Sequence length {length}'):
            randints = np.random.randint(0, len(cliffords[gate_type]), length)
            cliffs = cliffords[gate_type][randints]
            # Total RB sequence without inverting gate
            U_RB = ff.concatenate(cliffs, omega=omega)
            U_inv = find_inverse(U_RB.total_propagator, gate_type)
            U_ID = U_RB @ U_inv

            for k, a in enumerate(alpha):
                FF_infid_tot[a][m, j] = FF_state_infidelity(U_ID, S[k], omega)
                # Run Monte Carlo only on RB sequence since we take the
                # inversion gate to be perfect
                if run_MC:
                    if gate_type == 'naive':
                        x_coeff = U_RB.c_coeffs[0]
                        y_coeff = U_RB.c_coeffs[1]
                        z_coeff = None
                    elif gate_type == 'zyz':
                        x_coeff = None
                        y_coeff = U_RB.c_coeffs[0]
                        z_coeff = U_RB.c_coeffs[1]
                    elif gate_type == 'simple':
                        coeffs = []
                        i = 0
                        for j in ('X', 'Y', 'Z'):
                            if j in U_RB.c_oper_identifiers:
                                coeffs.append(U_RB.c_coeffs[i])
                                i += 1
                            else:
                                coeffs.append(None)

                        x_coeff, y_coeff, z_coeff = coeffs
                    elif gate_type == 'optimized':
                        x_coeff = U_RB.c_coeffs[0]
                        y_coeff = None
                        z_coeff = U_RB.c_coeffs[1]

                    # Don't forget to append inversion gate
                    MC_gates[a][m, j] = U_inv.total_propagator @ monte_carlo_gate(
                        N_MC, S0[alpha.index(a)], a,
                        1e-2/U_RB.t[-1], 1e2/U_RB.t[-1], U_RB.dt,
                        [x_coeff, y_coeff, z_coeff],
                        seed=seed, threads=threads
                    )
    print('------------------------------------------------')
    print(f'Finished simulation in {t_now[-1] - t_now[0]:2f} s')

    if run_MC:
        return MC_gates, FF_infid_tot
    else:
        return FF_infid_tot


def find_inverse(U: ndarray, gate_type) -> ndarray:
    """
    Function to find the inverting gate to take the input state back to itself.
    """
    eye = np.identity(U.shape[0])
    if util.oper_equiv(U, eye, eps=1e-8)[0]:
        return Id[gate_type]

    for i, gate in enumerate(permutation(cliffords[gate_type])):
        if util.oper_equiv(gate.total_propagator @ U, eye, eps=1e-8)[0]:
            return gate

    # Shouldn't reach this point because the major axis pi and pi/2 rotations
    # are in the Clifford group, the state is always an eigenstate of a Pauli
    # operator during the pulse sequence.
    raise Exception


def construct_clifford_group(gateset: Sequence[PulseSequence]) -> ndarray:
    """Construct clifford group from gateset (Id, X/2, Y/2)"""
    for gate in gateset:
        gate.cleanup('all')
        gate.diagonalize()

    X2, Y2 = gateset
    cliffords = np.array([
        Y2 @ Y2 @ Y2 @ Y2,                  # Id
        X2 @ X2,                            # X
        Y2 @ Y2,                            # Y
        Y2 @ Y2 @ X2 @ X2,                  # Z
        X2 @ Y2,                            # Y/2 ○ X/2
        X2 @ Y2 @ Y2 @ Y2,                  # -Y/2 ○ X/2
        X2 @ X2 @ X2 @ Y2,                  # Y/2 ○ -X/2
        X2 @ X2 @ X2 @ Y2 @ Y2 @ Y2,        # -Y/2 ○ -X/2
        Y2 @ X2,                            # X/2 ○ Y/2
        Y2 @ X2 @ X2 @ X2,                  # -X/2 ○ Y/2
        Y2 @ Y2 @ Y2 @ X2,                  # X/2 ○ -Y/2
        Y2 @ Y2 @ Y2 @ X2 @ X2 @ X2,        # -X/2 ○ -Y/2
        X2,                                 # X/2
        X2 @ X2 @ X2,                       # -X/2
        Y2,                                 # Y/2
        Y2 @ Y2 @ Y2,                       # -Y/2
        X2 @ Y2 @ Y2 @ Y2 @ X2 @ X2 @ X2,   # Z/2
        X2 @ X2 @ X2 @ Y2 @ Y2 @ Y2 @ X2,   # -Z/2
        X2 @ X2 @ Y2,                       # Y/2 ○ X
        X2 @ X2 @ Y2 @ Y2 @ Y2,             # -Y/2 ○ X
        Y2 @ Y2 @ X2,                       # X/2 ○ Y
        Y2 @ Y2 @ X2 @ X2 @ X2,             # -X/2 ○ Y
        X2 @ Y2 @ X2,                       # X/2 ○ Y/2 ○ X/2
        X2 @ Y2 @ Y2 @ Y2 @ X2              # X/2 ○ -Y/2 ○ X/2
    ], dtype=object)
    return cliffords


# %% Load data
gates = ['X2', 'Y2']
# Set up Hamiltonian for X2, Y2 gate
struct = {'X2': loadmat(DATA_PATH / 'X2ID.mat'),
          'Y2': loadmat(DATA_PATH / 'Y2ID.mat')}
eps = {key: np.asarray(struct[key]['eps'], order='C') for key in gates}
dt = {key: np.asarray(struct[key]['t'].ravel(), order='C') for key in gates}
B = {key: np.asarray(struct[key]['B'].ravel(), order='C') for key in gates}

J = {key: np.exp(eps[key]) for key in gates}
n_dt = {key: len(dt[key]) for key in gates}

d = 2
delta = d**2/(d - 1)/(d + 1)
H = np.empty((3, d, d), dtype=complex)

P0, Px, Py, Pz = util.paulis
H[0] = 1/2*Px
H[1] = 1/2*Py
H[2] = 1/2*Pz

# Target gates
U_t = {
    'Id': P0,
    'X2': (P0 - 1j*Px)/np.sqrt(2),  # sqrt X
    'Y2': (P0 + 1j*Py)/np.sqrt(2)   # sqrt Y
}
# %% Parameters
m_min = 1
m_max = 101

eps0 = 2.7241e-4

T = dt['X2'].sum()
omega = np.geomspace(1e-2/(7*m_max*T), 1e2/T, 500)*2*np.pi
# omega = np.geomspace(1e-2, 1e2/T.mean(), 250)*2*np.pi
# %% Optimized gates
opers = list(H)

c_opers = opers.copy()
n_opers = opers.copy()

identifiers = ['X', 'Y', 'Z']
c_coeffs = {key: [J[key][0], np.zeros(n_dt[key]),
                  B[key][0]*np.ones(n_dt[key])] for key in gates}
n_coeffs = {key: [np.ones(n_dt[key]), np.ones(n_dt[key]), np.ones(n_dt[key])]
            for key in gates}

# Add identity gate, choosing the X/2 operation on the right qubit
H_c = {'optimized': {key: list(zip(c_opers, val, identifiers))
                     for key, val in c_coeffs.items()}}
H_n = {'optimized': {key: list(zip(n_opers, val, identifiers))
                     for key, val in n_coeffs.items()}}
dt = {'optimized': dt}

# #######################
H_c['optimized']['Y2'][0] = list(H_c['optimized']['Y2'][0])
H_c['optimized']['Y2'][0][1] = H_c['optimized']['Y2'][0][1][::-1]
# #######################
# %% naive gates
sens = 1
H_c['naive'] = {'X2': [[H[0], [np.pi/2/T], 'X'],
                       [H[1], [0], 'Y'],
                       [H[2], [0], 'Z']],
                'Y2': [[H[0], [0], 'X'],
                       [H[1], [np.pi/2/T], 'Y'],
                       [H[2], [0], 'Z']]}
H_n['naive'] = {'X2': list(zip(H, [[sens]]*3, identifiers)),
                'Y2': list(zip(H, [[sens]]*3, identifiers))}
dt['naive'] = {'X2': [T],
               'Y2': [T]}
# %% PulseSequences
gate_types = ['naive', 'optimized']
X2 = {gate_type: ff.PulseSequence(H_c[gate_type]['X2'],
                                  H_n[gate_type]['X2'],
                                  dt[gate_type]['X2'])
      for gate_type in gate_types}
Y2 = {gate_type: ff.PulseSequence(H_c[gate_type]['Y2'],
                                  H_n[gate_type]['Y2'],
                                  dt[gate_type]['Y2'])
      for gate_type in gate_types}

X2_perfect = {
    gate_type: ff.PulseSequence(
        H_c[gate_type]['X2'],
        [[f(arg) for f, arg in
          zip([lambda arg: arg, np.zeros_like, lambda arg: arg], H)]
         for H in H_n[gate_type]['X2']],
        dt[gate_type]['X2']
    ) for gate_type in gate_types
}
Y2_perfect = {
    gate_type: ff.PulseSequence(
        H_c[gate_type]['Y2'],
        [[f(arg) for f, arg in
          zip([lambda arg: arg, np.zeros_like, lambda arg: arg], H)]
         for H in H_n[gate_type]['Y2']],
        dt[gate_type]['Y2']
    ) for gate_type in gate_types
}
# %% Diagonalize and cache control matrices
for gate_type in gate_types:
    X2[gate_type].cache_control_matrix(omega)
    X2_perfect[gate_type].cache_control_matrix(omega)
    Y2[gate_type].cache_control_matrix(omega)
    Y2_perfect[gate_type].cache_control_matrix(omega)

# %% Clifford group
cliffords = {
    k: construct_clifford_group((X2[k], Y2[k]))
    for k in gate_types
}
cliffords_perfect = {
    k: construct_clifford_group((X2_perfect[k], Y2_perfect[k]))
    for k in gate_types
}

Id = {gate_type: cliffords[gate_type][0] for gate_type in gate_types}
Id_perfect = {gate_type: cliffords_perfect[gate_type][0]
              for gate_type in gate_types}
# %% Find ZYZ Euler angles for Clifford group
cliffords['zyz'] = []
cliffords_perfect['zyz'] = []
cliffords['single'] = []
cliffords_perfect['single'] = []
dt_zyz = np.mean([c.t[-1] for c in cliffords['optimized']])/3
P = util.paulis
identifiers = ['X', 'Y', 'Z']

for i, c in enumerate(cliffords['naive']):
    # Random input state
    psi_i = (randn() + 1j*randn())*qt.basis(2)
    psi_i += (randn() + 1j*randn())*qt.basis(2, 1)
    psi_i /= np.linalg.norm(psi_i.full())
    # Output state
    psi_f = qt.Qobj(c.total_propagator)*psi_i
    # Bloch vectors
    b_i = np.array([qt.expect(qt.sigmax(), psi_i),
                    qt.expect(qt.sigmay(), psi_i),
                    qt.expect(qt.sigmaz(), psi_i)])
    b_f = np.array([qt.expect(qt.sigmax(), psi_f),
                    qt.expect(qt.sigmay(), psi_f),
                    qt.expect(qt.sigmaz(), psi_f)])

    coeffs = ff.basis.expand(c.total_propagator, P)
    coeffs[1:] /= -1j
    q = quat.quaternion(*coeffs.real)
    a = quat.as_euler_angles(q)
    r = quat.as_rotation_vector(q)

    r = ff.util.remove_float_errors(r, 10)
    axis = r.astype(bool)

    cliffords['zyz'].append(
        ff.PulseSequence(
            [[H[2], np.array([a[2], 0, a[0]])/dt_zyz, 'Z'],
             [H[1], np.array([0, a[1], 0])/dt_zyz, 'Y']],
            list(zip(H, [[sens]*3]*3, identifiers)),
            [dt_zyz, dt_zyz, dt_zyz]
        )
    )
    cliffords_perfect['zyz'].append(
        ff.PulseSequence(
            [[H[2], np.array([a[2], 0, a[0]])/dt_zyz, 'Z'],
             [H[1], np.array([0, a[1], 0])/dt_zyz, 'Y']],
            list(zip(H, [[0]*3]*3, identifiers)),
            [dt_zyz, dt_zyz, dt_zyz]
        )
    )
    # factor two already in r
    cliffords['single'].append(
        ff.PulseSequence(
            list(zip(compress(P[1:]/2, axis), [[r/3/dt_zyz] for r in compress(r, axis)],
                     compress(identifiers, axis))),
            list(zip(H, [[sens]]*3, identifiers)),
            [3*dt_zyz]
        )
    )
    cliffords_perfect['single'].append(
        ff.PulseSequence(
            list(zip(compress(P[1:]/2, axis), [[r/3/dt_zyz] for r in compress(r, axis)],
                     compress(identifiers, axis))),
            list(zip(H, [[0]]*3, identifiers)),
            [3*dt_zyz]
        )
    )

    # Diagonalize them also
    cliffords['zyz'][-1].cache_control_matrix(omega)
    cliffords_perfect['zyz'][-1].cache_control_matrix(omega)
    cliffords['single'][-1].cache_control_matrix(omega)
    cliffords_perfect['single'][-1].cache_control_matrix(omega)

gate_types += ['zyz', 'single']
cliffords['zyz'] = np.array(cliffords['zyz'], dtype=object)
cliffords['single'] = np.array(cliffords['single'], dtype=object)
cliffords_perfect['zyz'] = np.array(cliffords_perfect['zyz'], dtype=object)
cliffords_perfect['single'] = np.array(cliffords_perfect['single'], dtype=object)
Id['zyz'] = cliffords['zyz'][0]
Id['single'] = cliffords['single'][0]
Id_perfect['zyz'] = cliffords_perfect['zyz'][0]
Id_perfect['single'] = cliffords_perfect['single'][0]

# %% Scale noise
# Scale noise of unoptimized gates so that the average fidelity per clifford is
# the same for all gate types

alpha = [0.7, 0.0]
noise_scaling_factor = defaultdict(dict)

gate_types = ('optimized', 'single', 'naive', 'zyz',)
for gate_type in gate_types:
    for i, a in enumerate(alpha):
        S0 = 4e-11*(2*np.pi*1e-3)**a/eps0**2
        S = S0/omega**a

        if gate_type == 'optimized' and a == alpha[0]:
            optimized_correl_infids = np.array([ff.infidelity(c, S, omega)
                                                for c in cliffords[gate_type]]).mean(0)

        infids = np.array([ff.infidelity(c, S, omega) for c in cliffords[gate_type]]).mean(0)

        noise_scaling_factor[gate_type][a] = optimized_correl_infids / infids

# %% Run RB
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print('================================================')
print(f'Start time:\t\t{now}')
print('================================================')
print()

N_m = 11        # Number of sequence m
N_G = 100      # Number of RB sequences per length
N_MC = 100      # Number of Monte Carlo runs per RB sequence
run_MC = True

MC_gates = defaultdict(dict)
FF_infid_tot = defaultdict(dict)
single_clifford_infids = defaultdict(dict)
single_clifford_ffs = defaultdict(dict)

for gate_type in gate_types:
    print('================================================')
    print(f'Running RB with {gate_type} gates')
    print('================================================')

    single_clifford_ffs[gate_type] = np.mean(
        [c.get_filter_function(omega) for c in cliffords[gate_type]],
        axis=0
    )

    S0 = [4e-11*(2*np.pi*1e-3)**a/eps0**2 *
          noise_scaling_factor[gate_type][a]
          for a in alpha]

    result = run_randomized_benchmarking(
        N_G, N_m, N_MC, m_min, m_max, S0, alpha, omega,
        gate_type, run_MC
    )

    S = np.einsum('an,oa->ano', S0, np.power.outer(1/omega, alpha))

    for i, a in enumerate(alpha):
        single_clifford_infids[gate_type][a] = np.array([ff.infidelity(c, S[i], omega)
                                                         for c in cliffords[gate_type]])

    if run_MC:
        MC_gates[gate_type], FF_infid_tot[gate_type][a] = result
        np.savez_compressed(
            DATA_PATH / f'RB_normalized_XYZ_noise_MC_{gate_type}_gates_{now}',
            white=MC_gates[gate_type][0.0]
        )
    else:
        FF_infid_tot[gate_type] = result

    np.savez(
        DATA_PATH / f'RB_normalized_XYZ_noise_FF_{gate_type}_gates_{now}',
        white=FF_infid_tot[gate_type][alpha[1]],
        correlated=FF_infid_tot[gate_type][alpha[0]]
    )

# %% Load pickled files
"""current in paper
sha = '7fddd61'
date = '20200615-164906'
"""
sha = '6fac17f'  # thesis:'9754468'  # '3baa09e' 500 traces:'c0d179b'
date = '20200615-170254'  # thesis:'20190822-153458'  # '20190716-162957' 500 traces:'20191219-170932' 250 traces: '20191219-181356'
folder = 'RB_normalized_XYZ_noise_no_sensitivities'
m_min, m_max = 1, 101

dpath = Path(r'Z:/MA/data')
# dpath = Path(r'C:/Users/Tobias/Documents/Uni/Physik/Master/Thesis/data')
# dpath = Path('/home/tobias/Physik/Master/Thesis/data')
# dpath = Path('/run/media/tobias/janeway/Hangleiter/MA/data')
# spath = Path(r'Z:/Publication/efficient_calculation_of_generalized_filter_' +
#               r'functions_for_sequences_of_quantum_gates/img')
spath = Path(r'C:/Users/Tobias/Documents/Uni/Physik/Publication/efficient_' +
              r'calculation_of_generalized_filter_functions/img')
# spath = Path(r'/home/tobias/Physik/Publication/efficient_' +
#              r'calculation_of_generalized_filter_functions/img')

gate_types = ['naive', 'optimized', 'single', ]  # 'zyz']
noise_types = ['white', 'correlated']
infid_types = ['tot', 'nocorr']
calc_types = ['FF', 'MC']

MC_gates = {}
data = {'MC': {}, 'FF': {}, 'single_clifford': {}}
single_clifford_FF = {}
for gate_type in gate_types:
    MC_gates[gate_type] = {}
    single_clifford_FF[gate_type] = {}
    data['MC'][gate_type] = {'tot': {}}
    data['FF'][gate_type] = {'tot': {}, 'nocorr': {}}
    data['single_clifford'][gate_type] = {'tot': {}}
    with np.load(dpath / folder / sha / date /
                 f'RB_FF_infids_tot_{gate_type}_gates_m{m_min}-{m_max}.npz') as arch:
        for file in arch.files:
            data['FF'][gate_type]['tot'][file] = arch[file]

    with np.load(dpath / folder / sha / date /
                 f'RB_FF_infids_nocorr_{gate_type}_gates_m{m_min}-{m_max}.npz') as arch:
        for file in arch.files:
            data['FF'][gate_type]['nocorr'][file] = arch[file]

    data['single_clifford'][gate_type]['avg'] = {}
    fname = f'single_clifford_FF_en_infids_{gate_type}_gates.npz'
    with np.load(dpath / folder / sha / date / fname) as arch:
        for file in arch.files:
            data['single_clifford'][gate_type]['tot'][file] = arch[file]
            data['single_clifford'][gate_type]['avg'][file] = arch[file]*2/3

    try:
        with np.load(dpath / folder / sha / date
                     / f'RB_MC_gates_{gate_type}_gates_m{m_min}-{m_max}.npz') as arch:
            for file in arch.files:
                MC_gates[gate_type][file] = arch[file]

        data['MC'][gate_type]['tot']['white'] = 1 - MC_state_fidelity(MC_gates[gate_type]['white'])

        for infid_type, infid in data['MC'][gate_type].items():
            infid['white'] = infid['white'].mean(axis=-1)
    except FileNotFoundError:
        pass

    with np.load(dpath / folder / sha / date /
                 f'single_clifford_FF_avg_{gate_type}_gates.npz') as arch:
        for file in arch.files:
            single_clifford_FF[gate_type][file] = arch[file]

# %% Fit


def exponential(beta, x):
    return 0.5 + 0.5*(1 - 2*beta[1])**x
#    return beta[0] + 0.5*(1 - 2*beta[1])**x
#    return 0.5 + beta[0]*(1 - 2*beta[1])**x


def linear(beta, x):
    return 1 - beta[1]*x
#    return beta[0] - beta[1]*x
#    return 0.5 + beta[0]*(1 - 2*beta[1]*x)


n_m = data['FF']['naive']['tot']['white'].shape[0]
m = np.round(np.linspace(m_min, m_max, n_m)).astype(int)
exp_model = odr.Model(exponential)
lin_model = odr.Model(linear)
fit = {'MC': {}, 'FF': {}}
for calc_type, c in data.items():
    if calc_type != 'single_clifford':
        fit[calc_type] = {}
        for gate_type, g in c.items():
            fit[calc_type][gate_type] = {}
            for infid_type, I in g.items():
                fit[calc_type][gate_type][infid_type] = {}
                for noise_type, n in I.items():
                    print('======================================================')
                    print(f'{calc_type}\t{gate_type}\t{infid_type}\t{noise_type}')
                    print('------------------------------------------------------')
                    fit[calc_type][gate_type][infid_type][noise_type] = {'sep': {}}
                    for i in range(n.shape[-1]):
                        mean = 1 - n[..., i].mean(axis=1)
                        err = n[..., i].std(axis=1)/np.sqrt(n[..., i].shape[1])
                        odr_data = odr.RealData(m, mean, err)
                        if noise_type == 'white' and calc_type == 'MC':
                            model = exp_model
                        else:
                            model = lin_model

                        beta0 = [1, 1e-3]
                        ODR = odr.ODR(odr_data, model, beta0)
                        fit[calc_type][gate_type][infid_type][noise_type]['sep'][i] = ODR.run()
                        fit[calc_type][gate_type][infid_type][noise_type]['sep'][i].pprint()

                    mean = 1 - n.sum(-1).mean(axis=1)
                    err = n.sum(-1).std(axis=1)/np.sqrt(n.sum(-1).shape[1])
                    odr_data = odr.RealData(m, mean, err)
                    if noise_type == 'white' and calc_type == 'MC':
                        model = exp_model
                    else:
                        model = lin_model

                    beta0 = [1, 1e-3]
                    ODR = odr.ODR(odr_data, model, beta0)
                    fit[calc_type][gate_type][infid_type][noise_type]['tot'] = \
                        ODR.run()
                    fit[calc_type][gate_type][infid_type][noise_type]['tot'].pprint()
                    print('======================================================')

# %% all gates white vs correl with inset
gate_types = ['naive', 'optimized', 'single']  # , 'zyz'

fig, axes = plt.subplots(2, 1, sharex=True, sharey=True,
                         figsize=(figsize_narrow[0], figsize_narrow[1]*np.sqrt(2)),
                         gridspec_kw={'height_ratios': [4, 5]})

m = np.round(np.linspace(m_min, m_max, n_m)).astype(int)
for i, (ax, noise_type, subfig) in enumerate(zip(axes, noise_types, ('(a)', '(b)'))):
    means = []
    for gate_type, ms_style, ls_style in zip(gate_types, ms_cycle, ls_cycle):
        N_l, N_G, n_nops = data['FF'][gate_type]['tot'][noise_type].shape
        means.append(ax.errorbar(
            m, 1 - data['FF'][gate_type]['tot'][noise_type].sum(-1).mean(-1),
            (data['FF'][gate_type]['tot'][noise_type].sum(-1).std(-1)),
            linestyle='None', **ms_style
        ))

        fit_tot = ax.plot(
            m, linear(fit['FF'][gate_type]['tot'][noise_type]['tot'].beta, m),
            linewidth=1, **ls_style
        )

    rb_theory = ax.plot(
        m,
        1 - data['single_clifford'][gate_type]['avg'][noise_type].sum(-1).mean()*m,
        linestyle='-', zorder=4, color='k'
    )
    ax.grid(False)
    ax.text(0.90, 0.875 - .05*(1 - i), subfig, transform=ax.transAxes, fontsize=10)
    # ax.tick_params(top=True, bottom=True, left=True, right=True,
    #                direction='out')
    # ax.set_ylabel(r'Survival probability')

# Add proxy subplot for common axis labels
common_ax = fig.add_subplot(111, frameon=False)
common_ax.tick_params(labelcolor='none', top=False, bottom=False,
                      left=False, right=False)
common_ax.set_xticks([])
common_ax.set_yticks([])
common_ax.grid(False)
# common_ax.set_xlabel(r'Sequence length $m$', labelpad=15.)
common_ax.set_ylabel(r'Survival probability $p(\lvert\psi\rangle\!)$',
                     labelpad=35.)
axes[0].legend(loc='lower left', frameon=False,
               labels=gate_types + ['0th order SRB theory'],
               handles=means + rb_theory)
ax.set_xlim(0, 100)
axes[0].set_ylim(0.90, ymax=1)
ax.set_xlabel(r'Sequence length $m$')

# INSET
ins_ax = inset_axes(axes[1], 1, 1)
inset_position = InsetPosition(axes[1], [0.125, 0.125, 0.5, 0.4])
ins_ax.set_axes_locator(inset_position)

k = 2
identifier = 'Z'
for gate_type, ms_style, ls_style in zip(gate_types, ms_cycle, ls_cycle):
    N_l, N_G, n_nops = data['FF'][gate_type]['tot'][noise_type].shape
    mean_tot = ins_ax.errorbar(
        m[:-1], 1 - data['FF'][gate_type]['tot'][noise_type][..., k].mean(-1)[:-1],
        (data['FF'][gate_type]['tot'][noise_type][..., k].std(-1)[:-1]),
        linewidth=0.75, markersize=2, linestyle='None', **ms_style
    )

    fit_tot = ins_ax.plot(
        m, linear(fit['FF'][gate_type]['tot'][noise_type]['sep'][k].beta, m),
        linewidth=0.75, **ls_style
    )

rb_theory = ins_ax.plot(
    m,
    1 - data['single_clifford'][gate_type]['avg'][noise_type][..., k].mean()*m,
    linestyle='-', zorder=4, color='k', linewidth=0.75
)
ins_ax.grid(False)
ins_ax.set_xlim(0, 100)
ins_ax.set_ylim(ymax=1)
ins_ax.tick_params(direction='in', which='both', labelsize=6)
ins_ax.set_xticks(axes[1].get_xticks())
# ins_ax.set_xticklabels([])
ins_ax.set_yticks([0.96, 0.98, 1])
ins_ax.spines['right'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax.patch.set_alpha(0)

plt.draw()

yticks = axes[0].get_yticks()
yticklabels = axes[0].get_yticklabels()
yticklabels[0].set_text('')
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(yticklabels)

plt.draw()

fig.tight_layout(h_pad=0, w_pad=0, pad=0)
fname = 'RB_{}-{}-{}_gates_white_vs_correl_with_Z_noise_inset'.format(*gate_types)
fig.savefig(spath / (fname + '.pdf'))
fig.savefig(spath / (fname + '.eps'))

# # %% avg filter function for single clifford
# fig, ax = plt.subplots(3, 1, sharex=True, sharey=True,
#                        figsize=(figsize_narrow[0], figsize_narrow[0]))
#
# subfigs = ('(a)', '(b)', '(c)')
# for g, (subfig, gate_type) in enumerate(zip(subfigs, gate_types)):
#     dta = single_clifford_FF[gate_type]
#     ax[g].loglog(dta['omega'][250:],
#                  dta['filter_function'][range(3), range(3), 250:].real.T)
#     ax[g].text(0.9, 0.8, subfig, transform=ax[g].transAxes)
#
# ax[g].set_xlim(dta['omega'][250], dta['omega'][-1])
# ax[g].set_xlabel(r'$\omega$ (a.u.)')
# ax[1].legend(labels=['$F_{xx}$', '$F_{yy}$', '$F_{zz}$'], ncol=3,
#              frameon=False)
#
# # Add proxy subplot for common axis labels
# common_ax = fig.add_subplot(111, frameon=False)
# common_ax.tick_params(labelcolor='none', top=False, bottom=False,
#                       left=False, right=False)
# common_ax.set_xticks([])
# common_ax.set_yticks([])
# common_ax.grid(False)
# common_ax.set_ylabel(r'Filter function', labelpad=30.)
#
# fig.tight_layout(pad=0)
# fname = 'single_clifford_avg_FF_{}-{}-{}_gates'.format(*gate_types)
# fig.savefig(spath / (fname + '.pdf'))
# fig.savefig(spath / (fname + '.eps'))
