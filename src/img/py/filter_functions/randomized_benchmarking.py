# %% Imports
import datetime
import pathlib
import os
import sys
import time
from collections import defaultdict
from itertools import compress
from typing import Sequence

from cycler import cycler
import filter_functions as ff
import matplotlib.pyplot as plt
import numpy as np
import quaternion as quat
import qutil.linalg as qla
import qutip as qt
from filter_functions import PulseSequence, util
from numpy import ndarray
from numpy.random import permutation, randn
from qutil.plotting.colors import RWTH_COLORS
from scipy.integrate import trapezoid
from scipy.io import loadmat
from scipy.optimize import curve_fit
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, TOTALWIDTH, PATH, init, markerprops

LINE_COLORS = list(RWTH_COLORS.values())
DATA_PATH = PATH.parent / 'data/filter_functions'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)
RUN_SIMULATION = os.environ.get('RUN_SIMULATION', 'False') == 'True'

init(MAINSTYLE, backend := 'pgf')

# %% Function definitions


def MC_state_fidelity(gates, psi: ndarray = None):
    """Calculate state fidelity for input state psi"""
    if psi is None:
        psi = np.c_[1:-1:-1]

    fidelity = np.abs(psi.T @ gates @ psi).squeeze()
    return fidelity


def run_simulation(append_date=True, run_MC: bool = False, seed: int = 42, threads: int = 1):
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
        import lindblad_mc_tools as lmt

        # Indices of nonzero control vector components
        coeff_idx = np.array([coeff is not None for coeff in coeffs])

        samplers = []
        for i in range(len(coeffs)):
            if not coeff_idx[i]:
                samplers.append(None)
            else:
                sampler = lmt.noise.FFTSpectralSampler(
                    N_MC, lmt.noise.PowerLawSpectralFunction(S0[i], -alpha),
                    dt, f_max=f_max, f_min=f_min, seed=seed, threads=threads
                )
                samplers.append(sampler)
        if all(sampler is None for sampler in samplers):
            raise RuntimeError

        N_n = sampler.shape[-1] // dt.size
        dt = np.repeat(dt, N_n)/N_n
        a = np.zeros((coeff_idx.sum(), N_MC, dt.size))

        i = 0
        for idx, val in enumerate(coeff_idx):
            if val:
                # Control on this pauli vector component
                a[i] = np.tile(np.repeat(coeffs[idx], N_n), (N_MC, 1)) + samplers[idx]().squeeze()
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
        S = np.einsum('an,oa->ano', S0, np.power.outer(1/omega, alpha))

        t_start = time.perf_counter()
        print(f'Start simulation with {N_m} sequence lengths')
        print('------------------------------------------------')
        for m, length in enumerate(lengths):
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
                    if not run_MC:
                        continue

                    match gate_type:
                        case 'naive':
                            coeffs = [U_RB.c_coeffs[0], U_RB.c_coeffs[1], None]
                        case 'zyz':
                            coeffs = [U_RB.c_coeffs[0], U_RB.c_coeffs[1], None]
                        case 'single':
                            coeffs = []
                            i = 0
                            for ax in ('X', 'Y', 'Z'):
                                if ax in U_RB.c_oper_identifiers:
                                    coeffs.append(U_RB.c_coeffs[i])
                                    i += 1
                                else:
                                    coeffs.append(None)
                        case 'optimized':
                            coeffs = [U_RB.c_coeffs[0], None, U_RB.c_coeffs[1]]

                    # Don't forget to append inversion gate
                    MC_gates[a][m, j] = U_inv.total_propagator @ monte_carlo_gate(
                        N_MC, S0[alpha.index(a)], a,
                        1e-2/U_RB.t[-1], 1e2/U_RB.t[-1], U_RB.dt, coeffs,
                        seed=seed, threads=threads
                    )
        print('------------------------------------------------')
        print(f'Finished simulation in {time.perf_counter() - t_start:2f} s')

        if run_MC:
            return MC_gates, FF_infid_tot, lengths
        else:
            return FF_infid_tot, lengths

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

    # Load data
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
    H = np.empty((3, d, d), dtype=complex)

    P0, Px, Py, Pz = util.paulis
    H[0] = 1/2*Px
    H[1] = 1/2*Py
    H[2] = 1/2*Pz

    # Parameters
    m_min = 1
    m_max = 101

    eps0 = 2.7241e-4

    T = dt['X2'].sum()
    omega = np.geomspace(1e-2/(7*m_max*T), 1e2/T, 500)*2*np.pi
    # omega = np.geomspace(1e-2, 1e2/T.mean(), 250)*2*np.pi
    # Optimized gates
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
    # naive gates
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
    # PulseSequences
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
    # Diagonalize and cache control matrices
    for gate_type in gate_types:
        X2[gate_type].cache_control_matrix(omega)
        X2_perfect[gate_type].cache_control_matrix(omega)
        Y2[gate_type].cache_control_matrix(omega)
        Y2_perfect[gate_type].cache_control_matrix(omega)

    # Clifford group
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
    # Find ZYZ Euler angles for Clifford group
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

    # Scale noise
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

    # Run RB
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print('================================================')
    print(f'Start time:\t\t{now}')
    print('================================================')
    print()

    N_m = 11  # Number of sequence m
    N_G = 100  # Number of RB sequences per length
    N_MC = 100  # Number of Monte Carlo runs per RB sequence

    MC_gates = defaultdict(dict)
    FF_infid_tot = defaultdict(dict)
    single_clifford_infids = defaultdict(dict)

    for gate_type in gate_types:
        print('================================================')
        print(f'Running RB with {gate_type} gates')
        print('================================================')

        S0 = [4e-11*(2*np.pi*1e-3)**a/eps0**2*noise_scaling_factor[gate_type][a]
              for a in alpha]
        S = np.einsum('an,oa->ano', S0, np.power.outer(1/omega, alpha))

        result = run_randomized_benchmarking(
            N_G, N_m, N_MC, m_min, m_max, S0, alpha, omega, gate_type,
            run_MC, seed, threads
        )

        for i, a in enumerate(alpha):
            single_clifford_infids[gate_type][a] = np.array([ff.infidelity(c, S[i], omega)
                                                             for c in cliffords[gate_type]])

        ext = f'_{now}' if append_date else ''

        if run_MC:
            MC_gates[gate_type], FF_infid_tot[gate_type], lengths = result
            np.savez_compressed(
                DATA_PATH / (f'RB_normalized_XYZ_noise_MC_{gate_type}_gates' + ext),
                white=MC_gates[gate_type][alpha[1]],
                correlated=MC_gates[gate_type][alpha[0]],
                lengths=lengths,
            )
        else:
            FF_infid_tot[gate_type], lengths = result

        np.savez(
            DATA_PATH / (f'RB_normalized_XYZ_noise_FF_{gate_type}_gates' + ext),
            white=FF_infid_tot[gate_type][alpha[1]],
            correlated=FF_infid_tot[gate_type][alpha[0]],
            lengths=lengths,
            omega=omega,
            S=S
        )
        np.savez(
            DATA_PATH / (f'single_cliffords_normalized_XYZ_noise_FF_{gate_type}_gates' + ext),
            white=single_clifford_infids[gate_type][alpha[1]],
            correlated=single_clifford_infids[gate_type][alpha[0]],
            lengths=lengths,
            omega=omega,
            S=S
        )


# %% Run benchmark
if RUN_SIMULATION:
    run_simulation(append_date=False, run_MC=False, threads=None)

# %% Load pickled files
gate_types = ['naive', 'optimized', 'single']
noise_types = ['white', 'correlated']
infid_types = ['tot']
calc_types = ['FF', 'MC']

MC_gates = defaultdict(dict)
data = defaultdict(dict)
for gate_type in gate_types:
    data['FF'][gate_type] = defaultdict(dict)
    data['single_cliffords'][gate_type] = defaultdict(dict)
    MC_gates[gate_type] = defaultdict(dict)
    with np.load(DATA_PATH / f'RB_normalized_XYZ_noise_FF_{gate_type}_gates.npz') as arch:
        for file in arch.files:
            data['FF'][gate_type][file] = arch[file]

    with np.load(
            DATA_PATH / f'single_cliffords_normalized_XYZ_noise_FF_{gate_type}_gates.npz'
    ) as arch:
        for file in arch.files:
            data['single_cliffords'][gate_type][file] = arch[file]

    if 'MC' not in calc_types:
        continue

    data['MC'][gate_type] = defaultdict(dict)
    try:
        with np.load(
                DATA_PATH / f'RB_normalized_XYZ_noise_MC_{gate_type}_gates.npz'
        ) as arch:
            for file in arch.files:
                MC_gates[gate_type][file] = arch[file]

        data['MC'][gate_type]['white'] = 1 - MC_state_fidelity(
            MC_gates[gate_type]['white']
        ).mean(axis=-1, keepdims=True)
        data['MC'][gate_type]['correlated'] = 1 - MC_state_fidelity(
            MC_gates[gate_type]['correlated']
        ).mean(axis=-1, keepdims=True)
        data['MC'][gate_type]['lengths'] = MC_gates[gate_type]['lengths']
    except FileNotFoundError:
        calc_types.remove('MC')
        data.pop('MC')

# %% Fit


def exponential(m, r):
    return 0.5 + 0.5*(1 - 2*r)**m


def linear(m, r):
    return 1 - r*m


def fitit(m, mean, err, model):
    popt, pcov = curve_fit(model, m, mean, p0=[1e-3], sigma=err)
    return popt[0], np.sqrt(pcov[0, 0])


fit = dict()
for calc_type, c in data.items():
    if calc_type == 'single_cliffords':
        continue

    fit[calc_type] = dict()
    for gate_type, g in c.items():
        fit[calc_type][gate_type] = dict()
        for noise_type in ['white', 'correlated']:
            fit[calc_type][gate_type][noise_type] = dict()
            fit[calc_type][gate_type][noise_type]['sep'] = defaultdict(dict)
            fit[calc_type][gate_type][noise_type]['tot'] = defaultdict(dict)
            if calc_type == 'MC':
                model = exponential
            else:
                model = linear

            m = g['lengths']
            n = g[noise_type]
            r, sr = fitit(m,
                          1 - n.sum(-1).mean(axis=1),
                          n.sum(-1).std(axis=1)/np.sqrt(n.shape[1]),
                          model)
            fit[calc_type][gate_type][noise_type]['tot']['r'] = r
            fit[calc_type][gate_type][noise_type]['tot']['sr'] = sr

            for i in range(n.shape[-1]):
                r, sr = fitit(m,
                              1 - n[:, :, i].mean(axis=1),
                              n[:, :, i].std(axis=1)/np.sqrt(n.shape[1]),
                              model)
                fit[calc_type][gate_type][noise_type]['sep'][i]['r'] = r
                fit[calc_type][gate_type][noise_type]['sep'][i]['sr'] = sr

# %% Plot


def quantile_error(data, q=(0.159, 0.841), axis=-1):
    # return 1 - (mean := data.mean(axis)), abs(mean - np.quantile(data, q, axis=axis))
    return 1 - data.mean(axis), data.std(axis)


gate_types = ['naive', 'optimized', 'single']
cycle = cycler(marker=('.', 'x', '+', 's'), color=LINE_COLORS[:4])

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, layout='constrained',
                         figsize=(TOTALWIDTH, 2.))

for i, (ax, noise_type, subfig) in enumerate(zip(axes, noise_types, ('(a)', '(b)'))):
    means = []
    for gate_type, sty in zip(gate_types, cycle):
        means.append(ax.errorbar(
            data['FF'][gate_type]['lengths'],
            *quantile_error(data['FF'][gate_type][noise_type].sum(-1)),
            **markerprops(**sty),
        ))

        m = np.linspace(data['FF'][gate_type]['lengths'][0],
                        data['FF'][gate_type]['lengths'][-1]*1.1, 2)
        fit_tot = ax.plot(m, linear(m, fit['FF'][gate_type][noise_type]['tot']['r']),
                          color=sty['color'], alpha=0.6)

    rb_theory = ax.plot(
        m,
        # entanglement to average gate infidelity: 2/3
        1 - 2/3*data['single_cliffords'][gate_type][noise_type].sum(-1).mean()*m,
        linestyle='-', zorder=4, color='k'
    )

if backend == 'pgf':
    axes[0].set_ylabel(r'Survival probability $p(\ket{\psi})$')
else:
    axes[0].set_ylabel(r'Survival probability $p(|\psi\rangle)$')
axes[0].legend(loc='lower left', frameon=False,
               labels=gate_types + ['Theory'], handles=means + rb_theory)
axes[0].set_xlim(0, 105)
axes[0].set_ylim(top=1)
fig.supxlabel(r'Sequence length $m$', fontsize='medium')

# INSET
ins_ax = axes[1].inset_axes([0.1, 0.125, 0.4, 0.4])

k = 2
identifier = 'Z'
for gate_type, sty in zip(gate_types, cycle):
    mean_tot = ins_ax.errorbar(
        data['FF'][gate_type]['lengths'],
        *quantile_error(data['FF'][gate_type][noise_type][..., k]),
        **markerprops(**sty, markersize=2.5),
    )

    m = np.linspace(data['FF'][gate_type]['lengths'][0],
                    data['FF'][gate_type]['lengths'][-1]*1.1, 2)
    fit_tot = ins_ax.plot(
        m, linear(m, fit['FF'][gate_type][noise_type]['sep'][k]['r']),
        linewidth=0.75, color=sty['color'], alpha=0.6
    )

rb_theory = ins_ax.plot(
    m,
    1 - 2/3*data['single_cliffords'][gate_type][noise_type][..., k].mean()*m,
    linestyle='-', zorder=4, color='k', linewidth=0.75
)
ins_ax.tick_params(direction='in', which='both', labelsize='x-small')
ins_ax.set_xticks(axes[0].get_xticks())
ins_ax.set_xlim(axes[0].get_xlim())
ins_ax.set_yticks([0.96, 0.98, 1])
ins_ax.spines['right'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax.patch.set_alpha(0)

fig.savefig(SAVE_PATH / 'randomized_benchmarking.pdf')
