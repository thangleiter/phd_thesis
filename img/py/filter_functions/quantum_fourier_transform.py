# %% Imports
import pathlib
import sys
import time
from copy import deepcopy

import filter_functions as ff
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from filter_functions import plotting
from filter_functions.util import get_indices_from_identifiers
from matplotlib import colors, cycler, lines
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_50
from qutip.control import pulseoptim
from qutip.qip import operations
from qutip.qip.algorithms.qft import qft as qt_qft
from scipy import integrate

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, TOTALWIDTH, PATH, init

SHOW_PROGRESSBAR = False
LINE_COLORS = list(RWTH_COLORS.values())
FILE_PATH = pathlib.Path(__file__).relative_to(pathlib.Path(__file__).parents[3])
DATA_PATH = PATH.parent / 'data/filter_functions'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)

init(MAINSTYLE, backend := 'pgf')
pernanosecond = r' (\unit{\per\nano\second})' if backend == 'pgf' else r' (ns$^{-1}$)'
# %% GRAPE optimization

np.random.seed(10)
n_qubits = 4
# Single-qubit control Hamiltonian
H_c_single = [
    qt.sigmax(),
    qt.sigmay()
]
# Initial unitary
U_0_single = qt.qeye(2)
# No constant terms
H_d_single = U_0_single*0

# Two-qubit control Hamiltonian
H_c_two = [
    qt.tensor(qt.sigmax(), qt.qeye(2)),
    qt.tensor(qt.sigmay(), qt.qeye(2)),
    qt.tensor(qt.qeye(2), qt.sigmax()),
    qt.tensor(qt.qeye(2), qt.sigmay()),
    qt.tensor(qt.sigmaz(), qt.sigmaz())
]
# Initial unitary
U_0_two = qt.tensor(qt.qeye(2), qt.qeye(2))
# No constant terms
H_d_two = U_0_two*0

# Define the target unitaries
target_gates = {
    'X_pi2': operations.rotation(qt.sigmax(), np.pi/2),
    'Y_pi2': operations.rotation(qt.sigmay(), np.pi/2),
    'CZ_pi8': operations.cphase(np.pi/2**3)
}
H_c = {
    'X_pi2': H_c_single,
    'Y_pi2': H_c_single,
    'CZ_pi8': H_c_two,
}
H_d = {
    'X_pi2': H_d_single,
    'Y_pi2': H_d_single,
    'CZ_pi8': H_d_two,
}
U_0 = {
    'X_pi2': U_0_single,
    'Y_pi2': U_0_single,
    'CZ_pi8': U_0_two,
}

# Define some optimization parameters
t_sample = 1
n_sample = 30
t_clock = t_sample*n_sample

optim_options = dict(
    alg='GRAPE',                          # algorithm (could also do CRAB)
    num_tslots=n_sample,                  # number of time steps
    evo_time=t_clock,                     # total evolution time
    amp_lbound=0,                         # amplitudes should be positive
    fid_err_targ=1e-12,                   # target overlap
    max_iter=10**5,                       # maximum number of iterations
    init_pulse_type='RND',                # initial amplitude shapes
    init_pulse_params=dict(num_waves=2),  # number of wavelengths of init pulse
    phase_option='PSU',                   # ignore global phase
    method_params=dict(tol=1e-10),        # tolerance
    gen_stats=True,
)

grape_results = {
    name: pulseoptim.optimize_pulse_unitary(
        H_d=H_d[name],
        H_c=H_c[name],
        U_0=U_0[name],
        U_targ=target,
        **optim_options
    ) for name, target in target_gates.items()
}

identifiers = {
    'X_pi2': r'X$_{0}(\pi/2)$',
    'Y_pi2': r'Y$_{0}(\pi/2)$',
    'CZ_pi8': r'CR$_{01}(\pi/2^3)$'
}
c_identifiers = {
    'X_pi2': ['$I_{0}$', '$Q_{0}$'],
    'Y_pi2': ['$I_{0}$', '$Q_{0}$'],
    'CZ_pi8': ['$I_{0}$', '$Q_{0}$', '$I_{1}$', '$Q_{1}$', '$J_{01}$']
}
n_identifiers = {
    'X_pi2': [r'$\sigma_x^{(0)}$', r'$\sigma_y^{(0)}$'],
    'Y_pi2': [r'$\sigma_x^{(0)}$', r'$\sigma_y^{(0)}$'],
    'CZ_pi8': [r'$\sigma_x^{(0)}$', r'$\sigma_y^{(0)}$', r'$\sigma_x^{(1)}$',
               r'$\sigma_y^{(1)}$', r'$\sigma_z^{(0)}\sigma_z^{(1)}$']
}
pulses = {}
for name, result in grape_results.items():
    pulse = ff.PulseSequence(
        list(zip(H_c[name], result.final_amps.T, c_identifiers[name])),
        list(zip(H_c[name], np.ones((len(H_c[name]), n_sample)), n_identifiers[name])),
        [t_clock/n_sample]*n_sample,
        basis=ff.Basis.pauli(int(np.log2(U_0[name].shape[0])))
    )
    pulses[name] = pulse

    # print(name, '\t:', ff.util.oper_equiv(
    #     pulse.total_propagator, target_gates[name], eps=1e-10
    # ))
    # result.stats.report()

omega = np.concatenate([np.arange(0, 1e-4, 1e-4/10), np.geomspace(1e-4, 1e2, 1000)])
with np.errstate(divide='ignore'):
    S = 1e-9/omega
S[0] = S[1]
# %%% Extend pulses
t_start_fast = time.perf_counter()
pulses['X_pi2'].cache_control_matrix(omega, show_progressbar=SHOW_PROGRESSBAR)
pulses['Y_pi2'].cache_control_matrix(omega, show_progressbar=SHOW_PROGRESSBAR)
pulses['CZ_pi8'].cache_control_matrix(omega, show_progressbar=SHOW_PROGRESSBAR)

IDs = [qt.qeye(2)]*(n_qubits - 1)
four_qubit_X = [
    qt.tensor(*(IDs[:i] + [qt.sigmax()] + IDs[i:]))
    for i in range(1, n_qubits)
]
four_qubit_Y = [
    qt.tensor(*(IDs[:i] + [qt.sigmay()] + IDs[i:]))
    for i in range(1, n_qubits)
]
four_qubit_ZZ = [
    qt.tensor(*(IDs[:i] + [qt.sigmaz()]*2 + IDs[i:-1]))
    for i in range(n_qubits - 1)
] + [qt.tensor(*([qt.sigmaz()] + IDs[:n_qubits-2] + [qt.sigmaz()]))]

X_identifiers = [r'$\sigma_x^{{({})}}$'.format(q) for q in range(1, n_qubits)]
Y_identifiers = [r'$\sigma_y^{{({})}}$'.format(q) for q in range(1, n_qubits)]
ZZ_identifiers = [r'$\sigma_z^{{({})}}\sigma_z^{{({})}}$'.format(q, (q + 1) % n_qubits)
                  for q in range(n_qubits)]

# Identifier mapping maps every identifier to itself since we do not change the
# qubit the pulse acts on, only extend the Hilbert space
identifier_mapping = {identifier: identifier for identifier
                      in pulses['CZ_pi8'].n_oper_identifiers}
identifier_mapping.update(**{
    identifier: identifier
    for identifier in pulses['CZ_pi8'].c_oper_identifiers
})
# Additional noise Hamiltonian for one-qubit pulses
H_n_one = (
    list(zip(four_qubit_X, np.ones((n_qubits-1, n_sample)), X_identifiers)) +
    list(zip(four_qubit_Y, np.ones((n_qubits-1, n_sample)), Y_identifiers)) +
    list(zip(four_qubit_ZZ, np.ones((n_qubits, n_sample)), ZZ_identifiers))
)
# Additional noise Hamiltonian for two-qubit pulse
H_n_two = (
    list(zip(four_qubit_X[1:], np.ones((n_qubits-2, n_sample)), X_identifiers[1:])) +
    list(zip(four_qubit_Y[1:], np.ones((n_qubits-2, n_sample)), Y_identifiers[1:])) +
    list(zip(four_qubit_ZZ[1:], np.ones((n_qubits-1, n_sample)), ZZ_identifiers[1:]))
)

# Extend the pulses to four qubits and cache the filter functions for the
# dditional noise operators
four_qubit_pulses = {name: {} for name in pulses.keys()}
four_qubit_pulses['X_pi2'][0] = ff.extend([(pulses['X_pi2'], 0, identifier_mapping)],
                                          N=n_qubits, omega=omega,
                                          additional_noise_Hamiltonian=H_n_one,
                                          cache_filter_function=True,
                                          show_progressbar=SHOW_PROGRESSBAR)
four_qubit_pulses['Y_pi2'][0] = ff.extend([(pulses['Y_pi2'], 0, identifier_mapping)],
                                          N=n_qubits, omega=omega,
                                          additional_noise_Hamiltonian=H_n_one,
                                          cache_filter_function=True,
                                          show_progressbar=SHOW_PROGRESSBAR)
four_qubit_pulses['CZ_pi8'][(0, 1)] = ff.extend([(pulses['CZ_pi8'], (0, 1), identifier_mapping)],
                                                N=n_qubits, omega=omega,
                                                additional_noise_Hamiltonian=H_n_two,
                                                cache_filter_function=True,
                                                show_progressbar=SHOW_PROGRESSBAR)

four_qubit_pulses['hadamard'] = {}
four_qubit_pulses['CZ_pi4'] = {}
four_qubit_pulses['CZ_pi2'] = {}
four_qubit_pulses['CZ_pi'] = {}
four_qubit_pulses['CX_pi'] = {}
four_qubit_pulses['swap'] = {}

four_qubit_pulses['hadamard'][0] = ff.concatenate((four_qubit_pulses['Y_pi2'][0],
                                                   four_qubit_pulses['X_pi2'][0],
                                                   four_qubit_pulses['X_pi2'][0]),
                                                  show_progressbar=SHOW_PROGRESSBAR)
four_qubit_pulses['CZ_pi4'][(0, 1)] = ff.concatenate((four_qubit_pulses['CZ_pi8'][(0, 1)],
                                                      four_qubit_pulses['CZ_pi8'][(0, 1)]),
                                                     show_progressbar=SHOW_PROGRESSBAR)
four_qubit_pulses['CZ_pi2'][(0, 1)] = ff.concatenate((four_qubit_pulses['CZ_pi4'][(0, 1)],
                                                      four_qubit_pulses['CZ_pi4'][(0, 1)]),
                                                     show_progressbar=SHOW_PROGRESSBAR)
four_qubit_pulses['CZ_pi'][(0, 1)] = ff.concatenate((four_qubit_pulses['CZ_pi2'][(0, 1)],
                                                     four_qubit_pulses['CZ_pi2'][(0, 1)]),
                                                    show_progressbar=SHOW_PROGRESSBAR)
# CNOT with control on the second, target on the first qubit
four_qubit_pulses['CX_pi'][(1, 0)] = ff.concatenate((four_qubit_pulses['hadamard'][0],
                                                     four_qubit_pulses['CZ_pi'][(0, 1)],
                                                     four_qubit_pulses['hadamard'][0]),
                                                    show_progressbar=SHOW_PROGRESSBAR)


def cyclical_mapping(shift: int):
    """Shift qubit indices of identifiers by shift"""
    mapping = {}
    mapping.update({
        '$I_{{{}}}$'.format(i):
        '$I_{{{}}}$'.format((i+shift) % n_qubits)
        for i in range(n_qubits)
    })
    mapping.update({
        r'$\sigma_x^{{({})}}$'.format(i):
        r'$\sigma_x^{{({})}}$'.format((i+shift) % n_qubits)
        for i in range(n_qubits)
    })
    mapping.update({
        '$Q_{{{}}}$'.format(i):
        '$Q_{{{}}}$'.format((i+shift) % n_qubits)
        for i in range(n_qubits)
    })
    mapping.update({
        r'$\sigma_y^{{({})}}$'.format(i):
        r'$\sigma_y^{{({})}}$'.format((i+shift) % n_qubits)
        for i in range(n_qubits)
    })
    mapping.update({
        '$J_{{{}{}}}$'.format(i, (i+1) % n_qubits):
        '$J_{{{}{}}}$'.format((i+shift) % n_qubits, (i+shift+1) % n_qubits)
        for i in range(n_qubits)
    })
    mapping.update({
        r'$\sigma_z^{{({})}}\sigma_z^{{({})}}$'.format(i, (i+1) % n_qubits):
        r'$\sigma_z^{{({})}}\sigma_z^{{({})}}$'.format((i+shift) % n_qubits,
                                                       (i+shift+1) % n_qubits)
        for i in range(n_qubits)
    })
    return mapping


four_qubit_pulses['hadamard'][1] = ff.remap(four_qubit_pulses['hadamard'][0],
                                            order=(3, 0, 1, 2),
                                            oper_identifier_mapping=cyclical_mapping(1))
four_qubit_pulses['CX_pi'][(0, 1)] = ff.concatenate((four_qubit_pulses['hadamard'][1],
                                                     four_qubit_pulses['CZ_pi'][(0, 1)],
                                                     four_qubit_pulses['hadamard'][1]),
                                                    show_progressbar=SHOW_PROGRESSBAR)
four_qubit_pulses['swap'][(0, 1)] = ff.concatenate((four_qubit_pulses['CX_pi'][(1, 0)],
                                                    four_qubit_pulses['CX_pi'][(0, 1)],
                                                    four_qubit_pulses['CX_pi'][(1, 0)]),
                                                   show_progressbar=SHOW_PROGRESSBAR)

for q in range(2, n_qubits):
    # We remap the operators cyclically
    order = np.roll(range(n_qubits), q)
    mapping = cyclical_mapping(q)
    four_qubit_pulses['hadamard'][q] = ff.remap(four_qubit_pulses['hadamard'][0],
                                                order,
                                                oper_identifier_mapping=mapping)
for q in range(1, n_qubits-1):
    order = np.roll(range(n_qubits), q)
    mapping = cyclical_mapping(q)
    four_qubit_pulses['CZ_pi8'][(q, q+1)] = ff.remap(four_qubit_pulses['CZ_pi8'][(0, 1)],
                                                     order,
                                                     oper_identifier_mapping=mapping)
    four_qubit_pulses['CZ_pi4'][(q, q+1)] = ff.remap(four_qubit_pulses['CZ_pi4'][(0, 1)],
                                                     order,
                                                     oper_identifier_mapping=mapping)
    four_qubit_pulses['CZ_pi2'][(q, q+1)] = ff.remap(four_qubit_pulses['CZ_pi2'][(0, 1)],
                                                     order,
                                                     oper_identifier_mapping=mapping)
    four_qubit_pulses['CZ_pi'][(q, q+1)] = ff.remap(four_qubit_pulses['CZ_pi'][(0, 1)],
                                                    order,
                                                    oper_identifier_mapping=mapping)
    four_qubit_pulses['CX_pi'][(q, q+1)] = ff.remap(four_qubit_pulses['CX_pi'][(0, 1)],
                                                    order,
                                                    oper_identifier_mapping=mapping)
    four_qubit_pulses['CX_pi'][(q, q-1)] = ff.remap(four_qubit_pulses['CX_pi'][(1, 0)],
                                                    order,
                                                    oper_identifier_mapping=mapping)
    four_qubit_pulses['swap'][(q, q+1)] = ff.remap(four_qubit_pulses['swap'][(0, 1)],
                                                   order,
                                                   oper_identifier_mapping=mapping)

idle = ff.PulseSequence(
    list(zip(H_c_single, np.zeros((2, n_sample)), c_identifiers['X_pi2'])),
    list(zip(H_c_single, np.ones((2, n_sample)), n_identifiers['X_pi2'])),
    [t_clock/n_sample]*n_sample,
    basis=ff.Basis.pauli(1)
)

H_n_add = list(zip(four_qubit_pulses['CZ_pi4'][(0, 1)].n_opers,
                   four_qubit_pulses['CZ_pi4'][(0, 1)].n_coeffs,
                   four_qubit_pulses['CZ_pi4'][(0, 1)].n_oper_identifiers))

four_qubit_pulses['CZ_pi4_echo'] = {}
four_qubit_pulses['CZ_pi4_echo'][(1, 2)] = ff.extend(
    [(ff.concatenate([pulses['X_pi2'], pulses['X_pi2']]), 3,
      {'$I_{0}$': '$I_{3}$', '$Q_{0}$': '$Q_{3}$',
       '$\\sigma_x^{(0)}$': '$\\sigma_x^{(3)}$',
       '$\\sigma_y^{(0)}$': '$\\sigma_y^{(3)}$'}),
     (ff.concatenate([pulses['CZ_pi8'], pulses['CZ_pi8']]), (1, 2),
      {'$I_{0}$': '$I_{1}$',
       '$Q_{0}$': '$Q_{1}$',
       '$I_{1}$': '$I_{2}$',
       '$Q_{1}$': '$Q_{2}$',
       '$J_{01}$': '$J_{12}$',
       '$\\sigma_x^{(0)}$': '$\\sigma_x^{(1)}$',
       '$\\sigma_y^{(0)}$': '$\\sigma_y^{(1)}$',
       '$\\sigma_x^{(1)}$': '$\\sigma_x^{(2)}$',
       '$\\sigma_y^{(1)}$': '$\\sigma_y^{(2)}$',
       '$\\sigma_z^{(0)}\\sigma_z^{(1)}$': '$\\sigma_z^{(1)}\\sigma_z^{(2)}$'})],
    N=n_qubits,
    omega=omega,
    additional_noise_Hamiltonian=[[o, c, i] for o, c, i in H_n_add if i not in
                                  ['$\\sigma_x^{(1)}$', '$\\sigma_y^{(1)}$',
                                   '$\\sigma_x^{(2)}$', '$\\sigma_y^{(2)}$',
                                   '$\\sigma_x^{(3)}$', '$\\sigma_y^{(3)}$',
                                   '$\\sigma_z^{(1)}\\sigma_z^{(2)}$']],
    cache_filter_function=True,
    show_progressbar=SHOW_PROGRESSBAR
)

H_n_add = list(zip(four_qubit_pulses['CZ_pi2'][(0, 1)].n_opers,
                   four_qubit_pulses['CZ_pi2'][(0, 1)].n_coeffs,
                   four_qubit_pulses['CZ_pi2'][(0, 1)].n_oper_identifiers))

four_qubit_pulses['CZ_pi2_echo'] = {}
four_qubit_pulses['CZ_pi2_echo'][(0, 1)] = ff.extend(
    [(ff.concatenate([idle, pulses['X_pi2'], pulses['X_pi2'], idle]), 3,
      {'$I_{0}$': '$I_{3}$',
       '$Q_{0}$': '$Q_{3}$',
       '$\\sigma_x^{(0)}$': '$\\sigma_x^{(3)}$',
       '$\\sigma_y^{(0)}$': '$\\sigma_y^{(3)}$'}),
     (ff.concatenate_periodic(pulses['CZ_pi8'], 4), (0, 1),
      identifier_mapping)],
    N=n_qubits,
    omega=omega,
    additional_noise_Hamiltonian=[[o, c, i] for o, c, i in H_n_add if i not in
                                  ['$\\sigma_x^{(0)}$', '$\\sigma_y^{(0)}$',
                                   '$\\sigma_x^{(1)}$', '$\\sigma_y^{(1)}$',
                                   '$\\sigma_x^{(3)}$', '$\\sigma_y^{(3)}$',
                                   '$\\sigma_z^{(0)}\\sigma_z^{(1)}$']],
    cache_filter_function=True,
    show_progressbar=SHOW_PROGRESSBAR
)

four_qubit_pulses['hadamard-CZ_pi2_echo-swap'] = {}
four_qubit_pulses['hadamard-CZ_pi2_echo-swap'][(0, 1)] = ff.concatenate(
    (four_qubit_pulses['hadamard'][0],
     four_qubit_pulses['CZ_pi2_echo'][(0, 1)],
     four_qubit_pulses['swap'][(0, 1)]),
    show_progressbar=SHOW_PROGRESSBAR
)
four_qubit_pulses['hadamard-CZ_pi2-swap'] = {}
four_qubit_pulses['hadamard-CZ_pi2-swap'][(0, 1)] = ff.concatenate(
    (four_qubit_pulses['hadamard'][0],
     four_qubit_pulses['CZ_pi2'][(0, 1)],
     four_qubit_pulses['swap'][(0, 1)]),
    show_progressbar=SHOW_PROGRESSBAR
)
four_qubit_pulses['CZ_pi4_echo-swap'] = {}
four_qubit_pulses['CZ_pi4_echo-swap'][(1, 2)] = ff.concatenate(
    (four_qubit_pulses['CZ_pi4_echo'][(1, 2)],
     four_qubit_pulses['swap'][(1, 2)]),
    show_progressbar=SHOW_PROGRESSBAR
)
four_qubit_pulses['CZ_pi4-swap'] = {}
four_qubit_pulses['CZ_pi4-swap'][(1, 2)] = ff.concatenate(
    (four_qubit_pulses['CZ_pi4'][(1, 2)],
     four_qubit_pulses['swap'][(1, 2)]),
    show_progressbar=SHOW_PROGRESSBAR
)
four_qubit_pulses['CZ_pi8-swap'] = {}
four_qubit_pulses['CZ_pi8-swap'][(2, 3)] = ff.concatenate(
    (four_qubit_pulses['CZ_pi8'][(2, 3)],
     four_qubit_pulses['swap'][(2, 3)]),
    show_progressbar=SHOW_PROGRESSBAR
)
four_qubit_pulses['hadamard-CZ_pi2-swap-CZ_pi4-swap'] = {}
four_qubit_pulses['hadamard-CZ_pi2-swap-CZ_pi4-swap'][(0, 1, 2)] = ff.concatenate(
    (four_qubit_pulses['hadamard-CZ_pi2-swap'][(0, 1)],
     four_qubit_pulses['CZ_pi4-swap'][(1, 2)]),
    show_progressbar=SHOW_PROGRESSBAR
)
four_qubit_pulses['hadamard-CZ_pi2_echo-swap-CZ_pi4_echo-swap'] = {}
four_qubit_pulses['hadamard-CZ_pi2_echo-swap-CZ_pi4_echo-swap'][(0, 1, 2)] = ff.concatenate(
    (four_qubit_pulses['hadamard-CZ_pi2_echo-swap'][(0, 1)],
     four_qubit_pulses['CZ_pi4_echo-swap'][(1, 2)]),
    show_progressbar=SHOW_PROGRESSBAR
)

qft_pulse_echo = ff.concatenate(
    (
        # rotations on first qubit
        four_qubit_pulses['hadamard-CZ_pi2_echo-swap-CZ_pi4_echo-swap'][(0, 1, 2)],
        # ...
        four_qubit_pulses['CZ_pi8-swap'][(2, 3)],
        # rotations on second qubit
        four_qubit_pulses['hadamard-CZ_pi2_echo-swap-CZ_pi4_echo-swap'][(0, 1, 2)],
        # rotation on third qubit
        four_qubit_pulses['hadamard-CZ_pi2-swap'][(0, 1)],
        # rotation on fourth qubit
        four_qubit_pulses['hadamard'][0]
    ),
    show_progressbar=SHOW_PROGRESSBAR
)

qft_pulse = ff.concatenate(
    (
        # rotations on first qubit
        four_qubit_pulses['hadamard-CZ_pi2-swap-CZ_pi4-swap'][(0, 1, 2)],
        # ...
        four_qubit_pulses['CZ_pi8-swap'][(2, 3)],
        # rotations on second qubit
        four_qubit_pulses['hadamard-CZ_pi2-swap-CZ_pi4-swap'][(0, 1, 2)],
        # rotation on third qubit
        four_qubit_pulses['hadamard-CZ_pi2-swap'][(0, 1)],
        # rotation on fourth qubit
        four_qubit_pulses['hadamard'][0],
    ),
    show_progressbar=SHOW_PROGRESSBAR
)

assert ff.util.oper_equiv(qt_qft(4), qft_pulse.total_propagator)
assert ff.util.oper_equiv(qt_qft(4), qft_pulse_echo.total_propagator)
# %% Plot
arrowprops = dict(arrowstyle='->', mutation_scale=7.5, color=RWTH_COLORS['black'],
                  linewidth=0.75, shrinkA=0, shrinkB=0)
annotate_kwargs = dict(color=RWTH_COLORS['black'], arrowprops=arrowprops)
# %%% Atomic pulses and filter function
fig, axes = plt.subplots(3, 2, sharey='col', sharex='col',
                         gridspec_kw=dict(hspace=0., wspace=0.04))

pretty_names = {
    'X_pi2': r'$\mathrm{X}_{0}(\pi/2)$',
    'Y_pi2': r'$\mathrm{Y}_{0}(\pi/2)$',
    'CZ_pi8': r'$\mathrm{CZ}_{01}(\pi/8)$',
}

for i, (name, pulse) in enumerate(pulses.items()):
    cycle = cycler(color=[LINE_COLORS[c_identifiers['CZ_pi8'].index(identifier)]
                          for identifier in pulse.c_oper_identifiers])
    *_, leg = ff.plotting.plot_pulse_train(pulse, fig=fig, axes=axes[i, 0], cycler=cycle)
    leg.remove()

    axes[i, 0].grid(False)
    axes[i, 0].set_xlabel(None)
    axes[i, 0].set_ylabel(None)

    *_, leg = ff.plotting.plot_filter_function(pulse, fig=fig, axes=axes[i, 1], cycler=cycle)
    leg.remove()

    axes[i, 1].grid(False)
    axes[i, 1].set_xlabel(None)
    axes[i, 1].set_ylabel(None)
    axes[i, 1].yaxis.tick_right()
    axes[i, 1].yaxis.set_label_position('right')

    axes[i, 0].text(0.975, 0.85, pretty_names[name], transform=axes[i, 0].transAxes,
                    verticalalignment='top', horizontalalignment='right')
    axes[i, 1].text(0.975, 0.85, pretty_names[name], transform=axes[i, 1].transAxes,
                    verticalalignment='top', horizontalalignment='right')

axes[2, 0].set_xlabel('$t$ (ns)')
axes[2, 1].set_xlabel(r'$\omega$' + pernanosecond)
axes[1, 0].set_ylabel('Control' + pernanosecond)
axes[1, 1].set_ylabel(r'$\mathcal{F}(\omega)$')

axes[0, 0].legend(handles=axes[2, 0].get_lines(), labels=pulse.c_oper_identifiers.tolist(),
                  bbox_to_anchor=(-.075, 1.02, 2.2, .102), loc='lower left',
                  ncols=5, mode="expand", borderaxespad=0., frameon=False)

fig.savefig(SAVE_PATH / 'qft_atomic_pulses.pdf')
# %%% FF with cumulative FF
identifiers = [r'$\sigma_x^{(0)}$', r'$\sigma_y^{(0)}$', r'$\sigma_z^{(0)}\sigma_z^{(1)}$']
cycle = cycler(color=[LINE_COLORS[n_identifiers['CZ_pi8'].index(identifier)]
                      for identifier in identifiers])

fig, ax = plt.subplots(1, 1, figsize=(TOTALWIDTH, 2), layout='constrained')
*_, leg = plotting.plot_filter_function(qft_pulse, yscale='log', fig=fig, axes=ax,
                                        omega_in_units_of_tau=False, cycler=cycle,
                                        n_oper_identifiers=identifiers)
leg.remove()

for n in (1, 2, 3, 4):
    ax.axvline(n*2*np.pi/t_clock, color=RWTH_COLORS_50['black'], zorder=0, linestyle=':')

# calculate cumulative sensitivity, \int_0^\omega_c\dd{\omega} FS(\omega)
idx = get_indices_from_identifiers(qft_pulse.n_oper_identifiers, identifiers)
F = qft_pulse.get_filter_function(omega)[idx, idx].real
FS = integrate.cumulative_simpson(F, x=omega, axis=-1, initial=0)
FS /= FS[:, -1:]

ax2 = ax.twinx()
ax2.set_prop_cycle(cycle)
ax2.semilogx(omega, FS.T, linestyle='--')
ax2.set_ylabel(r'$\chi_\alpha(0, \omega) / \chi(0, \infty)$')

ax.annotate('', (1e-5*1.2, 5e6), (1e-5*1.2*2.5, 5e6), **annotate_kwargs)
ax2.annotate('', (1e2/1.2, 0.925), (1e2/1.2/2.5, 0.925), **annotate_kwargs)

ax.grid(False)
ax.set_xlabel(r'$\omega$' + pernanosecond)
ax.set_ylabel(r'$\mathcal{F}_{\alpha}(\omega)$')
ax.legend(loc="lower left", bbox_to_anchor=(0., 0.1),
          fancybox=False, frameon=False)

fig.savefig(SAVE_PATH / 'qft_filter_function.pdf')

# %% FF with & without echo
identifiers = ['$\\sigma_y^{(3)}$']
fig, ax, _ = plotting.plot_filter_function(qft_pulse,
                                           n_oper_identifiers=identifiers,
                                           yscale='log',
                                           omega_in_units_of_tau=False)
fig, ax, _ = plotting.plot_filter_function(qft_pulse_echo,
                                           n_oper_identifiers=identifiers,
                                           yscale='log',
                                           omega_in_units_of_tau=False,
                                           axes=ax,
                                           fig=fig)
ax.axvline(2*np.pi/qft_pulse.duration, color='black', linestyle='--', zorder=0)
ax.annotate(r'$2\pi/\tau$', (2*np.pi/qft_pulse.duration, 10), (2*np.pi/qft_pulse.duration*2, 0.05),
            arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc,angleA=120,armA=15,rad=10'})

ax.grid(False)
leg = ax.legend(['without echo', 'with echo'], frameon=False, framealpha=1.)
ax.set_xlabel(r'$\omega$ ($2\pi$GHz)')
ax.set_ylabel(r'$F(\omega)$ (ns/$2\pi)^2$')
ax.set_xscale('symlog', linthresh=1e-4)
ax.set_xlim(0)

fname = 'qft_filter_function_Y3_echo_vs_no'
# %% Correlations with & without echo

pls = ['hadamard', 'CZ_pi2', 'swap', 'CZ_pi4', 'swap', 'CZ_pi8', 'swap',
       'hadamard', 'CZ_pi2', 'swap', 'CZ_pi4', 'swap',
       'hadamard', 'CZ_pi2', 'swap',
       'hadamard']
pls_echo = ['hadamard', 'CZ_pi2_echo', 'swap', 'CZ_pi4_echo', 'swap', 'CZ_pi8', 'swap',
            'hadamard', 'CZ_pi2_echo', 'swap', 'CZ_pi4_echo', 'swap',
            'hadamard', 'CZ_pi2', 'swap',
            'hadamard']

qubits = [0, (0, 1), (0, 1), (1, 2), (1, 2), (2, 3), (2, 3),
          0, (0, 1), (0, 1), (1, 2), (1, 2),
          0, (0, 1), (0, 1),
          0]

qft_pulse_correl = ff.concatenate(
    [four_qubit_pulses[pls][qubits] for pls, qubits in zip(pls, qubits)],
    calc_pulse_correlation_FF=True,
    omega=omega,
    show_progressbar=SHOW_PROGRESSBAR
)
qft_pulse_echo_correl = ff.concatenate(
    [four_qubit_pulses[pls][qubits] for pls, qubits in zip(pls_echo, qubits)],
    calc_pulse_correlation_FF=True,
    omega=omega,
    show_progressbar=SHOW_PROGRESSBAR
)

# %%
S0 = 2.2265e-6
# S0 = 2e-6

infids = ff.infidelity(qft_pulse_correl, S, omega, which='correlations',
                       n_oper_identifiers=qft_pulse_correl.n_oper_identifiers).transpose(2, 0, 1)

infids_echo = ff.infidelity(qft_pulse_echo_correl, S, omega, which='correlations',
                            n_oper_identifiers=qft_pulse_echo_correl.n_oper_identifiers).transpose(2, 0, 1)

infids_white = ff.infidelity(qft_pulse_correl, np.ones_like(omega)*S0, omega, which='correlations',
                             n_oper_identifiers=qft_pulse_correl.n_oper_identifiers).transpose(2, 0, 1)

infids_white_echo = ff.infidelity(qft_pulse_echo_correl, np.ones_like(omega)*S0, omega, which='correlations',
                                  n_oper_identifiers=qft_pulse_echo_correl.n_oper_identifiers).transpose(2, 0, 1)

print('Total infidelities:')
print(f'1/f, no echo:\t{infids[7].sum():2.4e}')
print(f'1/f, echo:\t{infids_echo[7].sum():2.4e}')
print(f'white, no echo:\t{infids_white[7].sum():2.4e}')
print(f'white, echo:\t{infids_white_echo[7].sum():2.4e}')
# %%
# pls = [r'$\text{H}_0$', r'$\text{CR}_{01}(\pi/2)$', r'$\text{SWAP}_{01}$',
#        r'$\text{CR}_{12}(\pi/4)$', r'$\text{SWAP}_{12}$',
#        r'$\text{CR}_{23}(\pi/8)$', r'$\text{SWAP}_{23}$',
#        r'$\text{H}_0$', r'$\text{CR}_{01}(\pi/2)$', r'$\text{SWAP}_{01}$',
#        r'$\text{CR}_{12}(\pi/4)$', r'$\text{SWAP}_{12}$',
#        r'$\text{H}_0$', r'$\text{CR}_{01}(\pi/2)$', r'$\text{SWAP}_{01}$',
#        r'$\text{H}_0$', ]

comparison = np.stack((infids[7], infids_echo[7], infids_white[7], infids_white_echo[7]))
identifiers = [f'{c:.1e}' for c in comparison.sum((1, 2))]
identifiers = ['']*4
identifiers = ['(a) $1/f$, no echo', '(b) $1/f$, echo',
               '(c) white, no echo', '(d) white, echo']

with plt.rc_context(rc={'axes.titlesize': 9}):
    fig1, grid1 = plotting.plot_cumulant_function(
        cumulant_function=comparison,
        colorscale='log',
        n_oper_identifiers=identifiers,
        cbar_label=r"Pulse correlation infidelity $\mathcal{I}^{(gg')}$",
        # basis_labels=pls,
        # basis_labels=['' for p in pls],
        basis_labels=[p.replace('_', r'\_') for p in pls],
        basis_labelsize=7,
        figsize=(figsize_wide[0], figsize_wide[0]*2/5),
        grid_kw=dict(cbar_location='top', nrows_ncols=(1, 4), cbar_size="2%", cbar_pad=0.25),
        cbar_kw=dict(orientation='horizontal', ticklocation='top')
    )

cbar = grid1[-1].images[0].colorbar
labels = cbar.ax.get_xticklabels()
labels[len(labels) // 2] = ''
cbar.ax.set_xticklabels(labels, fontsize=7)

for g in grid1:
    ax = g.axes
    ax.title.set_x(0)

fig1.tight_layout()
for ext in ('pdf', 'eps', 'png'):
    if (not (save_path / f'correlation_infidelities_Y3_wide.{ext}').exists()
            or _force_overwrite):
        fig1.savefig(save_path / f'correlation_infidelities_Y3_wide.{ext}')

# %%
comparison = np.stack((infids[7], infids_echo[7], infids_white[7], infids_white_echo[7]))
identifiers = [f'{c:.1e}' for c in comparison.sum((1, 2))]
identifiers = ['']*4
identifiers = ['(a) $1/f$, no echo', '(b) $1/f$, echo',
               '(c) white, no echo', '(d) white, echo']

with plt.rc_context(rc={'axes.titlesize': 9}):
    fig2, grid2 = plotting.plot_cumulant_function(
        cumulant_function=comparison,
        colorscale='log',
        n_oper_identifiers=identifiers,
        cbar_label=r"Pulse correlation infidelity $\mathcal{I}^{(gg')}$",
        # basis_labels=pls,
        # basis_labels=['' for p in pls],
        basis_labels=[p.replace('_', r'\_') for p in pls],
        basis_labelsize=7,
        figsize=(figsize_narrow[0], figsize_narrow[0]*1.2),
        grid_kw=dict(cbar_location='top', nrows_ncols=(2, 2), cbar_size="5%", cbar_pad=0.25),
        cbar_kw=dict(orientation='horizontal', ticklocation='top')
    )

cbar = grid2[-1].images[0].colorbar
labels = cbar.ax.get_xticklabels()
labels[len(labels) // 2] = ''
cbar.ax.set_xticklabels(labels)

for g in grid2:
    ax = g.axes
    ax.title.set_x(0)

fig2.tight_layout()
for ext in ('pdf', 'eps', 'png'):
    if (not (save_path / f'correlation_infidelities_Y3.{ext}').exists()
            or _force_overwrite):
        fig2.savefig(save_path / f'correlation_infidelities_Y3.{ext}')

# %%% Only 1/f
comparison = np.stack((infids[7], infids_echo[7], ))
identifiers = [f'{c:.1e}' for c in comparison.sum((1, 2))]
identifiers = ['']*4
identifiers = ['(a) w/o echo', '(b) w/ echo']

with plt.rc_context(rc={'axes.titlesize': 9}):
    fig1, grid1 = plotting.plot_cumulant_function(
        cumulant_function=comparison,
        colorscale='log',
        n_oper_identifiers=identifiers,
        cbar_label=r"Pulse correlation infidelity $\mathcal{I}^{(gg')}$",
        # basis_labels=pls,
        basis_labels=[p for p in range(len(pls))],
        # basis_labels=[p.replace('_', r'\_') for p in pls],
        basis_labelsize=6,
        figsize=(figsize_narrow[0], figsize_narrow[0]),
        grid_kw=dict(cbar_location='top', nrows_ncols=(1, 2), cbar_size="4%", cbar_pad=0.25,
                     axes_pad=0.2),
        cbar_kw=dict(orientation='horizontal', ticklocation='top')
    )

cbar = grid1[-1].images[0].colorbar
labels = cbar.ax.get_xticklabels()
for txt in labels:
    if '10^{-6}' in txt.get_text():
        txt.set_text('')

for g in grid1:
    ax = g.axes
    ax.title.set_x(0)

cbar.ax.set_xticklabels(labels, fontsize=6)

fig1.tight_layout()
for ext in ('pdf', 'eps', 'png'):
    if (not (save_path / f'correlation_infidelities_Y3_corr_only.{ext}').exists()
            or _force_overwrite):
        fig1.savefig(save_path / f'correlation_infidelities_Y3_corr_only.{ext}')
# %%% Only 1/f cbar side
comparison = np.stack((infids[7], infids_echo[7],))
identifiers = [f'{c:.1e}' for c in comparison.sum((1, 2))]
identifiers = ['']*4
identifiers = ['(a) without echo', '(b) with echo']

with plt.rc_context(rc={'axes.titlesize': 8}):
    fig1, grid1 = plotting.plot_cumulant_function(
        cumulant_function=comparison,
        colorscale='log',
        n_oper_identifiers=identifiers,
        cbar_label="",
        # basis_labels=pls,
        basis_labels=[p for p in range(1, 1+len(pls))],
        # basis_labels=[p.replace('_', r'\_') for p in pls],
        basis_labelsize=5,
        figsize=(figsize_narrow[0], figsize_narrow[0]),
        grid_kw=dict(cbar_location='right', nrows_ncols=(1, 2), cbar_size="8%", cbar_pad=0.1,
                     axes_pad=0.15),
        cbar_kw=dict(orientation='vertical')
    )

cbar = grid1[-1].images[0].colorbar
labels = cbar.ax.get_yticklabels()
for txt in labels:
    t = txt.get_text()
    if '{0}' in t:
        txt.set_text('')
    elif '{10^' in t:
        ix = t.index('{10^')
        txt.set_text(t[:ix+1] + '+' + t[ix+1:])

for g in grid1:
    ax = g.axes
    ax.title.set_x(0)

cbar.ax.set_yticklabels(labels, fontsize=6)
# cbar.ax.tick_params(axis='y', which='major', pad=25)
cbar.ax.set_title(r"$\mathcal{I}^{(gg')}$", loc='left', fontsize=8)


for g in grid1:
    ax = g.axes
    ax.set_xlabel("$g$", fontsize=7)
    ax.set_ylabel("$g'$", fontsize=7)

fig1.tight_layout()
for ext in ('pdf', 'eps', 'png'):
    if (not (save_path / f'correlation_infidelities_Y3_corr_only_cbar_on_side.{ext}').exists()
            or _force_overwrite):
        fig1.savefig(save_path / f'correlation_infidelities_Y3_corr_only_cbar_on_side.{ext}')
# %%
# pls = [r'$\text{H}_0$', r'$\text{CR}_{01}(\pi/2)$', r'$\text{SWAP}_{01}$',
#        r'$\text{CR}_{12}(\pi/4)$', r'$\text{SWAP}_{12}$',
#        r'$\text{CR}_{23}(\pi/8)$', r'$\text{SWAP}_{23}$',
#        r'$\text{H}_0$', r'$\text{CR}_{01}(\pi/2)$', r'$\text{SWAP}_{01}$',
#        r'$\text{CR}_{12}(\pi/4)$', r'$\text{SWAP}_{12}$',
#        r'$\text{H}_0$', r'$\text{CR}_{01}(\pi/2)$', r'$\text{SWAP}_{01}$',
#        r'$\text{H}_0$', ]

comparison = np.stack((infids[10], infids_echo[10], infids_white[10], infids_white_echo[10]))
identifiers = [f'{c:.1e}' for c in comparison.sum((1, 2))]
identifiers = ['']*4
identifiers = ['(a) $1/f$, no echo', '(b) $1/f$, echo',
               '(c) white, no echo', '(d) white, echo']

with plt.rc_context(rc={'axes.titlesize': 9}):
    fig3, grid3 = plotting.plot_cumulant_function(
        cumulant_function=comparison,
        colorscale='log',
        n_oper_identifiers=identifiers,
        cbar_label=r"Pulse correlation infidelity $\mathcal{I}^{(gg')}$",
        # basis_labels=pls,
        # basis_labels=['' for p in pls],
        basis_labels=[p.replace('_', r'\_') for p in pls],
        basis_labelsize=7,
        figsize=(figsize_wide[0], figsize_wide[0]*2/5),
        grid_kw=dict(cbar_location='top', nrows_ncols=(1, 4), cbar_size="2%", cbar_pad=0.25),
        cbar_kw=dict(orientation='horizontal', ticklocation='top')
    )

cbar = grid3[-1].images[0].colorbar
labels = cbar.ax.get_xticklabels()
labels[len(labels) // 2] = ''
cbar.ax.set_xticklabels(labels)

for g in grid3:
    ax = g.axes
    ax.title.set_x(0)

fig3.tight_layout()
for ext in ('pdf', 'eps', 'png'):
    if (not (save_path / f'correlation_infidelities_CZ23_wide.{ext}').exists()
            or _force_overwrite):
        fig3.savefig(save_path / f'correlation_infidelities_CZ23_wide.{ext}')
# %%

comparison = np.stack((infids[10], infids_echo[10], infids_white[10], infids_white_echo[10]))
identifiers = [f'{c:.1e}' for c in comparison.sum((1, 2))]
identifiers = ['']*4
identifiers = ['(a) $1/f$, no echo', '(b) $1/f$, echo',
               '(c) white, no echo', '(d) white, echo']

with plt.rc_context(rc={'axes.titlesize': 9}):
    fig4, grid4 = plotting.plot_cumulant_function(
        cumulant_function=comparison,
        colorscale='log',
        n_oper_identifiers=identifiers,
        cbar_label=r"Pulse correlation infidelity $\mathcal{I}^{(gg')}$",
        # basis_labels=pls,
        # basis_labels=['' for p in pls],
        basis_labels=[p.replace('_', r'\_') for p in pls],
        basis_labelsize=7,
        figsize=(figsize_narrow[0], figsize_narrow[0]*1.2),
        grid_kw=dict(cbar_location='top', nrows_ncols=(2, 2), cbar_size="5%", cbar_pad=0.25),
        cbar_kw=dict(orientation='horizontal', ticklocation='top')
    )

cbar = grid4[-1].images[0].colorbar
labels = cbar.ax.get_xticklabels()
labels[len(labels) // 2] = ''
cbar.ax.set_xticklabels(labels)

for g in grid4:
    ax = g.axes
    ax.title.set_x(0)

fig4.tight_layout()
for ext in ('pdf', 'eps', 'png'):
    if (not (save_path / f'correlation_infidelities_CZ23.{ext}').exists()
            or _force_overwrite):
        fig4.savefig(save_path / f'correlation_infidelities_CZ23.{ext}')
