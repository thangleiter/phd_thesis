# %% Imports
import pathlib
import sys
import time

import filter_functions as ff
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import qutip as qt
from filter_functions import plotting
from filter_functions.util import get_indices_from_identifiers
from cycler import cycler
from mpl_toolkits.axes_grid1 import ImageGrid
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_50, make_diverging_colormap
from qutip.control import pulseoptim
from qutip.qip import operations
from qutip.qip.algorithms.qft import qft as qt_qft
from scipy import integrate

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, TEXTWIDTH, TOTALWIDTH, PATH, init

with np.errstate(divide='ignore', invalid='ignore'):
    DIVERGING_CMAP = make_diverging_colormap(('magenta', 'green'), endpoint='white')

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
                          for identifier in c_identifiers[name]])
    *_, leg = ff.plotting.plot_pulse_train(pulse, fig=fig, axes=axes[i, 0], cycler=cycle,
                                           c_op_identifiers=c_identifiers[name])
    leg.remove()

    axes[i, 0].grid(False)
    axes[i, 0].set_xlabel(None)
    axes[i, 0].set_ylabel(None)

    *_, leg = ff.plotting.plot_filter_function(pulse, fig=fig, axes=axes[i, 1], cycler=cycle,
                                               omega_in_units_of_tau=False,
                                               n_oper_identifiers=n_identifiers[name])
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
cycle = cycler(color=LINE_COLORS)

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
fig, ax, leg = plotting.plot_filter_function(
    qft_pulse, n_oper_identifiers=identifiers, yscale='log',
    omega_in_units_of_tau=False,
    layout='constrained', figsize=(TEXTWIDTH, 1.75)
)
leg.remove()
*_, leg = plotting.plot_filter_function(qft_pulse_echo, n_oper_identifiers=identifiers,
                                        omega_in_units_of_tau=False,
                                        yscale='log', axes=ax, fig=fig)
leg.remove()
ax.axvline(2*np.pi/qft_pulse.duration, color=RWTH_COLORS_50['black'], zorder=0, linestyle=':')
ax.annotate(r'$2\pi/\tau$',
            (2*np.pi/qft_pulse.duration, 1e-1), (2*np.pi/qft_pulse.duration/2.5, 1e-1),
            horizontalalignment='right', verticalalignment='center', **annotate_kwargs)

ax.grid(False)
leg = ax.legend(['without echo', 'with echo'], frameon=False, framealpha=1.)
ax.set_xlabel(r'$\omega$' + pernanosecond)
ax.set_ylabel(r'$\mathcal{F}(\omega)$')
ax.set_xscale('symlog', linthresh=1e-4)
ax.set_xlim(0)

fig.savefig(SAVE_PATH / 'qft_filter_function_Y3.pdf')
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

# %%% Infidelities
S0 = 2.2265e-6

infids = {}
infids['$1/f$ without echo'] = ff.infidelity(qft_pulse_correl, S, omega, which='correlations',
                                             n_oper_identifiers=['$\\sigma_y^{(3)}$'])
infids['$1/f$ with echo'] = ff.infidelity(qft_pulse_echo_correl, S, omega, which='correlations',
                                          n_oper_identifiers=['$\\sigma_y^{(3)}$'])
infids['White without echo'] = ff.infidelity(qft_pulse_correl, np.full_like(omega, S0), omega,
                                             which='correlations',
                                             n_oper_identifiers=['$\\sigma_y^{(3)}$'])
infids['White with echo'] = ff.infidelity(qft_pulse_echo_correl, np.full_like(omega, S0), omega,
                                          which='correlations',
                                          n_oper_identifiers=['$\\sigma_y^{(3)}$'])
# Normalize the infidelities for white noise so that the total infidelity without
# echos is the same as for 1/f.
normalization = infids['$1/f$ without echo'].sum() / infids['White without echo'].sum()
infids['White without echo'] *= normalization
infids['White with echo'] *= normalization

print('Total infidelities for Y noise on the third qubit:')
for k, v in infids.items():
    print(f'{k}\t{v.sum():.2e}')

# %%% Plot
fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH))
grid = ImageGrid(fig, 111, (2, 2), cbar_mode='single', axes_pad=0.1)
vmax = max(abs(np.max(val)) for val in infids.values())
vmin = -vmax
norm = mpl.colors.AsinhNorm(linear_width=1e-6, vmin=vmin, vmax=vmax)
titles = [key.split(' with') for key in infids]

for ax, (key, val) in zip(grid, infids.items()):
    img = ax.imshow(val, interpolation=None, norm=norm, cmap=DIVERGING_CMAP)
    ax.set_xticks(np.arange(0, 20, 5))
    ax.set_yticks(np.arange(0, 20, 5))
    ax.annotate(key.replace(' ', '\n', 1), (0.95, 0.95), xycoords='axes fraction',
                fontsize='small', horizontalalignment='right', verticalalignment='top')

fig.supxlabel('$g$', fontsize='medium', y=0.05)
fig.supylabel(r'$g^\prime$', fontsize='medium')
ax.cax.colorbar(img, label=r"Pulse correlation infidelity $I_{\alpha}^{(gg')}$")

fig.savefig(SAVE_PATH / 'qft_correlation_infids.pdf')
