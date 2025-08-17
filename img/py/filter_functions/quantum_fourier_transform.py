#!/usr/bin/env python
# coding: utf-8

# # Implementing a Quantum Fourier Transform (QFT)
# In this example we want to bring everything together to efficiently calculate filter functions of a QFT. We will optimize atomic gates using QuTiP and set up `PulseSequence`s with the optimized parameters. Using those, we will assemble a QFT circuit with interactions limited to nearest neighbors, thus requiring us to swap qubits around the registers to perform controlled rotations.
#
# The circuit for the algorithm on four qubits is as follows (for simplicity we run the transformations of each qubit sequentially):
#
# ![qft.png](../_static/qft.png)
#
# Here, unlike in the canonical circuit ([Wikipedia](https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Circuit_implementation)), the qubits are already swapped when the algorithm finishes.
#
# ## Physical model
# We will consider a qubit model where single-qubit operations are performed using I-Q manipulation and two-qubit operations using nearest-neighbor exchange interaction. Concretely, the single-qubit control Hamiltonian is given by
#
# $$
# H_c^{(1)}(t) = I(t)\:\sigma_x + Q(t)\:\sigma_y
# $$
#
# and the two-qubit control Hamiltonian by
#
# $$
# H_c^{(2)}(t) = I_1(t)\; \sigma_x \otimes \mathbb{1} + Q_1(t)\;\sigma_y \otimes \mathbb{1} + J(t)\:\sigma_z \otimes \sigma_z + I_2(t)\; \mathbb{1} \otimes \sigma_x + Q_2(t)\;\mathbb{1} \otimes \sigma_y.
# $$
#
# ## Optimizing pulses using GRAPE
# We would like to keep the size of our optimized gate set as small as possible and thus compile the required gates from the set $\left\lbrace\mathrm{X(\pi/2)}, \mathrm{Y(\pi/2)}, \mathrm{CZ(\pi/2^4)}\right\rbrace$ for a four-qubit QFT.

# In[1]:
from copy import deepcopy
from pathlib import Path
import string
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib import colors, cycler, lines
import numpy as np
import qutip as qt
from scipy import integrate
from qutip.control import pulseoptim
from qutip.qip import operations
from qutip.qip.algorithms.qft import qft as qt_qft

import filter_functions as ff
from filter_functions.util import get_indices_from_identifiers

warnings.simplefilter('ignore', DeprecationWarning)

plt.style.use('publication')
golden_ratio = (np.sqrt(5) - 1.)/2.
figsize_narrow = (3.40457, 3.40457*golden_ratio)
figsize_wide = (7.05826, 7.05826*golden_ratio)

cycle = plt.rcParams['axes.prop_cycle'][:5] + cycler(linestyle=('-','--','-.',':', (0, (5, 1))))

# thesis_path = Path('/home/tobias/Physik/Master/Thesis')
# thesis_path = Path('Z:/MA/')
# project_path = Path('/home/tobias/Physik/Publication/')
# project_path = Path('Z:/Publication/')
project_path = Path('C:/Users/Tobias/Documents/Uni/Physik/Publication')
name = ('efficient_calculation_of_generalized_filter_functions')
# name = ('efficient_calculation_of_generalized_filter_functions')
save_path = project_path / name / 'img'

_force_overwrite = False
# %%

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


# With the pulses optimized, we can now set up the `PulseSequence`s for each and assert their correct action:


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

    print(name, '\t:', ff.util.oper_equiv(
        pulse.total_propagator, target_gates[name], eps=1e-10
    ))

    result.stats.report()


# ## Assembling the circuit
# For simplicity, we are going to assume periodic boundary conditions, i.e., the qubits sit on a ring such that each qubit has two nearest neighbors. This allows us to make the best use of the caching of filter functions.
#
# To this end, we first explicitly cache the control matrices for the optimized, elementary pulses and then extend them to the four-qubit Hilbert space.

from filter_functions import plotting

omega = np.geomspace(1e-4, 1e2, 1000)
omega = np.concatenate([np.arange(0, omega[0], omega[0]/10), omega])

# omega = np.linspace(0, 1e2, 1000)
S = 1e-9/omega

fig, ax = plt.subplots(2, 3, sharey='row', sharex='row', figsize=figsize_wide)

print('Caching control matrices for single- and two-qubit pulses:')
for i, (name, pulse) in enumerate(pulses.items()):
    n = len(pulse.c_opers)
    cycle = cycler(color=plt.get_cmap('Blues')(np.linspace(1/n, n/n, n)[::-1]))

    pulse.cache_control_matrix(omega, show_progressbar=True)

    fig, ax[0, i], leg = plotting.plot_pulse_train(pulse, fig=fig, axes=ax[0, i],
                                                   cycler=cycle)
    if i == 2:
        leg = ax[0, i].legend(ncol=2, frameon=False)
    else:
        leg = ax[0, i].legend(frameon=False)

    ax[0, i].set_xlabel('$t$ (ns)')
    ax[0, i].grid(False)

    fig, ax[1, i], leg = plotting.plot_filter_function(pulse, fig=fig, axes=ax[1, i],
                                                       omega_in_units_of_tau=False,
                                                       cycler=cycle)
    if i == 2:
        leg = ax[1, i].legend(ncol=2, frameon=False)
    else:
        leg = ax[1, i].legend(frameon=False)

    # leg.set_title(identifiers[name])
    ax[1, i].set_xlabel(r'$\omega$ (ns$^{-1})$')
    ax[1, i].grid(False)

    ax[0, i].text(0.05 + 0.05*(i == 1), 0.90, f'({string.ascii_lowercase[2*i]})',
                  transform=ax[0, i].transAxes, fontsize=10)
    ax[1, i].text(0.05, 0.075, f'({string.ascii_lowercase[2*i+1]})',
                  transform=ax[1, i].transAxes, fontsize=10)

ax[0, 0].set_ylabel('Control (ns$^{-1}$)')
ax[0, 1].set_ylabel('')
ax[0, 2].set_ylabel('')
ax[1, 1].set_ylabel('')
ax[1, 2].set_ylabel('')

fig.tight_layout()
for ext in ('pdf', 'eps'):
    if (not (save_path / f'qft_atomic_pulses.{ext}').exists()
            or _force_overwrite):
        fig.savefig(save_path / f'qft_atomic_pulses.{ext}')
# ### Extending single- and two-qubit pulses to the four-qubit register
# In order to extend the pulses, we first need to define the noise Hamiltonians on the other qubits so that each four-qubit pulse has the complete set of noise operators.

t_start_fast = time.perf_counter()

print('Caching control matrices for single- and two-qubit pulses:')
pulses['X_pi2'].cache_control_matrix(omega, show_progressbar=True)
pulses['Y_pi2'].cache_control_matrix(omega, show_progressbar=True)
pulses['CZ_pi8'].cache_control_matrix(omega, show_progressbar=True)


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
print('Caching control matrices for four-qubit pulses:')
four_qubit_pulses['X_pi2'][0] = ff.extend([(pulses['X_pi2'], 0, identifier_mapping)],
                                          N=n_qubits, omega=omega,
                                          additional_noise_Hamiltonian=H_n_one,
                                          cache_filter_function=True,
                                          show_progressbar=True)
four_qubit_pulses['Y_pi2'][0] = ff.extend([(pulses['Y_pi2'], 0, identifier_mapping)],
                                          N=n_qubits, omega=omega,
                                          additional_noise_Hamiltonian=H_n_one,
                                          cache_filter_function=True,
                                          show_progressbar=True)
four_qubit_pulses['CZ_pi8'][(0, 1)] = ff.extend([(pulses['CZ_pi8'], (0, 1), identifier_mapping)],
                                                N=n_qubits, omega=omega,
                                                additional_noise_Hamiltonian=H_n_two,
                                                cache_filter_function=True,
                                                show_progressbar=True)

print('Correct action:')
print('X_pi2: ', ff.util.oper_equiv(four_qubit_pulses['X_pi2'][0].total_propagator,
                                    operations.rotation(qt.sigmax(), np.pi/2, N=4)))
print('Y_pi2: ', ff.util.oper_equiv(four_qubit_pulses['Y_pi2'][0].total_propagator,
                                    operations.rotation(qt.sigmay(), np.pi/2, N=4)))
print('CZ_pi8: ', ff.util.oper_equiv(four_qubit_pulses['CZ_pi8'][(0, 1)].total_propagator,
                                     operations.cphase(np.pi/2**4, N=4), eps=1e-9))



# ### Compiling the required gates
# Next, we compile all required single- and two-qubit gates from our elementary pulses. The Hadamard is given by
#
# $$
#     \mathrm{H}\doteq\mathrm{X(\pi/2)}\circ\mathrm{X(\pi/2)}\circ\mathrm{Y(\pi/2)},
# $$
#
# the controlled-X by
#
# $$
#     \mathrm{CX_{ij}(\phi)}\doteq\mathrm{H_j}\circ\mathrm{CZ_{ij}(\phi)}\circ\mathrm{H_j},
# $$
#
# and finally the SWAP by
#
# $$
#     \mathrm{SWAP_{ij}}\doteq\mathrm{CX_{ij}(\pi)}\circ\mathrm{CX_{ji}(\pi)}\circ\mathrm{CX_{ij}(\pi)}.
# $$
#
# Trivially, controlled rotations about multiples of $\pi/2^4$ are implemented by repeated applications of $\mathrm{CZ(\pi/2^4)}$.



four_qubit_pulses['hadamard'] = {}
four_qubit_pulses['CZ_pi4'] = {}
four_qubit_pulses['CZ_pi2'] = {}
four_qubit_pulses['CZ_pi'] = {}
four_qubit_pulses['CX_pi'] = {}
four_qubit_pulses['swap'] = {}

four_qubit_pulses['hadamard'][0] = ff.concatenate((four_qubit_pulses['Y_pi2'][0],
                                                   four_qubit_pulses['X_pi2'][0],
                                                   four_qubit_pulses['X_pi2'][0]),
                                                  show_progressbar=True)
four_qubit_pulses['CZ_pi4'][(0, 1)] = ff.concatenate((four_qubit_pulses['CZ_pi8'][(0, 1)],
                                                      four_qubit_pulses['CZ_pi8'][(0, 1)]),
                                                     show_progressbar=True)
four_qubit_pulses['CZ_pi2'][(0, 1)] = ff.concatenate((four_qubit_pulses['CZ_pi4'][(0, 1)],
                                                      four_qubit_pulses['CZ_pi4'][(0, 1)]),
                                                     show_progressbar=True)
four_qubit_pulses['CZ_pi'][(0, 1)] = ff.concatenate((four_qubit_pulses['CZ_pi2'][(0, 1)],
                                                     four_qubit_pulses['CZ_pi2'][(0, 1)]),
                                                    show_progressbar=True)
# CNOT with control on the second, target on the first qubit
four_qubit_pulses['CX_pi'][(1, 0)] = ff.concatenate((four_qubit_pulses['hadamard'][0],
                                                     four_qubit_pulses['CZ_pi'][(0, 1)],
                                                     four_qubit_pulses['hadamard'][0]),
                                                    show_progressbar=True)

print('Correct action:')
print('hadamard: ', ff.util.oper_equiv(four_qubit_pulses['hadamard'][0].total_propagator,
                                       operations.snot(4, 0), eps=1e-9))
print('CZ_pi4: ', ff.util.oper_equiv(four_qubit_pulses['CZ_pi4'][(0, 1)].total_propagator,
                                     operations.cphase(np.pi/2**2, N=4), eps=1e-8))
print('CZ_pi2: ', ff.util.oper_equiv(four_qubit_pulses['CZ_pi2'][(0, 1)].total_propagator,
                                     operations.cphase(np.pi/2**1, N=4), eps=1e-7))
print('CZ_pi: ', ff.util.oper_equiv(four_qubit_pulses['CZ_pi'][(0, 1)].total_propagator,
                                    operations.cphase(np.pi/2**0, N=4), eps=1e-7))
print('CX_pi: (1, 0)', ff.util.oper_equiv(four_qubit_pulses['CX_pi'][(1, 0)].total_propagator,
                                          operations.cnot(4, 1, 0), eps=1e-7))


# ### Remapping pulses to different qubits
# To get the CNOT with control and target interchanged, we simply remap the Hadamard pulse to the first qubit by cyclically moving the qubits


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
        r'$\sigma_z^{{({})}}\sigma_z^{{({})}}$'.format((i+shift) % n_qubits, (i+shift+1) % n_qubits)
        for i in range(n_qubits)
    })
    return mapping

four_qubit_pulses['hadamard'][1] = ff.remap(four_qubit_pulses['hadamard'][0],
                                            order=(3, 0, 1, 2),
                                            oper_identifier_mapping=cyclical_mapping(1))
four_qubit_pulses['CX_pi'][(0, 1)] = ff.concatenate((four_qubit_pulses['hadamard'][1],
                                                     four_qubit_pulses['CZ_pi'][(0, 1)],
                                                     four_qubit_pulses['hadamard'][1]),
                                                    show_progressbar=True)
four_qubit_pulses['swap'][(0, 1)] = ff.concatenate((four_qubit_pulses['CX_pi'][(1, 0)],
                                                    four_qubit_pulses['CX_pi'][(0, 1)],
                                                    four_qubit_pulses['CX_pi'][(1, 0)]),
                                                   show_progressbar=True)

print('Correct action:')
print('hadamard: (1)', ff.util.oper_equiv(four_qubit_pulses['hadamard'][1].total_propagator,
                                          operations.snot(4, 1), eps=1e-9))
print('CX_pi: (0, 1)', ff.util.oper_equiv(four_qubit_pulses['CX_pi'][(0, 1)].total_propagator,
                                          operations.cnot(4, 0, 1), eps=1e-7))
print('swap: (0, 1)', ff.util.oper_equiv(four_qubit_pulses['swap'][(0, 1)].total_propagator,
                                         operations.swap(4, [0, 1]), eps=1e-6))


# Now we can simply remap the four-qubit pulses to apply to qubits other than 0 and 1:



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


# ### Grouping reoccuring gates
# As a last step before finally calculating the complete pulse we precompute pulses that appear in the algorithm multiple times in order to salvage the concatenation performance gain. As a first step, we can precompute the gates $\mathrm{SWAP_{10}}\circ\mathrm{CZ_{10}(\pi/2)}\circ\mathrm{H_0}$ and $\mathrm{SWAP_{21}}\circ\mathrm{CZ_{21}(\pi/4)}$ as depicted below:
#
# ![qft-HR2R3-boxed-separately.png](../_static/qft_HR2R3_boxed_separately.png)
#
# Afterwards, we can precompute the gate combination $\mathrm{SWAP_{21}}\circ\mathrm{CZ_{21}(\pi/4)}\circ\mathrm{SWAP_{10}}\circ\mathrm{CZ_{10}(\pi/2)}\circ\mathrm{H_0}$ from those pulses:
#
# ![qft-HR2R3-boxed.png](../_static/qft_HR2R3_boxed.png)


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
    show_progressbar=True
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
    show_progressbar=True
)

four_qubit_pulses['hadamard-CZ_pi2_echo-swap'] = {}
four_qubit_pulses['hadamard-CZ_pi2_echo-swap'][(0, 1)] = ff.concatenate(
    (four_qubit_pulses['hadamard'][0],
     four_qubit_pulses['CZ_pi2_echo'][(0, 1)],
     four_qubit_pulses['swap'][(0, 1)]),
    show_progressbar=True
)
four_qubit_pulses['hadamard-CZ_pi2-swap'] = {}
four_qubit_pulses['hadamard-CZ_pi2-swap'][(0, 1)] = ff.concatenate(
    (four_qubit_pulses['hadamard'][0],
     four_qubit_pulses['CZ_pi2'][(0, 1)],
     four_qubit_pulses['swap'][(0, 1)]),
    show_progressbar=True
)
four_qubit_pulses['CZ_pi4_echo-swap'] = {}
four_qubit_pulses['CZ_pi4_echo-swap'][(1, 2)] = ff.concatenate(
    (four_qubit_pulses['CZ_pi4_echo'][(1, 2)],
     four_qubit_pulses['swap'][(1, 2)]),
    show_progressbar=True
)
four_qubit_pulses['CZ_pi4-swap'] = {}
four_qubit_pulses['CZ_pi4-swap'][(1, 2)] = ff.concatenate(
    (four_qubit_pulses['CZ_pi4'][(1, 2)],
     four_qubit_pulses['swap'][(1, 2)]),
    show_progressbar=True
)
four_qubit_pulses['CZ_pi8-swap'] = {}
four_qubit_pulses['CZ_pi8-swap'][(2, 3)] = ff.concatenate(
    (four_qubit_pulses['CZ_pi8'][(2, 3)],
     four_qubit_pulses['swap'][(2, 3)]),
    show_progressbar=True
)
four_qubit_pulses['hadamard-CZ_pi2-swap-CZ_pi4-swap'] = {}
four_qubit_pulses['hadamard-CZ_pi2-swap-CZ_pi4-swap'][(0, 1, 2)] = ff.concatenate(
    (four_qubit_pulses['hadamard-CZ_pi2-swap'][(0, 1)],
     four_qubit_pulses['CZ_pi4-swap'][(1, 2)]),
    show_progressbar=True
)
four_qubit_pulses['hadamard-CZ_pi2_echo-swap-CZ_pi4_echo-swap'] = {}
four_qubit_pulses['hadamard-CZ_pi2_echo-swap-CZ_pi4_echo-swap'][(0, 1, 2)] = ff.concatenate(
    (four_qubit_pulses['hadamard-CZ_pi2_echo-swap'][(0, 1)],
     four_qubit_pulses['CZ_pi4_echo-swap'][(1, 2)]),
    show_progressbar=True
)


# At last we can concatenate those pulses to get the quantum fourier transform and plot the filter function.


qft_pulse_echo = ff.concatenate(
    (four_qubit_pulses['hadamard-CZ_pi2_echo-swap-CZ_pi4_echo-swap'][(0, 1, 2)],  # rotations on first qubit
     four_qubit_pulses['CZ_pi8-swap'][(2, 3)],                          # ...
     four_qubit_pulses['hadamard-CZ_pi2_echo-swap-CZ_pi4_echo-swap'][(0, 1, 2)],  # rotations on second qubit
     four_qubit_pulses['hadamard-CZ_pi2-swap'][(0, 1)],                 # rotation on third qubit
     four_qubit_pulses['hadamard'][0]),                                 # rotation on fourth qubit
    show_progressbar=True
)
qft_pulse = ff.concatenate(
    (four_qubit_pulses['hadamard-CZ_pi2-swap-CZ_pi4-swap'][(0, 1, 2)],  # rotations on first qubit
     four_qubit_pulses['CZ_pi8-swap'][(2, 3)],                          # ...
     four_qubit_pulses['hadamard-CZ_pi2-swap-CZ_pi4-swap'][(0, 1, 2)],  # rotations on second qubit
     four_qubit_pulses['hadamard-CZ_pi2-swap'][(0, 1)],                 # rotation on third qubit
     four_qubit_pulses['hadamard'][0]),                                 # rotation on fourth qubit
    show_progressbar=True
)

print('Correct action:',
      ff.util.oper_equiv(qt_qft(4), qft_pulse.total_propagator), ff.util.oper_equiv(qt_qft(4), qft_pulse_echo.total_propagator))
print('Trace fidelity:',
      abs(np.trace(qt_qft(4).dag().full() @ qft_pulse.total_propagator))/2**4,
      abs(np.trace(qt_qft(4).dag().full() @ qft_pulse_echo.total_propagator))/2**4)
print('Filter function cached:', qft_pulse.is_cached('filter_function'))
qt.matrix_histogram_complex(qft_pulse.total_propagator)


# In[10]:


import matplotlib.pyplot as plt

single_qubit_identifiers = [
    i for i in qft_pulse.n_oper_identifiers if len(i) < 17
]
two_qubit_identifiers = [
    i for i in qft_pulse.n_oper_identifiers if len(i) > 16
]
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(3.4, 3.4/golden_ratio))
_ = plotting.plot_filter_function(qft_pulse, axes=ax[0],
                                  yscale='log', omega_in_units_of_tau=False,
                                  n_oper_identifiers=single_qubit_identifiers)
_ = plotting.plot_filter_function(qft_pulse, axes=ax[1],
                                  yscale='log', omega_in_units_of_tau=False,
                                  n_oper_identifiers=two_qubit_identifiers[:-1])

for n in (1, 2, 3, 4):
    ax[0].axvline(n*2*np.pi/30, color='k', zorder=0, linestyle='--', alpha=0.3)
    ax[1].axvline(n*2*np.pi/30, color='k', zorder=0, linestyle='--', alpha=0.3)

ax[0].set_title('Single-qubit filter functions')
ax[0].legend(ncol=4, fontsize=6)
ax[0].set_xlabel('')
ax[1].set_title('Multi-qubit filter functions')
ax[1].legend(ncol=4, fontsize=6)

fig.tight_layout(h_pad=0)
for ext in ('pdf', 'eps'):
    if (not (save_path / f'qft_filter_function.{ext}').exists()
            or _force_overwrite):
        fig.savefig(save_path / f'qft_filter_function.{ext}')

# Evidently, the DC regime is dominated by the $X_3$ and $Y_3$ filter functions. This is obvious, since the third qubit idles for most of the algorithm in this circuit arrangement. In a realistic setting, the idling periods would be filled with dynamical decoupling sequences, thus cancelling most of the slow noise on the third qubit. Similarly, the $ZZ_{23}$ exchange is turned on least frequently and thus dominates the exchange filter functions.
#
# The sharp peaks, some of which are indicated by grey dashed lines, are harmonics located at frequencies which are multiples of the inverse duration of a a single atomic pulse, $t_\mathrm{clock} = 30$, i.e. $\omega_n = 2\pi n/t_\mathrm{clock}$. Interestingly, the filter function has a baseline of around $10^4$ in the range $\omega\in[10^{-1}, 10^{1}]$ before it drops down to follow the usual $1/\omega^2$ behavior.

# In[10]:


import matplotlib.pyplot as plt

single_qubit_identifiers = [
    i for i in qft_pulse.n_oper_identifiers if len(i) < 17
]
two_qubit_identifiers = [
    i for i in qft_pulse.n_oper_identifiers if len(i) > 16
]
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=figsize_wide)
_ = plotting.plot_filter_function(qft_pulse, axes=ax[0],
                                  yscale='log', omega_in_units_of_tau=False,
                                  n_oper_identifiers=single_qubit_identifiers,
                                  plot_kw=dict(linewidth=1))
_ = plotting.plot_filter_function(qft_pulse, axes=ax[1],
                                  yscale='log', omega_in_units_of_tau=False,
                                  n_oper_identifiers=two_qubit_identifiers[:-1],
                                  plot_kw=dict(linewidth=1))

for n in (1, 2, 3, 4):
    ax[0].axvline(n*2*np.pi/30, color='k', zorder=0, linestyle='--', alpha=0.3,
                  linewidth=1)
    ax[1].axvline(n*2*np.pi/30, color='k', zorder=0, linestyle='--', alpha=0.3,
                  linewidth=1)

ax[0].set_title('Single-qubit filter functions')
ax[0].set_xlabel('')
ax[0].legend(ncol=2)
ax[1].set_title('Multi-qubit filter functions')

fig.tight_layout(h_pad=0)
for ext in ('pdf', 'eps'):
    if (not (save_path / f'qft_filter_function_wide.{ext}').exists()
            or _force_overwrite):
        fig.savefig(save_path / f'qft_filter_function_wide.{ext}')

# Evidently, the DC regime is dominated by the $X_3$ and $Y_3$ filter functions. This is obvious, since the third qubit idles for most of the algorithm in this circuit arrangement. In a realistic setting, the idling periods would be filled with dynamical decoupling sequences, thus cancelling most of the slow noise on the third qubit. Similarly, the $ZZ_{23}$ exchange is turned on least frequently and thus dominates the exchange filter functions.

# The sharp peaks, some of which are indicated by grey dashed lines, are harmonics located at frequencies which are multiples of the inverse duration of a a single atomic pulse, $t_\mathrm{clock} = 30$, i.e. $\omega_n = 2\pi n/t_\mathrm{clock}$. Interestingly, the filter function has a baseline of around $10^4$ in the range $\omega\in[10^{-1}, 10^{1}]$ before it drops down to follow the usual $1/\omega^2$ behavior.

# In[10]:


import matplotlib.pyplot as plt

identifiers = [r'$\sigma_x^{(0)}$', r'$\sigma_y^{(0)}$',
               r'$\sigma_z^{(0)}\sigma_z^{(1)}$']

fig, ax = plt.subplots(1, 1, figsize=(7, 3))
_ = plotting.plot_filter_function(qft_pulse, axes=ax,
                                  yscale='log', omega_in_units_of_tau=False,
                                  n_oper_identifiers=identifiers,
                                  plot_kw=dict(linewidth=1))

for n in (1, 2, 3, 4):
    ax.axvline(n*2*np.pi/30, color='k', zorder=0, linestyle='--', alpha=0.3,
               linewidth=1)

ax.legend(loc='lower left')
ax.set_xlabel(r'$\omega$ (ns)')

fig.tight_layout(h_pad=0)
# fig.savefig(save_path / 'qft_filter_function_first_qubit_wide.eps', dpi=600)
if not (save_path / 'qft_filter_function_first_qubit_wide.pdf').exists() or _force_overwrite:
    fig.savefig(save_path / 'qft_filter_function_first_qubit_wide.pdf')

# Evidently, the DC regime is dominated by the $X_3$ and $Y_3$ filter functions. This is obvious, since the third qubit idles for most of the algorithm in this circuit arrangement. In a realistic setting, the idling periods would be filled with dynamical decoupling sequences, thus cancelling most of the slow noise on the third qubit. Similarly, the $ZZ_{23}$ exchange is turned on least frequently and thus dominates the exchange filter functions.

# The sharp peaks, some of which are indicated by grey dashed lines, are harmonics located at frequencies which are multiples of the inverse duration of a a single atomic pulse, $t_\mathrm{clock} = 30$, i.e. $\omega_n = 2\pi n/t_\mathrm{clock}$. Interestingly, the filter function has a baseline of around $10^4$ in the range $\omega\in[10^{-1}, 10^{1}]$ before it drops down to follow the usual $1/\omega^2$ behavior.

# In[10]:
identifiers = [r'$\sigma_x^{(0)}$', r'$\sigma_y^{(0)}$',
               r'$\sigma_z^{(0)}\sigma_z^{(1)}$']

fig, ax, leg = plotting.plot_filter_function(qft_pulse, yscale='log', omega_in_units_of_tau=False,
                                             n_oper_identifiers=identifiers[0:3], figsize=(13.33, 7.1),
                                             plot_kw=dict(linewidth=1), xscale='log')

ax.set_prop_cycle(plt.rcParams['axes.prop_cycle'])
fig, ax, leg = plotting.plot_filter_function(four_qubit_pulses['X_pi2'][0], yscale='log',
                                             fig=fig, axes=ax, xscale='log',
                                             omega_in_units_of_tau=False,
                                             n_oper_identifiers=identifiers[0:2],
                                             plot_kw=dict(linewidth=0.75, alpha=1, linestyle='--'))

ax.set_prop_cycle(plt.rcParams['axes.prop_cycle'])
fig, ax, leg = plotting.plot_filter_function(four_qubit_pulses['Y_pi2'][0], yscale='log',
                                             fig=fig, axes=ax, xscale='log',
                                             omega_in_units_of_tau=False,
                                             n_oper_identifiers=identifiers[0:2],
                                             plot_kw=dict(linewidth=0.75, alpha=1, linestyle='-.'))

ax.set_prop_cycle(plt.rcParams['axes.prop_cycle'])
fig, ax, leg = plotting.plot_filter_function(four_qubit_pulses['CZ_pi8'][(0, 1)], yscale='log',
                                             fig=fig, axes=ax, xscale='log',
                                             omega_in_units_of_tau=False,
                                             n_oper_identifiers=identifiers[0:3],
                                             plot_kw=dict(linewidth=0.75, alpha=1, linestyle=':'))

leg = ax.legend(loc='lower left')
ax.set_xlabel(r'$\omega$ (ns)')
ax.grid(False)

fig.tight_layout(h_pad=0)
for ext in ('pdf', 'eps'):
    if (not (save_path / f'qft_filter_function_first_qubit_wide.{ext}').exists()
            or _force_overwrite):
        fig.savefig(save_path / f'qft_filter_function_first_qubit_wide.{ext}')

# Evidently, the DC regime is dominated by the $X_3$ and $Y_3$ filter functions. This is obvious, since the third qubit idles for most of the algorithm in this circuit arrangement. In a realistic setting, the idling periods would be filled with dynamical decoupling sequences, thus cancelling most of the slow noise on the third qubit. Similarly, the $ZZ_{23}$ exchange is turned on least frequently and thus dominates the exchange filter functions.

# The sharp peaks, some of which are indicated by grey dashed lines, are harmonics located at frequencies which are multiples of the inverse duration of a a single atomic pulse, $t_\mathrm{clock} = 30$, i.e. $\omega_n = 2\pi n/t_\mathrm{clock}$. Interestingly, the filter function has a baseline of around $10^4$ in the range $\omega\in[10^{-1}, 10^{1}]$ before it drops down to follow the usual $1/\omega^2$ behavior.

# %% FF with cumulative FF
identifiers = [r'$\sigma_x^{(0)}$', r'$\sigma_y^{(0)}$',
               r'$\sigma_z^{(0)}\sigma_z^{(1)}$']

fig, ax, leg = plotting.plot_filter_function(qft_pulse, yscale='log',
                                             figsize=(figsize_wide[0], figsize_wide[0]/3),
                                             omega_in_units_of_tau=False,
                                             n_oper_identifiers=identifiers)

# ax.set_xlabel(r'$\omega$ (\si{\per\nano\second})')
ax.set_xlabel(r'$\omega$ (ns$^{-1}$)')
ax.grid(False)

width = 4e-4
head_width = 1.5e-2
head_length = 1.5e-2
ax.arrow(.08, .85, -.04, 0, transform=ax.transAxes, length_includes_head=False,
         width=width, head_width=head_width, head_length=head_length, color='k')
ax.arrow(.92, .8, .04, 0, transform=ax.transAxes, length_includes_head=False,
         width=width, head_width=head_width, head_length=head_length, color='k')

for n in (1, 2, 3, 4):
    ax.axvline(n*2*np.pi/30, color='k', zorder=0, linestyle=':', alpha=0.5,
               linewidth=1)

# calculate cumulative sensitivity, \int_0^\omega_c\dd{\omega} FS(\omega)
idx = get_indices_from_identifiers(qft_pulse.n_oper_identifiers, identifiers)
F = qft_pulse.get_filter_function(omega)[idx, idx].real
FS = integrate.cumtrapz(F, omega, axis=-1, initial=0)

ax2 = ax.twinx()
ax2.semilogx(omega, FS.T, linestyle='--')
ax2.set_ylabel(r"$\displaystyle\int_0^\omega\mathrm{d}\omega'F(\omega')$")

ax.legend(loc="lower left", bbox_to_anchor=(0., 0.35), fancybox=False, frameon=False)
fig.tight_layout()
for ext in ('pdf', 'eps'):
    if (not (save_path / f'qft_filter_function_first_qubit_with_cumulative.{ext}').exists()
            or _force_overwrite):
        fig.savefig(save_path / f'qft_filter_function_first_qubit_with_cumulative.{ext}')
# %% FF with cumulative FF fraction
identifiers = [r'$\sigma_x^{(0)}$', r'$\sigma_y^{(0)}$',
               r'$\sigma_z^{(0)}\sigma_z^{(1)}$']

cols = np.array([colors.hex2color(colors.TABLEAU_COLORS['tab:blue']),
                 colors.hex2color(colors.TABLEAU_COLORS['tab:green']),
                 colors.hex2color(colors.TABLEAU_COLORS['tab:orange'])])
cols = colors.rgb_to_hsv(cols)
cols[1, 1] -= 0.10
cols[1, 2] += 0.10
cols[2, 1] -= 0.20
cols[2, 2] -= 0.00

# cycle = cycler(color=colors.hsv_to_rgb(cols))
cycle = cycler(color=plt.get_cmap('Blues')(np.linspace(1/3, 3/3, 3)[::-1]))

qft_pulse_copy = deepcopy(qft_pulse)
qft_pulse_copy.omega = np.insert(qft_pulse.omega, 0, 0)
qft_pulse_copy._control_matrix = np.concatenate([qft_pulse._control_matrix[..., 0:1],
                                                 qft_pulse._control_matrix], axis=-1)
qft_pulse_copy._filter_function = np.concatenate([qft_pulse._filter_function[..., 0:1],
                                                  qft_pulse._filter_function], axis=-1)

fig, ax, leg = plotting.plot_filter_function(qft_pulse_copy, yscale='log',
                                             figsize=(figsize_wide[0], figsize_wide[0]/3),
                                             omega_in_units_of_tau=False,
                                             n_oper_identifiers=identifiers,
                                             cycler=cycle)

# ax.set_xlabel(r'$\omega$ (\si{\per\nano\second})')
ax.set_xlabel(r'$\omega$ (ns$^{-1}$)')
ax.set_ylabel(r'$F_\alpha(\omega)$')
ax.grid(False)

ax.set_xlim(1e-4)

width = 4e-4
head_width = 1.5e-2
head_length = 1.0e-2
ax.arrow(.07, .825, -.04, 0, transform=ax.transAxes, length_includes_head=False,
         width=width, head_width=head_width, head_length=head_length, color='k')
ax.arrow(.93, .9, .04, 0, transform=ax.transAxes, length_includes_head=False,
         width=width, head_width=head_width, head_length=head_length, color='k')

for n in (1, 2, 3, 4):
    ax.axvline(n*2*np.pi/30, color='k', zorder=0, linestyle=':', alpha=0.5,
               linewidth=1)

# calculate cumulative sensitivity, \int_0^\omega_c\dd{\omega} FS(\omega)
idx = get_indices_from_identifiers(qft_pulse_copy.n_oper_identifiers, identifiers)
F = qft_pulse_copy.get_filter_function(qft_pulse_copy.omega)[idx, idx].real
FS = integrate.cumtrapz(F, qft_pulse_copy.omega, axis=-1, initial=0)

ax2 = ax.twinx()
ax2.set_prop_cycle(cycle)
ax2.semilogx(qft_pulse_copy.omega, FS.T / FS[:, -1], linestyle='--', zorder=0)
ax2.set_ylabel(r"$\mathcal{I}_\alpha(0, \omega) / \mathcal{I}_\alpha(0, \infty)$")

ax.legend(loc="lower left", bbox_to_anchor=(0., 0.35), fancybox=False, frameon=False)
fig.tight_layout()
for ext in ('pdf', 'eps'):
    if (not (save_path / f'qft_filter_function_first_qubit_with_cumulative_fraction.{ext}').exists()
            or _force_overwrite):
        fig.savefig(save_path / f'qft_filter_function_first_qubit_with_cumulative_fraction.{ext}')

# %% FF with cumulative FF fraction twocolored
identifiers = [r'$\sigma_x^{(0)}$', r'$\sigma_y^{(0)}$',
               r'$\sigma_z^{(0)}\sigma_z^{(1)}$']

qft_pulse_copy = deepcopy(qft_pulse)
qft_pulse_copy.omega = np.insert(qft_pulse.omega, 0, 0)
qft_pulse_copy._control_matrix = np.concatenate([qft_pulse._control_matrix[..., 0:1],
                                                 qft_pulse._control_matrix], axis=-1)
qft_pulse_copy._filter_function = np.concatenate([qft_pulse._filter_function[..., 0:1],
                                                  qft_pulse._filter_function], axis=-1)

fig, ax, leg = plotting.plot_filter_function(qft_pulse_copy, yscale='log',
                                             figsize=(figsize_wide[0], figsize_wide[0]/3),
                                             omega_in_units_of_tau=False,
                                             n_oper_identifiers=identifiers,
                                             cycler=cycler(linestyle=('-','--','-.'),
                                                           color=('tab:blue',)*3))

# ax.set_xlabel(r'$\omega$ (\si{\per\nano\second})')
ax.set_xlabel(r'$\omega$ (ns$^{-1}$)')
ax.set_ylabel(r'$F_\alpha(\omega)$')
ax.grid(False)

ax.set_xlim(1e-4)

width = 4e-4
head_width = 1.5e-2
head_length = 1.5e-2
ax.arrow(.08, .825, -.04, 0, transform=ax.transAxes, length_includes_head=False,
         width=width, head_width=head_width, head_length=head_length, color='tab:blue')
ax.arrow(.92, .9, .04, 0, transform=ax.transAxes, length_includes_head=False,
         width=width, head_width=head_width, head_length=head_length, color='tab:orange')

for n in (1, 2, 3, 4):
    ax.axvline(n*2*np.pi/30, color='k', zorder=0, linestyle=':', alpha=0.5,
               linewidth=1)

# calculate cumulative sensitivity, \int_0^\omega_c\dd{\omega} FS(\omega)
idx = get_indices_from_identifiers(qft_pulse_copy.n_oper_identifiers, identifiers)
F = qft_pulse_copy.get_filter_function(qft_pulse_copy.omega)[idx, idx].real
FS = integrate.cumtrapz(F, qft_pulse_copy.omega, axis=-1, initial=0)

ax2 = ax.twinx()
ax2.set_prop_cycle(cycler(linestyle=('-','--','-.'),
                          color=('tab:orange',)*3))
ax2.semilogx(qft_pulse_copy.omega, FS.T / FS[:, -1], zorder=0)
ax2.set_ylabel(r"$\mathcal{I}_\alpha(0, \omega) / \mathcal{I}_\alpha(0, \infty)$")

handles = [lines.Line2D([0], [0], linestyle='-', color='tab:grey'),
           lines.Line2D([0], [0], linestyle='--', color='tab:grey'),
           lines.Line2D([0], [0], linestyle='-.', color='tab:grey')]

ax.legend(loc="lower left", handles=handles, labels=identifiers,
          bbox_to_anchor=(0., 0.35), fancybox=False, frameon=False)
fig.tight_layout()
for ext in ('pdf', 'eps'):
    if (not (file := save_path / f'qft_filter_function_first_qubit_with_cumulative_fraction_twocolor.{ext}').exists()
            or _force_overwrite):
        fig.savefig(file)

# %% FF with & without echo
identifiers = ['$\\sigma_y^{(3)}$']
fig, ax, _ = plotting.plot_filter_function(qft_pulse,
                                           n_oper_identifiers=identifiers,
                                           yscale='log',
                                           omega_in_units_of_tau=False,
                                           figsize=(figsize_wide[0], figsize_wide[1]/2))
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
fig.tight_layout()
for ext in ('png', 'eps', 'pdf'):
    if (not (f := save_path / '.'.join([fname, ext])).exists() or _force_overwrite):
        fig.savefig(f)

# %% FF with & without echo with smoothed spectrum


def fermi(x, offset, width):
    return 1 / (np.exp((x - offset)/width*4) + 1)


def fermi_highpass(freqs, offset, width):
    return fermi(freqs, freqs[0] + offset, -width)


identifiers = ['$\\sigma_y^{(3)}$']
fig, ax, _ = plotting.plot_filter_function(qft_pulse,
                                           n_oper_identifiers=identifiers,
                                           yscale='log',
                                           omega_in_units_of_tau=False,
                                           figsize=figsize_narrow)
fig, ax, _ = plotting.plot_filter_function(qft_pulse_echo,
                                           n_oper_identifiers=identifiers,
                                           yscale='log',
                                           omega_in_units_of_tau=False,
                                           axes=ax,
                                           fig=fig)

ax2 = ax.twinx()
omega = qft_pulse.omega
bw = np.ptp(omega)
ax2.loglog(omega, fermi_highpass(omega, omega[0]/bw*30, omega[0]/bw*10) * 1e-9/omega,
           color='tab:green')
ax2.set_ylabel(r'$S(\omega)$ ($2\pi$ GHz$^2$/GHz)')

# ax.axvline(2*np.pi/qft_pulse.duration, color='black', linestyle='--')
# ax.annotate(r'$2\pi/\tau$', (2*np.pi/qft_pulse.duration, 20), (2*np.pi/qft_pulse.duration*4, 0.2),
#             arrowprops={'arrowstyle': '->',
#                         'connectionstyle': 'arc,angleA=120,armA=15,rad=10'})
ax.grid(False)
leg = ax.legend(['FF without echo', 'FF with echo'], frameon=False, loc='lower left')
ax2.legend(['Noise spectrum'], frameon=False, loc=(.027, .282))
ax.set_xlabel(r'$\omega$ ($2\pi$GHz)')
ax.set_ylabel(r'$F(\omega)$ (ns/$2\pi)^2$')

fname = 'qft_filter_function_Y3_echo_vs_no_with_spec'
fig.tight_layout()
for ext in ('png', 'eps', 'pdf'):
    if not (f := save_path / '.'.join([fname, ext])).exists() or _force_overwrite:
        fig.savefig(f)
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

# S, omega = S[::2], omega[::2]

qft_pulse_correl = ff.concatenate(
    [four_qubit_pulses[pls][qubits] for pls, qubits in zip(pls, qubits)],
    calc_pulse_correlation_FF=True,
    omega=omega,
    show_progressbar=True
)
qft_pulse_echo_correl = ff.concatenate(
    [four_qubit_pulses[pls][qubits] for pls, qubits in zip(pls_echo, qubits)],
    calc_pulse_correlation_FF=True,
    omega=omega,
    show_progressbar=True
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
