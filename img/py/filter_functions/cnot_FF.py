# %% Imports
import pathlib
import sys
from itertools import product

import filter_functions as ff
import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from cycler import cycler
from filter_functions import plotting, util
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from qutil.plotting.colors import RWTH_COLORS, make_diverging_colormap
from scipy import constants, io

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, TEXTWIDTH, TOTALWIDTH, PATH, init

with np.errstate(divide='ignore', invalid='ignore'):
    DIVERGING_CMAP = make_diverging_colormap(('green', 'magenta'), endpoint='white')

LINE_COLORS = list(RWTH_COLORS.values())[1:]
DATA_PATH = PATH.parent / 'data/filter_functions'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)

init(MAINSTYLE, backend := 'pgf')

cycle = mpl.rcParams['axes.prop_cycle'][:4] + cycler(linestyle=('-', '-.', '--', ':'))

_force_overwrite = True

# %% set up the operators
gates = ['X2ID', 'Y2ID', 'CNOT']
struct = {'X2ID': io.loadmat(str(DATA_PATH / 'X2ID.mat')),
          'Y2ID': io.loadmat(str(DATA_PATH / 'Y2ID.mat')),
          'CNOT': io.loadmat(str(DATA_PATH / 'CNOT.mat'))}
eps = {key: np.asarray(struct[key]['eps'], order='C', dtype=float)
       for key in gates}
dt = {key: np.asarray(struct[key]['t'].ravel(), order='C', dtype=float)
      for key in gates}
B = {key: np.asarray(struct[key]['B'].ravel(), order='C', dtype=float)
     for key in gates}
B_avg = {key: struct[key]['BAvg'].ravel() for key in gates}
infid_fast = {key: struct[key]['infid_fast'].ravel() for key in gates}
# B_avg same for all
B_avg = B_avg['X2ID']
T = {key: val.sum() for key, val in dt.items()}

J = {key: np.exp(eps[key]) for key in gates}
ndt = {key: len(dt[key]) for key in gates}

d = 16
H = np.empty((6, d, d), dtype=complex)

Id, Px, Py, Pz = util.paulis
# Exchange Hamiltonians
H[0] = 1/4*sum(util.tensor(P, P, Id, Id) for P in (Px, Py, Pz)).real
H[1] = 1/4*sum(util.tensor(Id, P, P, Id) for P in (Px, Py, Pz)).real
H[2] = 1/4*sum(util.tensor(Id, Id, P, P) for P in (Px, Py, Pz)).real
# Zeeman Hamiltonians
H[3] = 1/8*(util.tensor(Pz, Id, Id, Id)*(-3) +
            util.tensor(Id, Pz, Id, Id) +
            util.tensor(Id, Id, Pz, Id) +
            util.tensor(Id, Id, Id, Pz)).real
H[4] = 1/4*(util.tensor(Pz, Id, Id, Id)*(-1) +
            util.tensor(Id, Pz, Id, Id)*(-1) +
            util.tensor(Id, Id, Pz, Id) +
            util.tensor(Id, Id, Id, Pz)).real
H[5] = 1/8*(util.tensor(Pz, Id, Id, Id)*(-1) +
            util.tensor(Id, Pz, Id, Id)*(-1) +
            util.tensor(Id, Id, Pz, Id)*(-1) +
            util.tensor(Id, Id, Id, Pz)*3).real

# Technically there would also be H_0 (the mean magnetic field), but on the
# m_s = 0 subspace it is zero.

# %% Set up Hamiltonian
opers = list(H)

# Reduce to 6x6 subspace
zerospin_subspace_inds = ((3, 5, 6, 9, 10, 12), (3, 5, 6, 9, 10, 12))
d_zerospin_subspace = 6
qubit_subspace_inds = ((1, 2, 3, 4), (1, 2, 3, 4))
opers = [H[np.ix_(*zerospin_subspace_inds)] for H in H]

# Subtract identity to make Hamiltonian traceless (always allowed since we are
# not interested in absolute energies)
opers = [oper - np.trace(oper)/d_zerospin_subspace*np.eye(d_zerospin_subspace)
         for oper in opers]

# The coefficients are the exchange couplings and B-field gradients
c_coeffs = {key: [J[key][0],
                  J[key][1],
                  J[key][2],
                  B[key][0]*np.ones(ndt[key]),
                  B[key][1]*np.ones(ndt[key]),
                  B[key][2]*np.ones(ndt[key])] for key in gates}
# We include the exponential dependence of J on epsilon by a first-order
# derivative (just J back) as noise sensitivity.
n_coeffs = {key: [J[key][0],
                  J[key][1],
                  J[key][2],
                  np.ones(ndt[key]),
                  np.ones(ndt[key]),
                  np.ones(ndt[key])] for key in gates}

identifiers = [r'$\epsilon_{12}$', r'$\epsilon_{23}$', r'$\epsilon_{34}$',
               '$b_{12}$', '$b_{23}$', '$b_{34}$']
H_c = {key: list(zip(opers, val, identifiers))
       for key, val in c_coeffs.items()}
H_n = {key: list(zip(opers, val, identifiers))
       for key, val in n_coeffs.items()}

# Basis for qubit subspace (just pad a two-qubit pauli basis with zeros on the
# leakage space)
qubit_subspace_mask = np.pad(np.ones((4, 4)), 1, 'constant').astype(bool)
qubit_subspace_basis = ff.Basis(
    [np.pad(b, 1, 'constant') for b in ff.Basis.pauli(2)],
    btype='Pauli'
)
complete_basis = ff.Basis.from_partial(qubit_subspace_basis, btype='Pauli')
# Initialize the PulseSequences
X2ID_complete = ff.PulseSequence(H_c['X2ID'], H_n['X2ID'], dt['X2ID'],
                                 basis=complete_basis)
Y2ID_complete = ff.PulseSequence(H_c['Y2ID'], H_n['Y2ID'], dt['Y2ID'],
                                 basis=complete_basis)
CNOT_complete = ff.PulseSequence(H_c['CNOT'], H_n['CNOT'], dt['CNOT'],
                                 basis=complete_basis)
pulses_complete = {'X2ID': X2ID_complete, 'Y2ID': Y2ID_complete,
                   'CNOT': CNOT_complete}

X2ID_subspace = ff.PulseSequence(H_c['X2ID'], H_n['X2ID'], dt['X2ID'],
                                 basis=qubit_subspace_basis)
Y2ID_subspace = ff.PulseSequence(H_c['Y2ID'], H_n['Y2ID'], dt['Y2ID'],
                                 basis=qubit_subspace_basis)
CNOT_subspace = ff.PulseSequence(H_c['CNOT'], H_n['CNOT'], dt['CNOT'],
                                 basis=qubit_subspace_basis)
pulses_subspace = {'X2ID': X2ID_subspace, 'Y2ID': Y2ID_subspace,
                   'CNOT': CNOT_subspace}

X2ID_ggm = ff.PulseSequence(H_c['X2ID'], H_n['X2ID'], dt['X2ID'])
Y2ID_ggm = ff.PulseSequence(H_c['Y2ID'], H_n['Y2ID'], dt['Y2ID'])
CNOT_ggm = ff.PulseSequence(H_c['CNOT'], H_n['CNOT'], dt['CNOT'],)
pulses_ggm = {'X2ID': X2ID_ggm, 'Y2ID': Y2ID_ggm, 'CNOT': CNOT_ggm}

for key in gates:
    pulses_subspace[key].diagonalize()
    pulses_complete[key].diagonalize()
    pulses_ggm[key].diagonalize()

# %% constants
infid = {key: [np.ones(2)] for key in gates}
delta_infid = {key: [] for key in gates}
xi = {key: [] for key in gates}

eps0 = 2.7241e-4
mu_B = constants.value('Bohr magneton')
t_unit = 1e-9
h_unit = 1/(2*np.pi*t_unit)
B_unit = mu_B*0.44/constants.h/h_unit
sigma_eps = 8e-6/eps0
sigma_B = 3e-4*B_unit
# S(f) = S_0 f^{-\alpha}
alpha = np.array([0, 0.7])
# At f = 1 MHz = 1e-3 GHz, S = S_0 = 4e-20 V^2/Hz = 4e-11 1/GHz
# Correspondingly, S(\omega) = A \omega^{-\alpha} such that at
# \omega = 2\pi 10^{-3} GHz, S_0 = 4e-11 1/GHz
S_0 = 4e-11/eps0**2
A = S_0*(2*np.pi*1e-3)**alpha

n = 1001
# %% function to cache projected filter function

d_c = 4
d_l = 2
d = d_c + d_l
PI_c_op = np.zeros((6, 6))
PI_l_op = np.zeros((6, 6))
PI_c_op[range(1, 5), range(1, 5)] = 1
PI_l_op[(0, 5), (0, 5)] = 1


def cache_filter_function(pulse, omega, proj=None):
    if proj is None:
        proj = ff.superoperator.liouville_representation(PI_c_op, pulse.basis)

    pulse.cache_filter_function(omega, filter_function=ff.numeric.calculate_filter_function(
        np.einsum('aio,ij->ajo', pulse.get_control_matrix(omega), proj)
    ))


# %% calculate transfer matrix
K_complete = {}
K_ggm = {}
P_complete = {}
P_ggm = {}
for key, pulse in pulses_complete.items():
    K_complete[key] = {}
    K_ggm[key] = {}
    P_complete[key] = {}
    P_ggm[key] = {}
    for i, a in enumerate(alpha):
        omega = np.geomspace(1e-2/T[key], 1e2, n)
        S = A[i]/omega**a
        K_complete[key][a] = ff.numeric.calculate_cumulant_function(pulse, S, omega,
                                                                    show_progressbar=True)
        K_ggm[key][a] = ff.numeric.calculate_cumulant_function(pulses_ggm[key], S, omega,
                                                               show_progressbar=True)
        P_complete[key][a] = ff.error_transfer_matrix(pulse,
                                                      cumulant_function=K_complete[key][a][:3])
        P_ggm[key][a] = ff.error_transfer_matrix(pulses_ggm[key],
                                                 cumulant_function=K_ggm[key][a][:3])

basis_labels = ['']
basis_labels.extend([''.join(tup) for tup in
                     product(['I', 'X', 'Y', 'Z'], repeat=2)][1:])
basis_labels.extend(['$C_{{{}}}$'.format(i) for i in range(16, 36)])

# %% everything together
key = 'CNOT'
a = 0.7
K = K_complete[key][a][:3].real
colorscale = 'linear'
# colorscale = 'log'

fig = plt.figure(figsize=(TOTALWIDTH, 4))
gs = GridSpec(5, 10, figure=fig, width_ratios=[1.5]*9 + [1])

# Pulse train
pt_ax = fig.add_subplot(gs[:2, :4])
*_, leg = plotting.plot_pulse_train(
    pulses_complete[key], c_oper_identifiers=identifiers[:3],
    axes=pt_ax, fig=fig,
    # cycler=cycle
)
pt_ax.set_xlabel(r'$t$ (ns)')
pt_ax.set_ylabel(r'$J(\epsilon_{ij})$ ($2\pi$GHz)')
pt_ax.grid(False)
pt_ax.text(0.015, 0.875, '(a)', transform=pt_ax.transAxes, fontsize=10)
leg.remove()

# Filter functions
ff_ax = fig.add_subplot(gs[:2, 4:])
omega = np.geomspace(1/T[key], 1e2, n)

cache_idx = np.diag([0, *[1]*15, *[0]*20])
cache_filter_function(pulses_complete[key], omega, cache_idx)

*_, leg = plotting.plot_filter_function(
    pulses_complete[key], omega, n_oper_identifiers=identifiers[:3],
    yscale='log', omega_in_units_of_tau=False, axes=ff_ax, fig=fig,
    # cycler=cycle
)
# ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# ax.set_ylim(1e-15, ax.get_ylim()[1])
ff_ax.legend(identifiers[:3], frameon=False, loc='upper right')
ff_ax.grid(False)
ff_ax.set_xlabel(ff_ax.get_xlabel() + r' (ns$^{-1}$)')
ff_ax.set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
ff_ax.tick_params(direction='out', which='both', labelsize=8)
ff_ax.set_ylim(bottom=1e-9, top=1e3)

omega = np.linspace(0, 1e2/T[key], n)
cache_filter_function(pulses_complete[key], omega, cache_idx)

F = pulses_complete[key].get_filter_function(omega)
ins_ax = ff_ax.inset_axes([0.1, 0.15, 0.5, 0.5])

for i, props in enumerate(cycle[:3]):
    ins_ax.plot(omega, F[i, i].real, linewidth=1)#, **props)

ins_ax.set_xlim(0, omega.max())
ins_ax.set_yscale('linear')
ins_ax.tick_params(direction='in', which='both', labelsize=6)
ins_ax.spines['right'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax.patch.set_alpha(0)

ff_ax.text(0.015, 0.875, '(b)', transform=ff_ax.transAxes, fontsize=10)

# Transfer matrices
Kmax = abs(K[:, :16, :16]).max()
Kmin = -Kmax
if colorscale == 'log':
    linthresh = np.abs(K).mean()/10
    norm = colors.AsinhNorm(linear_width=linthresh, vmin=Kmin, vmax=Kmax, base=10)
elif colorscale == 'linear':
    norm = colors.Normalize(vmin=Kmin, vmax=Kmax)

imshow_kw = {}
imshow_kw.setdefault('origin', 'upper')
imshow_kw.setdefault('interpolation', 'nearest')
imshow_kw.setdefault('cmap', plt.get_cmap('RdBu'))
imshow_kw.setdefault('norm', norm)
basis_labels = np.array([
    ''.join(tup) for tup in product(['I', 'X', 'Y', 'Z'], repeat=2)
])

tm_axes = [fig.add_subplot(gs[2:, 3*i:3*(i+1)]) for i in range(3)]
subfigs = ['(c)', '(d)', '(e)']
for i, (n_oper_identifier, tm_ax) in enumerate(zip(identifiers, tm_axes)):
    if i == 2:
        idx = [np.ravel_multi_index((j, i), (4, 4))
               for i in range(4) for j in range(4)]
    else:
        idx = [np.ravel_multi_index((i, j), (4, 4))
               for i in range(4) for j in range(4)]
    im = tm_ax.imshow(K[i][..., idx][idx, ...], origin='upper', interpolation='nearest', norm=norm,
                      cmap=DIVERGING_CMAP, rasterized=True)
    tm_ax.text(0.01, 1.05, subfigs[i] + ' ' + n_oper_identifier,
               transform=tm_ax.transAxes, fontsize=10)
    tm_ax.set_xticks(np.arange(16))
    tm_ax.set_yticks(np.arange(16))
    tm_ax.set_xticklabels(basis_labels[idx], rotation='vertical', fontsize=8)
    tm_ax.set_yticklabels(basis_labels[idx], fontsize=8)
    tm_ax.spines['left'].set_visible(False)
    tm_ax.spines['right'].set_visible(False)
    tm_ax.spines['top'].set_visible(False)
    tm_ax.spines['bottom'].set_visible(False)

gs.tight_layout(fig, h_pad=0., w_pad=0., pad=0.)

# cb_ax = fig.add_axes([1, 0.105, 0.015, 0.375])
cb_ax = fig.add_axes([1, 0.105, 0.015, 0.405])
# divider = make_axes_locatable(tm_axes)
# cb_ax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(im, cax=cb_ax)
cb.set_label(r"Cumulant function $\mathcal{K}_{\epsilon_{ij}}(\tau)$")

fig.savefig(SAVE_PATH / f'CNOT_FF.pdf')
# %% filter function unitary vs complete
key = 'CNOT'

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True,
                         figsize=(figsize_wide[0], figsize_wide[1]/1.75))
omega = np.geomspace(1/T[key], 4, n)

# Excluding identity element
cache_filter_function(pulses_complete[key], omega,
                      np.diag([0, *[1]*15, *[0]*20]))

*_, leg = plotting.plot_filter_function(
    pulses_complete[key], omega, n_oper_identifiers=identifiers[:3],
    yscale='log', omega_in_units_of_tau=False, axes=axes[0], fig=fig,
    cycler=cycle
)
leg.remove()
# ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# ax.set_ylim(1e-15, ax.get_ylim()[1])
axes[0].grid(False)
axes[0].set_xlabel(axes[0].get_xlabel() + r' (ns$^{-1}$)')
axes[0].set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
axes[0].tick_params(direction='out', which='both', labelsize=8)
axes[0].set_ylim(bottom=1e-2, top=1e2)
# axes[0].set_title(r'Excluding $\mathrm{diag}(1,1,1,1,0,0)$')
axes[0].text(0.90, 0.90, '(a)', transform=axes[0].transAxes, fontsize=10)


# Including identity element
cache_filter_function(pulses_complete[key], omega,
                      np.diag([1, *[1]*15, *[0]*20]))

*_, leg = plotting.plot_filter_function(
    pulses_complete[key], omega, n_oper_identifiers=identifiers[:3],
    yscale='log', omega_in_units_of_tau=False, axes=axes[1], fig=fig,
    cycler=cycle
)
# ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# ax.set_ylim(1e-15, ax.get_ylim()[1])
axes[1].legend(identifiers[:3], frameon=True, loc='lower left', framealpha=1)
axes[1].grid(False)
axes[1].set_xlabel(axes[1].get_xlabel() + r' (ns$^{-1}$)')
axes[1].set_ylabel('')
# axes[1].set_title(r'Including $\mathrm{diag}(1,1,1,1,0,0)$')
# ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
# axes[1].tick_params(direction='out', which='both', labelsize=8)
# axes[1].set_ylim(bottom=1e-9, top=1e3)
axes[1].text(0.90, 0.90, '(b)', transform=axes[1].transAxes, fontsize=10)
axes[1].legend(handlelength=1.95, frameon=False)

fig.tight_layout()
fname = f'filter_function-{key}_unitary_v_complete'
for ext in ('pdf', 'eps', 'png'):
    if (not (file := save_path / '.'.join([fname, ext])).exists()
            or _force_overwrite):
        fig.savefig(file)

# %% fidelities
# repo = git.Repo('Z:/Code/filter_functions')
repo = git.Repo('~/Code/filter_functions')
sha = repo.head.object.hexsha

fname = 'fidelities_commit-{}.txt'.format(sha[:7])

PI_c_liouville_ggm = ff.superoperator.liouville_representation(PI_c_op,
                                                               ff.Basis.ggm(6))

with open(save_path / fname, 'w+') as f:
    infidelities = np.empty((2, 3, 6))
    f.write(f'Git commit hash: {sha}\n\n')
    f.write('==========================================================\n')
    f.write('calculated using ff.infidelity (subspace)\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_subspace.items()):
            pulse.d = 4
            omega = np.geomspace(1e-2/T[key], 1e2, n)
            S = A[i]/omega**a
            infidelities[i, j] = ff.infidelity(pulse, S, omega)
            f.write(f'{key}\t{a}\t\t{infidelities[i, j, :3].sum():.3e}\t\t\t' +
                    f'{infidelities[i, j, :3].sum()*4/5:.3e}\n')
            pulse.d = 6
    f.write('==========================================================\n')
    f.write('calculated using ff.error_transfer_matrix, separated basis\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            infid = 1 - np.einsum('...ii',
                                  P_complete[key][a][:16, :16]).sum().real/d_c**2
            f.write(f'{key}\t{a}\t\t{infid:.3e}\t\t\t{infid*4/5:.3e}\n')
    f.write('==========================================================\n')
    f.write('calculated using cumulant function, excl. identity\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            infid = np.einsum('...ii',
                              -K_complete[key][a][:3, 1:16, 1:16]).sum().real/d_c**2
            f.write(f'{key}\t{a}\t\t{infid:.3e}\t\t\t{infid*4/5:.3e}\n')
    f.write('==========================================================\n')
    f.write('calculated using cumulant function, separated basis\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            infid = np.einsum('...ii',
                              -K_complete[key][a][:3, :16, :16]).sum().real/d_c**2
            f.write(f'{key}\t{a}\t\t{infid:.3e}\t\t\t{infid*4/5:.3e}\n')
    f.write('==========================================================\n')
    f.write('calculated using ff.error_transfer_matrix, excl. identity\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            infid = 1 - np.einsum('...ii',
                                  P_complete[key][a][1:16, 1:16]).sum().real/d_c**2
            f.write(f'{key}\t{a}\t\t{infid:.3e}\t\t\t{infid*4/5:.3e}\n')
    f.write('==========================================================\n')
    f.write('MC\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            F_e = infid_fast[key][i]
            f.write(f'{key}\t{a}\t\t{F_e:.3e}\t\t\t{F_e*4/5:.3e}\n')
    f.write('==========================================================')

# %% leakage
repo = git.Repo('~/Code/filter_functions')
# repo = git.Repo('Z:/Code/filter_functions')
sha = repo.head.object.hexsha

fname = 'leakage_commit-{}.txt'.format(sha[:7])
pulse = pulses_complete['CNOT']

d_c = 4
d_l = 2
PI_c_op = np.zeros((6, 6))
PI_l_op = np.zeros((6, 6))
PI_c_op[range(1, 5), range(1, 5)] = 1
PI_l_op[(0, 5), (0, 5)] = 1
PI_c_vec = ff.basis.expand(PI_c_op, pulse.basis)
PI_l_vec = ff.basis.expand(PI_l_op, pulse.basis)


def leakage(E):
    return (PI_l_vec.T @ E @ PI_c_vec/d_c).real


def seepage(E):
    return (1 - PI_l_vec.T @ E @ PI_l_vec/d_l).real


with open(save_path / fname, 'w+') as f:
    infidelities = np.empty((2, 3, 6))
    f.write(f'Git commit hash: {sha}\n\n')
    f.write('====================================================\n')
    f.write('                   CNOT (eps_23)\n')
    f.write('====================================================\n')
    f.write('----------------------------------------------------\n')
    f.write('\t\tSystematic (L(Q))\n')
    f.write('alpha\tLeakage\t\t\tSeepage\n')
    f.write('----------------------------------------------------\n')
    for i, a in enumerate(alpha):
        Q_liouville = pulse.total_propagator_liouville.real
        Uerr_liouville = np.eye((d_c + d_l)**2) + K_complete['CNOT'][a][1]
        L_c = leakage(Q_liouville)
        L_l = seepage(Q_liouville)
        f.write(f'{a}\t\t{L_c:.3e}\t\t{L_l:.3e}\n')

    f.write('----------------------------------------------------\n')
    f.write('\t\tDecoherence (L(Uerr))\n')
    f.write('alpha\tLeakage\t\t\tSeepage\n')
    f.write('----------------------------------------------------\n')
    for i, a in enumerate(alpha):
        Q_liouville = pulse.total_propagator_liouville.real
        Uerr_liouville = np.eye((d_c + d_l)**2) + K_complete['CNOT'][a][1]
        L_c = leakage(Uerr_liouville)
        L_l = seepage(Uerr_liouville)
        f.write(f'{a}\t\t{L_c:.3e}\t\t{L_l:.3e}\n')

    f.write('----------------------------------------------------\n')
    f.write('\t\tBoth (L(Q @ Uerr))\n')
    f.write('alpha\tLeakage\t\t\tSeepage\n')
    f.write('----------------------------------------------------\n')
    for i, a in enumerate(alpha):
        Q_liouville = pulse.total_propagator_liouville.real
        Uerr_liouville = np.eye((d_c + d_l)**2) + K_complete['CNOT'][a][1]
        L_c = leakage(Q_liouville @ Uerr_liouville)
        L_l = seepage(Q_liouville @ Uerr_liouville)
        f.write(f'{a}\t\t{L_c:.3e}\t\t{L_l:.3e}\n')
    f.write('====================================================')

# %% Leakage Wood & Gambetta


def apply_channel(dmat, ops):
    if dmat.shape[0] == d:
        res = ops @ dmat @ ops.conj().swapaxes(-1, -2)
        return res.sum(0) if ops.ndim == 3 else res
    else:
        return ops @ dmat


def leakage_op(channel, PI_l, PI_c, d_c):
    return np.trace(PI_l @ apply_channel(PI_c, channel)).real/d_c


def leakage_vec(channel, PI_lvec, PI_cvec, d_c):
    return np.einsum('i,...i', PI_lvec, apply_channel(PI_cvec, channel)).real/d_c


d_c = 4
d_l = 2
d = d_c + d_l
# basis = ff.Basis.ggm(d)

d_c = 4
d_l = 2
PI_c_op = np.zeros((6, 6))
PI_l_op = np.zeros((6, 6))
PI_c_op[range(1, 5), range(1, 5)] = 1
PI_l_op[(0, 5), (0, 5)] = 1

# %%% basis transform
computational_basis_vec = np.eye(d**2)
computational_basis_c = ff.Basis(
    [np.reshape(qt.basis(d**2, i).full(), (d, d), order='C')
     for i in range(d**2)],
)
computational_basis_r = ff.Basis(
    [np.reshape(qt.basis(d**2, i).full(), (d, d), order='F')
     for i in range(d**2)],
)

# Convert from column stacking to complete basis and v.v.
T_sc_complete = np.einsum('ai,aj', computational_basis_vec,
                          ff.basis.expand(computational_basis_c, complete_basis))
# Convert from row stacking to complete basis and v.v.
T_sr_complete = np.einsum('ai,aj', computational_basis_vec,
                          ff.basis.expand(computational_basis_r, complete_basis))

# Convert from column stacking to GGM basis and v.v.
T_sc_ggm = np.einsum('ai,aj', computational_basis_vec,
                     ff.basis.expand(computational_basis_c, ff.Basis.ggm(d)))
# Convert from row stacking to GGM basis and v.v.
T_sr_ggm = np.einsum('ai,aj', computational_basis_vec,
                     ff.basis.expand(computational_basis_r, ff.Basis.ggm(d)))

# Convert from complete basis to GGM basis and v.v.
T_ssp = np.einsum('ai,aj', computational_basis_vec,
                  ff.basis.expand(complete_basis, ff.Basis.ggm(d)))

a = np.random.randn(6, 6) + np.random.randn(6, 6)*1j
a += a.conj().T

A_rvec = np.zeros((36), dtype=complex)
eye = np.eye(6)
for i in range(6):
    for j in range(6):
        A_rvec += a[i, j]*np.kron(eye[i], eye[j]).T

A_cvec = np.zeros((36), dtype=complex)
eye = np.eye(6)
for i in range(6):
    for j in range(6):
        A_cvec += a[i, j]*np.kron(eye[j], eye[i])

A_svec_complete = ff.basis.expand(a, complete_basis)
A_svec_ggm = ff.basis.expand(a, ff.Basis.ggm(d))

PI_cvec_complete = ff.basis.expand(PI_c_op, complete_basis)
PI_lvec_complete = ff.basis.expand(PI_l_op, complete_basis)
PI_cvec_ggm = ff.basis.expand(PI_c_op, ff.Basis.ggm(d))
PI_lvec_ggm = ff.basis.expand(PI_l_op, ff.Basis.ggm(d))

P_col = {key: {a: T_sc_ggm @ P @ T_sc_ggm.conj().T for a, P in p.items()}
         for key, p in K_ggm.items()}
P_row = {key: {a: T_sr_ggm @ P @ T_sr_ggm.conj().T for a, P in p.items()}
         for key, p in K_ggm.items()}

# %%% test correct transform of vectors
print('complete vs row:', np.allclose(T_sr_complete @ A_svec_complete, A_rvec))
print('complete vs col:', np.allclose(T_sc_complete @ A_svec_complete, A_cvec))
print('ggm vs row:', np.allclose(T_sr_ggm @ A_svec_ggm, A_rvec))
print('ggm vs col:', np.allclose(T_sc_ggm @ A_svec_ggm, A_cvec))
print('inverse complete vs row:', np.allclose(T_sr_complete.conj().T @ A_rvec, A_svec_complete))
print('inverse complete vs col:', np.allclose(T_sc_complete.conj().T @ A_cvec, A_svec_complete))
print('inverse ggm vs row:', np.allclose(T_sr_ggm.conj().T @ A_rvec, A_svec_ggm))
print('inverse ggm vs col:', np.allclose(T_sc_ggm.conj().T @ A_cvec, A_svec_ggm))

# %%% test correct transform of superoperators
S_s_complete = -K_complete['CNOT'][0.0].sum(0)
S_s_ggm = -K_ggm['CNOT'][0.0].sum(0)
S_c = -P_col['CNOT'][0.0].sum(0)
S_r = -P_row['CNOT'][0.0].sum(0)

print('superoperator ggm to complete:', np.allclose(T_ssp@S_s_ggm@T_ssp.conj().T, S_s_complete))
print('superoperator complete to ggm:', np.allclose(T_ssp.conj().T@S_s_complete@T_ssp, S_s_ggm))

# %%% leakage
for a in (0.0, 0.7):
    L = leakage_vec(np.eye(d**2) + K_complete['CNOT'][a], PI_lvec_complete,
                    PI_cvec_complete, d_c)
    print(f'leakage complete\t {a}:', L[:3])
    L = leakage_vec(np.eye(d**2) + K_ggm['CNOT'][a], PI_lvec_ggm,
                    PI_cvec_ggm, d_c)
    print(f'leakage ggm\t\t {a}:', L[:3])

# %%% process fidelity
for key in pulses_ggm.keys():
    for a in (0.0, 0.7):
        i = np.einsum('...ii', -K_complete[key][a][:3, :16, :16])/d_c**2
        print(f'fidelity complete \t {key} {a}:', i.real)
        i = np.einsum('ij,...ji', np.kron(PI_c_op, PI_c_op), -P_col[key][a][:3])/d_c**2
        print(f'fidelity col\t\t {key} {a}:', i.real)
        i = np.einsum('ij,...ji', np.kron(PI_c_op, PI_c_op), -P_row[key][a][:3])/d_c**2
        print(f'fidelity row\t\t {key} {a}:', i.real)
        i = np.einsum('ij,...ji',
                      ff.superoperator.liouville_representation(PI_c_op, complete_basis),
                      -K_complete[key][a][:3])/d_c**2
        print(f'fidelity complete proj.\t {key} {a}:', i.real)
        i = np.einsum('ij,...ji',
                      ff.superoperator.liouville_representation(PI_c_op, ff.Basis.ggm(d)),
                      -K_ggm[key][a][:3])/d_c**2
        print(f'fidelity ggm proj.\t {key} {a}:', i.real)

# %%% average gate fidelity without leakage
for key in pulses_complete.keys():
    for a in (0.0, 0.7):
        l = leakage_vec(np.eye(d**2) + K_complete['CNOT'][a][:3],
                        PI_lvec_complete, PI_cvec_complete, d_c)
        i = np.einsum('...ii', -K_complete[key][a][:3, :16, :16])/d_c**2
        f = (d_c*(1 - i.real) + 1 - l.real)/(d_c + 1)
        print(f'avg gate fidelity complete \t {key} {a}:', 1 - f)
        i = np.einsum('ij,...ji', np.kron(PI_c_op, PI_c_op), -P_col[key][a][:3])/d_c**2
        f = (d_c*(1 - i.real) + 1 - l.real)/(d_c + 1)
        print(f'avg gate fidelity col\t\t {key} {a}:', 1 - f)

# %%% save average gate fidelity without leakage
# repo = git.Repo('Z:/Code/filter_functions')
sha = repo.head.object.hexsha

fname = 'fidelity-leakage_commit-{}.txt'.format(sha[:7])
pulse = pulses_complete['CNOT']

with open(save_path / fname, 'w+') as f:
    infidelities = np.empty((2, 3, 6))
    f.write(f'Git commit hash: {sha}\n\n')
    f.write('============================================================\n')
    f.write('Only eps contributions\n')
    f.write('============================================================\n\n')

    f.write('============================================================\n')
    f.write('calculated using ff.error_transfer_matrix separated basis\n')
    f.write('============================================================\n')
    f.write('Gate\talpha\tF_ent\t\tF_avg\t\tF_avg sans L\n')
    for k, a in enumerate(alpha):
        f.write('------------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            l = leakage_vec(np.eye(d**2) + K_complete[key][a][:3],
                            PI_lvec_complete, PI_cvec_complete, d_c).real
            proj = ff.superoperator.liouville_representation
            i = np.einsum('...ii', -K_complete[key][a][:3, :16, :16]).real/d_c**2
            iav = 1 - (d_c*(1 - i) + 1 - l)/(d_c + 1)
            f.write(f'{key}\t{a}\t\t{i.sum():.3e}\t{i.sum()*4/5:.3e}\t{iav.sum():.3e}\n')
    f.write('============================================================\n')
    f.write('calculated using ff.error_transfer_matrix projected \n')
    f.write('============================================================\n')
    f.write('Gate\talpha\tF_ent\t\tF_avg\t\tF_avg sans L\n')
    for k, a in enumerate(alpha):
        f.write('------------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            l = leakage_vec(np.eye(d**2) + K_complete[key][a][:3],
                            PI_lvec_complete, PI_cvec_complete, d_c).real
            infid = np.einsum('ij,...ji', PI_c_liouville_ggm,
                              -K_ggm[key][a][:3]).real/d_c**2
            iav = 1 - (d_c*(1 - infid) + 1 - l)/(d_c + 1)
            f.write(f'{key}\t{a}\t\t{infid.sum():.3e}\t{infid.sum()*4/5:.3e}\t{iav.sum():.3e}\n')
    f.write('============================================================\n')
    f.write('\t\t\tMC\n')
    f.write('============================================================\n')
    f.write('Gate\talpha\tF_ent\t\tF_avg\t\tF_avg sans L\n')
    for i, a in enumerate(alpha):
        f.write('------------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            F_e = infid_fast[key][i]
            f.write(f'{key}\t{a}\t\t{F_e:.3e}\t{F_e*4/5:.3e}\t{F_e*4/5:.3e}\n')
    f.write('============================================================')
