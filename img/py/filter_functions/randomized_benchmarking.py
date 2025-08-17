# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:09:45 2018

@author: Tobias Hangleiter (tobias.hangleiter@rwth-aachen.de)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from filter_functions import util
from scipy import odr
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import cycler

from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition)

plt.style.use('publication')
# Hotfix latex preamble
for key in ('text.latex.preamble', 'pgf.preamble'):
    plt.rcParams.update({key: '\n'.join(plt.rcParams.get(key).split('|'))})

golden_ratio = (np.sqrt(5) - 1.)/2.
figsize_narrow = (3.40457, 3.40457*golden_ratio)
figsize_wide = (7.05826, 7.05826*golden_ratio)

ls_cycle = plt.rcParams['axes.prop_cycle'][:4] + cycler(linestyle=('-.','--',':','-',))
ms_cycle = plt.rcParams['axes.prop_cycle'][:4] + cycler(marker=('.', 'x', '+', 's'))
# %% Functions


def ent_fidelity(gates, target):
    """Calculate the entanglement fidelity"""
    d = gates.shape[-1]
    return util.abs2(np.einsum('...ll', gates @ target.conj().T)/d)


def avg_gate_fidelity(gates, target):
    """Calculate the average gate fidelity"""
    d = gates.shape[-1]
    return (d*ent_fidelity(gates, target) + 1)/(d + 1)


def state_fidelity(gates, psi: np.ndarray = None):
    """Calculate state fidelity for input state psi"""
    if psi is None:
        psi = np.c_[1:-1:-1]

    fidelity = util.abs2(psi.T @ gates @ psi).squeeze()
    return fidelity


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.

    https://stackoverflow.com/a/20528097
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpc.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def shift_zero_bwr_colormap(z: float, transparent: bool = True):
    """shifted bwr colormap"""
    if (z < 0) or (z > 1):
        raise ValueError('z must be between 0 and 1')

    cdict1 = {'red': ((0.0, max(-2*z + 1, 0), max(-2*z + 1, 0)),
                      (z,   1.0, 1.0),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, max(-2*z + 1, 0), max(-2*z + 1, 0)),
                        (z,   1.0, 1.0),
                        (1.0, max(2*z - 1, 0),  max(2*z - 1, 0))),

              'blue': ((0.0, 1.0, 1.0),
                       (z,   1.0, 1.0),
                       (1.0, max(2*z - 1, 0), max(2*z - 1, 0))),
              }
    if transparent:
        cdict1['alpha'] = ((0.0, 1-max(-2*z + 1, 0), 1-max(-2*z+1, 0)),
                           (z,   0.0, 0.0),
                           (1.0, 1-max(2*z - 1, 0),  1-max(2*z - 1, 0)))

    return mpc.LinearSegmentedColormap('shifted_rwb', cdict1)


# %% Load pickled files
"""current in paper
sha = '7fddd61'
date = '20200615-164906'
"""
sha = '6fac17f'  # thesis:'9754468'  # '3baa09e' 500 traces:'c0d179b'
date = '20200615-170254'  # thesis:'20190822-153458'  # '20190716-162957' 500 traces:'20191219-170932' 250 traces: '20191219-181356'
folder = 'RB_normalized_XYZ_noise_no_sensitivities'
m_min, m_max = 1, 101

# dpath = Path(r'Z:/MA/data')
dpath = Path(r'C:/Users/Tobias/Documents/Uni/Physik/Master/Thesis/data')
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
        with np.load(dpath / folder / sha / date /
                     f'RB_MC_gates_{gate_type}_gates_m{m_min}-{m_max}.npz') as arch:
            for file in arch.files:
                MC_gates[gate_type][file] = arch[file]

        data['MC'][gate_type]['tot']['white'] = \
            1 - state_fidelity(MC_gates[gate_type]['white'])

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
                        fit[calc_type][gate_type][infid_type][noise_type]['sep'][i] = \
                            ODR.run()
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

# %% Plot white vs correlated, gate type vs gate type

# gates = gate_types[:2]

# fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7, 3))
# # First row is white noise, second correlated.
# # First column is naive gates, second optimized
# for i, noise_type in enumerate(noise_types):
#     for j, gate_type in enumerate(gates):
#         N_l, N_G, n_nops = data['FF'][gate_type]['tot'][noise_type].shape

#         mean_tot = ax[i, j].errorbar(
#             m, 1 - data['FF'][gate_type]['tot'][noise_type].sum(-1).mean(-1),
#             (data['FF'][gate_type]['tot'][noise_type].sum(-1).std(-1) /
#              np.sqrt(N_G)),
#             fmt='.', color='tab:red'
#         )

#         fit_tot = ax[i, j].plot(
#             m, linear(fit['FF'][gate_type]['tot'][noise_type]['tot'].beta, m),
#             color='tab:red', linestyle='--', linewidth=1
#         )

#         mean_nocorr = ax[i, j].errorbar(
#             m, 1 - data['FF'][gate_type]['nocorr'][noise_type].sum(-1).mean(-1),
#             (data['FF'][gate_type]['nocorr'][noise_type].sum(-1).std(-1) /
#              np.sqrt(N_G)),
#             fmt='.', color='tab:green'
#         )

#         fit_nocorr = ax[i, j].plot(
#             m, linear(fit['FF'][gate_type]['nocorr'][noise_type]['tot'].beta, m),
#             color='tab:green', linestyle='--', linewidth=1
#         )

#         rb_theory = ax[i, j].plot(
#             m,
#             1 - data['single_clifford'][gate_type]['avg'][noise_type].sum(-1).mean()*m,
#             '-', zorder=4, color='tab:blue'
#         )

#         ax[i, j].grid(zorder=0)
#         ax[i, j].set_ylim(top=1)
#         ax[i, j].tick_params(direction='in', which='both')

#     ax[0, 0].set_xlim(0, 80)
#     # ax[0, 0].ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')

# for i in range(2):
#     ax[0, i].set_xlabel(gates[i])
#     ax[0, i].xaxis.set_label_position("top")
# for j in range(2):
#     ax[j, 1].set_ylabel(noise_types[j])
#     ax[j, 1].yaxis.set_label_position("right")

# # Add proxy subplot for common axis labels
# common_ax = fig.add_subplot(111, frameon=False)
# common_ax.tick_params(labelcolor='none', top=False, bottom=False,
#                       left=False, right=False)
# common_ax.set_xticks([])
# common_ax.set_yticks([])
# common_ax.grid(False)
# common_ax.set_xlabel(r'Sequence length $m$', labelpad=15.)
# common_ax.set_ylabel(r'Survival probability', labelpad=30.)

# handles = [mean_tot, mean_nocorr, fit_tot[0], fit_nocorr[0], rb_theory[0]]
# labels = ['Total state fidelity', 'State fidelity w/o corr.', 'Fit', 'Fit',
#           'SRB theory']

# # ax[0, 1].legend(handles, labels, loc='lower left', framealpha=1)
# common_ax.legend(handles, labels, bbox_to_anchor=(0, 1.1, 1, 0.2),
#                  loc="lower left", mode="expand", borderaxespad=0, ncol=5,
#                  fancybox=False, frameon=True)

# fig.tight_layout(w_pad=0.5, h_pad=0.5)
# fname = 'RB_{}_vs_{}_gates_white_vs_correl_{}_{}_{}'.format(*gates, folder,
#                                                             sha, date)
# fig.savefig(spath / (fname + '.pdf'))
# fig.savefig(spath / (fname + '.eps'),
#             transparent=False)

# %% Plot white vs correlated, gate type vs gate type

# gates = gate_types[:2]
# N_l, N_G, n_nops = data['FF']['naive']['tot']['white'].shape

# for k, identifier in zip(range(n_nops), ('X', 'Y', 'Z')):
#     fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7, 3))
#     # First row is white noise, second correlated.
#     # First column is naive gates, second optimized
#     for i, noise_type in enumerate(noise_types):
#         for j, gate_type in enumerate(gates):
#             N_l, N_G, n_nops = data['FF'][gate_type]['tot'][noise_type].shape

#             mean_tot = ax[i, j].errorbar(
#                 m, 1 - data['FF'][gate_type]['tot'][noise_type][..., k].mean(-1),
#                 (data['FF'][gate_type]['tot'][noise_type][..., k].std(-1) /
#                  np.sqrt(N_G)),
#                 fmt='.', color='tab:red'
#             )

#             fit_tot = ax[i, j].plot(
#                 m, linear(fit['FF'][gate_type]['tot'][noise_type]['sep'][k].beta, m),
#                 color='tab:red', linestyle='--', linewidth=1
#             )

#             mean_nocorr = ax[i, j].errorbar(
#                 m, 1 - data['FF'][gate_type]['nocorr'][noise_type][..., k].mean(-1),
#                 (data['FF'][gate_type]['nocorr'][noise_type][..., k].std(-1) /
#                  np.sqrt(N_G)),
#                 fmt='.', color='tab:green'
#             )

#             fit_nocorr = ax[i, j].plot(
#                 m, linear(fit['FF'][gate_type]['nocorr'][noise_type]['sep'][k].beta, m),
#                 color='tab:green', linestyle='--', linewidth=1
#             )

#             rb_theory = ax[i, j].plot(
#                 m,
#                 1 - data['single_clifford'][gate_type]['avg'][noise_type][..., k].mean()*m,
#                 '-', zorder=4, color='tab:blue'
#             )

#             ax[i, j].grid(zorder=0)
#             ax[i, j].set_ylim(top=1)
#             ax[i, j].tick_params(direction='in', which='both')

#         ax[0, 0].set_xlim(0, 80)
#         # ax[0, 0].ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')

#     for i in range(2):
#         ax[0, i].set_xlabel(gates[i])
#         ax[0, i].xaxis.set_label_position("top")
#     for j in range(2):
#         ax[j, 1].set_ylabel(noise_types[j])
#         ax[j, 1].yaxis.set_label_position("right")

#     # Add proxy subplot for common axis labels
#     common_ax = fig.add_subplot(111, frameon=False)
#     common_ax.tick_params(labelcolor='none', top=False, bottom=False,
#                           left=False, right=False)
#     common_ax.set_xticks([])
#     common_ax.set_yticks([])
#     common_ax.grid(False)
#     common_ax.set_xlabel(r'Sequence length $m$', labelpad=15.)
#     common_ax.set_ylabel(r'Survival probability', labelpad=30.)

#     handles = [mean_tot, mean_nocorr, fit_tot[0], fit_nocorr[0], rb_theory[0]]
#     labels = ['Total state fidelity', 'State fidelity w/o corr.', 'Fit', 'Fit',
#               'SRB theory']

#     # ax[0, 1].legend(handles, labels, loc='lower left', framealpha=1)
#     common_ax.legend(handles, labels, bbox_to_anchor=(0, 1.1, 1, 0.2),
#                      loc="lower left", mode="expand", borderaxespad=0, ncol=5,
#                      fancybox=False, frameon=True)

#     fig.tight_layout(w_pad=0.5, h_pad=0.5)
#     fname = 'RB_{}_vs_{}_gates_white_vs_correl_{}_{}_{}_{}-noise'.format(
#         *gates, folder, sha, date, identifier
#     )
#     fig.savefig(spath / (fname + '.pdf'))
#     fig.savefig(spath / (fname + '.eps'),
#                 transparent=False)

# # %% Correlation infidelities
# infids = {}
# for gate_type in gate_types:
#     infids[gate_type] = {}
#     with np.load(dpath / folder / sha / date /
#                  f'RB-correl_infids_{gate_type}_gates_m30.npz') as f:
#         for file in f:
#             infids[gate_type][file] = f[file]
#
#
# K, m, m, n_nops = infids[gate_type][file].shape
# m -= 1
#
# gates = gate_types[:2]
# alpha = (0.0, 0.7)
# means = np.empty((2, 2, m, m, n_nops))
# errs = np.empty((2, 2, m, m, n_nops))
# for i, (a, n) in enumerate(zip(alpha, noise_types)):
#     for j, g in enumerate(gates):
#         means[i, j] = infids[g][n][:, :-1, :-1].mean(axis=0)
#         errs[i, j] = infids[g][n][:, :-1, :-1].std(axis=0)/np.sqrt(K)
#
# # %% imagegrid plot
# mask = ~np.eye(m, dtype=bool)
# vmax = np.abs(means[:, :, mask].sum(-1).max())
# # vmin = means[:, :, mask].sum(-1).min()
# vmin = -vmax
# # cmap = shift_zero_bwr_colormap(1 + vmin/vmax)  # naive vs optimized
# # cmap = shift_zero_bwr_colormap(.580)  # optimized vs zyz
# cmap = plt.get_cmap('bwr')
# cmap.set_over((1/256, 0, 0, 1))
#
# fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4.8, 4))
# for i, (a, n) in enumerate(zip(alpha, noise_types)):
#     for j, g in enumerate(gates):
#         ax[i, j].set_aspect('equal', 'box')
#         mean = means[i, j].sum(-1)
#
#         pcm = ax[i, j].pcolor(
#             mean, norm=mpc.SymLogNorm(linthresh=1e-7, vmin=vmin, vmax=vmax),
#             cmap=cmap
#         )
#
# for i in range(2):
#     ax[0, i].set_xlabel(gates[i])
#     ax[0, i].xaxis.set_label_position("top")
# for j in range(2):
#     ax[j, 1].set_ylabel(noise_types[j])
#     ax[j, 1].yaxis.set_label_position("right")
#
# ax[0, 0].set_xticks(np.linspace(0, m, 6))
# ax[0, 1].set_xticks(np.linspace(0, m, 6))
# ax[0, 0].set_yticks(np.linspace(0, m, 6))
# ax[1, 0].set_yticks(np.linspace(0, m, 6))
# ax[1, 0].set_xlabel(r"$g$")
# ax[1, 1].set_xlabel(r"$g$")
# ax[0, 0].set_ylabel(r"$g'$")
# ax[1, 0].set_ylabel(r"$g'$")
#
# fig.tight_layout(h_pad=0, w_pad=0)
#
# cb = fig.colorbar(pcm, ax=ax.ravel().tolist(), fraction=0.045, pad=0.1)
# cb.set_label(r"$\mathcal{I}^{(gg')}$")
#
# fname = 'correlation_infids_{}_vs_{}_gates_white_vs_correl_one_cbar'.format(
#     *gates,
# )
# fig.savefig(spath / (fname + '.pdf'))
# fig.savefig(spath / (fname + '.eps'))
#
# # %% imagegrid plots each noise operator separately
# for k, identifier in zip(range(n_nops), ('X', 'Y', 'Z')):
#     mask = ~np.eye(m, dtype=bool)
#     vmax = np.abs(means[:, :, mask, k].max())
#     # vmin = means[:, :, mask, k].min()
#     vmin = -vmax
#     # cmap = shift_zero_bwr_colormap(.375)  # naive vs optimized
#     # cmap = shift_zero_bwr_colormap(.580)  # optimized vs zyz
#     cmap = plt.get_cmap('bwr')
#     cmap.set_over((1/256, 0, 0, 1))
#
#     fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4.8, 4))
#     for i, (a, n) in enumerate(zip(alpha, noise_types)):
#         for j, g in enumerate(gates):
#             ax[i, j].set_aspect('equal', 'box')
#             mean = means[i, j, ..., k]
#
#             pcm = ax[i, j].pcolor(
#                 mean, norm=mpc.SymLogNorm(linthresh=1e-7, vmin=vmin, vmax=vmax),
#                 cmap=cmap
#             )
#
#     for i in range(2):
#         ax[0, i].set_xlabel(gates[i])
#         ax[0, i].xaxis.set_label_position("top")
#     for j in range(2):
#         ax[j, 1].set_ylabel(noise_types[j])
#         ax[j, 1].yaxis.set_label_position("right")
#
#     ax[0, 0].set_xticks(np.linspace(0, m, 6))
#     ax[0, 1].set_xticks(np.linspace(0, m, 6))
#     ax[0, 0].set_yticks(np.linspace(0, m, 6))
#     ax[1, 0].set_yticks(np.linspace(0, m, 6))
#     ax[1, 0].set_xlabel(r"$g$")
#     ax[1, 1].set_xlabel(r"$g$")
#     ax[0, 0].set_ylabel(r"$g'$")
#     ax[1, 0].set_ylabel(r"$g'$")
#
#     fig.tight_layout(h_pad=0, w_pad=0)
#
#     cb = fig.colorbar(pcm, ax=ax.ravel().tolist(), fraction=0.045, pad=0.1)
#     cb.set_label(r"$\mathcal{I}^{(gg')}$")
#
#     fname = 'correlation_infids_{}_vs_{}_gates_white_vs_correl_one_cbar_{}-noise'.format(
#         *gates, identifier
#     )
#     fig.savefig(spath / (fname + '.pdf'))
#     fig.savefig(spath / (fname + '.eps'))
#
# # %% linecuts
# rng = np.arange(m)
# fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
# color = ('tab:green', 'tab:orange')
# for i, (a, n) in enumerate(zip(alpha, noise_types)):
#     for j, g in enumerate(gates):
#         markers, caps, bars = ax[i].errorbar(
#             rng, np.diag(means[i, j].sum(-1)[:, ::-1]*1e6, 0),
#             yerr=np.diag(errs[i, j].sum(-1)[:, ::-1]*1e6, 0),
#             fmt='.-', linewidth=1, label=f'{g}',
#             color=color[j]
#         )
#         [bar.set_alpha(0.5) for bar in bars]

#         axy = ax[i].twinx()
#         axy.set_ylabel(f'{n}', fontsize=12)
#         axy.set_yticks([], [])
#         axy.yaxis.set_label_position("right")
#         ax[i].set_ylabel(r"$\mathcal{I}^{(gg')}\times 10^6$")
#         ax[i].grid(True)

# ax[1].legend()
# ax[1].set_xticklabels
# ax[1].set_xlim(rng.min(), rng.max()+1)
# ax[1].set_xlabel(r"$g$")
# axx = ax[0].twiny()
# axx.set_xlim(rng.max(), rng.min()-1)
# axx.set_xticks(ax[0].get_xticks()-1)
# axx.set_xlabel(r"$g'$")

# fig.tight_layout()
# fname = 'correlation_infids_{}_vs_{}_gates_white_vs_correl_linecuts'.format(
#     *gates,
# )
# fig.savefig(spath / (fname + '.pdf'))
# fig.savefig(spath / (fname + '.eps'))

# %% linecuts
# for k, identifier in zip(range(n_nops), ('X', 'Y', 'Z')):
#     rng = np.arange(m)
#     fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
#     color = ('tab:green', 'tab:orange')
#     for i, (a, n) in enumerate(zip(alpha, noise_types)):
#         for j, g in enumerate(gates):
#             markers, caps, bars = ax[i].errorbar(
#                 rng, np.diag(means[i, j, ..., k][:, ::-1]*1e6, 0),
#                 yerr=np.diag(errs[i, j, ..., k][:, ::-1]*1e6, 0),
#                 fmt='.-', linewidth=1, label=f'{g}',
#                 color=color[j]
#             )
#             [bar.set_alpha(0.5) for bar in bars]

#             axy = ax[i].twinx()
#             axy.set_ylabel(f'{n}', fontsize=12)
#             axy.set_yticks([], [])
#             axy.yaxis.set_label_position("right")
#             ax[i].set_ylabel(r"$\mathcal{I}^{(gg')}\times 10^6$")
#             ax[i].grid(True)

#     ax[1].legend()
#     ax[1].set_xticklabels
#     ax[1].set_xlim(rng.min(), rng.max()+1)
#     ax[1].set_xlabel(r"$g$")
#     axx = ax[0].twiny()
#     axx.set_xlim(rng.max(), rng.min()-1)
#     axx.set_xticks(ax[0].get_xticks()-1)
#     axx.set_xlabel(r"$g'$")

#     fig.tight_layout()
#     fname = 'correlation_infids_{}_vs_{}_gates_white_vs_correl_linecuts_{}-noise'.format(
#         *gates, identifier
#     )
#     fig.savefig(spath / (fname + '.pdf'))
#     fig.savefig(spath / (fname + '.eps'))

# # %% all gate types in one
# colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')
#
# fig, axes = plt.subplots(2, 1, sharex=True, sharey=True,
#                          figsize=(3.4, 3.4))
#
# m = np.round(np.linspace(m_min, m_max, n_m)).astype(int)
# for ax, noise_type, subfig in zip(axes, noise_types, ('(a)', '(b)')):
#     for g, gate_type in enumerate(gate_types):
#         N_l, N_G, n_nops = data['FF'][gate_type]['tot'][noise_type].shape
#         mean_tot = ax.errorbar(
#             m, 1 - data['FF'][gate_type]['tot'][noise_type].sum(-1).mean(-1),
#             (data['FF'][gate_type]['tot'][noise_type].sum(-1).std(-1) /
#              np.sqrt(N_G)),
#             fmt='.', color=colors[g]
#         )
#
#         fit_tot = ax.plot(
#             m, linear(fit['FF'][gate_type]['tot'][noise_type]['tot'].beta, m),
#             color=colors[g], linestyle='--', linewidth=1
#         )
#
#     rb_theory = ax.plot(
#         m,
#         1 - data['single_clifford'][gate_type]['avg'][noise_type].sum(-1).mean()*m,
#         '-', zorder=4, color='k'
#     )
#     ax.grid(False)
#     ax.text(0.85, 0.8, subfig, transform=ax.transAxes)
#     ax.tick_params(top=True, bottom=True, left=True, right=True,
#                    direction='in')
#     # ax.set_ylabel(r'Survival probability')
#
# # Add proxy subplot for common axis labels
# common_ax = fig.add_subplot(111, frameon=False)
# common_ax.tick_params(labelcolor='none', top=False, bottom=False,
#                       left=False, right=False)
# common_ax.set_xticks([])
# common_ax.set_yticks([])
# common_ax.grid(False)
# # common_ax.set_xlabel(r'Sequence length $m$', labelpad=15.)
# common_ax.set_ylabel(r'Survival probability', labelpad=25.)
# axes[0].legend(loc='lower left', frameon=False,
#                labels=gate_types + ['0th order SRB theory'])
# ax.set_xlim(0, 100)
# ax.set_ylim(ymax=1)
# ax.set_xlabel(r'Sequence length $m$')
#
# fig.tight_layout(h_pad=0, w_pad=0)
# fname = 'RB_all_gates_white_vs_correl'
# fig.savefig(spath / (fname + '.pdf'))
# fig.savefig(spath / (fname + '.eps'))
#
# # %% all gate types in one
# colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')

# for k, identifier in zip(range(n_nops), ('X', 'Y', 'Z')):
#     fig, axes = plt.subplots(2, 1, sharex=True, sharey=True,
#                              figsize=(3.4, 3.4))

#     m = np.round(np.linspace(m_min, m_max, n_m)).astype(int)
#     for ax, noise_type, subfig in zip(axes, noise_types, ('(a)', '(b)')):
#         for g, gate_type in enumerate(gate_types):
#             N_l, N_G, n_nops = data['FF'][gate_type]['tot'][noise_type].shape
#             mean_tot = ax.errorbar(
#                 m, 1 - data['FF'][gate_type]['tot'][noise_type][..., k].mean(-1),
#                 (data['FF'][gate_type]['tot'][noise_type][..., k].std(-1) /
#                  np.sqrt(N_G)),
#                 fmt='.', color=colors[g]
#             )

#             fit_tot = ax.plot(
#                 m, linear(fit['FF'][gate_type]['tot'][noise_type]['sep'][k].beta, m),
#                 color=colors[g], linestyle='--', linewidth=1
#             )

#         rb_theory = ax.plot(
#             m,
#             1 - data['single_clifford'][gate_type]['avg'][noise_type][..., k].mean()*m,
#             '-', zorder=4, color='k'
#         )
#         ax.grid(False)
#         ax.text(0.85, 0.8, subfig, transform=ax.transAxes)
#         ax.tick_params(top=True, bottom=True, left=True, right=True,
#                        direction='in')
#         # ax.set_ylabel(r'Survival probability')

#     # Add proxy subplot for common axis labels
#     common_ax = fig.add_subplot(111, frameon=False)
#     common_ax.tick_params(labelcolor='none', top=False, bottom=False,
#                           left=False, right=False)
#     common_ax.set_xticks([])
#     common_ax.set_yticks([])
#     common_ax.grid(False)
#     # common_ax.set_xlabel(r'Sequence length $m$', labelpad=15.)
#     common_ax.set_ylabel(r'Survival probability', labelpad=30.)
#     axes[0].legend(loc='lower left', frameon=False,
#                    labels=gate_types + ['0th order SRB theory'])
#     ax.set_xlim(0, 100)
#     ax.set_ylim(ymax=1)
#     ax.set_xlabel(r'Sequence length $m$')

#     fig.tight_layout()
#     fname = 'RB_all_gates_white_vs_correl_{}-noise'.format(identifier)
#     fig.savefig(spath / (fname + '.pdf'))
#     fig.savefig(spath / (fname + '.eps'))

# # %% all gate types in one, individual noise operator contributions
# colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')
#
# fig, axes = plt.subplots(2, 2, sharex=True, sharey=False,
#                          figsize=figsize_wide)
#
# m = np.round(np.linspace(m_min, m_max, n_m)).astype(int)
# noise_type = 'correlated'
# for k, (ax, subfig, identifier) in enumerate(zip(axes.ravel(),
#                                                  ('(a)', '(b)', '(c)'),
#                                                  ('X', 'Y', 'Z'))):
#     for g, gate_type in enumerate(gate_types):
#         N_l, N_G, n_nops = data['FF'][gate_type]['tot'][noise_type].shape
#         mean_tot = ax.errorbar(
#             m, 1 - data['FF'][gate_type]['tot'][noise_type][..., k].mean(-1),
#             (data['FF'][gate_type]['tot'][noise_type][..., k].std(-1) /
#              np.sqrt(N_G)),
#             fmt='.', color=colors[g]
#         )
#
#         fit_tot = ax.plot(
#             m, linear(fit['FF'][gate_type]['tot'][noise_type]['sep'][k].beta, m),
#             color=colors[g], linestyle='--', linewidth=1
#         )
#
#     rb_theory = ax.plot(
#         m,
#         1 - data['single_clifford'][gate_type]['avg'][noise_type][..., k].mean()*m,
#         '-', zorder=4, color='k'
#     )
#     ax.grid(False)
#     ax.text(0.90, 0.85, subfig, transform=ax.transAxes)
#     ax.tick_params(top=True, bottom=True, left=True, right=True,
#                    direction='in')
#     ax.set_title(f'{identifier}-noise')
#
# ax = axes.ravel()[3]
# subfig = '(d)'
# for g, gate_type in enumerate(gate_types):
#     N_l, N_G, n_nops = data['FF'][gate_type]['tot'][noise_type].shape
#     mean_tot = ax.errorbar(
#         m, 1 - data['FF'][gate_type]['tot'][noise_type].sum(-1).mean(-1),
#         (data['FF'][gate_type]['tot'][noise_type].sum(-1).std(-1) /
#          np.sqrt(N_G)),
#         fmt='.', color=colors[g]
#     )
#
#     fit_tot = ax.plot(
#         m, linear(fit['FF'][gate_type]['tot'][noise_type]['tot'].beta, m),
#         color=colors[g], linestyle='--', linewidth=1
#     )
#
# rb_theory = ax.plot(
#     m,
#     1 - data['single_clifford'][gate_type]['avg'][noise_type].sum(-1).mean()*m,
#     '-', zorder=4, color='k'
# )
# ax.grid(False)
# ax.text(0.90, 0.85, subfig, transform=ax.transAxes)
# ax.tick_params(top=True, bottom=True, left=True, right=True,
#                direction='in')
# ax.set_title(f'Sum of all noise')
#
# # Add proxy subplot for common axis labels
# common_ax = fig.add_subplot(111, frameon=False)
# common_ax.tick_params(labelcolor='none', top=False, bottom=False,
#                       left=False, right=False)
# common_ax.set_xticks([])
# common_ax.set_yticks([])
# common_ax.grid(False)
# common_ax.set_xlabel(r'Sequence length $m$', labelpad=15.)
# common_ax.set_ylabel(r'Survival probability', labelpad=30.)
# axes[0, 0].legend(loc='lower left', frameon=False,
#                   labels=gate_types + ['0th order SRB theory'])
# ax.set_xlim(0, 100)
# ax.set_ylim(ymax=1)
#
# fig.tight_layout()
# fname = 'RB_all_gates_noise_comparison'
# fig.savefig(spath / (fname + '.pdf'))
# fig.savefig(spath / (fname + '.eps'))
#
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
