# %% Imports
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import elementwise

from qutil import const, functools, math
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_50

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, TEXTWIDTH, PATH, init,  # noqa
                    E_AlGaAs, effective_mass)

SAVE_PATH = PATH / 'pdf/experiment'
SAVE_PATH.mkdir(exist_ok=True)

LINE_COLORS = [color for name, color in RWTH_COLORS.items() if name not in ('blue',)]
LINE_COLORS_50 = [color for name, color in RWTH_COLORS_50.items() if name not in ('blue',)]

init(MARGINSTYLE, backend := 'pgf')
# %% Functions


@functools.wraps(sc.special.airy)
def airy(*args, **kwargs):
    return sc.special.airy(*args, **kwargs)[0]


@functools.wraps(sc.special.ai_zeros)
def airy_zeros(*args, **kwargs):
    return sc.special.ai_zeros(*args, **kwargs)[0]


def eps_tilde(F, m):
    # Eq. (2.2) of Rabinovitch & Zak (1971)
    return np.float_power((const.e*const.hbar*F)**2/(2*m), 1/3, dtype=complex).real


def z_tilde(F, m):
    # Eq. (2.2) of Rabinovitch & Zak (1971)
    return eps_tilde(F, m) / (const.e*F)


def Z(z, eps, F, m):
    # Eq. (2.2) of Rabinovitch & Zak (1971)
    if np.isscalar(F):
        return -np.inf if F == 0 else (z*const.e*F - eps)/eps_tilde(F, m)

    with np.errstate(divide='ignore'):
        # z/z_tilde - eps/eps_tilde
        res = (z*const.e*F - eps)/eps_tilde(F, m)
    res[np.isnan(res)] = -np.inf
    return res


def eps_square(L, m, n: int):
    return (const.hbar * np.pi * np.arange(1, n+1) / L)**2 / (2 * m)


def eps_triangular(F, L, m):
    # Eq. (2.4) of Rabinovitch & Zak (1971)

    def fn(eps, FF):
        Z_0 = Z(0, eps, FF, m)
        Z_L = Z(L, eps, FF, m)

        Ai_0, Aip_0, Bi_0, Bip_0 = sc.special.airy(Z_0.real)
        Ai_L, Aip_L, Bi_L, Bip_L = sc.special.airy(Z_L.real)

        deps = Z_0/eps
        return (
            Ai_0*Bi_L - Ai_L*Bi_0,
            deps*(Ai_0*Bip_L + Aip_0*Bi_L - Ai_L*Bip_0 - Aip_L*Bi_0)
        )

    squeeze = np.isscalar(F)
    F = np.atleast_1d(F)
    eps = np.empty(F.size, dtype=complex)
    eps_sq = eps_square(L, m, 1).item()
    for i, FF in enumerate(F):
        if FF == 0:
            eps[i] = eps_sq
            continue
        elif i == 0:
            x0 = eps_sq
        else:
            x0 = eps[i-1]

        result = sc.optimize.root_scalar(fn, args=(FF,), fprime=True, x0=x0, xtol=1e-4*eps_sq)
        if not result.converged:
            print(result)
            raise RuntimeError("Not converged")
        eps[i] = result.root.item()
    return eps.real.item() if squeeze else eps.real


def eps_triangular_vec(F, L, m, n: int):
    # Vectorized Eq. (2.4) of Rabinovitch & Zak (1971)

    def fn(eps, FF):
        eps, FF = np.broadcast_arrays(eps, FF)
        val, grad = np.empty((2, *FF.shape), dtype=complex)
        mask = FF != 0

        Z_0 = Z(0, eps[mask], FF[mask], m)
        Z_L = Z(L, eps[mask], FF[mask], m)

        # Compute the Airy functions Ai and Bi for Zp and Zm
        Ai_0, _, Bi_0, _ = sc.special.airy(Z_0)
        Ai_L, _, Bi_L, _ = sc.special.airy(Z_L)

        val[mask] = Ai_0 * Bi_L - Ai_L * Bi_0
        val[~mask] = 0
        return val.real

    assert n > 0
    F = np.atleast_1d(F)
    eps = np.empty((n, *F.shape))
    eps[0] = eps_triangular(F, L, m).real
    eps_sq = eps_square(L, m, n)
    eps_tri = eps_triangular_large_field(F, m, n)

    a_n, *_ = sc.special.ai_zeros(n)

    mask = F != 0
    for i in range(1, n):
        xl0 = xmin = eps[i-1] * 1.01
        # this needs a bit of fine-tuning
        xmax = eps_sq[i] + eps_tri[i]

        res_bracket = elementwise.bracket_root(fn, xl0, xmin=xmin, xmax=xmax, args=(F,))
        res_root = elementwise.find_root(fn, res_bracket.bracket, args=(F,))

        if not res_root.success.all():
            raise RuntimeError("Not converged.\n", res_root, res_bracket)
        else:
            eps[i, mask] = res_root.x[mask]
            eps[i, ~mask] = eps_sq[i]

    return eps


def eps_triangular_large_field(F, m, n: int):
    # Eq (4.42), Davies 1998
    epst = eps_tilde(F, m).real
    return np.expand_dims(-airy_zeros(n), list(range(1, np.ndim(epst) + 1)))*epst


def psi_square(z, L, n: int):
    return np.sqrt(2/L)*np.sin(np.arange(1, n+1)[:, None]*np.pi*z/L)


def psi_triangular(z, F: float, L, m, n: int):
    if F == 0:
        return psi_square(z, L, n)

    eps = eps_triangular_vec(F, L, m, n)

    Z_ = Z(z, eps, F, m)
    Z_L = Z(L, eps, F, m)

    Ai, _, Bi, _ = sc.special.airy(Z_)
    Ai_L, _, Bi_L, _ = sc.special.airy(Z_L)

    res = Ai - Ai_L/Bi_L*Bi
    # Consistent phase
    res *= np.where(np.diff(res[..., :2]) < 0, -1, 1)
    return res


def psi_triangular_large_field(z, F, m, n: int):
    # airy(z/z_tilde - eps/eps_tilde)
    return airy(const.e*F*z/eps_tilde(F, m) + airy_zeros(n)[:, None]).real


def plot_states(ax, z, z0, F, E0, sgn, n, scale):
    E = sgn*(.5*E_g_GaAs - E0 + eps_triangular_vec(F, L, m[int(np.signbit(sgn))], n)/e)
    P = sgn*psi_triangular(z - z0, F, L, m[int(np.signbit(sgn))], n)[:, ::sgn]
    P /= np.sqrt(np.trapezoid(math.abs2(P), z - z0)[:, None])
    for i in range(n):
        ax.plot([95, 125], [E[i]]*2, ls='--', color=LINE_COLORS_50[i], alpha=0.66)
        ax.plot(z*1e9, E[i] + P[i]*scale, color=LINE_COLORS[i])


def sketch_style(fig, axs):
    fig.subplots_adjust(hspace=0.05)

    ax = axs[0]
    ax.set_xticks([10, 100, 120, 210])
    ax.set_yticks(np.around([.5*E_g_GaAs + ΔE_c, .5*E_g_GaAs], 2))
    ax.set_xlim(80, 140)
    ax.set_ylabel(r'$E - \mu$ (eV)', rotation='horizontal',  horizontalalignment='left')
    ax.yaxis.set_label_coords(-.2, 1.05)

    ax.tick_params(bottom=False, labelbottom=False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.right.set_visible(False)

    ax = axs[1]
    ax.set_yticks(np.around([-.5*E_g_GaAs, -.5*E_g_GaAs - ΔE_v], 2))
    ax.set_xlabel('$z$ (nm)', verticalalignment='top')
    ax.xaxis.set_label_coords(.9, -0.1)

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    # Axis break
    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=6,
                  linestyle="none", color='k', mec='k', mew=.75, clip_on=False)
    axs[0].plot([0], [0], transform=axs[0].transAxes, **kwargs)
    axs[1].plot([0], [1], transform=axs[1].transAxes, **kwargs)


# %% Parameters
E_g_GaAs = E_AlGaAs(0)
ΔE_g = E_AlGaAs(0.33) - E_g_GaAs
Q_e = 0.57
ΔE_c = Q_e * ΔE_g
ΔE_v = (1 - Q_e) * ΔE_g

n = 3
L = 20e-9
m = effective_mass()
e = const.e

# 10 MV/m = 100 kV/cm = 2 V/200nm
F = np.linspace(0, 10, 1001)*1e6
# %% Parameter sweeps
z = np.linspace(0, L, 101)
F = np.linspace(0, 25, 201)*1e6

with np.errstate(invalid='ignore'):
    eps = np.zeros((2, n, F.size))
    eps_a = np.zeros((2, n, F.size))
    for i in range(2):
        eps[i] = eps_triangular_vec(F, L, m[i, 0], n)
        eps_a[i] = eps_triangular_large_field(F, m[i, 0], n)

    psi = np.zeros((n, F.size, z.size))
    psi_a = np.zeros((n, F.size, z.size))
    for i in range(F.size):
        psi[:, i] = psi_triangular(z, F[i], L, m[0, 0], n)
        psi[:, i] /= np.sqrt(np.trapezoid(math.abs2(psi[:, i]), z))[:, None]
        psi_a[:, i] = psi_triangular_large_field(z, F[i], m[0, 0], n)
        psi_a[:, i] /= np.sqrt(np.trapezoid(math.abs2(psi_a[:, i]), z))[:, None]

# %%% Energy levels
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, layout='constrained')
for i in range(2):
    # subtract e*F*L/2 to obtain the quadratic shift
    # (corresponds to tilting about the middle of the well rather than the edge)
    axs[0, i].plot(F, (eps[i]/const.e - F*L/2).T)
    axs[1, i].plot(F, (eps_a[i]/const.e - F*L/2).T)
# %%% Wave functions
fig, axs = plt.subplots(2, n, sharex=True, sharey=True, layout='constrained')
for i in range(n):
    img = axs[0, i].pcolormesh(z, F, psi[i], cmap='RdBu', norm=mpl.colors.CenteredNorm(0))
    img = axs[1, i].pcolormesh(z, F, psi_a[i], cmap='RdBu', norm=mpl.colors.CenteredNorm(0))

# %% Sketch band structure
# Plot the first n levels
n = 2
z = np.array([-100, -L/2*1e9, L/2*1e9, 100]) + 110
z0 = np.mean(z)
zw = np.linspace(z[1], z[2], 101)
# %%% Plot untilted well
F = E0 = 0

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, axs = plt.subplots(2, sharex=True)

    # Conduction band
    ax = axs[0]
    ax.plot(z[0:2], np.array([.5*E_g_GaAs + ΔE_c]*2), color=RWTH_COLORS['blue'])
    ax.plot(z[[1, 1]], np.array([.5*E_g_GaAs + ΔE_c, .5*E_g_GaAs]), color=RWTH_COLORS['blue'])
    ax.plot(z[1:3], np.array([.5*E_g_GaAs]*2), color=RWTH_COLORS['blue'])
    ax.plot(z[[2, 2]], np.array([.5*E_g_GaAs, .5*E_g_GaAs + ΔE_c]), color=RWTH_COLORS['blue'])
    ax.plot(z[2:4], np.array([.5*E_g_GaAs + ΔE_c]*2), color=RWTH_COLORS['blue'])
    ax.annotate('$E_c$', (z[2] + 15, .5*E_g_GaAs + ΔE_c), verticalalignment='top')
    # Wave function
    plot_states(ax, zw*1e-9, z[1]*1e-9, F, E0, +1, n, scale=1e-5)

    ax.set_ylim(0.7, 1.05)

    # Valence band
    ax = axs[1]
    ax.plot(z[0:2], np.array([-.5*E_g_GaAs - ΔE_v]*2), color=RWTH_COLORS['blue'])
    ax.plot(z[[1, 1]], np.array([-.5*E_g_GaAs - ΔE_v, -.5*E_g_GaAs]), color=RWTH_COLORS['blue'])
    ax.plot(z[1:3], np.array([-.5*E_g_GaAs]*2), color=RWTH_COLORS['blue'])
    ax.plot(z[[2, 2]], np.array([-.5*E_g_GaAs, -.5*E_g_GaAs - ΔE_v]), color=RWTH_COLORS['blue'])
    ax.plot(z[2:4], np.array([-.5*E_g_GaAs - ΔE_v]*2), color=RWTH_COLORS['blue'])
    ax.annotate('$E_v$', (z[2] + 15,  -.5*E_g_GaAs - ΔE_v), verticalalignment='bottom')
    # Wave function
    plot_states(ax, zw*1e-9, z[1]*1e-9, F, E0, -1, n, scale=1e-5)

    ax.set_ylim(-1.0, -0.65)

    sketch_style(fig, axs)
    fig.savefig(SAVE_PATH / 'qw_undoped_0V.pdf')

# %%% Plot tilted well
zz = [np.linspace(*z[i:i+2], 101) for i in range(3)]
F = 1/200e-9  # 1V/200nm
# Stark shift of the edge of the WE wrt the center
E0 = F*L/2

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, axs = plt.subplots(2, sharex=True)

    # Conduction band
    ax = axs[0]
    ax.plot(zz[0], F*1e-9*(zz[0] - z0) + .5*E_g_GaAs + ΔE_c, color=RWTH_COLORS['blue'])
    ax.plot(z[[1, 1]], np.array([.5*E_g_GaAs + ΔE_c - E0, .5*E_g_GaAs - E0]),
            color=RWTH_COLORS['blue'])
    ax.plot(zz[1], F*1e-9*(zz[1] - z0) + .5*E_g_GaAs, color=RWTH_COLORS['blue'])
    ax.plot(z[[2, 2]], np.array([.5*E_g_GaAs + E0, .5*E_g_GaAs + ΔE_c + E0]),
            color=RWTH_COLORS['blue'])
    ax.plot(zz[2], F*1e-9*(zz[2] - z0) + .5*E_g_GaAs + ΔE_c, color=RWTH_COLORS['blue'])
    ax.annotate(r'$E_\mathrm{c}$', (z[2] + 15, F*1e-9*(z[2] - z0 + 15) + .5*E_g_GaAs + ΔE_c),
                verticalalignment='top')

    # Wave function
    plot_states(ax, zw*1e-9, z[1]*1e-9, F, E0, +1, n, scale=1e-5)

    ax.set_ylim(0.675, 1.175)
    # ax.grid(axis='y')

    # Valence band
    ax = axs[1]
    ax.plot(zz[0], F*1e-9*(zz[0] - z0) - .5*E_g_GaAs - ΔE_v, color=RWTH_COLORS['blue'])
    ax.plot(z[[1, 1]], np.array([-.5*E_g_GaAs - ΔE_v - E0, -.5*E_g_GaAs - E0]),
            color=RWTH_COLORS['blue'])
    ax.plot(zz[1], F*1e-9*(zz[1] - z0) - .5*E_g_GaAs, color=RWTH_COLORS['blue'])
    ax.plot(z[[2, 2]], np.array([-.5*E_g_GaAs + E0, -.5*E_g_GaAs - ΔE_v + E0]),
            color=RWTH_COLORS['blue'])
    ax.plot(zz[2], F*1e-9*(zz[2] - z0) - .5*E_g_GaAs - ΔE_v, color=RWTH_COLORS['blue'])
    ax.annotate(r'$E_\mathrm{v}$', (z[2] + 15, F*1e-9*(z[2] - z0 + 20) - .5*E_g_GaAs - ΔE_v),
                verticalalignment='bottom')

    # Wave function
    plot_states(ax, zw*1e-9, z[1]*1e-9, F, E0, -1, n, scale=1e-5)

    ax.set_ylim(-1.125, -0.625)
    # ax.grid(axis='y')

    sketch_style(fig, axs)
    fig.savefig(SAVE_PATH / 'qw_undoped_1V.pdf')
