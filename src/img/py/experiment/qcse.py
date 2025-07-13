# %% Imports
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import elementwise

from qutil import const, functools
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_50

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, TEXTWIDTH, PATH, init,  # noqa
                    apply_sketch_style)

SAVE_PATH = PATH / 'pdf/experiment'
SAVE_PATH.mkdir(exist_ok=True)

init(MARGINSTYLE, backend := 'pgf')
# %% Functions


def E_AlGaAs(x):
    # https://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/bandstr.html#Temperature
    return 1.519 + 1.155*x + 0.37*x**2


@functools.wraps(sc.special.airy)
def airy(*args, **kwargs):
    return sc.special.airy(*args, **kwargs)[0]


@functools.wraps(sc.special.ai_zeros)
def airy_zeros(*args, **kwargs):
    return sc.special.ai_zeros(*args, **kwargs)[0]


def eps_square(L, q, m, n: int):
    return np.sign(q) * (const.hbar * np.pi * np.arange(1, n+1) / L)**2 / (2 * m)


def eps_triangular_large_field(F, q, m, n: int):
    # Eq (4.42), Davies 1998
    with np.errstate(divide='ignore', invalid='ignore'):
        eps_0 = np.float_power(.5 * (q * F * const.hbar)**2 / m, 1/3)
    return np.expand_dims(-airy_zeros(n), list(range(1, np.ndim(eps_0) + 1))) * eps_0


def psi_square(z, L, n: int):
    return np.sqrt(2/L)*np.sin(n*np.pi*z/L)


def psi_triangular(z, L, F, q, m, n: int):
    # Miller's f, W_1
    W_1 = eps_square(L, q, m, 1)
    f = F / W_1 * q * L
    # Miller's W is Rabinovitch's eps
    eps = eps_triangular_normalized_vec(np.atleast_1d(f), n)[-1]*W_1
    # Rabinovitch's α_1, \bar{ε}
    alpha_1 = 2*m*q*F/const.hbar**2
    eps_bar = 2*m*eps/const.hbar**2
    Z = (np.float_power(alpha_1, 1/3, dtype=complex)*z
         - eps_bar/np.float_power(alpha_1, 2/3, dtype=complex))
    ZL = (np.float_power(alpha_1, 1/3, dtype=complex)*L
          - eps_bar/np.float_power(alpha_1, 2/3, dtype=complex))

    Ai, _, Bi, _ = sc.special.airy(Z)
    Ai_L, _, Bi_L, _ = sc.special.airy(ZL)

    return Ai - Ai_L/Bi_L*Bi


def psi_triangular_davies(z, L, F, q, m, n: int):
    # Miller's f, W_1
    W_1 = eps_square(L, q, m, 1)
    f = F / W_1 * q * L
    # Miller's W is Rabinovitch's eps
    eps = eps_triangular_normalized_vec(np.atleast_1d(f), n)[-1]*W_1
    eps0 = np.float_power(.5*(q*F*const.hbar)**2/m, 1/3, dtype=complex)

    Z = (q*F*z + eps)/eps0
    return sc.special.airy(Z)[0]


def Eq_B7(w, f):
    # Miller 1985
    f = float(f)
    if f == 0:
        return 0, 0

    c = -(-np.pi / f)**(2/3)  # scaling factor
    Zp = c * (w + f/2)
    Zm = c * (w - f/2)

    # Compute the Airy functions Ai and Bi for Zp and Zm
    Ai_Zp, Aip_Zp, Bi_Zp, Bip_Zp = sc.special.airy(Zp)
    Ai_Zm, Aip_Zm, Bi_Zm, Bip_Zm = sc.special.airy(Zm)

    return (
        Ai_Zp * Bi_Zm - Ai_Zm * Bi_Zp,
        c*(Ai_Zp*Bip_Zm + Aip_Zp*Bi_Zm - Ai_Zm*Bip_Zp - Aip_Zm*Bi_Zp)
    )


def Eq_B7_vec(w, f):
    # Miller 1985
    w, f = np.broadcast_arrays(w, f)
    mask = f != 0

    val, grad = np.empty((2, *f.shape), dtype=complex)

    c = -np.float_power(-np.pi / f[mask], 2/3, dtype=complex)  # scaling factor
    Zp = c * (w + f/2)[mask]
    Zm = c * (w - f/2)[mask]

    # Compute the Airy functions Ai and Bi for Zp and Zm
    Ai_Zp, _, Bi_Zp, _ = sc.special.airy(Zp)
    Ai_Zm, _, Bi_Zm, _ = sc.special.airy(Zm)

    val[mask] = Ai_Zp * Bi_Zm - Ai_Zm * Bi_Zp
    val[~mask] = 0
    return val.real


def Eq_2d5(eps, F, L, q, m):
    # Rabinovitch 1971
    if F == 0:
        return 0, 0

    alpha_1 = 2*m*e*F/const.hbar**2
    eps_bar = 2*m*eps/const.hbar**2

    alphaell = np.float_power(alpha_1, 1/3, dtype=complex)*L
    alphaeps = -np.float_power(alpha_1, -2/3, dtype=complex)*eps_bar
    deps = -np.float_power(alpha_1, -2/3, dtype=complex)*2*m/const.hbar**2

    Ai_1, Aip_1, Bi_1, Bip_1 = sc.special.airy(alphaeps.real)
    Ai_2, Aip_2, Bi_2, Bip_2 = sc.special.airy(alphaeps.real + alphaell.real)

    return (
        Ai_1 * Bi_2 - Ai_2 * Bi_1,
        deps*(Ai_1*Bip_2 + Aip_1*Bi_2 - Ai_2*Bip_1 - Aip_2*Bi_1)
    )


def eps_triangular(F, L, q, m):
    # Rabinovitch 1971
    eps = np.empty(F.size, dtype=complex)
    for i in range(len(F)):
        if i == 0:
            x0 = eps_square(L, e, m, 1)
        else:
            x0 = eps[i-1]
        result = sc.optimize.root_scalar(Eq_2d5, args=(F[i], L, q, m), fprime=True, x0=x0)
        if not result.converged:
            raise RuntimeError("Not converged.\n", result)
        eps[i] = result.root.item()
    return eps


def eps_triangular_normalized(f):
    # Miller 1985
    w = np.empty(f.size, dtype=complex)
    for i in range(len(f)):
        if i == 0:
            x0 = 1
        else:
            x0 = w[i-1]
        result = sc.optimize.root_scalar(Eq_B7, args=(f[i],), fprime=True, x0=x0)
        if not result.converged:
            raise RuntimeError("Not converged.\n", result)
        w[i] = result.root
    return w


def eps_triangular_normalized_vec(f, n: int):
    assert n > 0
    f = np.asarray(f)
    eps = np.empty((n, *f.shape))

    mask = f == 0

    for i in range(n):
        if i == 0:
            # initial run
            eps[i] = eps_triangular_normalized(f).real
            continue

        xl0 = xmin = eps[i-1] + 1e-2
        # this needs a bit of fine-tuning
        xmax = 2*(i + 1)**2

        res_bracket = elementwise.bracket_root(Eq_B7_vec, xl0, xmin=xmin, xmax=xmax, args=(f,))
        res_root = elementwise.find_root(Eq_B7_vec, res_bracket.bracket, args=(f,))

        if not res_root.success.all():
            raise RuntimeError("Not converged.\n", res_root, res_bracket)
        else:
            eps[i, ~mask] = res_root.x[~mask]
            eps[i, mask] = (i + 1)**2

    return eps


# %% Parameters
E_g_GaAs = E_AlGaAs(0)
ΔE_g = E_AlGaAs(0.33) - E_g_GaAs
Q_e = 0.57
ΔE_c = Q_e * ΔE_g
ΔE_v = (1 - Q_e) * ΔE_g

n = 2
L = 20e-9
# masses from 10.1103/PhysRevB.29.7085
m_ep = 0.0665
m_hp = 0.34
m = np.array([[m_ep, m_hp]]).T * const.m_e
e = const.e

# 10 MV/m = 100 kV/cm = 2 V/200nm
F = np.linspace(0, 10, 1001)*1e6

W_1 = eps_square(L, e, m, 1)
f = F / W_1 * const.e * L
w = np.empty((2, n, F.size))
for i in range(2):
    w[i] = eps_triangular_normalized_vec(f[i], n)

W = w * W_1[:, None]

# %% Energy levels
# %%% Square
fig, ax = plt.subplots(layout='constrained')
for (E_e, E_h), c in zip(eps_square(L, e, m, n).T, plt.rcParams['axes.prop_cycle']):
    ax.axhline(E_e/const.e, 0, 1, **c)
    ax.axhline(E_h/const.e, 0, 1, ls='--', **c)

# %%% Triangular (Davies)
fig, ax = plt.subplots(layout='constrained')
for (E_e, E_h), c in zip(eps_triangular(F, e, m, n), plt.rcParams['axes.prop_cycle']):
    ax.plot(F, E_e/W_1[0], **c)
    ax.plot(F, E_h/W_1[1], ls='--', **c)

# %%% Triangular (Miller)
fig, axs = plt.subplots(2, sharex=True)
for (E_e, E_h), c in zip(W.swapaxes(0, 1), plt.rcParams['axes.prop_cycle']):
    axs[0].grid(True)
    axs[0].plot(F, +E_e/const.e, **c)
    axs[0].plot(F, -E_h/const.e, ls='--', **c)

    axs[1].grid(True)
    axs[1].plot(F, (E_e - (-E_h)) / const.e, **c)

# %% Sketch band structure
F = 1/200  # 1V/200nm
# %%% Plot untilted well
fig, ax = plt.subplots()

apply_sketch_style(ax)

z = np.array([-100, -L/2*1e9, L/2*1e9, 100]) + 110
z0 = np.mean(z)
E_off = 0.5

# Conduction band
ax.plot(z[0:2], np.array([.5*ΔE_g + ΔE_c]*2), color=RWTH_COLORS['blue'])
ax.plot(z[[1, 1]], np.array([.5*ΔE_g + ΔE_c, .5*ΔE_g]), color=RWTH_COLORS['blue'])
ax.plot(z[1:3], np.array([.5*ΔE_g]*2), color=RWTH_COLORS['blue'])
ax.plot(z[[2, 2]], np.array([.5*ΔE_g, .5*ΔE_g + ΔE_c]), color=RWTH_COLORS['blue'])
ax.plot(z[2:4], np.array([.5*ΔE_g + ΔE_c]*2), color=RWTH_COLORS['blue'])
ax.annotate('$E_c$', (z[3] + 5, .5*ΔE_g + ΔE_c), verticalalignment='center')

# Valence band
ax.plot(z[0:2], np.array([-.5*ΔE_g - ΔE_v]*2), color=RWTH_COLORS['blue'])
ax.plot(z[[1, 1]], np.array([-.5*ΔE_g - ΔE_v, -.5*ΔE_g]), color=RWTH_COLORS['blue'])
ax.plot(z[1:3], np.array([-.5*ΔE_g]*2), color=RWTH_COLORS['blue'])
ax.plot(z[[2, 2]], np.array([-.5*ΔE_g, -.5*ΔE_g - ΔE_v]), color=RWTH_COLORS['blue'])
ax.plot(z[2:4], np.array([-.5*ΔE_g - ΔE_v]*2), color=RWTH_COLORS['blue'])
ax.annotate('$E_v$', (z[3] + 5,  -.5*ΔE_g - ΔE_v), verticalalignment='center')

# Wave functions
zw = np.linspace(z[1], z[2], 101)
ax.plot([90, 130], [.5*ΔE_g + eps_square(L, e, m[0], 1)/e]*2,
        ls='--', color=RWTH_COLORS_50['magenta'], alpha=0.66)
ax.plot([90, 130], [-.5*ΔE_g - eps_square(L, e, m[1], 1)/e]*2,
        ls='--', color=RWTH_COLORS_50['magenta'], alpha=0.66)
ax.plot(
    zw,
    .5*ΔE_g + eps_square(L, e, m[0], 1)/e + psi_square(zw - z[1], L*1e9, n=1)**2,
    color=RWTH_COLORS['magenta']
)
ax.plot(
    zw,
    -.5*ΔE_g - eps_square(L, e, m[1], 1)/e - psi_square(zw - z[1], L*1e9, n=1)**2,
    color=RWTH_COLORS['magenta']
)

ax.set_xticks([10, 100, 120, 210])
# ax.set_yticks([.5*ΔE_g + ΔE_c, .5*ΔE_g, -.5*ΔE_g, -.5*ΔE_g - ΔE_v])
ax.set_xlim(0, 250)
ax.set_xlabel('$z$ (nm)', x=0.95, verticalalignment='bottom')
ax.set_ylabel('$E$ (eV)', rotation='vertical')

# %%% Plot tilted well
fig, ax = plt.subplots()

apply_sketch_style(ax)

zz = [np.linspace(*z[i:i+2], 101) for i in range(3)]
E0 = F * np.diff(z)[1] / 2

# Conduction band
ax.plot(zz[0], F*(zz[0] - z0) + .5*ΔE_g + ΔE_c, color=RWTH_COLORS['blue'])
ax.plot(z[[1, 1]], np.array([.5*ΔE_g + ΔE_c - E0, .5*ΔE_g - E0]), color=RWTH_COLORS['blue'])
ax.plot(zz[1], F*(zz[1] - z0) + .5*ΔE_g, color=RWTH_COLORS['blue'])
ax.plot(z[[2, 2]], np.array([.5*ΔE_g + E0, .5*ΔE_g + ΔE_c + E0]), color=RWTH_COLORS['blue'])
ax.plot(zz[2], F*(zz[2] - z0) + .5*ΔE_g + ΔE_c, color=RWTH_COLORS['blue'])
ax.annotate('$E_c$', (zz[2][-1] + 5, F*(zz[2][-1] - z0) + .5*ΔE_g + ΔE_c))

# Valence band
ax.plot(zz[0], F*(zz[0] - z0) - .5*ΔE_g - ΔE_v, color=RWTH_COLORS['blue'])
ax.plot(z[[1, 1]], np.array([-.5*ΔE_g - ΔE_v - E0, -.5*ΔE_g - E0]), color=RWTH_COLORS['blue'])
ax.plot(zz[1], F*(zz[1] - z0) - .5*ΔE_g, color=RWTH_COLORS['blue'])
ax.plot(z[[2, 2]], np.array([-.5*ΔE_g + E0, -.5*ΔE_g - ΔE_v + E0]), color=RWTH_COLORS['blue'])
ax.plot(zz[2], F*(zz[2] - z0) - .5*ΔE_g - ΔE_v, color=RWTH_COLORS['blue'])
ax.annotate('$E_v$', (zz[2][-1] + 5, F*(zz[2][-1] - z0) - .5*ΔE_g - ΔE_v))

# Wave functions
zw = np.linspace(z[1], z[2], 101)
ax.plot(
    zw,
    .5*ΔE_g - E0 + eps_triangular_normalized(F*1e9/W_1[0]*const.e*L).real*W_1[0]/e
    + psi_triangular((zw - z[1])*1e-9, L, F*1e9, e, m[0], n)**2,
    color=RWTH_COLORS['magenta']
)
ax.plot(
    zw,
    -.5*ΔE_g + E0 - eps_triangular_normalized(F*1e9/W_1[1]*const.e*L).real*W_1[1]/e
    - psi_triangular((zw - z[1])*1e-9, L, F*1e9, e, m[1], n)**2,
    color=RWTH_COLORS['magenta']
)

ax.set_xticks([10, 100, 120, 210])
# ax.set_yticks([.5*ΔE_g + ΔE_c, .5*ΔE_g, -.5*ΔE_g, -.5*ΔE_g - ΔE_v])
ax.set_xlim(0, 250)
ax.set_xlabel('$z$ (nm)', x=0.95, verticalalignment='bottom')
ax.set_ylabel('$E$ (eV)', rotation='vertical')
