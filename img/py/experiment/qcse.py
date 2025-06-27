import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import elementwise

from qutil import const, functools
# %%


def E_AlGaAs(x):
    # https://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/bandstr.html#Temperature
    return 1.519 + 1.155*x + 0.37*x**2


@functools.wraps(sc.special.airy)
def airy(*args, **kwargs):
    return sc.special.airy(*args, **kwargs)[0]


@functools.wraps(sc.special.ai_zeros)
def airy_zeros(*args, **kwargs):
    return sc.special.ai_zeros(*args, **kwargs)[0]


def psi_square(z, L, q, n):
    return -np.sign(q) * np.sqrt(2/L) * np.cos(np.arange(1, n+1)[:, None]*np.pi*z/L)


def eps_square(L, q, m, n: int):
    return np.sign(q) * (const.hbar * np.pi * np.arange(1, n+1) / L)**2 / (2 * m)


def eps_triangular(F, q, m, n: int):
    # Eq (4.42), Davies 1998
    with np.errstate(divide='ignore', invalid='ignore'):
        eps_0 = np.float_power(.5 * (q * F * const.hbar)**2 / m, 1/3)
    return np.expand_dims(-airy_zeros(n), list(range(1, np.ndim(eps_0) + 1))) * eps_0


def Eq_B7(w, f):
    """
    Evaluate the function
      Ai(Z_+) Bi(Z_-) - Ai(Z_-) Bi(Z_+)
    and its derivative with
      Z_± = (-π/f)^(2/3) * (w ± f/2)
    """
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
    """
    Evaluate the function
      Ai(Z_+) Bi(Z_-) - Ai(Z_-) Bi(Z_+)
    and its derivative with
      Z_± = (-π/f)^(2/3) * (w ± f/2)
    """
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
    ax.axhline(E_e/W_1[0], 0, 1, **c)
    ax.axhline(E_h/W_1[1], 0, 1, ls='--', **c)

# %%% Triangular (Davies)
fig, ax = plt.subplots(layout='constrained')
for (E_e, E_h), c in zip(eps_triangular(F, e, m, n), plt.rcParams['axes.prop_cycle']):
    ax.plot(F, E_e/W_1[0], **c)
    ax.plot(F, E_h/W_1[1], ls='--', **c)

# %%% Triangular (Miller)
fig, axs = plt.subplots(2, sharex=True)
for (E_e, E_h), c in zip(W.swapaxes(0, 1), plt.rcParams['axes.prop_cycle']):
    axs[0].grid()
    axs[0].plot(F, +E_e/const.e, **c)
    axs[0].plot(F, -E_h/const.e, ls='--', **c)

    axs[1].grid()
    axs[1].plot(F, (E_e - (-E_h)) / const.e, **c)
