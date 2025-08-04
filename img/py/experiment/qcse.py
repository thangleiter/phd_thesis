# %% Imports
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import elementwise
from mpl_toolkits.axes_grid1 import ImageGrid

import PyMoosh as pm
from qutil import const, functools, math
from qutil.plotting.colors import (make_sequential_colormap, make_diverging_colormap,
                                   RWTH_COLORS, RWTH_COLORS_75, RWTH_COLORS_50, RWTH_COLORS_25)

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, TEXTWIDTH, PATH, init,  # noqa
                    E_AlGaAs, effective_mass, reduced_mass)

SAVE_PATH = PATH / 'pdf/experiment'
SAVE_PATH.mkdir(exist_ok=True)

LINE_COLORS = [color for name, color in RWTH_COLORS.items() if name not in ('blue',)]
LINE_COLORS_75 = [color for name, color in RWTH_COLORS_75.items() if name not in ('blue',)]
LINE_COLORS_50 = [color for name, color in RWTH_COLORS_50.items() if name not in ('blue',)]
LINE_COLORS_25 = [color for name, color in RWTH_COLORS_25.items() if name not in ('blue',)]
with np.errstate(divide='ignore', invalid='ignore'):
    SEQUENTIAL_CMAP = make_sequential_colormap('magenta', endpoint='blackwhite').reversed()
    DIVERGING_CMAP = make_diverging_colormap(('green', 'magenta'), endpoint='white')
with np.errstate(divide='ignore', invalid='ignore'):
    CYCLIC_CMAP = mpl.colors.ListedColormap(np.concatenate(
        [make_sequential_colormap('purple', endpoint='blackwhite').colors[:-1],
         make_sequential_colormap('teal', endpoint='blackwhite').reversed().colors]
    ))

init(MARGINSTYLE, backend := 'pgf')
# %% Functions


def coulomb(r, eps_r):
    with np.errstate(divide='ignore'):
        return -const.e**2/(4*np.pi*const.epsilon_0*eps_r*abs(r))


def oscillator_strength(z, i, j, Delta_E, m, in_plane=True):
    return 2*m*Delta_E/const.hbar**2*math.abs2(
        sc.integrate.simpson(j*i*(z if not in_plane else 1), z)
    )


def tunneling_probability(F, ΔV, E, m):
    with np.errstate(divide='ignore'):
        # WKB gives a factor 4/3*sqrt{2} instead of 2
        result = np.exp(-np.sqrt(4*m*(ΔV - E)**3)/(const.e*F*const.hbar))
    result[np.isnan(result)] = 0
    return result


def tunneling_rate(F, ΔV, E, L, m):
    # Adapted from Larsson (1988)
    # k = sqrt(2mE)/hbar
    # v = hbar k/m
    # f = v / 2L
    f = np.sqrt(0.5*E/m)/L
    P = tunneling_probability(F, ΔV, E, m)
    return P * f


def thermionic_emission_rate(F, T, ΔV, E, L, m):
    # For completeness, though at mK zero
    # Fox (1991), from Schneider & v. Klitzing (1988)
    H = ΔV - (E - const.e*F*L/2)
    kT = const.k*T
    return np.sqrt(kT/(2*np.pi*m*L**2))*np.exp(-H/kT)


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


def eps_harmonic_oscillator(omega, p=0, ell=0):
    # p = 0.5*(n - abs(ell)), where n = n_x + n_y in cartesian coordinates.
    return const.hbar*omega*(2*p + abs(ell) + 1)


def eps_square(L, m, N: int):
    return (const.hbar * np.pi * np.arange(1, N+1) / L)**2 / (2 * m)


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


def eps_triangular_vec(F, L, m, N: int):
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

    assert N > 0
    F = np.atleast_1d(F)
    eps = np.empty((N, *F.shape))
    eps[0] = eps_triangular(F, L, m).real
    eps_sq = eps_square(L, m, N)
    eps_tri = eps_triangular_large_field(F, m, N)

    a_n, *_ = sc.special.ai_zeros(N)

    mask = F != 0
    for n in range(1, N):
        xl0 = xmin = eps[n-1] * 1.01
        # this needs a bit of fine-tuning
        xmax = eps_sq[n] + eps_tri[n]

        res_bracket = elementwise.bracket_root(fn, xl0, xmin=xmin, xmax=xmax, args=(F,))
        res_root = elementwise.find_root(fn, res_bracket.bracket, args=(F,))

        if not res_root.success.all():
            raise RuntimeError("Not converged.\n", res_root, res_bracket)
        else:
            eps[n, mask] = res_root.x[mask]
            eps[n, ~mask] = eps_sq[n]

    return eps


def eps_triangular_large_field(F, m, N: int):
    # Eq (4.42), Davies 1998
    epst = eps_tilde(F, m).real
    return np.expand_dims(-airy_zeros(N), list(range(1, np.ndim(epst) + 1)))*epst


def psi_harmonic_oscillator(rho, phi, m, omega, p=0, ell=0):
    # 10.1103/PhysRevA.89.063813 with slight changes of notation:
    # alpha -> 1/xi (oscillator length)
    omega = np.atleast_1d(omega)[:, None, None]
    with np.errstate(divide='ignore'):
        xi = np.sqrt(const.hbar/(m*omega))
    A_n_ell = (
        np.sqrt(2*sc.special.factorial(p)/(xi**2*sc.special.factorial(p + abs(ell))))
        * np.exp(-0.5*(rho/xi)**2)
        * (rho/xi)**abs(ell)
        * sc.special.genlaguerre(p, abs(ell))((rho/xi)**2)
    )
    return np.where(omega != 0, A_n_ell*math.cexp(ell*phi)/np.sqrt(2*np.pi), 1)


def psi_square(z, L, N: int):
    return np.sqrt(2/L)*np.sin(np.arange(1, N+1)[:, None]*np.pi*z/L)


def psi_triangular(z, F: float, L, m, N: int):
    if F == 0:
        return psi_square(z, L, N)

    eps = eps_triangular_vec(F, L, m, N)

    Z_ = Z(z, eps, F, m)
    Z_L = Z(L, eps, F, m)

    Ai, _, Bi, _ = sc.special.airy(Z_)
    Ai_L, _, Bi_L, _ = sc.special.airy(Z_L)

    res = Ai - Ai_L/Bi_L*Bi
    # Consistent phase
    res *= np.where(np.diff(res[..., :2]) < 0, -1, 1)
    return res


def psi_triangular_large_field(z, F, m, N: int):
    # airy(z/z_tilde - eps/eps_tilde)
    return airy(const.e*F*z/eps_tilde(F, m) + airy_zeros(N)[:, None]).real


def plot_states(ax, z, z0, F, E0, sgn, N, scale):
    E = sgn*(.5*E_g_GaAs - E0 + eps_triangular_vec(F, L, m[int(np.signbit(sgn))], N)/const.e)
    P = sgn*psi_triangular(z - z0, F, L, m[int(np.signbit(sgn))], N)[:, ::sgn]
    P /= np.sqrt(np.trapezoid(math.abs2(P), z - z0)[:, None])
    for n in range(N):
        ax.plot([95, 125], [E[n]]*2, ls='--', color=LINE_COLORS_50[n], alpha=0.66)
        ax.plot(z*1e9, E[n] + P[n]*scale, color=LINE_COLORS[n])


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

N = 2
L = 20e-9
m = effective_mass()
eps_r = pm.Material(['main', 'GaAs', 'Papatryfonos'], specialType="RII").get_permittivity(800).real

# 10 MV/m = 100 kV/cm = 2 V/200nm
F = np.linspace(0, 10, 201)*1e6
z = np.linspace(0, L, 101)
# %% Parameter sweeps
with np.errstate(invalid='ignore'):
    eps = np.zeros((2, N, F.size))
    eps_a = np.zeros((2, N, F.size))
    psi = np.zeros((2, N, F.size, z.size))
    psi_a = np.zeros((2, N, F.size, z.size))
    for i in range(2):
        eps[i] = eps_triangular_vec(F, L, m[i, 0], N)
        eps_a[i] = eps_triangular_large_field(F, m[i, 0], N)
        for j in range(F.size):
            psi[i, :, j] = psi_triangular(z, F[j], L, m[i, 0], N)
            psi[i, :, j] /= np.sqrt(np.trapezoid(math.abs2(psi[i, :, j]), z))[:, None]
            psi[i, :, j, [0, -1]] = 0  # get rid of artefacts, per BC zero
            psi_a[i, :, j] = psi_triangular_large_field(z, F[j], m[i, 0], N)
            psi_a[i, :, j] /= np.sqrt(np.trapezoid(math.abs2(psi_a[i, :, j]), z))[:, None]

# %%% 1d Wave functions
fig, axs = plt.subplots(2, N, sharex=True, sharey=True, layout='constrained')
for n in range(N):
    img = axs[0, n].pcolormesh(z, F, psi[0, n], cmap=DIVERGING_CMAP,
                               norm=mpl.colors.CenteredNorm(0))
    img = axs[1, n].pcolormesh(z, F, psi_a[0, n], cmap=DIVERGING_CMAP,
                               norm=mpl.colors.CenteredNorm(0))

# %%% Energy levels
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, layout='constrained')
for i in range(2):
    # subtract e*F*L/2 to obtain the quadratic shift
    # (corresponds to tilting about the middle of the well rather than the edge)
    axs[i, 0].plot(F, (eps[i]/const.e - F*L/2).T)
    axs[i, 1].plot(F, (eps_a[i]/const.e - F*L/2).T)

# Well height
axs[0, 0].axhline(ΔE_c, color='k')
axs[1, 0].axhline(ΔE_v, color='k')
# %%% Tunneling probability estimate (Davies Eq. (7.40))
ΔV = np.array([ΔE_c, ΔE_v])[:, None, None]*const.e
T = tunneling_probability(F, ΔV, eps, m[:, None])
Γ_T = tunneling_rate(F, ΔV, eps, L, m[:, None])
Γ_E = thermionic_emission_rate(F, 10, ΔV, eps, L, m[:, None])
# %% 3d wavefunctions in a trap
omega = 7.38e11 * F / 5e6
rho = np.linspace(0, 75e-9, 151)
zz = z - L/2
mu = reduced_mass()
P = 3

f = np.empty((P, F.size), dtype=complex)
eps_com = np.array([eps_harmonic_oscillator(omega, p) for p in np.arange(P)])
psi_com = np.empty((P, F.size, z.size, rho.size), dtype=complex)
for p in range(P):
    # phi = 0 since ell != 0 does not couple
    psi_com[p] = psi_harmonic_oscillator(np.hypot(*np.meshgrid(rho, zz)), 0, m.sum(), omega, p, 0)
    psi_com[p] /= np.sqrt(
        sc.integrate.simpson(2*np.pi*rho*math.abs2(psi_com[p]), rho, axis=-1)
    )[..., None]

# Axes (e/h, n, p, F, z, rho)
Delta_Ez = E_g_GaAs*const.e + (eps - const.e*F*L/2).sum(axis=0)
Delta_E = Delta_Ez[:, None] + eps_com
# 3d wavefunction
PSI = psi[:, :, None, :, :, None] * psi_com
# ψ_e * ψ_h
psi_exc = psi[0] * psi[1, ..., ::-1]
# 2π\int dρ ρ χ(r)
psi_com_perp = sc.integrate.simpson(2*np.pi*rho*psi_com, rho, axis=-1)
f_3d = oscillator_strength(zz, psi_exc[:, None], psi_com_perp, Delta_E, mu, in_plane=True)
f_1d = oscillator_strength(zz, psi[0], psi[1, ..., ::-1], Delta_Ez, mu, in_plane=False)

# %%% For fun plot higher orbital angular momenta (unused)
phi = np.arange(0, 2*np.pi, 2.5e-2)

fig, axs = plt.subplots(2, 3, subplot_kw={'projection': 'polar'}, layout='constrained')
for p in range(P):
    wf = psi_harmonic_oscillator(rho[:, None], phi, m.sum(), omega=7.38e11, p=p, ell=1)
    img = axs[0, p].pcolormesh(phi, rho * 1e9, wf[0].real, cmap=DIVERGING_CMAP,
                               norm=mpl.colors.CenteredNorm(0))
    img = axs[1, p].pcolormesh(phi, rho * 1e9, wf[0].imag, cmap=DIVERGING_CMAP,
                               norm=mpl.colors.CenteredNorm(0))
    axs[0, p].tick_params(labelbottom=False)
    axs[1, p].tick_params(labelbottom=False)

# %%% 3d wavefunction
fig = plt.figure(figsize=(TEXTWIDTH, 1.8))
grid = ImageGrid(fig, 111, (min(P, 3), 2), cbar_mode='single', cbar_pad=0.075, cbar_size='2.5%',
                 axes_pad=0.1)
norm = mpl.colors.CenteredNorm(0)
n = 0
F0 = 20
F1 = 100

for p in range(min(3, P)):
    # plot holes since they couple more strongly to the field
    img1 = grid.axes_row[p][0].contourf(rho*1e9, zz*1e9, PSI[1, n, p, F0, :, :].real,
                                        cmap=DIVERGING_CMAP,
                                        norm=norm,
                                        levels=21)
    img2 = grid.axes_row[p][1].contourf(rho*1e9, zz*1e9, PSI[1, n, p, F1, :, :].real,
                                        cmap=DIVERGING_CMAP,
                                        norm=norm,
                                        levels=21)
    grid.axes_row[p][0].text(rho[-1]*1e9 - 1.5, -8.5,
                             '\n'.join(['$n=0$', f'$p={p}$', r'$\ell=0$']),
                             fontsize='small', horizontalalignment='right')
    grid.axes_row[p][1].text(rho[-1]*1e9 - 1.5, -8.5,
                             '\n'.join(['$n=0$', f'$p={p}$', r'$\ell=0$']),
                             fontsize='small', horizontalalignment='right')

cbar = grid.cbar_axes[0].colorbar(img1)
cbar.set_ticks([0])
cbar.set_label(r'$\mathrm{Re}\Psi_{np\ell}(r, z_{\mathrm{h}})$', loc='top')
cbar.ax.yaxis.set_label_coords(1.5, 1)

grid.axes_row[-1][0].set_ylabel(r'$z_{\mathrm{h}}$ (nm)')
grid.axes_row[-1][0].set_xlabel(r'$\rho$ (nm)')

match backend:
    case 'pgf':
        grid.axes_row[0][0].set_title(rf'$F=\qty{{{F[F0]*1e-6:.0f}}}{{\volt\per\micro\meter}}$')
        grid.axes_row[0][1].set_title(rf'$F=\qty{{{F[F1]*1e-6:.0f}}}{{\volt\per\micro\meter}}$')
    case _:
        grid.axes_row[0][0].set_title(f'$F={F[F0]*1e-6:.0f}$ V/μm')
        grid.axes_row[0][1].set_title(f'$F={F[F1]*1e-6:.0f}$ V/μm')

fig.savefig(SAVE_PATH / 'wavefunction.pdf')

# %%% Energy shift & oscillator strength
with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax1 = plt.subplots(2, sharex=True,
                            layout='constrained',
                            figsize=(MARGINWIDTH, 1.75),
                            gridspec_kw=dict(height_ratios=[2, 2]))
    ax2 = [ax1[0].twinx(), ax1[1].twinx()]

    for i in (1, 0):
        # i are the energy levels
        ax2[1].plot(F * 1e-6, Γ_T[0, i]*1e-6, ls='--', color=LINE_COLORS_50[i])
        ax2[1].plot(F * 1e-6, Γ_T[1, i]*1e-6, ls='-.', color=LINE_COLORS_50[i])

        ax2[0].plot(F * 1e-6, np.gradient(Delta_Ez[i], F, axis=-1).T/const.e * 1e9,
                    ls='--', color=LINE_COLORS_50[i])
        for p, line_colors in enumerate([LINE_COLORS_50, LINE_COLORS_75, LINE_COLORS]):
            if i != 0 and p != 2:
                continue
            ax1[0].plot(F * 1e-6, (Delta_E[i, -p-1]/const.e - E_g_GaAs) * 1e3,
                        color=line_colors[i])
            ax1[1].plot(F * 1e-6, f_3d[i, -p-1] / f_3d[0, 0, 0],
                        color=line_colors[i])

    ax2[0].set_yticks([0, -5, -10])
    ax2[1].set_ylim(1)
    ax2[1].set_yscale('log')
    ax2[1].set_yticks([1e-6, 1, 1e6])
    ax1[1].set_ylim(1.25e-3, 1.5)
    ax1[1].set_yscale('log')
    ax1[0].set_ylim(ylim := ax1[0].get_ylim())  # to freeze them
    ax1[0].plot([0, 3.7165, 3.7165], [0, 0, -ylim[1]], ls=':', color=RWTH_COLORS_50['black'],
                zorder=1)

    ax1[1].set_ylabel(r'$\tilde{f}_{np}$')
    ax2[1].set_ylabel(r'$\Gamma$ (MHz)')
    match backend:
        case 'pgf':
            ax1[0].set_ylabel(r'$\Delta E_{np} - E_\mathrm{g}$ (\unit{\milli\electronvolt})')
            ax1[1].set_xlabel(r'$F$ (\unit{\volt\per\micro\meter})')
            ax2[0].set_ylabel(r'$\pdv*{\Delta E}{F}$ (\unit{\electron\nano\meter})')
        case _:
            ax1[1].set_xlabel('$F$ (V/μm)')
            ax1[0].set_ylabel(r'$\Delta E_{n} - E_\mathrm{g}$ (meV)')
            ax2[0].set_ylabel(r'$\partial\Delta E/\partial F$ ($e$nm)')

    fig.get_layout_engine().set(h_pad=1/72)
    fig.savefig(SAVE_PATH / 'qcse_field_dependence.pdf')

# %% Sketch in-plane field
F0 = 1e5
r = np.linspace(-100, 100, 1001)*1e-9
a_B = 2*np.pi*const.epsilon_0*eps_r*(const.hbar/const.e)**2/mu
phi = coulomb(r, eps_r)/const.e

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained', figsize=(MARGINWIDTH, 1.25))

    ax.axhline(-8, ls=':', color=RWTH_COLORS_50['black'])
    ax.axline((0, 0), slope=0, ls='--', color=RWTH_COLORS_50['black'], alpha=0.66)
    ax.axline((1, (coulomb(1, eps_r)/const.e + F0*a_B) * 1e3), slope=F0*a_B * 1e3, ls='--',
              color=RWTH_COLORS_50['black'], alpha=0.66)
    ax.plot(r/a_B, phi * 1e3, color=LINE_COLORS[0])
    ax.plot(r/a_B, (phi + F0*r) * 1e3, color=LINE_COLORS[1])
    ax.plot(r/a_B, 12.5*np.exp(-abs(r)/a_B) - 8, color=RWTH_COLORS['black'])
    ax.set_ylim(-14, 10)

    ax.set_xlabel(r'$r/a_\mathrm{B}$')
    ax.set_ylabel(r'$E$ (meV)')

    fig.savefig(SAVE_PATH / 'in_plane_field.pdf')

# %% Sketch band structure
# Plot the first N levels
N = 2
z = np.array([-100, -L/2*1e9, L/2*1e9, 100]) + 110
z0 = np.mean(z)
zw = np.linspace(z[1], z[2], 101)

band_color = RWTH_COLORS['black']
# %%% Plot untilted well
F0 = E0 = 0

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, axs = plt.subplots(2, sharex=True)

    # Conduction band
    ax = axs[0]
    ax.plot(z[0:2], np.array([.5*E_g_GaAs + ΔE_c]*2), color=band_color)
    ax.plot(z[[1, 1]], np.array([.5*E_g_GaAs + ΔE_c, .5*E_g_GaAs]), color=band_color)
    ax.plot(z[1:3], np.array([.5*E_g_GaAs]*2), color=band_color)
    ax.plot(z[[2, 2]], np.array([.5*E_g_GaAs, .5*E_g_GaAs + ΔE_c]), color=band_color)
    ax.plot(z[2:4], np.array([.5*E_g_GaAs + ΔE_c]*2), color=band_color)
    ax.annotate(r'$E_\mathrm{c}$', (z[2] + 14, .5*E_g_GaAs + ΔE_c - 0.02),
                verticalalignment='top')
    # Wave function
    plot_states(ax, zw*1e-9, z[1]*1e-9, F0, E0, +1, N, scale=1e-5)

    ax.set_ylim(0.7, 1.05)

    # Valence band
    ax = axs[1]
    ax.plot(z[0:2], np.array([-.5*E_g_GaAs - ΔE_v]*2), color=band_color)
    ax.plot(z[[1, 1]], np.array([-.5*E_g_GaAs - ΔE_v, -.5*E_g_GaAs]), color=band_color)
    ax.plot(z[1:3], np.array([-.5*E_g_GaAs]*2), color=band_color)
    ax.plot(z[[2, 2]], np.array([-.5*E_g_GaAs, -.5*E_g_GaAs - ΔE_v]), color=band_color)
    ax.plot(z[2:4], np.array([-.5*E_g_GaAs - ΔE_v]*2), color=band_color)
    ax.annotate(r'$E_\mathrm{hh}$', (z[2] + 14,  -.5*E_g_GaAs - ΔE_v + 0.02),
                verticalalignment='bottom')
    # Wave function
    plot_states(ax, zw*1e-9, z[1]*1e-9, F0, E0, -1, N, scale=1e-5)

    ax.set_ylim(-1.0, -0.65)

    sketch_style(fig, axs)
    fig.savefig(SAVE_PATH / f'qw_undoped_{F0*200e-9:1g}V.pdf')

# %%% Plot tilted well
zz = [np.linspace(*z[i:i+2], 101) for i in range(3)]
F0 = 1/200e-9  # 1V/200nm
# Stark shift of the edge of the WE wrt the center
E0 = F0*L/2

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, axs = plt.subplots(2, sharex=True)

    # Conduction band
    ax = axs[0]
    ax.plot(zz[0], F0*1e-9*(zz[0] - z0) + .5*E_g_GaAs + ΔE_c, color=band_color)
    ax.plot(z[[1, 1]], np.array([.5*E_g_GaAs + ΔE_c - E0, .5*E_g_GaAs - E0]),
            color=band_color)
    ax.plot(zz[1], F0*1e-9*(zz[1] - z0) + .5*E_g_GaAs, color=band_color)
    ax.plot(z[[2, 2]], np.array([.5*E_g_GaAs + E0, .5*E_g_GaAs + ΔE_c + E0]),
            color=band_color)
    ax.plot(zz[2], F0*1e-9*(zz[2] - z0) + .5*E_g_GaAs + ΔE_c, color=band_color)
    ax.annotate(r'$E_\mathrm{c}$', (z[2] + 14, F0*1e-9*(z[2] - z0) + .5*E_g_GaAs + ΔE_c + 0.05),
                verticalalignment='top')

    # Wave function
    plot_states(ax, zw*1e-9, z[1]*1e-9, F0, E0, +1, N, scale=1e-5)

    ax.set_ylim(0.675, 1.175)

    # Valence band
    ax = axs[1]
    ax.plot(zz[0], F0*1e-9*(zz[0] - z0) - .5*E_g_GaAs - ΔE_v, color=band_color)
    ax.plot(z[[1, 1]], np.array([-.5*E_g_GaAs - ΔE_v - E0, -.5*E_g_GaAs - E0]), color=band_color)
    ax.plot(zz[1], F0*1e-9*(zz[1] - z0) - .5*E_g_GaAs, color=band_color)
    ax.plot(z[[2, 2]], np.array([-.5*E_g_GaAs + E0, -.5*E_g_GaAs - ΔE_v + E0]), color=band_color)
    ax.plot(zz[2], F0*1e-9*(zz[2] - z0) - .5*E_g_GaAs - ΔE_v, color=band_color)
    ax.annotate(r'$E_\mathrm{hh}$', (z[2] + 14, F0*1e-9*(z[2] - z0) - .5*E_g_GaAs - ΔE_v + 0.125),
                verticalalignment='bottom')

    # Wave function
    plot_states(ax, zw*1e-9, z[1]*1e-9, F0, E0, -1, N, scale=1e-5)

    ax.set_ylim(-1.125, -0.625)

    sketch_style(fig, axs)
    fig.savefig(SAVE_PATH / f'qw_undoped_{F0*200e-9:1g}V.pdf')
