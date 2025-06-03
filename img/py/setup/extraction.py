import pathlib
import sys
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from qutil.plotting import colors
from qutil.plotting.colors import RWTH_COLORS
from sympy import (sqrt, exp, cos, sin, asin, atan2, symbols, Matrix, trigsimp, lambdify,
                   Eq, solve, integrate, I, pi, re)

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, PATH, TEXTWIDTH, TOTALWIDTH, init,  # noqa
                    markerprops, n_GaAs)

SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

with np.errstate(divide='ignore'):
    SEQUENTIAL_CMAPS = {
        color: colors.make_sequential_colormap(color, endpoint='blackwhite')
        for color in RWTH_COLORS if color != 'black'
    }

CYCLIC_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    'magentagreen', np.concatenate((SEQUENTIAL_CMAPS['magenta'].colors[:-1],
                                    SEQUENTIAL_CMAPS['green'].reversed().colors[:-1]))
)

PLOT_CONTOURS = True

λ = 800e-9
d = 110e-9
nn = n_GaAs(0).real
NAn = 0.7
f_ob = 3.1e-3

init(MARGINSTYLE, backend := 'pgf')
# %% Sympy
z, zp, r, rp, rho, rhop, theta, vartheta, k, c, n, NA, eps, mu = symbols(
    r"z z' r r' rho \rho' theta vartheta k c n NA epsilon mu",
    positive=True, real=True
)
x, y, xp, yp, phi, varphi = symbols("x y x' y' phi varphi", real=True)

# %%% Extraction
# Field components of a dipole oriented along z (share a common factor)
E_r = cos(vartheta) * (2/(k*r)**2 - 2*I/(k*r))
E_theta = sin(vartheta) * (1/(k*r)**2 - I/(k*r) - 1)
E_phi = 0

H_r = 0
H_theta = 0
H_phi = sin(vartheta) * (-I/(k*r) - 1)

# Unit vectors in spherical coordinates
e_r = Matrix([[sin(vartheta)*cos(varphi), sin(vartheta)*sin(varphi), cos(vartheta)]]).T
e_theta = Matrix([[cos(vartheta)*cos(varphi), cos(vartheta)*sin(varphi), -sin(vartheta)]]).T
e_phi = Matrix([[-sin(varphi), cos(varphi), 0]]).T
# Rotation matrix to convert form cartesian to spherical coordinates
R = Matrix([e_r.T, e_theta.T, e_phi.T])

# Field vector
E_0 = 1 / (4 * pi * eps) * exp(I*k*r) / r * k**2
H_0 = (E_0 * sqrt(eps/(mu))).simplify()
E_cartesian = E_0 * trigsimp(E_r * e_r + E_theta * e_theta + E_phi * e_phi)
E_spherical = trigsimp(R * E_cartesian)

H_cartesian = H_0 * trigsimp(H_r * e_r + H_theta * e_theta + H_phi * e_phi)
H_spherical = trigsimp(R * H_cartesian)

S_cartesian = trigsimp(re(E_cartesian.cross(H_cartesian.conjugate()))/2)
S_spherical = trigsimp(re(E_spherical.cross(H_spherical.conjugate()))/2)

assert trigsimp(E_spherical - E_0*Matrix([[E_r, E_theta, E_phi]]).T) == Matrix([[0, 0, 0]]).T

# Rotate coordinate system (xyz) to have z along growth direction and xy plane in the QW; (yzx)
# x' || z
# y' || x
# z' || y
# unprimed angles to cartesian coordinates
cartesian = {
    vartheta: atan2(sqrt(x**2 + y**2), z),
    varphi: atan2(y, x)
}
# Transformation from unprimed to primed
trafo = {
    x: yp,
    y: zp,
    z: xp
}
# Spherical coordinates in the primed system
spherical = {
    xp: r*sin(theta)*cos(phi),
    yp: r*sin(theta)*sin(phi),
    zp: r*cos(theta)
}
# Perform the substitutions to convert between coordinate systems
E_cartesian_prime = trigsimp(E_cartesian.subs(cartesian).subs(trafo).subs(spherical))
# Finally switch axes of the vector
E_cartesian_prime = Matrix([[E_cartesian_prime[2], E_cartesian_prime[0], E_cartesian_prime[1]]]).T

S_spherical_prime = trigsimp(
    S_spherical.subs(cartesian).subs(trafo).subs(spherical).subs(sqrt(eps*mu), c)
)
# %%% Coupling
x_prime = x * n * cos(theta) / sqrt(1 - n**2 * sin(theta)**2)
y_prime = y * n * cos(theta) / sqrt(1 - n**2 * sin(theta)**2)

subs = {x**2 + y**2: rho**2,
        theta: atan2(rho, z)}
x_prime = x_prime.subs(subs).subs(rho**2 + z**2, r**2).simplify()
y_prime = y_prime.subs(subs).subs(rho**2 + z**2, r**2).simplify()

subs = {r**2: x**2 + y**2 + z**2,
        rho**2: x**2 + y**2}
solved = solve([Eq(xp, x_prime.subs(subs)), Eq(yp, y_prime.subs(subs))], [x, y])
x_sol, y_sol = solved[0]

subs = {xp**2 + yp**2: rhop**2,
        xp**2 + yp**2 + z**2: rp**2}
x_sol = x_sol.collect(n**2).subs(subs)
y_sol = y_sol.collect(n**2).subs(subs)

zeta = x_sol / xp
assert (zeta - y_sol / yp == 0)

# %% Functions


def transform_lateral_coordinates(xxp, yyp, zz, nn):
    Zeta = lambdify((rho, r, z), zeta.subs({rhop: rho, rp: r, xp: x, yp: y, n: nn}))

    XXp, YYp = np.meshgrid(xxp, yyp)
    ρp = np.hypot(XXp, YYp)
    Rp = np.hypot(ρp, zz)

    with np.errstate(divide='ignore'):
        Zeta_eval = Zeta(ρp, Rp, zz)
    Zeta_eval[(XXp == 0) & (YYp == 0)] = 0
    return Zeta_eval * XXp, Zeta_eval * YYp


def E_dipole(x, y, z):
    subs = {k: 2*pi/λ, eps: 1}
    E = lambdify([theta, phi, r], E_cartesian_prime.subs(subs))
    ρ = np.hypot(x, y)
    R = np.hypot(ρ, z)
    θ = np.arctan2(ρ, z)
    φ = np.arctan2(y, x)
    with np.errstate(invalid='ignore', divide='ignore'):
        E_eval = E(θ, φ, R)

    E_eval[np.isinf(E_eval)] = np.nan
    mask = np.isnan(E_eval).nonzero()
    E_eval[mask] = E_eval[(*mask[:-1], mask[-1] + 1)]
    E_eval /= np.nanmax(np.abs(E_eval))
    return E_eval


def fractional_radiosity(S_r, NAn, nn):
    P_tot = integrate(S_r*sin(theta), (phi, 0, 2*pi), (theta, 0, pi))
    P_frac = integrate(S_r*sin(theta), (phi, 0, 2*pi), (theta, 0, asin(NA/n)))
    return (P_frac / P_tot).simplify().evalf(subs={n: nn, NA: NAn})


# %% Plot
abslevels = np.linspace(0.4, 1, 13)
absnorm = mpl.colors.Normalize(*abslevels[[0, -1]])
anglelevels = np.linspace(-np.pi, np.pi, 13)
anglenorm = mpl.colors.Normalize(*anglelevels[[0, -1]])

s = 1.5
θ_m = np.arcsin(0.7 / nn)
ρ_m = d * np.tan(θ_m)
ws = [ρ_m, 2.5e-3]
zs = [d, f_ob]


def _get_data(z, w, s, i):
    xp = yp = np.linspace(-w*s, w*s, 251)
    if i == 1:
        xx, yy = transform_lateral_coordinates(xp, yp, z, nn)
    else:
        xx, yy = np.meshgrid(xp, yp)
    xxp, yyp = np.meshgrid(xp, yp)

    E = E_dipole(xx, yy, z)
    return xp, yp, xxp, yyp, E


def _plot_common(ax, s):
    rect = mpl.patches.Rectangle((-s, -s), 2 * s, 2 * s,
                                 facecolor='white',
                                 edgecolor=None,
                                 hatch='//',
                                 rasterized=PLOT_CONTOURS)
    ax.add_patch(rect)
    ax.plot(np.cos(φ := np.linspace(0, 2 * np.pi, 1001)), np.sin(φ), color='white')
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_xticks(np.arange(-int(s), int(s) + 1))
    ax.set_yticks(np.arange(-int(s), int(s) + 1))
    ax.label_outer()
    match backend:
        case 'pgf':
            ax.set_xlabel(r'$\flatfrac{x}{w}$')
            ax.set_ylabel(r'$\flatfrac{x}{w}$')
        case _:
            ax.set_xlabel('$x/w$')
            ax.set_ylabel('$y/w$')


def _plot_img(ax, xp, yp, xxp, yyp, cc, w, z, levels, norm, cmap, rast=False):
    mask = np.hypot(xxp, + yyp) <= w

    if PLOT_CONTOURS:
        im1 = ax.contourf(xxp / w, yyp / w, np.ma.array(cc, mask=mask),
                          alpha=0.75, levels=levels, cmap=cmap, norm=norm)
        im2 = ax.contourf(xxp / w, yyp / w, np.ma.array(cc, mask=~mask),
                          levels=levels, cmap=cmap, norm=norm)
    else:
        im1 = ax.pcolormesh(xp / w, yp / w, np.ma.array(cc, mask=mask),
                            alpha=0.75, cmap=cmap, norm=norm)
        im2 = ax.pcolormesh(xp / w, yp / w, np.ma.array(cc, mask=~mask),
                            cmap=cmap, norm=norm)

    # Warns but works, https://github.com/matplotlib/matplotlib/pull/29582
    im1.set_rasterized(rast)
    im2.set_rasterized(rast)
    return im1, im2


def _label_cbar_abs(cb):
    cb.set_ticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    match backend:
        case 'pgf':
            cb.set_label(r'$\flatfrac{\abs{E(x, y)}}{\abs{E(0,0)}}$')
        case _:
            cb.set_label('$|E(x, y)|/|E(0,0)|$')


def _label_cbar_arg(cb):
    cb.set_ticks(
        ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        labels=[r'$\mathdefault{-\pi}$', r'$\mathdefault{-\pi/2}$', r'$\mathdefault{0}$',
                r'$\mathdefault{\pi/2}$', r'$\mathdefault{\pi}$'],
        va='center'
    )
    cb.ax.tick_params(axis='x', pad=6.4)
    cb.set_label(r'$\arg E(x, y)$')


# %%% Plot abs only
fig = plt.figure(figsize=(MARGINWIDTH, 4))
grid = ImageGrid(fig, 111, nrows_ncols=(2, 1), share_all=True, aspect=True, axes_pad=0.1,
                 cbar_pad=0.075, cbar_mode='single', cbar_location="top", cbar_size='5%')

ims = []

for i, (ax, z0, w) in enumerate(zip(grid, zs, ws)):
    _plot_common(ax, s)

    *args, E = _get_data(z0, w, s, i)
    ims.append(_plot_img(ax, *args, np.linalg.norm(E[:2], axis=(0, 1)), w, z0, abslevels, absnorm,
                         SEQUENTIAL_CMAPS['magenta']))

_label_cbar_abs(grid.cbar_axes[0].colorbar(ims[0][0], ticks=abslevels[::2]))

fig.savefig(SAVE_PATH / 'modes_2d_abs.pdf')

# %%% Plot arg only
fig = plt.figure(figsize=(MARGINWIDTH, 4))
grid = ImageGrid(fig, 111, nrows_ncols=(2, 1), share_all=True, aspect=True, axes_pad=0.1,
                 cbar_pad=0.075, cbar_mode='single', cbar_location="top", cbar_size='5%')

ims = []

for i, (ax, z0, w) in enumerate(zip(grid, zs, ws)):
    _plot_common(ax, s)

    *args, E = _get_data(z0, w, s, i)
    ims.append(_plot_img(ax, *args, np.angle(E[0, 0]) - np.angle(E[1, 0]), w, z0,
                         anglelevels, anglenorm, CYCLIC_CMAP,
               rast=i == 1))

_label_cbar_arg(grid.cbar_axes[0].colorbar(ims[0][0]))

fig.savefig(SAVE_PATH / 'modes_2d_arg.pdf')

# %%% Plot abs and arg
with mpl.style.context(MAINSTYLE, after_reset=True):
    fig = plt.figure(figsize=(TEXTWIDTH, 4.5))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), share_all=True, aspect=True, axes_pad=0.175,
                     cbar_pad=0.1, cbar_mode='edge', cbar_location="top", cbar_size='5%')

    for i, (axs, z0, w) in enumerate(zip(grid.axes_row, zs, ws)):
        _plot_common(axs[0], s)
        _plot_common(axs[1], s)

        *args, E = _get_data(z0, w, s, i)
        ims1 = _plot_img(axs[0], *args, np.abs(E), w, z0, abslevels, absnorm,
                         SEQUENTIAL_CMAPS['magenta'])
        ims2 = _plot_img(axs[1], *args, np.angle(E), w, z0, anglelevels, anglenorm, CYCLIC_CMAP,
                         rast=i == 1)

    _label_cbar_abs(grid.cbar_axes[0].colorbar(ims1[0]))
    _label_cbar_arg(grid.cbar_axes[1].colorbar(ims2[0]))

    fig.savefig(SAVE_PATH / 'modes_2d_both.pdf')
