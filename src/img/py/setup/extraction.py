import pathlib
import sys
import matplotlib as mpl
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from qutil import math
from qutil.plotting import colors
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_25
from sympy import (sqrt, exp, cos, sin, asin, atan2, symbols, trigsimp, simplify, lambdify,
                   ImmutableDenseMatrix, Function, rot_axis1, rot_axis3, integrate, I, pi, re)

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

# lengths are in mm
# suffix n means numeric value rather than symbol
nn = n_GaAs(0).real
λ = 800e-6
kn = 2*np.pi/λ
dn = 110e-6
NAn = 0.7
CAn = 5
f_ob = 3.1
f_oc = 18.4
MFD = 5e-3

init(MARGINSTYLE, backend := 'pgf')
# %% Sympy
z, zp, r, rp, rho, rhop, d, f, k, c, n, NA, eps, mu = symbols(
    r"z z' r r' rho \rho' d f k c n NA epsilon mu",
    positive=True, real=True
)
x, y, xp, yp, phi, varphi, theta, vartheta, thetap = symbols(
    r"x y x' y' phi varphi theta vartheta \theta'",
    real=True
)
nu = Function('nu', real=True)(theta)  # sqrt(1 - n**2 * sin(theta)**2)
f_r = Function('f_r')(k, r)  # (2/(k*r)**2 - 2*I/(k*r))
f_vartheta = Function(r'f_\vartheta')(k, r)  # (1/(k*r)**2 - I/(k*r) - 1)
f_varphi = Function(r'f_\varphi')(k, r)  # (-I/(k*r) - 1)

csubs = {1/sqrt(eps*mu): c}
nusub = {sqrt(1 - n**2 * sin(theta)**2): nu}
fsubs = {f_r: (2/(k*r)**2 - 2*I/(k*r)),
         f_vartheta: (1/(k*r)**2 - I/(k*r) - 1),
         f_varphi: (-I/(k*r) - 1)}
# %%% Extraction
# Field components of a dipole oriented along z (share a common factor)
E_r = cos(vartheta) * f_r
E_vartheta = sin(vartheta) * f_vartheta
E_varphi = 0

H_r = 0
H_vartheta = 0
H_varphi = sin(vartheta) * f_varphi

# Unit vectors in spherical coordinates
e_r = ImmutableDenseMatrix([[sin(vartheta)*cos(varphi),
                             sin(vartheta)*sin(varphi),
                             cos(vartheta)]]).T
e_vartheta = ImmutableDenseMatrix([[cos(vartheta)*cos(varphi),
                                    cos(vartheta)*sin(varphi),
                                    -sin(vartheta)]]).T
e_varphi = ImmutableDenseMatrix([[-sin(varphi), cos(varphi), 0]]).T
# Rotation matrix to convert form cartesian to spherical coordinates
R_cart_sphe = ImmutableDenseMatrix([e_r.T, e_vartheta.T, e_varphi.T])

# Field vector
E_0 = 1 / (4 * pi * eps) * exp(I*k*r) / r * k**2
H_0 = (E_0 * sqrt(eps/(mu))).subs(csubs).simplify()
E_cartesian = E_0 * simplify(E_r * e_r + E_vartheta * e_vartheta + E_varphi * e_varphi)
E_spherical = simplify(R_cart_sphe * E_cartesian)

H_cartesian = H_0 * simplify(H_r * e_r + H_vartheta * e_vartheta + H_varphi * e_varphi)
H_spherical = simplify(R_cart_sphe * H_cartesian)

S_cartesian = simplify(re(E_cartesian.cross(H_cartesian.conjugate()).subs(fsubs))/2)
S_spherical = simplify(re(E_spherical.cross(H_spherical.conjugate()).subs(fsubs))/2)

assert trigsimp(
    E_spherical - E_0*ImmutableDenseMatrix([E_r, E_vartheta, E_varphi])
) == ImmutableDenseMatrix([0, 0, 0])

# %%% Rotation of the coordinate system
# Rotate coordinate system (xyz) to have z along growth direction and xy plane in the QW; (yzx)
# x' || z
# y' || x
# z' || y
R_cart_cart_prime = rot_axis3(-pi/2) * rot_axis1(-pi/2)

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
total = {
    vartheta: atan2(sqrt(sin(theta)**2*sin(phi)**2 + cos(theta)**2), sin(theta)*cos(phi)),
    varphi: atan2(cos(theta), sin(theta)*sin(phi))
}
# Perform the substitutions to convert between coordinate systems
H_cartesian_prime = trigsimp(R_cart_cart_prime * H_cartesian.subs(total))
S_cartesian_prime = trigsimp(R_cart_cart_prime * S_cartesian.subs(total))
E_cartesian_prime = trigsimp(E_cartesian.subs(cartesian).subs(trafo).subs(spherical))
# Finally switch axes of the vector
E_cartesian_prime = ImmutableDenseMatrix([E_cartesian_prime[2],
                                          E_cartesian_prime[0],
                                          E_cartesian_prime[1]])
# Just a sanity check
assert (
    trigsimp(E_cartesian_prime - R_cart_cart_prime * E_cartesian.subs(total))
    == ImmutableDenseMatrix([0, 0, 0])
)
# Another reality check
assert (E_cartesian.dot(e_r) / E_0 - E_r).simplify() == 0
assert (E_cartesian.dot(e_vartheta) / E_0 - E_vartheta).simplify() == 0
assert (E_cartesian.dot(e_varphi) / E_0 - E_varphi).simplify() == 0

e_r_prime = e_r.subs({vartheta: theta, varphi: phi})
e_theta_prime = e_vartheta.subs({vartheta: theta, varphi: phi})
e_phi_prime = e_varphi.subs({vartheta: theta, varphi: phi})
R_cart_prime_sphe_prime = ImmutableDenseMatrix([e_r_prime.T, e_theta_prime.T, e_phi_prime.T])

E_spherical_prime = E_0 * ImmutableDenseMatrix(
    [(E_cartesian_prime.dot(e_r_prime) / E_0).simplify(),
     (E_cartesian_prime.dot(e_theta_prime) / E_0).simplify(),
     (E_cartesian_prime.dot(e_phi_prime) / E_0).simplify()]
)
H_spherical_prime = H_0 * ImmutableDenseMatrix(
    [(H_cartesian_prime.dot(e_r_prime) / H_0).simplify(),
     (H_cartesian_prime.dot(e_theta_prime) / H_0).simplify(),
     (H_cartesian_prime.dot(e_phi_prime) / H_0).simplify()]
)
S_spherical_prime = ImmutableDenseMatrix(
    [S_cartesian_prime.dot(e_r_prime).simplify(),
     S_cartesian_prime.dot(e_theta_prime).simplify(),
     S_cartesian_prime.dot(e_phi_prime).simplify()]
)

assert (
    trigsimp(E_spherical_prime - R_cart_prime_sphe_prime * E_cartesian_prime)
    == ImmutableDenseMatrix([0, 0, 0])
)
# Radial component is invariant, so simple substition of angles should give the correct result
assert (S_spherical_prime - S_spherical.subs(total)).simplify() == ImmutableDenseMatrix([0, 0, 0])
# %%% Fresnel transmission at the surface
t_s = 2 * cos(theta) * sin(thetap) / sin(theta + thetap)
t_p = t_s / cos(theta - thetap)

snellius = {thetap: asin(n * sin(theta))}
t_s_prime = t_s.subs(snellius).simplify().subs(nusub)
t_p_prime = t_p.subs(snellius).simplify().subs(nusub)

# Transmitted quantities. Awkward variable convention
snellius = {theta: asin(n * sin(theta))}
e_r_prime_trans = e_r_prime.subs(snellius).subs(nusub)
e_theta_prime_trans = e_theta_prime.subs(snellius).subs(nusub)
e_phi_prime_trans = e_phi_prime.subs(snellius).subs(nusub)

E_spherical_prime_trans = ImmutableDenseMatrix([
    simplify(t_p_prime * E_spherical_prime[0].subs(snellius).subs(nusub)),
    simplify(t_p_prime * E_spherical_prime[1].subs(snellius).subs(nusub)),
    simplify(t_s_prime * E_spherical_prime[2].subs(snellius).subs(nusub))
])
H_spherical_prime_trans = ImmutableDenseMatrix([
    simplify(t_p_prime * H_spherical_prime[0].subs(snellius).subs(nusub)),
    simplify(t_p_prime * H_spherical_prime[1].subs(snellius).subs(nusub)),
    simplify(t_s_prime * H_spherical_prime[2].subs(snellius).subs(nusub))
])
E_cartesian_prime_trans = simplify(
    E_spherical_prime[0] * e_r_prime_trans
    + E_spherical_prime[1] * e_theta_prime_trans
    + E_spherical_prime[2] * e_phi_prime_trans
)
H_cartesian_prime_trans = simplify(
    H_spherical_prime[0] * e_r_prime_trans
    + H_spherical_prime[1] * e_theta_prime_trans
    + H_spherical_prime[2] * e_phi_prime_trans
)
S_cartesian_prime_trans = simplify(
    re(E_cartesian_prime_trans.cross(H_cartesian_prime_trans.conjugate()).subs(fsubs))/2
)
S_spherical_prime_trans = simplify(
    re(E_spherical_prime_trans.cross(H_spherical_prime_trans.conjugate()).subs(fsubs))/2
)
# %%% Collimation
subs = {nu: sqrt(1 - n**2 * sin(theta)**2)}
# The lens collimates, which means the transverse components along θ and φ turn into x and y by
# evaluating the unit vectors for θ = 0.
E_cartesian_prime_trans_coll = (
    # The radial component falls off with 1/kr and is therefore negligible in the far field
    (E_spherical_prime_trans[0] * e_r_prime_trans.subs(subs | {theta: 0}))
    + (E_spherical_prime_trans[1] * e_theta_prime_trans.subs(subs | {theta: 0}))
    + (E_spherical_prime_trans[2] * e_phi_prime_trans.subs(subs | {theta: 0}))
).simplify()

E_cartesian_prime_trans_coll_approx = ImmutableDenseMatrix([
    E_cartesian_prime_trans_coll[0].subs({
        f_vartheta: -1,  # other components fall off with at least 1/r
        exp(I*k*r): exp(I*k*z)  # spherical wave front is converted to a plane wave front
    }),
    E_cartesian_prime_trans_coll[1].subs({
        f_vartheta: -1,  # other components fall off with at least 1/r
        exp(I*k*r): exp(I*k*z)  # spherical wave front is converted to a plane wave front
    }),
    E_cartesian_prime_trans_coll[2].subs({
        f_r: 0,  # other components fall off with at least 1/r
        exp(I*k*r): exp(I*k*z)  # spherical wave front is converted to a plane wave front
    }),
])
# %% Functions


def _to_spherical(x, y, z):
    ρ = np.hypot(x, y)
    R = np.hypot(ρ, z)
    θ = np.arctan2(ρ, z)
    φ = np.arctan2(y, x)
    return R, θ, φ


def _sanitize(E):
    E[np.isinf(E)] = np.nan
    if (nan := np.isnan(E)).all():
        E[nan] = 1
    elif nan.any():
        mask = nan.nonzero()
        E[mask] = E[(*mask[:-1], mask[-1] + 1)]
    E /= np.nanmax(np.linalg.norm(E, axis=(0,) if np.ndim(E) == 1 else (0, 1)))
    return E


def E_dipole(x, y, z, nn):
    # Field inside the sample in the rotated coordinate system
    R, θ, φ = _to_spherical(x, y, z)

    subs = {k: 2*pi/λ, eps: 1, n: nn, nu: sqrt(1 - nn**2 * sin(theta)**2)}
    E = lambdify([r, theta, phi], E_spherical_prime.subs(fsubs).subs(subs))
    with np.errstate(invalid='ignore', divide='ignore'):
        E_eval = E(R, θ, φ)

    return _sanitize(E_eval)


def E_collimated(x, y, fn, nn):
    # Field ε after the collimating lens
    R, θp, φ = _to_spherical(x, y, fn)
    θ = np.arcsin(np.sin(θp) / nn)

    subs = {k: 2*pi/λ, eps: 1, n: nn, nu: sqrt(1 - nn**2 * sin(theta)**2), z: fn}
    E = lambdify([r, theta, phi], E_cartesian_prime_trans_coll_approx.subs(subs)[:2])
    with np.errstate(invalid='ignore', divide='ignore'):
        E_eval = E(R, θ, φ)

    E_eval = np.stack(E_eval + [np.zeros(R.shape, dtype=complex)])[:, None]
    return _sanitize(E_eval)


def E_dipole_radial_func(fn, nn):
    subs1 = {k: 2*pi/λ, eps: 1, nu: sqrt(1 - nn**2 * sin(theta)**2), z: fn,
             r: sqrt(fn**2 + rho**2),
             # the integrator is not smart enough to recognize the integrals averaging over phi are
             # actually very simple, so we just manually replace the phi-dependence with the result
             sin(phi)**2: pi, cos(phi)**2: pi}
    subs2 = {theta: asin(sin(theta) / n)}
    subs3 = {theta: atan2(rho, fn), n: nn}
    return lambdify(
        [rho],
        E_cartesian_prime_trans_coll_approx[0].subs(fsubs).subs(subs1).subs(subs2).subs(subs3)
    )


def E_gaussian(q, w_0, k, z=0, n=1):
    z = np.atleast_1d(z)
    with np.errstate(divide='ignore', invalid='ignore'):
        λ = 2 * np.pi / k
        z_0 = rayleigh_range(2*w_0, λ / n)
        w = w_0 * np.sqrt(1 + z**2 / z_0**2)
        R = z * (1 + z_0**2 / z**2)
    if np.size(z) > 1:
        R[np.isnan(R)] = np.inf
    elif np.isnan(R):
        R = np.inf

    return w_0 / w * np.exp(
        -1j * (k * z - np.arctan2(z, z_0))
        - q**2 * (1 / w**2 + 1j * k / (2 * R))
    )


def E_flattop(q):
    return np.ones_like(q)


def E_airy(q, R, a, k, E_fun):
    # Hecht: Optics (2017) Sec. 10.2.5
    # ρ is the radial dimension in the aperture
    # q is the radial dimension on the screen
    # R is the distance between screen and aperture
    # a is the radius of the aperture

    def _aperture(ρ, q, R):
        return ρ*sp.special.j0(k*q*ρ/R)

    return sp.integrate.quad_vec(lambda ρ: _aperture(ρ, q, R)*E_fun(ρ), 0, a)[0]


def E_airy_flattop(q, f_oc, w, k):
    with np.errstate(divide='ignore', invalid='ignore'):
        E = 2 * np.pi * w**2 / f_oc * math.cexp(k * f_oc) * sp.special.j1(x := k*w*q/f_oc) / x
    E /= np.pi * w**2 / f_oc  # normalize to 1 at center
    return _sanitize(E)


def P_cumulative(fn, q, *args):
    E = fn(q, *args)
    return sp.integrate.cumulative_simpson(q*np.abs(E)**2, x=q, initial=0)


def rayleigh_range(MFD, λ):
    return np.pi/λ*(MFD/2)**2


def beam_diameter_gaussian(MFD, z, λ):
    return MFD * np.sqrt(1 + (z/rayleigh_range(MFD, λ))**2)


def fractional_radiosity(S_r):
    P_tot = integrate(S_r*sin(theta), (phi, 0, 2*pi), (theta, 0, pi))
    P_frac = integrate(S_r*sin(theta), (phi, 0, 2*pi), (theta, 0, asin(NA/n)))
    return (P_frac / P_tot).simplify()


def mode_overlap_dipole(w0, w, f_oc, k, f_ob, n, N=1001):
    # https://www.rp-photonics.com/mode_matching.html
    def i1(q, w0, k):
        return 2*np.pi*q*math.abs2(E_gaussian(q, w0, k).squeeze())

    def i2(q, w, f_oc, k, E_fun):
        return 2*np.pi*q*math.abs2(E_airy(q, f_oc, w, k, E_fun).squeeze())

    def i3(q, w0, w, f_oc, k, E_fun):
        return (2*np.pi*q*E_gaussian(q, w0, k)*E_airy(q, f_oc, w, k, E_fun)).squeeze()

    # quadrature integration does not converge in a reasonable amount of time
    q = np.geomspace(1e-4, 1000*w0, N-1)
    q = np.insert(q, 0, 0)
    I1 = sp.integrate.simpson(i1(q, w0, k), q)
    I2 = sp.integrate.simpson(i2(q, w, f_oc, k, E_dipole_radial_func(f_ob, n)), q)
    I3 = sp.integrate.simpson(i3(q, w0, w, f_oc, k, E_dipole_radial_func(f_ob, n)), q)
    return math.abs2(I3) / I1 / I2


def mode_overlap_flattop(w0, w, f_oc, k, n, N=1001):
    # https://www.rp-photonics.com/mode_matching.html
    def i1(q, w0, k):
        return 2*np.pi*q*math.abs2(E_gaussian(q, w0, k).squeeze())

    def i2(q, w, f_oc, k, n):
        return 2*np.pi*q*math.abs2(E_airy_flattop(q, f_oc, w, k).squeeze())

    def i3(q, w0, w, f_oc, k, n):
        return (2*np.pi*q*E_gaussian(q, w0, k)*E_airy_flattop(q, f_oc, w, k)).squeeze()

    # quadrature integration does not converge in a reasonable amount of time
    q = np.geomspace(1e-4, 1000*w0, N-1)
    q = np.insert(q, 0, 0)
    I1 = sp.integrate.simpson(i1(q, w0, k), q)
    I2 = sp.integrate.simpson(i2(q, w, f_oc, k, n), q)
    I3 = sp.integrate.simpson(i3(q, w0, w, f_oc, k, n), q)
    return math.abs2(I3) / I1 / I2


def collection_efficiency(NA, n):
    return (1/8)*(4*n**3 + np.sqrt(-NA**2 + n**2)*(NA**2 - 4*n**2))/n**3


# %% Print measured efficiencies
# Measured values
P_down = 900e-9
P_up = 340e-9
P_ana = 230e-9
P_fiber = 90e-9
P_spect = 20e-9

η_up = P_up/P_down
η_ana = P_ana/P_up
η_fiber = P_fiber/P_ana
η_spect = P_spect/P_fiber
η_total = η_up * η_ana * η_fiber * η_spect

print(f'η_up = {η_up:.3g}')
print(f'η_ana = {η_ana:.3g}')
print(f'η_fiber = {η_fiber:.3g}')
print(f'η_spect = {η_spect:.3g}')
print(f'η_total = {η_total:.3g}')
# %% Plot
abslevels = np.linspace(0.4, 1, 13)
absnorm = mpl.colors.Normalize(*abslevels[[0, -1]])

s = 1.5
θ_m = np.arcsin(NAn / nn)
ρ_m = dn * np.tan(θ_m)
w = CAn / 2
w0 = MFD / 2
ws = [ρ_m, w]
zs = [dn, f_ob]
# %%% Compute values
result = sp.optimize.minimize_scalar(lambda f: (1 - mode_overlap_flattop(w0, w, f, kn, nn))**2,
                                     bounds=[1e-3, 1e3])
f_opt = result.x

print("Mode matching (flattop, f_oc=6.2mm): "
      f"η = {mode_overlap_flattop(w0, w, 6.2, kn, nn):.2g}")
print("Mode matching (flattop, f_oc=18.4mm): "
      f"η = {mode_overlap_flattop(w0, w, f_oc, kn, nn):.2g}")
print("Mode matching (flattop, f_oc=opt.): "
      f"η = {mode_overlap_flattop(w0, w, f_opt, kn, nn):.2g}")
print("Mode matching (dipole, f_oc=18.4mm): "
      "η = {:.2g}".format(η_m := mode_overlap_dipole(w0, w, f_oc, kn, 3.1, nn)))
print("Collection efficiency: η = {:.2g}".format(η_c := collection_efficiency(NAn, nn)))

T = .87
print(f"Total efficiency: T η_c η_m = {T*η_c*η_m:.3g}")

# %%% Functions


def _get_data(z, w, s, nn, i):
    x = y = np.linspace(-w*s, w*s, 251)
    xx, yy = np.meshgrid(x, y)

    if i == 0:
        E = E_dipole(xx, yy, z, nn)
    else:
        E = E_collimated(xx, yy, z, nn)
    return x, y, xx, yy, E


def _plot_common(ax, s, sub):
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
    # ax.label_outer()
    match backend:
        case 'pgf':
            ax.set_xlabel(rf'$\flatfrac{{x}}{{w_{sub}}}$')
            ax.set_ylabel(rf'$\flatfrac{{y}}{{w_{sub}}}$')
        case _:
            ax.set_xlabel(f'$x/w_{sub}$')
            ax.set_ylabel(f'$y/w_{sub}$')


def _plot_img(ax, xp, yp, xxp, yyp, cc, w, z, levels, norm, cmap):
    if PLOT_CONTOURS:
        img = ax.contourf(xxp / w, yyp / w, cc, levels=levels, cmap=cmap, norm=norm)
    else:
        img = ax.pcolormesh(xp / w, yp / w, cc, cmap=cmap, norm=norm)
    return img


def _plot_mask(ax, xxp, yyp, w, levels, norm, cmap):
    mask = np.hypot(xxp, + yyp) <= w
    img = ax.contourf(xxp / w, yyp / w, np.ma.array(np.ones_like(xxp), mask=mask), alpha=0.5,
                      levels=levels, cmap=cmap, norm=norm)
    return img


def _plot_abs(ax, xp, yp, xxp, yyp, E, w, z, levels, norm, cmap):
    im1 = _plot_img(ax, *args, np.linalg.norm(E, axis=(0, 1)), w, z0, levels, norm, cmap)

    step = 25
    ax.quiver(xxp[::step, ::step] / w, yyp[::step, ::step] / w,
              np.abs(E[0, 0, ::step, ::step]), np.abs(E[1, 0, ::step, ::step]),
              angles='xy', pivot='middle')

    im2 = _plot_mask(ax, xxp, yyp, w, levels, norm, cmap)

    return im1, im2


def _label_cbar_abs(cb):
    match backend:
        case 'pgf':
            cb.set_label(r'$\flatfrac{\abs{\symbf{E}(x, y, z)}}{\max\abs{\symbf{E}(x,y,z)}}$')
        case _:
            cb.set_label(r'$|\mathbf{E}(x, y, z)|/\max|\mathbf{E}(x,y,z)|$')


# %%% Plot absolute value of field in the sample plane and the objective lens plane
fig = plt.figure(figsize=(MARGINWIDTH, 4))
grid = ImageGrid(fig, 111, nrows_ncols=(2, 1), share_all=True, aspect=True, axes_pad=0.1,
                 cbar_pad=0.075, cbar_mode='single', cbar_location="top", cbar_size='5%')

ims = []

for i, (ax, z0, w, sub) in enumerate(zip(grid, zs, ws, ['i', 'f'])):
    _plot_common(ax, s, sub)

    *args, E = _get_data(z0, w, s, nn, i)
    ims.append(_plot_abs(ax, *args, E, w, z0, abslevels, absnorm, SEQUENTIAL_CMAPS['magenta']))

_label_cbar_abs(grid.cbar_axes[0].colorbar(ims[0][0], ticks=abslevels[::2]))

fig.savefig(SAVE_PATH / 'modes_2d.pdf')

# %%% Plot mode profile in aperture and on screen
kn = 2*np.pi/λ
w0 = MFD / 2
w = CAn / 2
fill_style = dict(alpha=0.5, color=RWTH_COLORS_25['black'], hatch='//')

fig, axs = plt.subplots(nrows=3, layout='constrained', figsize=(MARGINWIDTH, 3))

# radial intensity profile
ρ = np.linspace(0, 3*w, 1001)
ax = axs[0]
ax.plot(ρ[ρ <= w] / w,
        (x := abs(E_dipole_radial_func(f_ob, nn)(ρ))**2)[ρ <= w]/x[0], color=RWTH_COLORS['blue'])
ax.plot(ρ[ρ > w] / w,
        abs(E_dipole_radial_func(f_ob, nn)(ρ))[ρ > w]**2/x[0], '--', color=RWTH_COLORS['blue'])
ax.plot(ρ[ρ <= w] / w, ρ[ρ <= w] < w, color=RWTH_COLORS['magenta'])
ax.plot(ρ[ρ > w] / w, ρ[ρ > w] <= w, '--', color=RWTH_COLORS['magenta'])
ax.plot(
    ρ[ρ <= w] / w,
    (x := abs(E_gaussian(ρ, beam_diameter_gaussian(MFD, f_oc, λ)/2, kn)))
    [ρ <= w]/x[0],
    color=RWTH_COLORS['green']
)
ax.plot(ρ[ρ > w] / w,
        abs(E_gaussian(ρ, beam_diameter_gaussian(MFD, f_oc, λ)/2, kn))[ρ > w]/x[0],
        '--', color=RWTH_COLORS['green'])

ax.set_xlim(xlim := ax.get_xlim())
ax.set_ylim(ylim := ax.get_ylim())
ax.fill_between([1, xlim[1]], *ylim, **fill_style)
ax.set_xlabel(r'$\rho/w_f$')
ax.set_ylabel(r'$I(\rho)/I(0)$')

# diffraction pattern and gaussian mode
q = np.linspace(0, 3*w0, 1001)
ax = axs[1]
ax.plot(q / w0, q*(x := abs(E_airy(q, f_oc, w, kn, E_dipole_radial_func(f_ob, nn)))**2)/(x[0]*w0),
        color=RWTH_COLORS['blue'])
ax.plot(q / w0, q*(x := abs(E_airy_flattop(q, f_oc, w, kn))**2)/(x[0]*w0),
        color=RWTH_COLORS['magenta'])
ax.plot(q / w0, q*(x := abs(E_gaussian(q, w0, kn))**2)/(x[0]*w0), color=RWTH_COLORS['green'])

ax.set_xlim(xlim := ax.get_xlim())
ax.set_ylim(ylim := ax.get_ylim())
ax.set_xticks([0, 1, 2, 3], labels=[])
ax.set_ylabel(r'$\rho I(\rho)/(w_0 I(0))$')

# power included in circle of radius q
q = np.linspace(0, MFD*10, 1001)
ax = axs[2]
ax.sharex(axs[1])
ax.plot(q / w0, (x := P_cumulative(E_airy, q, f_oc, w, kn, E_dipole_radial_func(f_ob, nn)))/x[-1],
        color=RWTH_COLORS['blue'])
ax.plot(q / w0, (x := P_cumulative(E_airy_flattop, q, f_oc, w, kn))/x[-1],
        color=RWTH_COLORS['magenta'])
ax.plot(q / w0, (x := P_cumulative(E_gaussian, q, w0, kn))/x[-1],
        color=RWTH_COLORS['green'])

ax.set_xlim(xlim)
ax.set_ylim(ylim := ax.get_ylim())
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel(r'$\rho/w_0$')
ax.set_ylabel(r'$P(\rho)/P(\infty)$')

fig.savefig(SAVE_PATH / 'modes_1d.pdf')
