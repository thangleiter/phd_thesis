# %% Imports
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker, colors as mcolors
from matplotlib.lines import Line2D
from qutil import math
from qutil.functools import partial
from qutil.misc import filter_warnings
from qutil.plotting import colors
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_25
from scipy import integrate, special

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, PATH, TEXTWIDTH, TOTALWIDTH, init,  # noqa
                    markerprops, n_GaAs)

DATA_PATH = PATH.parent / 'data/lenses'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

with np.errstate(divide='ignore'):
    SEQUENTIAL_CMAPS = {
        color: colors.make_sequential_colormap(color, endpoint='blackwhite')
        for color in RWTH_COLORS if color != 'black'
    }

init(MARGINSTYLE, backend := 'pgf')

# Everything is in mm!
n = n_GaAs(0).real
λ0 = 800e-6
λ = λ0/n
k0 = 2*np.pi/λ0
k = 2*np.pi/λ
MFD = 5e-3
w0 = MFD / 2
AR_COATING = 'B'
# %% Functions


def sm_fiber_angle(MFD, λ):
    # https://www.sukhamburg.com/support/technotes/fiberoptics/coupling/collimatingsm/divergence.html
    # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14489
    # Far field, z >> z_R
    return np.arctan(2*λ/(np.pi*MFD))


def lens_divergence_angle(D, f):
    # Same as SM fiber angle with D == MFD
    return lens_angle_geom(D, f)


def lens_angle_geom(D, f):
    with np.errstate(divide='ignore'):
        return np.arctan(D/(2*f))


def lens_na_geom(D, f):
    return np.sin(lens_angle_geom(D, f))


def lens_focal_length_geom(D, NA):
    with np.errstate(divide='ignore'):
        return D/(2*np.tan(np.arcsin(NA)))


def lens_focal_length_diffraction(D, d, λ):
    return np.pi*D*d/(4*λ)


def lens_focal_length_divergence_angle(D, θ):
    return lens_focal_length_geom(D, np.sin(θ))


def diffraction_limit(D, f, λ):
    return 4*λ*f/(np.pi*D)


def rayleigh_range(MFD, λ):
    return np.pi/λ*(MFD/2)**2


def beam_diameter_gaussian(MFD, z, λ):
    return MFD * np.sqrt(1 + (z/rayleigh_range(MFD, λ))**2)


def collimated_beam_diameter_gaussian(f, MFD, λ, z=0):
    return beam_diameter_gaussian(diffraction_limit(MFD, f, λ), z, λ)


def beam_diameter_geom(f, NA, z=None):
    return 2*f*np.tan(np.arcsin(NA))


def optimal_focal_length(D, MFD, λ):
    return np.pi*D*MFD/(4*λ)


def instantaneous_divergence_gaussian(MFD, f, λ, z=None):
    w_0 = MFD/2
    if z is None:
        return λ*f/(np.pi*w_0)
    return np.arctan(np.pi*z*w_0**3/(f**3*λ*np.sqrt(1 + np.pi**2*z**2*w_0**4/(f**4*λ**2))))


def instantaneous_diameter_gaussian(θ, MFD, λ, z=None):
    w_0 = MFD/2
    if z is None:
        return 2*λ/(np.pi*np.tan(θ))
    return np.tan(f**3*θ*λ*np.sqrt(-1/((f*θ - w_0)*(f*θ + w_0)))/(np.pi*w_0**2))


# %% Load catalogs
# %%% Thorlabs
# HTML is copied from the website source code ("Selection Guide" tab) at
# https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=3811
dfs = pd.read_html(DATA_PATH / 'thorlabs_3811.html', index_col=0, header=[1, 2], encoding='utf-8')
df = dfs[0]

CA = df['Clear Aperture of Unmounted Lens']['Clear Aperture of Unmounted Lens'].str.split(
    ' S2: Ø', expand=True
)
S1 = (
    CA[0]
    .str.removeprefix('S1:\xa0Ø')
    .str.removeprefix('S1: Ø')
    .str.removeprefix('Ø')
)
S2 = CA[1]

WD_U = df['Working Distance', 'Unmounted'].str.split().str[0]
WD_M = df['Working Distance', 'Mounted'].str.split().str[0]

df[('Clear Aperture of Unmounted Lens', 'S1')] = S1
df[('Clear Aperture of Unmounted Lens', 'S2')] = S2
df[('Working Distance', 'Unmounted')] = WD_U
df[('Working Distance', 'Mounted')] = WD_M
df.drop(columns=('Clear Aperture of Unmounted Lens', 'Clear Aperture of Unmounted Lens'),
        inplace=True)
df.drop(index='355440', inplace=True)  # not plano-convex, don't want to deal with that
df.drop(df[df['AR Coating Options'][AR_COATING] != 'X'].index, inplace=True)
df.drop(columns='AR Coating Options', inplace=True, level=0)
df.replace(to_replace={None: np.nan}, inplace=True)
for name, col in df.items():
    try:
        df[name] = col.str.rstrip(' mm').astype(float)
    except ValueError:
        pass

df_thorlabs = df
# %%% Edmund
# XLSX is exported from
# https://www.edmundoptics.com/f/lightpath-geltech-molded-aspheric-lenses/14752/
df = pd.read_excel(DATA_PATH / 'LightPath-Geltech-Molded-Aspheric-Lenses.xlsx',
                   index_col='Lightpath Lens Code')
df.drop(columns=[
    'Title', 'Dimensions (mm)', 'Substrate', 'FL Tolerance (%)',
    'Coating Specification', 'Asphere Figure Error (μm RMS)', 'Surface Quality', 'ET (mm)',
    'CT (mm)', 'f/#', 'Bevel', 'Compatible Window', 'Distance from Window to Lens (D) (mm)',
    'Operating Temperature (°C)', 'Type', 'Wavelength Range (nm)', 'Typical Applications',
    'Conjugate Distance', 'Transmitted Wavefront, P-V', 'Note',
    'Transmitted Wavefront Error (λ, RMS)', 'RoHS', 'Certificate of Conformance', 'RoHS 2015',
    'Reach 240', 'Reach 233'
], inplace=True)
df.dropna(subset='Dia. (mm)', inplace=True)

df['EFL (mm)'] = df['EFL (mm)'].str.split().str[0]
df['Dia. (mm)'] = df['Dia. (mm)'].str.split().str[0]
df['Coating'] = df['Coating'].str.rstrip()

coatings = {
    'BBAR (350-700nm)': 'A',
    'BBAR (600-1050nm)': 'B',
    'BBAR (1050-1600nm)': 'C',
    'VIS': 'A'
}
df.replace({'Coating': coatings}, inplace=True)
df = df[df['Coating'] == AR_COATING]
df.drop(columns='Coating', inplace=True)
df.index = df.index.astype(str)
df.drop(index='355440', inplace=True)  # not plano-convex, don't want to deal with that

for name, col in df.items():
    try:
        df[name] = col.astype(float)
    except ValueError:
        pass

df_edmund = df

# %%% Merge
df_edmund_compat = pd.DataFrame({
    ('Effective Focal Length', 'Effective Focal Length'): df_edmund['EFL (mm)'],
    ('NA', 'NA'): df_edmund['NA'],
    ('Outer Diameter of Unmounted Lens', 'Outer Diameter of Unmounted Lens'):
        df_edmund['Dia. (mm)'],
    ('Working Distance', 'Unmounted'): df_edmund['WD (mm)'],
    ('Clear Aperture of Unmounted Lens', 'S2'): df_edmund['CA (mm)'],

})

df_all = pd.concat([df_thorlabs, df_edmund_compat], verify_integrity=False)
df_all = df_all[~df_all.index.duplicated(keep='first')]

# %%% Select
available_lenses = df_all[
    df_all[('Clear Aperture of Unmounted Lens', 'S2')] <= collimated_beam_diameter_gaussian(
        df_all['Effective Focal Length']['Effective Focal Length'].max(),
        MFD, λ0
    )
]
available_lenses[[('Effective Focal Length', 'Effective Focal Length'),
                  ('Working Distance', 'Mounted'),
                  ('Working Distance', 'Unmounted'),
                  ('NA', 'NA')]].sort_values(by=('NA', 'NA'))

# %% NA contours


def parse_labels(label):
    if backend != 'pgf':
        return label
    if label in df_thorlabs.index:
        url = f'https://www.thorlabs.com/thorproduct.cfm?partnumber={label}-{AR_COATING}'
    elif label in df_edmund_compat.index:
        stock_number = df_edmund.loc[label]['Stock Number']
        url = f'https://www.edmundoptics.com/search/?criteria={stock_number}'
    return rf'\href{{{url}}}{{{label}}}'


def plot_lens_choosing(df, D_min=1.5, D_max=5.3, f_max=30, dipole: bool = True, ylog: bool = False,
                       fill_cbar: bool = True, cbar_loc: str = 'bottom', figsize=(4, 3),
                       legendfontsize='small'):

    def plot_markers(ax, gaussian):
        handles, labels = [], []
        for i, (model, row) in enumerate(df.sort_index().iterrows()):
            f, NA, S1, S2, WD_u, _ = row[
                ['Effective Focal Length', 'NA', 'Clear Aperture of Unmounted Lens',
                 'Working Distance']
            ]
            # TODO
            if gaussian:
                with filter_warnings(action='ignore', category=RuntimeWarning):
                    D0 = beam_diameter_gaussian(MFD, z=f, λ=λ0)
                    D1 = np.nanmin((D0, S1, S2))
                    if not np.isnan([S1, S2]).any():
                        D2 = D1 / min(S1, S2) * max(S1, S2)
                    else:
                        D2 = D1
                    D = beam_diameter_gaussian(D2, z=1.5e3, λ=λ0)
            else:
                with filter_warnings(action='ignore', category=RuntimeWarning):
                    D = np.nanmax((S1, S2))
            if not np.isnan((S1, S2)).all():
                edgecolors = 'black'
            else:
                edgecolors = 'red'
            if D > D_max or D < D_min:
                continue

            h = ax.scatter(f, D, label=model, zorder=5, marker=markers[i], color=colors[i],
                           edgecolors=edgecolors)
            handles.append(h)
            labels.append(model)
        return handles, labels

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, layout='constrained', figsize=figsize)

    D = np.geomspace(D_min, D_max + 1, 1001)
    f = np.geomspace(1, f_max, 1001)
    DD, ff = np.meshgrid(D, f)

    markers = Line2D.filled_markers * (len(df) // len(Line2D.filled_markers) + 1)
    colors = (mpl.color_sequences['rwth100'][:-1]
              + mpl.color_sequences['rwth75'][:-1]
              + mpl.color_sequences['rwth50'][:-1]
              + mpl.color_sequences['rwth25'][:-1])
    colors = colors * (len(df) // len(colors) + 1)

    # Excitation: Gaussian
    MFD = 5e-3
    ax = axes[0]
    ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    cs1 = ax.contour(ff, DD, diffraction_limit(DD, ff, λ0)*1e3, levels=np.linspace(0, 5.5, 12),
                     cmap=SEQUENTIAL_CMAPS['green'])

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('$f$ (mm)')
    ax.set_ylabel('$D$ (mm)')
    ax.grid(True, which='both', axis='both', alpha=0.5)

    handles1, labels1 = plot_markers(ax, gaussian=True)

    ax.set_ylim(ax.get_ylim())
    ax.plot(f, collimated_beam_diameter_gaussian(f, MFD, λ0, z=1.5e3), ls='--', color='tab:grey')

    # Objective detection
    # TODO
    MFD = 3e-3
    ax = axes[1]

    cs2 = ax.contour(ff, DD, lens_na_geom(DD, ff), levels=np.linspace(0, 1, 11),
                     cmap=SEQUENTIAL_CMAPS['magenta'])

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('$f$ (mm)')
    ax.grid(True, which='both', axis='both', alpha=0.5)

    handles2, labels2 = plot_markers(ax, gaussian=False)

    if cbar_loc:
        match cbar_loc:
            case 'bottom':
                axins1 = axes[1].inset_axes([1.05, 0.0, 0.45, 0.05])
                axins2 = axes[1].inset_axes([1.05, 0.15, 0.45, 0.05])
            case 'side vertical':
                axins1 = axes[1].inset_axes([1.05, 0.0, 0.05, 0.475])
                axins2 = axes[1].inset_axes([1.05, 0.525, 0.05, 0.475])
            case _:
                axins1 = axes[0].inset_axes([0.00, 1.05, 1.00, 0.05])
                axins2 = axes[1].inset_axes([0.00, 1.05, 1.00, 0.05])

        if fill_cbar:
            norm1 = mcolors.Normalize(vmin=cs1.cvalues.min(), vmax=cs1.cvalues.max())
            norm2 = mcolors.Normalize(vmin=cs2.cvalues.min(), vmax=cs2.cvalues.max())
            m1 = plt.cm.ScalarMappable(norm=norm1, cmap=cs1.cmap)
            m2 = plt.cm.ScalarMappable(norm=norm2, cmap=cs2.cmap)
        else:
            m1 = cs1
            m2 = cs2

        cbar1 = fig.colorbar(
            m1, cax=axins1,
            orientation='horizontal' if 'vertical' not in cbar_loc else 'vertical',
            ticklocation='top' if 'vertical' not in cbar_loc else 'right',
            ticks=ticker.FixedLocator(cs1.levels, nbins=10)
        )
        cbar2 = fig.colorbar(
            m2, cax=axins2,
            orientation='horizontal' if 'vertical' not in cbar_loc else 'vertical',
            ticklocation='top' if 'vertical' not in cbar_loc else 'right',
            ticks=ticker.FixedLocator(cs2.levels, nbins=10)
        )
        cbar1.set_label('Spot size diffraction limit (μm)')
        cbar2.set_label('Geometric NA')

    axes[0].clabel(cs1, manual=((lens_focal_length_diffraction(D_max + 0.5, d*1e-3, λ0),
                                 D_max + 0.5) for d in cs1.levels[1:-2]))
    axes[1].clabel(cs2, manual=((lens_focal_length_geom(D_max + 0.5, NA), D_max + 0.5)
                                for NA in cs2.levels[1:-1]))

    handles, labels = [], []
    for h, l in zip(handles1 + handles2, labels1 + labels2):
        if (label := parse_labels(l)) not in labels:
            handles.append(h)
            labels.append(label)
    if 'vertical' in cbar_loc:
        lax = axes[1].inset_axes([1.15, 0.0, 0.1, 1.0])
        lax.axis('off')
    else:
        lax = axes[1]

    leg = lax.legend(handles=handles, labels=labels,
                     loc='upper left', bbox_to_anchor=(1.0, 1.29), borderaxespad=0., ncol=2,
                     frameon=False, title='Lens codes', fontsize=legendfontsize)

    return fig, axes, leg


# %%% Plot
with mpl.style.context(MAINSTYLE, after_reset=True):
    fig, axes, leg = plot_lens_choosing(df_all, ylog=False, fill_cbar=True, cbar_loc='top',
                                        D_min=1.25, figsize=(TOTALWIDTH, 3.4),
                                        legendfontsize='x-small')
    fig.savefig(SAVE_PATH / 'choosing.pdf')

# %% Mode matching
lenses = {'ob': df_all.loc['354330'],
          'oc': df_all.loc['A280']}
f = {'ob': lenses['ob']['Effective Focal Length'].item(),
     'oc': lenses['oc']['Effective Focal Length'].item()}
w = lenses['ob']['Clear Aperture of Unmounted Lens', 'S2'].item() / 2


def E_gaussian(x, y, w_0):
    return np.exp(-(x**2 + y**2)/w_0**2)


def E_gaussian_circular(q, w_0, k, z=0, n=1):
    z = np.atleast_1d(z)
    with np.errstate(divide='ignore', invalid='ignore'):
        λ = 2 * np.pi / k
        z_0 = np.pi * w_0**2 * n / λ
        w = w_0 * np.sqrt(1 + z**2 / z_0**2)
        R = z * (1 + z_0**2 / z**2)
    if np.size(z) > 1:
        R[np.isnan(R)] = np.inf
    elif np.isnan(R):
        R = np.inf

    return w_0 / w * np.exp(
        -1j * (k * z - np.arctan(z / z_0))
        - q**2 * (1 / w**2 + 1j * k / (2 * R))
    )


def E_airy_flattop(q, f_oc, w, k):
    with np.errstate(invalid='ignore'):
        E = 2 * np.pi * w**2 / f_oc * math.cexp(k * f_oc) * special.j1(x := k*w*q/f_oc) / x
    E /= np.pi * w**2 / f_oc  # normalize to 1 at center

    if np.size(E) > 1:
        E[np.isnan(E)] = 1
    elif np.isnan(E):
        E = 1
    return E


def E_spherical(q, z, k, n=1):
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp = n**2*(1 + (z/q)**2) - 1
        eta = np.hypot(1, 1 / np.sqrt(tmp))
        eta[np.isinf(eta)] = np.nan
        r = z * eta
        E = math.cexp(k * r) / r
    E *= z  # normalize to 1 at center
    E[np.isnan(E)] = 1
    return E


def _fresnel_kirchoff_integrand(rho, q, f_oc, f_ob, k, n=1):
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp = n**2*(1 + (f_ob/q)**2) - 1
        eta = np.hypot(1, 1/np.sqrt(tmp))
    if np.size(eta) > 1:
        eta[np.isinf(eta)] = 1
    elif np.isinf(eta):
        eta = 1

    amp = 1j*k/(2*f_oc*f_ob)*math.cexp(k*f_oc)
    exp = math.cexp(k*eta*f_ob)
    bes = special.j0(k*rho*q/f_oc)
    bra = 1 + eta - (n*f_ob*rho/(n**2*(rho**2 + f_ob**2) - rho**2))**2/eta
    return amp*rho*exp*bes*bra


def E_fresnel_kirchoff(q, f_oc, f_ob, w, k, n=1):
    return integrate.quad_vec(
        partial(_fresnel_kirchoff_integrand, q=q, f_oc=f['oc'], f_ob=f['ob'], k=k, n=n), 0, w
    )[0]


def P_cumulative(fn, q, *args):
    E = fn(q, *args)
    return integrate.cumulative_simpson(q*np.abs(E)**2, x=q, initial=0)


def mode_overlap(w0, w, f_oc, k, shift=0, lower=0, upper=np.inf):
    # https://www.rp-photonics.com/mode_matching.html
    def i1(q, w0, k):
        return 2*np.pi*q*np.abs(E_gaussian_circular(q, w0, k).squeeze())**2

    def i2(q, w, f_oc, k):
        return 2*np.pi*q*np.abs(E_airy_flattop(q, f_oc, w, k).squeeze())**2

    def i3(q, w0, w, f_oc, k):
        return (
            2*np.pi*q*E_gaussian_circular(q, w0, k)*E_airy_flattop(q, f_oc, w, k)
        ).squeeze()

    I1 = integrate.quad(i1, lower, upper, (w0, k))[0]
    I2 = integrate.quad(i2, lower, upper, (w, f_oc, k,), limit=100)[0]
    I3 = integrate.quad(i3, lower, upper, (w0, w, f_oc, k), limit=100, complex_func=True)[0]
    return np.abs(I3)**2 / I1 / I2


def mode_overlap_fresnel_kirchoff(w0, w, f_oc, f_ob, k, n, N=1001):
    def i1(q, w0, k):
        return 2*np.pi*q*np.abs(E_gaussian_circular(q, w0, k)).squeeze()**2

    def i2(q, w, f_oc, f_ob, k, n):
        return 2*np.pi*q*np.abs(E_fresnel_kirchoff(q, f_oc, f_ob, w, k, n)).squeeze()**2

    def i3(q, w0, w, f_oc, f_ob, k, n):
        return (
            2*np.pi*q*E_gaussian_circular(q, w0, k)*E_fresnel_kirchoff(q, f_oc, f_ob, w, k, n)
        ).squeeze()

    # quadrature integration does not converge in a reasonable amount of time
    q = np.geomspace(1e-4, 1000*w0, N-1)
    q = np.insert(q, 0, 0)
    I1 = integrate.simpson(i1(q, w0, k), q)
    I2 = integrate.simpson(i2(q, w, f_oc, f_ob, k, n), q)
    I3 = integrate.simpson(i3(q, w0, w, f_oc, f_ob, k, n), q)
    return np.abs(I3)**2 / I1 / I2


def collection_efficiency(NA, n):
    return 0.5 * (1 - (1 - (NA / n)**2)**1.5)


# TODO: estimate decrease in overlap from lateral shift due to vibrations
# %%% Compute efficiencies
print(f"Collection efficiency: η = {collection_efficiency(lenses['ob']['NA'].item(), n):.3g}")
print(f"Mode matching overlap: η = {mode_overlap(w0, w, f['oc'], k0):.3g}")
# %%% Plot mode profile in aperture and on screen
fill_style = dict(alpha=0.5, color=RWTH_COLORS_25['black'], hatch='//')

fig, axs = plt.subplots(nrows=3, layout='constrained', figsize=(MARGINWIDTH, 3))

# radial intensity profile
ρ = np.linspace(0, 3*w, 1001)
ax = axs[0]
ax.plot(ρ[ρ <= w] / w,
        (x := abs(E_spherical(ρ, f['ob'], k0, n))**2)[ρ <= w]/x[0], color=RWTH_COLORS['blue'])
ax.plot(ρ[ρ > w] / w,
        abs(E_spherical(ρ, f['ob'], k0, n))[ρ > w]**2/x[0], '--', color=RWTH_COLORS['blue'])
ax.plot(
    ρ[ρ <= w] / w,
    (x := abs(E_gaussian_circular(ρ, beam_diameter_gaussian(MFD, f['oc'], λ0)/2, k0)))
    [ρ <= w]/x[0],
    color=RWTH_COLORS['magenta']
)
ax.plot(ρ[ρ > w] / w,
        abs(E_gaussian_circular(ρ, beam_diameter_gaussian(MFD, f['oc'], λ0)/2, k0))[ρ > w]/x[0],
        '--', color=RWTH_COLORS['magenta'])
ax.plot(ρ[ρ <= w] / w, ρ[ρ <= w] < w, color=RWTH_COLORS['green'])
ax.plot(ρ[ρ > w] / w, ρ[ρ > w] <= w, '--', color=RWTH_COLORS['green'])

ax.set_xlim(xlim := ax.get_xlim())
ax.set_ylim(ylim := ax.get_ylim())
ax.fill_between([1, xlim[1]], *ylim, **fill_style)
ax.set_xlabel(r'$\rho/w$')
ax.set_ylabel(r'$I(\rho)/I(0)$')

# diffraction pattern and gaussian mode
q = np.linspace(0, 3*w0, 1001)
ax = axs[1]
# ax.plot(q / w0, q*(x := abs(E_fresnel_kirchoff(q, f['oc'], f['ob'], w, k0, n))**2)/(x[0]*w0))
ax.plot(q / w0, q*(x := abs(E_gaussian_circular(q, w0, k0))**2)/(x[0]*w0),
        color=RWTH_COLORS['magenta'])
ax.plot(q / w0, q*(x := abs(E_airy_flattop(q, f['oc'], w, k0))**2)/(x[0]*w0),
        color=RWTH_COLORS['green'])

ax.set_xlim(xlim := ax.get_xlim())
ax.set_ylim(ylim := ax.get_ylim())
ax.set_xticks([0, 1, 2, 3], labels=[])
ax.set_ylabel(r'$\rho I(\rho)/(w_0 I(0))$')

# power included in circle of radius q
q = np.linspace(0, MFD*10, 1001)
ax = axs[2]
ax.sharex(axs[1])
# ax.plot(q / w0, (x := P_cumulative(E_fresnel_kirchoff, q, f['oc'], f['ob'], w, k0, n))/x[-1])
ax.plot(q / w0, (x := P_cumulative(E_gaussian_circular, q, w0, k0))/x[-1],
        color=RWTH_COLORS['magenta'])
ax.plot(q / w0, (x := P_cumulative(E_airy_flattop, q, f['oc'], w, k0))/x[-1],
        color=RWTH_COLORS['green'])

ax.set_xlim(xlim)
ax.set_ylim(ylim := ax.get_ylim())
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel(r'$\rho/w_0$')
ax.set_ylabel(r'$P(\rho)/P(\infty)$')

fig.savefig(SAVE_PATH / 'modes_1d.pdf')
