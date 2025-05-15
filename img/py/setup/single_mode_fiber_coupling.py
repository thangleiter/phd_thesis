# %% Imports
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker, colors as mcolors
from matplotlib.lines import Line2D
from qutil import const
from qutil.functools import chain, wraps
from qutil.misc import filter_warnings
from qutil.plotting import colors
from qutil.plotting.colors import RWTH_COLORS
from scipy import integrate

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, PATH, TEXTWIDTH, TOTALWIDTH, init,  # noqa
                    markerprops)

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
λ = 800e-6
MFD = 5e-3
AR_COATING = 'B'
# %% Functions


def sm_fiber_angle(MFD, λ):
    return sm_fiber_divergence_angle(MFD, λ)


def sm_fiber_divergence_angle(D, λ):
    # https://www.sukhamburg.com/support/technotes/fiberoptics/coupling/collimatingsm/divergence.html
    # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14489
    # Far field, z >> z_R
    return np.arctan(2*λ/(np.pi*D))


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
        MFD, λ
    )
]
available_lenses[[('Effective Focal Length', 'Effective Focal Length'),
                  ('Working Distance', 'Mounted'),
                  ('Working Distance', 'Unmounted'),
                  ('NA', 'NA')]].sort_values(by=('NA', 'NA'))

# %% Matching beam diameter and focal length
f = np.linspace(0, 30, 1001)
D = np.linspace(0, beam_diameter_geom(f[-1], np.sin(sm_fiber_angle(MFD, λ))), 1001)
DD, ff = np.meshgrid(D, f)

with mpl.style.context(MAINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained')

    with np.errstate(divide='ignore', invalid='ignore'):
        NA = lens_na_geom(DD, ff)
        NA = np.ma.masked_where(NA >= np.sin(sm_fiber_angle(MFD, λ)), NA)
    im1 = ax.contourf(ff, DD, np.rad2deg(np.arcsin(NA)),
                      levels=np.linspace(0, np.rad2deg(sm_fiber_angle(MFD, λ)), 11),
                      # cmap='viridis')
                      cmap=SEQUENTIAL_CMAPS['magenta'])

    with np.errstate(divide='ignore', invalid='ignore'):
        DL = diffraction_limit(DD, ff, λ)
        DL = np.ma.masked_where(DL >= MFD, DL)*1e3
    im2 = ax.contourf(ff, DD, DL, levels=np.linspace(0, MFD*1e3, 11),
                      # cmap='inferno')
                      cmap=SEQUENTIAL_CMAPS['green'])

    ax.plot(optimal_focal_length(D, MFD, λ), D, color='k', label='Optimal focal length')

    cbar2 = fig.colorbar(im2, location='top', aspect=20 * const.golden)
    cbar1 = fig.colorbar(im1, pad=0.0)

    cax = cbar1.ax.secondary_yaxis(9, functions=(chain(np.deg2rad, np.sin),
                                                 chain(np.arcsin, np.rad2deg)))
    # plt.pause(0.1)  # ticks aren't updated otherwise
    cbar1.set_ticks(
        cbar1.get_ticks().tolist()[:-1] + [np.rad2deg(sm_fiber_angle(MFD, λ))],
        labels=[
            rf'$\mathdefault{{{t:.1f}^\circ}}$' for t in cbar1.get_ticks().tolist()[:-1]
        ] + [r'$\mathdefault{\theta_\mathrm{SM}}$']
    )

    cbar2.set_ticks(cbar2.get_ticks().tolist()[:-1] + [MFD*1e3],
                    labels=cbar2.ax.get_xticklabels()[:-1] + [plt.Text(1, MFD*1e3, 'MFD')])
    ax.set_xlabel('$f$ (mm)')
    ax.set_ylabel('$D$ (mm)')
    cbar1.set_label('Lens acceptance angle')
    cbar2.set_label('Spot size diffraction limit (μm)')
    cax.set_ylabel(r'$\mathrm{NA}$')

    ax.set_xscale('linear')
    ax.set_yscale('linear')

    fig.savefig(SAVE_PATH / 'matching.pdf')

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
                       fill_cbar: bool = True, cbar_loc: str = 'bottom'):

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
                    D0 = beam_diameter_gaussian(MFD, z=f, λ=λ)
                    D1 = np.nanmin((D0, S1, S2))
                    if not np.isnan([S1, S2]).any():
                        D2 = D1 / min(S1, S2) * max(S1, S2)
                    else:
                        D2 = D1
                    D = beam_diameter_gaussian(D2, z=1.5e3, λ=λ)
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

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, layout='constrained',
                             figsize=(TOTALWIDTH, TOTALWIDTH / const.golden / 1.4))

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

    cs1 = ax.contour(ff, DD, diffraction_limit(DD, ff, λ)*1e3, levels=np.linspace(0, 5.5, 12),
                     cmap=SEQUENTIAL_CMAPS['green'])

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('$f$ (mm)')
    ax.set_ylabel('$D$ (mm)')
    ax.grid(True, which='both', axis='both', alpha=0.5)

    handles1, labels1 = plot_markers(ax, gaussian=True)

    ax.set_ylim(ax.get_ylim())
    ax.plot(f, collimated_beam_diameter_gaussian(f, MFD, λ, z=1.5e3), ls='--', color='tab:grey')

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
        cbar2.set_label('Lens NA')

    axes[0].clabel(cs1, manual=((lens_focal_length_diffraction(D_max + 0.5, d*1e-3, λ),
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
                     loc='upper left', bbox_to_anchor=(1.0, 1.25), borderaxespad=0., ncol=2,
                     frameon=False, title='Lens codes')

    return fig, axes, leg


# %%% Plot
fig, axes, leg = plot_lens_choosing(df_all, ylog=False, fill_cbar=True, cbar_loc='top', D_min=1.25)
fig.savefig(SAVE_PATH / 'choosing.pdf')

# %% Light emission from a point-like dipole in GaAs (820 nm)
n_GaAs = 3.59  # 4K, see Wu 2023
n_AlGaAs = 3.38
d = np.array([10, 90, 10])*1e-6
dx = 0
# WD = 2.1e-3
# CA = 5.5e-3
WD = df_all.loc['354330']['Working Distance']['Mounted']
CA = df_all.loc['354330']['Clear Aperture of Unmounted Lens']['S1']
NA = df_all.loc['354330']['NA']['NA']
T = 0.7
T = 0.7


def δ(ρ, z):
    # Acceptance cone for lens
    return np.arctan(ρ/z)


def α(δ, n):
    # Approximates source of point-dipole dz below surface of the device.
    # Good since:
    #   - QW and cap << barrier thickness ==> approximately constant angle of emission along z
    #   - HS thickness << WD ==> source is basically point-like from that POV
    return np.arcsin(np.sin(δ)/n)


def η(α):
    # Fraction of light emitted into solid half angle α
    return (np.sin(α) + np.sin(3*α)) / 16


print("Theoretical maximum collection efficiency by objective lens: "
      f"{η(α(δ(CA/2, WD), n_GaAs))*100*T:.2g}%")


# %% Coupling dipole-light into a single mode fiber
# Everything is in units of the mode field radius MFD/2

MFD = 5e-3
# D = df_all.loc['354330']['Clear Aperture of Unmounted Lens']['S2']/MFD*2
# f = df_all.loc['354330']['Effective Focal Length']['Effective Focal Length']/MFD*2
D = 5/MFD*2
f = 3.1/MFD*2
magnification = D/2
ρ = np.linspace(-D*2, D*2, 1001)
ρp = ρ/magnification

aperture = (-1, 1)
# aperture = (-np.inf, np.inf)


def normalized(lower=-np.inf, upper=np.inf):
    def wrap(func):
        @wraps(func)
        def wrapped(x, *args):
            def abs2(x, *args):
                return np.square(func(x, *args))
            norm, _ = np.sqrt(integrate.quad(abs2, lower, upper, args))
            return func(x, *args) / norm
        return wrapped
    return wrap


def magnified(magnification=1):
    def wrap(func):
        @wraps(func)
        def wrapped(x, *args):
            return func(x*magnification, *args)
        return wrapped
    return wrap


def truncated(lower=-np.inf, upper=np.inf):
    def wrap(func):
        @wraps(func)
        def wrapped(x, *args):
            result = func(x, *args)
            if np.isscalar(x):
                return result if lower <= x <= upper else 0
            if (mask := (x < lower) & (x > upper)).any():
                result[mask] = 0
            return result
        return wrapped
    return wrap


# @normalized(*aperture)
@truncated(*aperture)
@magnified(magnification)
def radial_profile_dipole(ρ, f, n, d):
    ξ = ρ/f
    l = d/np.sqrt(1 - (ξ/n)**2/(1 + ξ**2)) + f*np.sqrt(1 + ξ**2)
    return np.sqrt(n**2 - ξ**2/(1 + ξ**2)) / (l*n)


# @normalized()
def radial_profile_gaussian(ρ, ω):
    return np.exp(-(ρ/ω)**2)


def overlap(ρ, ω, f, n, d, lower=-np.inf, upper=np.inf):
    # https://www.rp-photonics.com/mode_matching.html
    def integrand1(ρ, ω, f, n, d):
        return radial_profile_dipole(ρ, f, n, d)*radial_profile_gaussian(ρ, ω)
    def integrand2(ρ, f, n, d):
        return radial_profile_dipole(ρ, f, n, d)**2
    def integrand3(ρ, ω):
        return radial_profile_gaussian(ρ, ω)**2
    return (integrate.quad(integrand1, lower, upper, (ω, f, n, d))[0]**2
            / integrate.quad(integrand2, lower, upper, (f, n, d))[0]
            / integrate.quad(integrand3, lower, upper, (ω,))[0])


dipole = (radial_profile_dipole(ρp, f, n_GaAs, d.sum())
          / radial_profile_dipole(0, f, n_GaAs, d.sum()))
gaussian = radial_profile_gaussian(ρp, 1)
integral_str = r'$\eta=\frac{\left|\int\mathrm{d}\rho\,E_\mathrm{G}(\rho)E_\mathrm{D}(\rho)\right|^2}{\int\mathrm{d}\rho\,\left|E_\mathrm{G}(\rho)\right|^2\int\mathrm{d}\rho\,\left|E_\mathrm{D}(\rho)\right|^2}$'

fig, ax = plt.subplots()
ax.axhline(1/np.exp(1), ls='--', color='tab:grey')
ax.axvline(-1, ls='--', color='tab:grey', label='Aperture\nMode field')
ax.axvline(+1, ls='--', color='tab:grey')
ax.plot(ρp, dipole, label='Dipole')
ax.plot(ρp, gaussian, label='Gaussian')
ax.set_xlabel(r'Radial distance $\rho/\omega_0$')
ax.set_ylabel(r'Electric field $E(\rho/\omega_0)/E_0$')
ax.set_title(f'{integral_str} = {overlap(ρp, 1, f, n_GaAs, d.sum()):.2g}')
ax.legend()
