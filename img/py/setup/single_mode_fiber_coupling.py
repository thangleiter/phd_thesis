# %% Imports
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker, colors as mcolors
from matplotlib.lines import Line2D
from qutil.misc import filter_warnings
from qutil.plotting import colors
from qutil.plotting.colors import RWTH_COLORS

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, PATH, TEXTWIDTH, TOTALWIDTH, init,  # noqa
                    markerprops, n_GaAs)

DATA_PATH = PATH.parent / 'data/lenses'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)
SAVE_PATH_PGF = PATH / 'pgf/setup'
SAVE_PATH_PGF.mkdir(exist_ok=True)

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


def diffraction_limit(D, f, λ):
    return 4*λ*f/(np.pi*D)


def rayleigh_range(MFD, λ):
    return np.pi/λ*(MFD/2)**2


def beam_diameter_gaussian(MFD, z, λ):
    return MFD * np.sqrt(1 + (z/rayleigh_range(MFD, λ))**2)


def collimated_beam_diameter_gaussian(f, MFD, λ, z=0):
    return beam_diameter_gaussian(beam_diameter_gaussian(MFD, f, λ), z, λ)


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
            if gaussian:
                with filter_warnings(action='ignore', category=RuntimeWarning):
                    # Collimated beam diameter at the lens
                    D0 = beam_diameter_gaussian(MFD, z=f, λ=λ0)
                    # Clip by the smallest aperture
                    D1 = np.nanmin((D0, S1, S2))
                    if not np.isnan([S1, S2]).any():
                        # Scale the beam by the ratio of the apertures
                        D2 = D1 / min(S1, S2) * max(S1, S2)
                    else:
                        D2 = D1
                    # Account for beam divergence at the objective plane
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
                           edgecolors=edgecolors, alpha=0.85)
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
    fig.savefig(SAVE_PATH_PGF / 'choosing.pgf')
