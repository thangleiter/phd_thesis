# %% Imports
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from qutil import const, misc
from qutil.plotting import colors
from qcodes.dataset import initialise_or_create_database_at
from mjolnir.helpers import save_to_hdf5
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import MAINSTYLE, PATH, init  # noqa

EXTRACT_DATA = False
DATA_PATH = PATH.parent / 'data/rejection'
DATA_PATH.mkdir(exist_ok=True)
ORIG_DATA_PATH = pathlib.Path(
    r"\\janeway\User AG Bluhm\Common\GaAs\PL Lab\Data\Triton\db\Fig_F10.db"
)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

with misc.filter_warnings(action='ignore', category=RuntimeWarning):
    SEQUENTIAL_CMAP = colors.make_sequential_colormap('yellow', endpoint='black').reversed()

init(MAINSTYLE, backend := 'pgf')
# %% Parameters
η_QE = 0.65  # SPCM, optimistic estimate neglecting losses at the 50:50 BS and fiber coupling.
η_O = 2.2e-2  # Optical losses up to the spectrometer exit port, see extraction.py
# %% Functions


def count_rate_to_extinction_ratio(cr, λ, P_0):
    power = cr * const.lambda2eV(λ) * const.e
    power_corrected = power / (η_O * η_QE)
    return P_0 / power_corrected


def sanitize(ds):
    coords = {coord: ds[coord] - ds[coord].mean() for coord in ds.coords}
    coords['rotators_axis_1_position'].attrs = ds['rotators_axis_1_position'].attrs
    coords['rotators_axis_2_position'].attrs = ds['rotators_axis_2_position'].attrs

    coords['rotators_axis_1_position'].attrs['name'] += '_shifted'
    coords['rotators_axis_2_position'].attrs['name'] += '_shifted'

    coords['rotators_axis_1_position'].attrs['long_name'] = r'$\Delta\phi_{\lambda/2}$'
    coords['rotators_axis_2_position'].attrs['long_name'] = r'$\Delta\phi_{\lambda/4}$'

    ctxts = json.loads(ds.attrs['custom_metadata'])['measurement_parameter_contexts']
    extinction = count_rate_to_extinction_ratio(ds.count_rate,
                                                ctxts['excitation_path_wavelength'] * 1e-9,
                                                ctxts['excitation_path_power_at_sample'])
    extinction.name = 'extinction_ratio'
    extinction.attrs['full_name'] = 'Extinction ratio'

    return ds.assign(extinction_ratio=extinction).assign_coords(**coords)


def rotated_vertex_paraboloid(coords, a, b, h, k, c, theta):
    """
    Rotated Vertex Form of a 2D Paraboloid.

    Parameters
    ----------
    coords:
        tuple of arrays (x, y)
    a :
        Curvature along the rotated x' axis
    b :
        Curvature along the rotated y' axis
    h :
        Translation along the x-axis
    k :
        Translation along the y-axis
    c :
        Vertical shift
    theta :
        Rotation angle in radians
    """
    x, y = coords
    cos_theta = np.cos(np.pi/4 + theta)
    sin_theta = np.sin(np.pi/4 + theta)

    x_shift = x - h
    y_shift = y - k

    x_rot = x_shift * cos_theta - y_shift * sin_theta
    y_rot = x_shift * sin_theta + y_shift * cos_theta

    return a * x_rot**2 + b * y_rot**2 + c


# %% Load data
if EXTRACT_DATA:
    initialise_or_create_database_at(ORIG_DATA_PATH)
    save_to_hdf5(66, DATA_PATH / 'rejection_vs_angles.h5')

ds = sanitize(xr.load_dataset(DATA_PATH / 'rejection_vs_angles.h5'))
# %% Fit paraboloid
p0 = {
    'A': 1,
    'a': 1e-4,
    'b': 1e-5,
    'h': 0,
    'k': 0,
    'c': 1/ds.extinction_ratio.max(),
    'theta': 0
}
bounds = {
    'A': (0, np.inf),
    'a': (-np.inf, np.inf),
    'b': (-np.inf, np.inf),
    'h': (-np.inf, np.inf),
    'k': (-np.inf, np.inf),
    'c': (-np.inf, np.inf),
    'theta': (-np.pi/4, np.pi / 4)
}
ds_fit = ds.extinction_ratio.curvefit(
    ds.coords,
    lambda x, A, *args: A/rotated_vertex_paraboloid(x, *args).ravel(),
    param_names=list(p0), p0=p0, bounds=bounds
)

popt = ds_fit.curvefit_coefficients
pcov = ds_fit.curvefit_covariance
print(f"a = {popt.sel(param='a'):.2g}")
print(f"b = {popt.sel(param='b'):.2g}")
print(f"θ = {np.rad2deg(popt.sel(param='theta')):.2g}º")
# %% Plot
levels = np.log10(np.geomspace(ds.extinction_ratio.min(), ds.extinction_ratio.max(), 9))
levels = levels.round(1)[:-1]

fig = plt.figure()
grid = ImageGrid(fig, 111, (1, 2), axes_pad=0.225)

ax = grid.axes_all[0]
ax.axline((-0.5, -0.5), (0.5, 0.5), ls='--', color=colors.RWTH_COLORS_25['black'])
cs = np.log10(ds.extinction_ratio).plot.contour(ax=ax, levels=levels, add_colorbar=False,
                                                cmap=SEQUENTIAL_CMAP)
ax.set_xlabel(xlabel := ax.get_xlabel().replace('[', '(').replace(']', ')'))
ax.set_ylabel(ylabel := ax.get_ylabel().replace('[', '(').replace(']', ')'))
ax.set_xticks([-0.5, 0, 0.5])
ax.set_yticks([-0.5, 0, 0.5])
ax.clabel(cs)

ax = grid.axes_all[1]
ax.axline((-0.5, -0.5), (0.5, 0.5), ls='--', color=colors.RWTH_COLORS_25['black'])
cs = ax.contour(
    *ds.coords.values(),
    np.log10(
        ds_fit.curvefit_coefficients[0]
        / rotated_vertex_paraboloid(ds.coords.values(), *ds_fit.curvefit_coefficients[1:])
    ),
    levels=levels,
    cmap=SEQUENTIAL_CMAP
)
ax.set_xlabel(xlabel)
ax.set_ylabel(None)
ax.set_xticks([-0.5, 0, 0.5])
ax.set_yticks([-0.5, 0, 0.5])
ax.clabel(cs)

fig.savefig(SAVE_PATH / 'rejection.pdf')
